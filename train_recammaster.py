import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from datetime import datetime
from wandb_module import WandBVideoLogger, VideoDecoder

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, paths=None, deterministic=False, fixed_length=None):
        metadata = pd.read_csv('./metadata.csv')
        if paths is None:
            self.path = [os.path.join(file_name) for file_name in metadata["video_absolute_path"]]
            print(len(self.path), "videos in metadata.")
            self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        else:
            self.path = paths
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.deterministic = deterministic
        self.fixed_length = fixed_length


    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)


    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


    def __getitem__(self, index):
        # Return: 
        # data['latents']: torch.Size([16, 21*2, 60, 104])
        # data['camera']: torch.Size([21, 3, 4])
        # data['prompt_emb']['context'][0]: torch.Size([512, 4096])
        while True:
            try:
                data = {}
                if self.deterministic:
                    data_id = index % len(self.path)
                else:
                    data_id = torch.randint(0, len(self.path), (1,))[0]
                    data_id = (data_id + index) % len(self.path) # For fixed seed.
                path_tgt = self.path[data_id]
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                if self.deterministic:
                    cond_idx = 1 if tgt_idx != 1 else 2
                else:
                    cond_idx = random.randint(1, 10)
                    while cond_idx == tgt_idx:
                        cond_idx = random.randint(1, 10)
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                data['latents'] = torch.cat((data_tgt['latents'],data_cond['latents']),dim=1)
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb'] = {}

                # Extract scene information from path
                scene_match = re.search(r'scene(\d+)', path_tgt.lower())
                scene_id = scene_match.group(1) if scene_match else 'unknown'
                
                # Store camera type information
                data['scene_id'] = scene_id
                data['condition_cam_type'] = f"cam{cond_idx:02d}"
                data['target_cam_type'] = f"cam{tgt_idx:02d}"
                data['path'] = path_tgt  # Keep original path for reference

                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")              
                with open(tgt_camera_path, 'r') as file:
                    cam_data = json.load(file)
                multiview_c2ws = []
                cam_idx = list(range(81))[::4] 
                for view_idx in [cond_idx, tgt_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 100
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                
                # --- Start: Calculate relative camera poses ---
                # The reference pose is the first frame of the conditional camera trajectory.
                ref_cam = cond_cam_params[0]

                # Calculate relative poses for the conditional trajectory
                relative_cond_poses = []
                for i in range(len(cond_cam_params)):
                    # get_relative_pose expects a list of [reference_camera, target_camera]
                    # It returns an array where the second element is the relative pose of the target.
                    relative_pose_matrix = self.get_relative_pose([ref_cam, cond_cam_params[i]])
                    relative_cond_poses.append(torch.as_tensor(relative_pose_matrix)[1, :3, :])

                # Calculate relative poses for the target trajectory
                relative_tgt_poses = []
                for i in range(len(tgt_cam_params)):
                    relative_pose_matrix = self.get_relative_pose([ref_cam, tgt_cam_params[i]])
                    relative_tgt_poses.append(torch.as_tensor(relative_pose_matrix)[1, :3, :])

                # Stack, rearrange, and concatenate to form the final pose embedding
                cond_embedding = torch.stack(relative_cond_poses, dim=0)
                tgt_embedding = torch.stack(relative_tgt_poses, dim=0)
                cond_embedding = rearrange(cond_embedding, 'b c d -> b (c d)')
                tgt_embedding = rearrange(tgt_embedding, 'b c d -> b (c d)')
                pose_embedding = torch.cat([cond_embedding, tgt_embedding], dim=0)
                # --- End: Calculate relative camera poses ---
                
                
                data['camera'] = pose_embedding.to(torch.bfloat16)
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data
    

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        latent_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        wandb_video_strategy="selective",  # "none", "selective", "quality_based", "all"
        wandb_max_videos_per_epoch=5,
        wandb_video_quality_threshold=25.0,  # PSNR threshold for quality-based selection
        wandb_compress_videos=True,
        wandb_video_fps=4,  # Reduced FPS for WandB uploads
        wandb_video_scale=0.5  # Scale factor for WandB videos (0.5 = half resolution)
    ):
        super().__init__()
        self.latent_path = latent_path
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        models_to_load = [vae_path]
        if os.path.isfile(dit_path):
            models_to_load.append(dit_path)
        else:
            dit_path = dit_path.split(",")
            models_to_load.extend(dit_path)
        model_manager.load_models(models_to_load)
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            print(f"Loading checkpoint from: {resume_ckpt_path}")
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")

            # If the checkpoint is a deepspeed checkpoint, it might have a different format
            if "module" in state_dict:
                state_dict = state_dict["module"]

            # Strip prefix from state_dict keys if necessary
            prefixes_to_remove = ["module.", "model."]
            has_prefix = any(any(key.startswith(p) for p in prefixes_to_remove) for key in state_dict.keys())
            if has_prefix:
                print("Prefix detected in checkpoint keys. Stripping prefixes...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    for p in prefixes_to_remove:
                        if k.startswith(p):
                            k = k[len(p):]
                            break
                    new_state_dict[k] = v
                state_dict = new_state_dict
                print("Prefixes stripped.")
                self.pipe.dit.load_state_dict(state_dict, strict=False)
            else:
                self.pipe.dit.load_state_dict(state_dict, strict=True)

            
            print("Have Loaded Checkpoint")

        self.freeze_parameters()
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                # print(f"Trainable: {name}")
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.last_decode_step = -1
        self._has_started_training = False
        
        # Initialize WandB video logger
        self.wandb_logger = WandBVideoLogger(
            strategy=wandb_video_strategy,
            max_videos_per_epoch=wandb_max_videos_per_epoch,
            quality_threshold=wandb_video_quality_threshold,
            compress_videos=wandb_compress_videos,
            video_fps=wandb_video_fps,
            video_scale=wandb_video_scale,
            output_dir=os.path.join(latent_path, "wandb_temp")
        )
        
        # Initialize video decoder
        self.video_decoder = VideoDecoder(self.pipe)
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()



    def decode_video_and_log(self, noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch):
        """Simplified video decoding using the VideoDecoder module"""
        # Extract metadata for file naming
        try:
            scene_id = batch.get('scene_id', ['unknown'])[0] if isinstance(batch.get('scene_id'), list) else str(batch.get('scene_id', 'unknown'))
            condition_cam_type = batch.get('condition_cam_type', ['unknown'])[0] if isinstance(batch.get('condition_cam_type'), list) else str(batch.get('condition_cam_type', 'unknown'))
            target_cam_type = batch.get('target_cam_type', ['unknown'])[0] if isinstance(batch.get('target_cam_type'), list) else str(batch.get('target_cam_type', 'unknown'))
        except Exception as e:
            print(f"Error extracting metadata from batch: {e}")
            scene_id = "unknown"
            condition_cam_type = "unknown"
            target_cam_type = "unknown"
        
        # Create output path
        os.makedirs(self.latent_path, exist_ok=True)
        combined_video_path = os.path.join(
            self.latent_path, 
            f"step{self.global_step}_SceneID{scene_id}_SourcePoseType{condition_cam_type}_TargetPoseType{target_cam_type}.mp4"
        )
        
        # Use VideoDecoder to handle the complex decoding logic
        psnr_value, combined_frames, metadata = self.video_decoder.decode_and_create_combined_video(
            noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch, combined_video_path
        )
        
        return psnr_value, None, None, combined_video_path, combined_frames
        
    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        cam_emb = batch["camera"].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )

        # Remove ad-hoc decode/eval from training loop; validation loop will handle evaluation

        loss = torch.nn.functional.mse_loss(noise_pred[:, :, :tgt_latent_len, ...].float(), training_target[:, :, :tgt_latent_len, ...].float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=latents.shape[0])
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Store current batch_idx for video naming
        self._current_batch_idx = batch_idx
        
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        cam_emb = batch["camera"].to(self.device)

        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]

        noise_pred = self.pipe.denoising_model()(noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload)

        # Decode, compute PSNR, save videos locally
        psnr_value, pred_frames, gt_frames, combined_path, combined_frames = self.decode_video_and_log(
            noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch
        )

        # Accumulate for epoch-average
        if not hasattr(self, "_val_psnr_sum"):
            self._val_psnr_sum = 0.0
            self._val_count = 0
        self._val_psnr_sum += float(psnr_value)
        self._val_count += 1

        # Queue video for WandB upload (only rank 0)
        if self.global_rank == 0:
            try:
                # Extract metadata from batch
                scene_id = batch.get('scene_id', ['unknown'])[0] if isinstance(batch.get('scene_id'), list) else str(batch.get('scene_id', 'unknown'))
                condition_cam_type = batch.get('condition_cam_type', ['unknown'])[0] if isinstance(batch.get('condition_cam_type'), list) else str(batch.get('condition_cam_type', 'unknown'))
                target_cam_type = batch.get('target_cam_type', ['unknown'])[0] if isinstance(batch.get('target_cam_type'), list) else str(batch.get('target_cam_type', 'unknown'))
                
                metadata = {
                    'scene_id': scene_id,
                    'condition_cam_type': condition_cam_type,
                    'target_cam_type': target_cam_type
                }
                
                # Queue video using WandB logger
                self.wandb_logger.queue_video(combined_frames, metadata, psnr_value, batch_idx)
                
            except Exception as e:
                print(f"Failed to queue validation video for WandB: {e}")
        
        # Print info for all ranks about saved videos
        print(f"Rank {self.global_rank}: Saved validation videos for batch {batch_idx} at step {self.global_step}")

        return {"psnr": psnr_value}

    def on_validation_epoch_start(self):
        # Reset accumulators
        self._val_psnr_sum = 0.0
        self._val_count = 0
        
        # Reset WandB video tracking for new epoch
        if self.global_rank == 0:
            self.wandb_logger.reset_epoch()
            print(f"Rank {self.global_rank}: Starting validation epoch. WandB strategy: {self.wandb_logger.strategy}, max videos: {self.wandb_logger.max_videos_per_epoch}")
        
        # Save checkpoint during validation (except for the initial validation before training)
        if hasattr(self, '_has_started_training') and self._has_started_training and self.global_rank == 0:
            self.save_validation_checkpoint()

    def on_validation_epoch_end(self):
        # Log average PSNR across the validation set
        if getattr(self, "_val_count", 0) > 0:
            avg_psnr = self._val_psnr_sum / self._val_count
            self.log("val/psnr", avg_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True, batch_size=self._val_count)
        
        # Upload all queued videos to WandB at epoch end (only rank 0)
        if self.global_rank == 0 and hasattr(self.logger, "experiment") and self.logger is not None:
            self.wandb_logger.upload_videos(self.logger, self.global_step)

    def save_validation_checkpoint(self):
        """Save checkpoint during validation"""
        try:
            checkpoint_dir = self.trainer.checkpoint_callback.dirpath if self.trainer.checkpoint_callback else "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            current_step = self.global_step
            state_dict = self.pipe.denoising_model().state_dict()
            
            checkpoint_path = os.path.join(checkpoint_dir, f"validation_step{current_step}.ckpt")
            torch.save(state_dict, checkpoint_path)
            print(f"Saved validation checkpoint at step {current_step}: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save validation checkpoint: {e}")

    def on_train_start(self):
        """Called when training starts"""
        self._has_started_training = True
        print("Training started - validation checkpoints will be saved from now on")


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        if not (os.path.exists(os.path.join(checkpoint_dir))):
            os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=False,
        choices=["train"],
        help="Task. Only `train` is supported.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )

    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_wandb",
        default=True,
        action="store_true",
        help="Whether to use WandB logger.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ReCamMaster",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="train",
        help="WandB run name.",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--wandb_video_strategy",
        type=str,
        default="selective",
        choices=["none", "selective", "quality_based", "all"],
        help="Strategy for uploading validation videos to WandB. 'none': no upload, 'selective': first few + every Nth, 'quality_based': based on PSNR thresholds, 'all': upload all videos"
    )
    parser.add_argument(
        "--wandb_max_videos_per_epoch",
        type=int,
        default=5,
        help="Maximum number of validation videos to upload to WandB per epoch"
    )
    parser.add_argument(
        "--wandb_video_quality_threshold",
        type=float,
        default=25.0,
        help="PSNR threshold for quality-based video selection (only used with quality_based strategy)"
    )
    parser.add_argument(
        "--wandb_compress_videos",
        action="store_true",
        default=True,
        help="Whether to compress videos before uploading to WandB (reduces resolution and FPS)"
    )
    parser.add_argument(
        "--wandb_video_fps",
        type=int,
        default=4,
        help="FPS for WandB video uploads (lower FPS reduces file size)"
    )
    parser.add_argument(
        "--wandb_video_scale",
        type=float,
        default=0.5,
        help="Scale factor for WandB videos (0.5 = half resolution, reduces file size)"
    )
    args = parser.parse_args()
    return args



    
    
def train(args):
    if args.debug:
        print("Debug mode is enabled.") 
        import debugpy
        debugpy.listen(6862)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    # Build datasets: create deterministic 10-sample validation split
    full_dataset = TensorDataset(
        args.dataset_path,
        "/data1/lcy/projects/ReCamMaster/metadata_inference.csv",
        steps_per_epoch=args.steps_per_epoch,
    )
    # Use actual dataset paths, not the string value of dataset_path
    all_paths = list(full_dataset.path)
    # Deterministic first 48 samples for validation
    val_paths = all_paths[:48] if len(all_paths) >= 48 else all_paths[:len(all_paths)]
    train_paths = [p for p in all_paths if p not in val_paths]

    train_dataset = TensorDataset(
        args.dataset_path,
        "/data1/lcy/projects/ReCamMaster/metadata.csv",
        steps_per_epoch=args.steps_per_epoch,
        paths=train_paths,
        deterministic=False,
        fixed_length=None,
    )
    val_dataset = TensorDataset(
        args.dataset_path,
        "/data1/lcy/projects/ReCamMaster/metadata.csv",
        steps_per_epoch=48,
        paths=val_paths,
        deterministic=True,
        fixed_length=48 if len(val_paths) >= 48 else len(val_paths),
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    ## TODO: lantent_path = './latents/{date-time}/   if don't exist, then mkdir
    time_str = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime('%m-%d-%H%M%S'))
    folder_name = f"{time_str}_{args.wandb_name}"
    latent_path = os.path.join("./wandb", folder_name, "video_debug")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        os.makedirs(latent_path, exist_ok=True)
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        latent_path=latent_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        wandb_video_strategy=getattr(args, 'wandb_video_strategy', 'selective'),
        wandb_max_videos_per_epoch=getattr(args, 'wandb_max_videos_per_epoch', 5),
        wandb_video_quality_threshold=getattr(args, 'wandb_video_quality_threshold', 25.0),
        wandb_compress_videos=getattr(args, 'wandb_compress_videos', True),
        wandb_video_fps=getattr(args, 'wandb_video_fps', 4),
        wandb_video_scale=getattr(args, 'wandb_video_scale', 0.5),
    )

    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_name = f"{time_str}_{args.wandb_name}"
        run_dir = os.path.join("./wandb", wandb_name)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            os.makedirs(run_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=wandb_name,
            id=wandb_name,
            config=vars(args),
            save_dir=run_dir,
        )
        logger = [wandb_logger]
    else:
        logger = None
        run_dir = args.output_path
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=300,
        limit_val_batches=48,
        num_sanity_val_steps=0,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(
            save_top_k=-1,
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="{epoch}-{step}"
        )],
        logger=logger,
        log_every_n_steps=1,
    )
    # Run an initial validation at step 0 for debugging/baseline
    trainer.validate(model, val_dataloader)
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    train(args)