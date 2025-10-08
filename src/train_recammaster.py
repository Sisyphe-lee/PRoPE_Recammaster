"""
Train ReCamMaster With PRoPE Attention
"""
import copy
import os
import torch, os, imageio, argparse
from torchvision.transforms import v2
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
import time
import csv


from src.wandb_module import WandBVideoLogger, VideoDecoder
from src.dataset import create_datasets


def set_global_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        latent_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        ckpt_type="recammaster",
        enable_cam_layers=False,
        wandb_video_strategy="selective",  # "none", "selective", "quality_based", "all"
        wandb_max_videos_per_epoch=5,
        wandb_video_quality_threshold=25.0,  # PSNR threshold for quality-based selection
        wandb_compress_videos=True,
        wandb_video_fps=4,  # Reduced FPS for WandB uploads
        wandb_video_scale=0.5,  # Scale factor for WandB videos (0.5 = half resolution)
        global_seed=42,
        enable_test_step=False,
        test_samples=10,
        test_inference_steps=5,
        t_highfreq_ratio=0.0,
        frame_downsample_to=0,
    ): 
        super().__init__()
        self.latent_path = latent_path
        self.global_seed = global_seed
        self.enable_test_step = enable_test_step
        self.test_samples = test_samples
        self.test_inference_steps = test_inference_steps
        self.test_dataset = None  # Will be set later
        self.t_highfreq_ratio = t_highfreq_ratio
        self.frame_downsample_to = frame_downsample_to
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        models_to_load = [vae_path]
        if os.path.isfile(dit_path):
            models_to_load.append(dit_path)
        else:
            dit_path = dit_path.split(",")
            models_to_load.extend(dit_path)
        model_manager.load_models(models_to_load)
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.train_timesteps = 1000
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)

        # Store parameters for later use
        self.ckpt_type = ckpt_type
        self.enable_cam_layers = enable_cam_layers
        # Only inject camera layers if enabled (for ReCamMaster training)
        if self.enable_cam_layers:
            dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            for block in self.pipe.dit.blocks:
                block.cam_encoder = nn.Linear(12, dim)
                block.projector = nn.Linear(dim, dim)
                block.cam_encoder.weight.data.zero_()
                block.cam_encoder.bias.data.zero_()
                block.projector.weight = nn.Parameter(torch.eye(dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))
                # Set enable_cam_layers to True so the forward method uses projector
                block.enable_cam_layers = True
        
        if resume_ckpt_path is not None:
            print(f"Loading checkpoint from: {resume_ckpt_path}")
            print(f"Checkpoint type: {self.ckpt_type}")
            
            if self.ckpt_type == "wan21":
                # Check if it's a .ckpt file (PyTorch format) or .safetensors file
                if resume_ckpt_path.endswith('.ckpt'):
                    # Load PyTorch checkpoint from previous wan21 training
                    print("Loading wan21 checkpoint from PyTorch format...")
                    state_dict = torch.load(resume_ckpt_path, map_location="cpu")
                    
                    # Handle different checkpoint formats
                    if "state_dict" in state_dict:
                        # Lightning checkpoint format
                        model_state_dict = state_dict["state_dict"]
                        # Remove 'pipe.dit.' prefix if present
                        dit_state_dict = {}
                        for k, v in model_state_dict.items():
                            if k.startswith("pipe.dit."):
                                dit_state_dict[k[9:]] = v  # Remove 'pipe.dit.' prefix
                        self.pipe.dit.load_state_dict(dit_state_dict, strict=False)
                    else:
                        # Direct state dict
                        self.pipe.dit.load_state_dict(state_dict, strict=False)
                else:
                    # Load Wan2.1 original model - use safetensors format
                    from safetensors.torch import load_file
                    state_dict = load_file(resume_ckpt_path)
                    print("Loading Wan2.1 original model from safetensors...")
                    self.pipe.dit.load_state_dict(state_dict, strict=True)
            else:
                # Load ReCamMaster fine-tuned model - use torch format
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
        # Only set camera layers as trainable if they are enabled
        if self.enable_cam_layers:
            for name, module in self.pipe.denoising_model().named_modules():
                if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                    # print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            # If camera layers are not enabled, only make self_attn trainable
            for name, module in self.pipe.denoising_model().named_modules():
                if "self_attn" in name:
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



    def decode_video(self, noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch):
        """Decode video and calculate PSNR without saving"""
        # Use VideoDecoder to handle the complex decoding logic
        psnr_value, combined_frames, metadata = self.video_decoder.decode_and_create_combined_video(
            noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch
        )
        
        return psnr_value, combined_frames, metadata
    
    def save_video_with_naming(self, combined_frames, batch, video_type="val"):
        """Save video with appropriate naming based on type"""
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
        
        # Create output path based on video type
        os.makedirs(self.latent_path, exist_ok=True)
        if video_type == "val":
            video_path = os.path.join(
                self.latent_path, 
                f"step{self.global_step}_Scene{scene_id}_S{condition_cam_type}_T{target_cam_type}.mp4"
            )
        elif video_type == "test":
            video_path = os.path.join(
                self.latent_path, 
                f"test_epoch{self.current_epoch}_Scene{scene_id}_S{condition_cam_type}_T{target_cam_type}.mp4"
            )
        else:
            raise ValueError(f"Unknown video_type: {video_type}")
        
        # Compress frames for local storage (optimized for storage)
        local_scale = 0.5   # 1/2 resolution
        frame_skip = 2      # Skip every other frame (1/2 frames)
        local_fps = 4       # Reduced FPS from 8 to 4
        local_quality = 4   # Lower quality for smaller file size
        
        compressed_combined_frames = []
        for i, frame in enumerate(combined_frames):
            # Skip frames for further compression
            if i % frame_skip != 0:
                continue
                
            # Resize frame using PIL for better quality
            h, w = frame.shape[:2]
            new_h, new_w = int(h * local_scale), int(w * local_scale)
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            compressed_frame = np.array(frame_pil)
            compressed_combined_frames.append(compressed_frame)
        
        # Save compressed video
        imageio.mimsave(video_path, compressed_combined_frames, fps=local_fps, quality=local_quality)
        print(f"Saved {video_type} video: {video_path}")
        
        return video_path
        
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

        # Optional external frame downsampling (two-halves: take base then base+per_half)
        if isinstance(self.frame_downsample_to, int) and self.frame_downsample_to > 0:
            F_total = latents.shape[2]
            halves = 2
            per_half = F_total // halves
            base = torch.linspace(0, per_half - 1, steps=self.frame_downsample_to, device=self.device, dtype=torch.float32).round().long()
            index_full = torch.cat([base, base + per_half], dim=0)
            # apply to latents (B, C, F, H, W)
            latents = latents.index_select(2, index_full)
            # sync cam_emb if its N matches F_total or per_half
            if cam_emb is not None and cam_emb.dim() >= 2:
                if cam_emb.shape[1] == F_total:
                    cam_emb = cam_emb.index_select(1, index_full.to(cam_emb.device))
                elif cam_emb.shape[1] == per_half:
                    cam_emb = cam_emb.index_select(1, base.to(cam_emb.device))

        # Loss
        self.pipe.device = self.device
        # Ensure training timesteps are set (validation/test may change it)
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)
        # Use deterministic generators for reproducibility
        gen_cuda = torch.Generator(device=self.device).manual_seed(self.global_seed + self.global_step)
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=gen_cuda)
        gen_cpu = torch.Generator(device='cpu').manual_seed(self.global_seed + self.global_step)
        tlen = len(self.pipe.scheduler.timesteps)
        timestep_idx = torch.randint(0, tlen, (1,), generator=gen_cpu)
        timestep = self.pipe.scheduler.timesteps[timestep_idx].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss with model-internal downsampling; match targets by selecting same indices
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            t_highfreq_ratio=self.t_highfreq_ratio,
            frame_downsample_to=self.frame_downsample_to,
        )

        # Build per-half indices to match model's internal downsampling (two-halves scheme on target half)
        if isinstance(self.frame_downsample_to, int) and self.frame_downsample_to > 0 and self.frame_downsample_to < tgt_latent_len:
            base_indices = torch.linspace(0, tgt_latent_len - 1, steps=self.frame_downsample_to, device=self.device, dtype=torch.float32).round().long()
            tgt_sel = training_target[:, :, base_indices, ...]
            pred_sel = noise_pred[:, :, :base_indices.numel(), ...]
            loss = torch.nn.functional.mse_loss(pred_sel.float(), tgt_sel.float())
        else:
            loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, :tgt_latent_len, ...].float(),
                training_target[:, :, :tgt_latent_len, ...].float()
            )
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

        
        # Multi-step denoising in validation (like test_step)
        tgt_latent_len = latents.shape[2] // 2
        target_latents = latents[:, :, :tgt_latent_len, ...]
        condition_latents = latents[:, :, tgt_latent_len:, ...]
        
        # External frame downsampling at start (two-halves scheme). Apply to latents and cam_emb.
        frame_downsample_to = getattr(self, 'frame_downsample_to', 0)
        base_indices = None
        if isinstance(frame_downsample_to, int) and frame_downsample_to > 0:
            per_half = tgt_latent_len
            base_indices = torch.linspace(0, per_half - 1, steps=frame_downsample_to, device=self.device, dtype=torch.float32).round().long()
            # Apply to target/condition
            target_latents = target_latents.index_select(2, base_indices)
            condition_latents = condition_latents.index_select(2, base_indices)
            # Apply to cam_emb
            if cam_emb is not None and cam_emb.dim() >= 2:
                if cam_emb.shape[1] == per_half * 2:
                    idx_full = torch.cat([base_indices, base_indices + per_half], dim=0)
                    cam_emb = cam_emb.index_select(1, idx_full.to(cam_emb.device))
                elif cam_emb.shape[1] == per_half:
                    cam_emb = cam_emb.index_select(1, base_indices.to(cam_emb.device))
            # Update target half length
            tgt_latent_len = base_indices.numel()
        
        # Deterministic seed per step/batch (use downsampled target shape)
        val_seed = self.global_seed + self.global_step + batch_idx
        noise = self.pipe.generate_noise(
            target_latents.shape,
            seed=val_seed,
            device=self.device,
            dtype=torch.float32
        ).to(dtype=self.pipe.torch_dtype, device=self.device)

        # Use multi-step scheduler
        self.pipe.scheduler.set_timesteps(self.test_inference_steps, denoising_strength=1.0)
        latents_gen = noise
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            latents_input = torch.cat([latents_gen, condition_latents], dim=2)
            extra_input = self.pipe.prepare_extra_input(latents_input)
            noise_pred = self.pipe.denoising_model()(
                latents_input,
                timestep=timestep,
                cam_emb=cam_emb,
                **prompt_emb,
                **extra_input,
                **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                t_highfreq_ratio=self.t_highfreq_ratio,
                frame_downsample_to=self.frame_downsample_to,
            )
            # With external downsampling, model outputs already match target temporal length
            pred_tgt_full = noise_pred[:, :, :tgt_latent_len, ...]
            latents_gen = self.pipe.scheduler.step(
                pred_tgt_full,
                self.pipe.scheduler.timesteps[progress_id],
                latents_input[:, :, :tgt_latent_len, ...]
            )
        
        # Decode video and calculate PSNR (reuse decode logic)
        dummy_timestep = torch.tensor([0], device=self.device, dtype=self.pipe.torch_dtype)
        
        noisy_latents = torch.cat([latents_gen, condition_latents], dim=2)
        origin_latents = torch.cat([target_latents, condition_latents], dim=2)
        psnr_value, combined_frames, metadata = self.decode_video(
            latents_gen, noisy_latents, tgt_latent_len, origin_latents, dummy_timestep, batch
        )
        
        # Save video with validation naming
        combined_path = self.save_video_with_naming(combined_frames, batch, video_type="val")

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
                    'target_cam_type': target_cam_type,
                    'current_step': self.global_step  # Store current step in metadata
                }
                
                # Queue video using WandB logger
                self.wandb_logger.queue_video(combined_frames, metadata, psnr_value, batch_idx)
                
            except Exception as e:
                print(f"Failed to queue validation video for WandB: {e}")
        
        # Print info for all ranks about saved videos
        print(f"Rank {self.global_rank}: Saved validation videos for batch {batch_idx} at step {self.global_step}")

        # Restore training timesteps after validation to avoid affecting training_step
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)

        return {"psnr": psnr_value}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Test step using inference mode (multi-step denoising from pure noise)"""
        if not self.enable_test_step:
            return {}
            
        # Only test on rank 0 to avoid duplicate work
        if self.global_rank != 0:
            return {}
            
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
        
        # Get target and condition latents
        tgt_latent_len = latents.shape[2] // 2
        target_latents = latents[:, :, :tgt_latent_len, ...]
        condition_latents = latents[:, :, tgt_latent_len:, ...]
        
        # Reuse pipeline's noise generation and scheduler setup
        test_seed = self.global_seed + self.current_epoch + batch_idx
        noise_shape = target_latents.shape
        
        # Use pipeline's generate_noise method (reuse from call())
        noise = self.pipe.generate_noise(
            noise_shape, 
            seed=test_seed, 
            device=self.device, 
            dtype=torch.float32
        ).to(dtype=self.pipe.torch_dtype, device=self.device)
        
        # Use pipeline's scheduler setup (reuse from call())
        self.pipe.scheduler.set_timesteps(self.test_inference_steps, denoising_strength=1.0)
        
        # Multi-step denoising loop (reuse from call())
        latents = noise
        
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            
            # Prepare input latents (target + condition) - same as call()
            latents_input = torch.cat([latents, condition_latents], dim=2)
            
            # Predict noise using the same method as call()
            extra_input = self.pipe.prepare_extra_input(latents_input)
            noise_pred = self.pipe.denoising_model()(
                latents_input, 
                timestep=timestep, 
                cam_emb=cam_emb, 
                **prompt_emb, 
                **extra_input, 
                **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                frame_downsample_to=self.frame_downsample_to,
            )
            
            # Scheduler step (same as call())
            latents = self.pipe.scheduler.step(
                noise_pred[:,:,:tgt_latent_len,...], 
                self.pipe.scheduler.timesteps[progress_id], 
                latents_input[:,:,:tgt_latent_len,...]
            )
        
        # Reuse decode logic for consistency with validation_step
        # Prepare inputs for decode_video_only
        dummy_timestep = torch.tensor([0], device=self.device, dtype=self.pipe.torch_dtype)
        
        # Create noisy_latents format: [target_latents, condition_latents] 
        noisy_latents = torch.cat([latents, condition_latents], dim=2)
        
        # Create origin_latents format: [target_latents, condition_latents]
        origin_latents = torch.cat([target_latents, condition_latents], dim=2)
        
        # Decode video and calculate PSNR (same as validation_step)
        test_psnr, combined_frames, metadata = self.decode_video(
            latents, noisy_latents, tgt_latent_len, origin_latents, dummy_timestep, batch
        )
        
        # Save video with test naming (only first 3 samples)
        if batch_idx < 3:
            try:
                self.save_video_with_naming(combined_frames, batch, video_type="test")
            except Exception as e:
                print(f"Failed to save test video: {e}")
        
        # Log test PSNR
        self.log("test_psnr", test_psnr, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_psnr": test_psnr}

    def test_dataloader(self):
        """Return test dataloader for automatic test_step execution after each epoch"""
        if not self.enable_test_step or self.test_dataset is None:
            return None
        
        # Create a subset of test dataset for testing
        test_indices = list(range(min(self.test_samples, len(self.test_dataset))))
        test_subset = torch.utils.data.Subset(self.test_dataset, test_indices)
        
        return torch.utils.data.DataLoader(
            test_subset,
            shuffle=False,
            batch_size=1,
            num_workers=4,  # Use reasonable number of workers
            persistent_workers=True
        )

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
        default=12,
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
    # Replaced metadata_file_name with metadata_path
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Absolute path to the metadata CSV file.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=42, 
        help="Number of samples to use for validation (taken from the beginning of metadata).",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        default="recammaster",
        choices=["wan21", "recammaster"],
        help="Type of checkpoint to load: 'wan21' for original Wan2.1 model, 'recammaster' for ReCamMaster fine-tuned model"
    )
    parser.add_argument(
        "--enable_cam_layers",
        action="store_true",
        help="Enable camera encoder and projector layers injection (only for ReCamMaster training)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--wandb_video_strategy",
        type=str,
        default="all",
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
    parser.add_argument(
        "--use_validation_dataset",
        action="store_true",
        default=False,
        help="Use ValidationDataset (3 scenes, 10x10 combinations each = 300 samples) instead of simple split"
    )
    parser.add_argument(
        "--num_val_scenes",
        type=int,
        default=3,
        help="Number of scenes to use for ValidationDataset"
    )
    parser.add_argument(
        "--cameras_per_scene",
        type=int,
        default=10,
        help="Number of cameras per scene for ValidationDataset"
    )
    parser.add_argument(
        "--global_seed",
        type=int,
        default=42,
        help="Global random seed for all random operations (training, validation, data loading, etc.)"
    )
    parser.add_argument(
        "--enable_test_step",
        action="store_true",
        default=False,
        help="Enable test step with inference mode (multi-step denoising) at the end of each epoch"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10,
        help="Number of random samples to test in test_step (default: 10)"
    )
    parser.add_argument(
        "--test_inference_steps",
        type=int,
        default=5,
        help="Number of inference steps for test_step (default: 5)"
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=50,
        help="Number of training steps between validation runs (default: 50)"
    )
    parser.add_argument(
        "--val_check_interval_batches",
        type=int,
        default=None,
        help="Number of batches between validation runs (overrides val_check_interval if set)"
    )
    parser.add_argument(
        "--t_highfreq_ratio",
        type=float,
        default=0.0,
        help="Temporal low-frequency masking ratio for self-attention (passed via **kwargs to blocks)"
    )
    parser.add_argument(
        "--frame_downsample_to",
        type=int,
        default=0,
        help="Per-half frames to sample (two-halves scheme). Use 0 to disable downsampling (default: 0)"
    )

    args = parser.parse_args()
    return args



    
    
def train(args):
    # Set global seed for reproducibility
    set_global_seed(args.global_seed)
    print(f"Global seed set to: {args.global_seed}")
    
    if args.debug:
        print("Debug mode is enabled.") 
        import debugpy
        debugpy.listen(6862)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    # Create datasets using the new create_datasets function

    train_dataset, val_dataset = create_datasets(
        metadata_path=args.metadata_path,
        val_size=args.val_size,
        steps_per_epoch=args.steps_per_epoch,
        use_validation_dataset=args.use_validation_dataset,
        num_val_scenes=args.num_val_scenes,
        cameras_per_scene=args.cameras_per_scene,
        seed=args.global_seed
    )

    def worker_init_fn(worker_id):
        """Initialize worker with deterministic seed"""
        worker_seed = args.global_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.global_seed)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn
    )
    


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
        ckpt_type=args.ckpt_type,
        enable_cam_layers=args.enable_cam_layers,
        wandb_video_strategy=getattr(args, 'wandb_video_strategy', 'selective'),
        wandb_max_videos_per_epoch=getattr(args, 'wandb_max_videos_per_epoch', 5),
        wandb_video_quality_threshold=getattr(args, 'wandb_video_quality_threshold', 25.0),
        wandb_compress_videos=getattr(args, 'wandb_compress_videos', True),
        wandb_video_fps=getattr(args, 'wandb_video_fps', 4),
        wandb_video_scale=getattr(args, 'wandb_video_scale', 0.5),
        global_seed=args.global_seed,
        enable_test_step=args.enable_test_step,
        test_samples=args.test_samples,
        test_inference_steps=args.test_inference_steps,
        t_highfreq_ratio=getattr(args, 't_highfreq_ratio', 0.0),
        frame_downsample_to=getattr(args, 'frame_downsample_to', 5),
    )
    
    # Set test dataset for automatic test_step execution
    if args.enable_test_step:
        model.test_dataset = val_dataset
    
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_name = f"{time_str}_{args.wandb_name}"
        run_dir = os.path.join("./wandb", wandb_name)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            os.makedirs(run_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            ## project=args.wandb_project+wandb_name
            project=f"{args.wandb_project}_{wandb_name[:5]}",
            name=wandb_name,
            id=wandb_name,
            config=vars(args),
            save_dir=run_dir,
        )
        logger = [wandb_logger]
    else:
        logger = None
        run_dir = args.output_path
    # Concise timing callback to record validate/train totals and per-step averages on rank 0 only
    class ConciseTimingCallback(pl.Callback):
        def __init__(self, out_dir, train_log_every_n_batches=20):
            self.out_dir = out_dir
            self.train_log_every_n_batches = max(1, int(train_log_every_n_batches))
            os.makedirs(self.out_dir, exist_ok=True)
            self.csv_path = os.path.join(self.out_dir, "timings.csv")
            if os.environ.get("LOCAL_RANK", "0") == "0" and not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["phase", "global_step", "epoch", "count", "total_ms", "avg_ms"])  # concise schema
            # validation timers/accumulators
            self._val_t0 = None
            self._val_step_t0 = None
            self._val_step_durations = []
            # training timers/accumulators (per-epoch)
            self._train_epoch_t0 = None
            self._train_step_t0 = None
            self._train_step_durations = []
        def _is_rank0(self, trainer):
            return getattr(trainer, "global_rank", 0) == 0
        # ========== Training (per epoch totals + avg step) ==========
        def on_train_epoch_start(self, trainer, pl_module):
            self._train_epoch_t0 = time.perf_counter()
            self._train_step_durations = []
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self._train_step_t0 = time.perf_counter()
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._train_step_t0 is not None:
                dt = (time.perf_counter() - self._train_step_t0) * 1000.0
                self._train_step_durations.append(dt)
                self._train_step_t0 = None
            # running output every N batches on rank0
            if self._is_rank0(trainer) and (len(self._train_step_durations) % self.train_log_every_n_batches == 0):
                count = len(self._train_step_durations)
                avg_ms = (sum(self._train_step_durations) / count) if count > 0 else 0.0
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["train_step_avg_running", trainer.global_step, trainer.current_epoch, count, "", f"{avg_ms:.3f}"])
                if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                    try:
                        trainer.logger.experiment.log({
                            "timing/train_step_avg_ms_running": avg_ms,
                            "trainer/global_step": trainer.global_step,
                            "trainer/epoch": trainer.current_epoch,
                        })
                    except Exception:
                        pass
        def on_train_epoch_end(self, trainer, pl_module):
            if not self._is_rank0(trainer):
                return
            total_ms = None
            if self._train_epoch_t0 is not None:
                total_ms = (time.perf_counter() - self._train_epoch_t0) * 1000.0
            count = len(self._train_step_durations)
            avg_ms = (sum(self._train_step_durations) / count) if count > 0 else 0.0
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["train_epoch_total", trainer.global_step, trainer.current_epoch, count, f"{total_ms:.3f}", f"{avg_ms:.3f}"])
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                try:
                    trainer.logger.experiment.log({
                        "timing/train_epoch_total_ms": total_ms,
                        "timing/train_step_avg_ms": avg_ms,
                        "trainer/global_step": trainer.global_step,
                        "trainer/epoch": trainer.current_epoch,
                    })
                except Exception:
                    pass
        # ========== Validation (total + avg step) ==========
        def on_validation_start(self, trainer, pl_module):
            self._val_t0 = time.perf_counter()
            self._val_step_durations = []
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self._val_step_t0 = time.perf_counter()
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if self._val_step_t0 is not None:
                dt = (time.perf_counter() - self._val_step_t0) * 1000.0
                self._val_step_durations.append(dt)
                self._val_step_t0 = None
        def on_validation_end(self, trainer, pl_module):
            if not self._is_rank0(trainer):
                return
            total_ms = None
            if self._val_t0 is not None:
                total_ms = (time.perf_counter() - self._val_t0) * 1000.0
            count = len(self._val_step_durations)
            avg_ms = (sum(self._val_step_durations) / count) if count > 0 else 0.0
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["validate_total", trainer.global_step, trainer.current_epoch, count, f"{total_ms:.3f}", f"{avg_ms:.3f}"])
            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                try:
                    trainer.logger.experiment.log({
                        "timing/validate_total_ms": total_ms,
                        "timing/validation_step_avg_ms": avg_ms,
                        "trainer/global_step": trainer.global_step,
                        "trainer/epoch": trainer.current_epoch,
                    })
                except Exception:
                    pass
    concise_timing_cb = ConciseTimingCallback(out_dir=os.path.join(run_dir, "profiler"))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval_batches if args.val_check_interval_batches is not None else args.val_check_interval,  # Use batch-based or step-based validation frequency
        limit_val_batches=len(val_dataset),
        num_sanity_val_steps=0,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=-1,
                dirpath=os.path.join(run_dir, "checkpoints"),
                filename="{epoch}-{step}"
            ),
            concise_timing_cb,
        ],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=0.05,
    )
    # Run an initial validation at step 0 for debugging/baseline
    trainer.validate(model, val_dataloader)
    
    # Run an initial test at step 0 if test_step is enabled
    # if test_dataloader is not None:
    #     trainer.test(model, test_dataloader)
    
    # Lightning will automatically call test_step after each epoch if test_dataloader() returns a DataLoader
    
    # Fit the model
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    train(args)
