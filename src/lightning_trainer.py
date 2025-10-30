"""Lightning module for ReCamMaster training."""

import copy
import inspect
import os
import sys
from pathlib import Path

import imageio
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline

from src.wandb_module import WandBVideoLogger, VideoDecoder


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
        wandb_video_scale=0.5,  # Scale factor for WandB videos (0.5 = half resolution)
        global_seed=42,
        val_steps=5,
        t_highfreq_ratio=0.0,
        frame_downsample_to=0,
        use_real_temporal_indices=False,
        use_physical_index=False,
        pipeline_type="v2v",
    ): 
        super().__init__()
        if resume_ckpt_path in (None, "", "none"):
            resume_ckpt_path = None

        self.latent_path = latent_path
        self.global_seed = global_seed
        self.val_steps = val_steps
        self.t_highfreq_ratio = t_highfreq_ratio
        self.frame_downsample_to = frame_downsample_to
        self.use_real_temporal_indices = use_real_temporal_indices
        self.use_physical_index = use_physical_index
        self.pipeline_type = pipeline_type
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        models_to_load = [vae_path]
        if isinstance(dit_path, (list, tuple)):
            shard_paths = list(dit_path)
        else:
            dit_path = str(dit_path)
            shard_paths = [p.strip() for p in dit_path.split(",") if p.strip()]
        if len(shard_paths) == 1 and os.path.isfile(shard_paths[0]):
            models_to_load.append(shard_paths[0])
        else:
            models_to_load.append(shard_paths)
        model_manager.load_models(models_to_load)

        self.pipe = self._init_pipeline(model_manager, pipeline_type)
        self._ensure_pipeline_compat()
        _ = self._get_denoising_model()
        self.train_timesteps = 1000
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)
        self._generate_noise_params = set(inspect.signature(self.pipe.generate_noise).parameters.keys())

        # Store parameters for later use
        self.ckpt_type = "wan21" if pipeline_type == "v2v" else "wan22"
        if resume_ckpt_path is not None:
            print(f"Loading checkpoint from: {resume_ckpt_path}")
            print(f"Checkpoint type: {self.ckpt_type}")

            if self.ckpt_type not in {"wan21", "wan22"}:
                raise ValueError(f"Unsupported ckpt_type '{self.ckpt_type}'. Expected 'wan21' or 'wan22'.")

            if resume_ckpt_path.endswith('.ckpt'):
                print(f"Loading {self.ckpt_type} checkpoint from PyTorch format...")
                state_dict = torch.load(resume_ckpt_path, map_location="cpu")

                if "state_dict" in state_dict:
                    model_state_dict = state_dict["state_dict"]
                    dit_state_dict = {}
                    for k, v in model_state_dict.items():
                        if k.startswith("pipe.dit."):
                            dit_state_dict[k[9:]] = v
                    dit_state_dict = self._strip_cam_layer_weights(dit_state_dict)
                    self.pipe.dit.load_state_dict(dit_state_dict, strict=False)
                else:
                    state_dict = self._strip_cam_layer_weights(state_dict)
                    self.pipe.dit.load_state_dict(state_dict, strict=True)
            else:
                from safetensors.torch import load_file
                state_dict = load_file(resume_ckpt_path)
                print(f"Loading {self.ckpt_type} original model from safetensors...")
                state_dict = self._strip_cam_layer_weights(state_dict)
                self.pipe.dit.load_state_dict(state_dict, strict=True)

            print("Have Loaded Checkpoint")

        self.freeze_parameters()
        for name, module in self._get_denoising_model().named_modules():
            if "self_attn" in name:
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self._get_denoising_model().named_modules():
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
        self.video_decoder = VideoDecoder(self.pipe, pipeline_type=self.pipeline_type)
        
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
        cam_intrinsics = batch.get("intrinsics")
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.to(self.device)
        is_i2v = self.pipeline_type == "i2v"
        # Optional external frame downsampling
        temporal_indices = None
        if isinstance(self.frame_downsample_to, int) and self.frame_downsample_to > 0:
            latents, cam_emb, cam_intrinsics, selected = self._apply_frame_downsample(
                latents, cam_emb, cam_intrinsics
            )
            if selected is not None:
                temporal_indices = selected.to(self.device)
        
        # 如果启用了真实时序索引但没有降采样，使用连续索引
        if self.use_real_temporal_indices and temporal_indices is None:
            F_total = latents.shape[2]
            temporal_indices = torch.arange(F_total, device=self.device, dtype=torch.long)

        # 物理索引：将前半段索引复制到后半段，使两半不共享时间戳
        if not is_i2v and getattr(self, 'use_physical_index', False):
            per_half = latents.shape[2] // 2
            if temporal_indices is None:
                half = torch.arange(per_half, device=self.device, dtype=torch.long)
            else:
                half = temporal_indices[:per_half]
            temporal_indices = torch.cat([half, half], dim=0)

        # Loss
        self.pipe.device = self.device
        # Ensure training timesteps are set (validation/test may change it)
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)
        # Use deterministic generators for reproducibility
        gen_cuda = torch.Generator(device=self.device).manual_seed(self.global_seed + self.global_step)
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=gen_cuda)
        if is_i2v:
            noise[:, :, 0, ...] = 0
        gen_cpu = torch.Generator(device='cpu').manual_seed(self.global_seed + self.global_step)
        tlen = len(self.pipe.scheduler.timesteps)
        timestep_idx = torch.randint(0, tlen, (1,), generator=gen_cpu)
        timestep = self.pipe.scheduler.timesteps[timestep_idx].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        if is_i2v:
            tgt_latent_len = noisy_latents.shape[2]
            noisy_latents[:, :, 0, ...] = origin_latents[:, :, 0, ...]
        else:
            tgt_latent_len = noisy_latents.shape[2] // 2
            noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss with model-internal downsampling; match targets by selecting same indices
        noise_pred = self._get_denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            t_highfreq_ratio=self.t_highfreq_ratio,
            frame_downsample_to=self.frame_downsample_to,
            cam_intrinsics=cam_intrinsics,
            temporal_indices=temporal_indices
        )

        # Build per-half indices to match model's internal downsampling (two-halves scheme on target half)
        if is_i2v:

            loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, 1:, ...].float(),
                training_target[:, :, 1:, ...].float()
            )
        else:
            if isinstance(self.frame_downsample_to, int) and self.frame_downsample_to > 0 and self.frame_downsample_to < tgt_latent_len:
                base_indices = torch.linspace(
                    0, tgt_latent_len - 1, steps=self.frame_downsample_to, device=self.device, dtype=torch.float32
                ).round().long()
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
        cam_intrinsics = batch.get("intrinsics")
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.to(self.device)

        self.pipe.device = self.device

        is_i2v = self.pipeline_type == "i2v"
        # External frame downsampling (align with training_step)
        frame_downsample_to = getattr(self, 'frame_downsample_to', 0)
        temporal_indices = None
        F_total = latents.shape[2]
        if isinstance(frame_downsample_to, int) and frame_downsample_to > 0:
            latents, cam_emb, cam_intrinsics, selected = self._apply_frame_downsample(
                latents, cam_emb, cam_intrinsics
            )
            if selected is not None:
                temporal_indices = selected.to(self.device)

        if self.use_real_temporal_indices and temporal_indices is None:
            temporal_indices = torch.arange(latents.shape[2], device=self.device, dtype=torch.long)

        if not is_i2v and getattr(self, 'use_physical_index', False):
            per_half = latents.shape[2] // 2
            if temporal_indices is None:
                half = torch.arange(per_half, device=self.device, dtype=torch.long)
            else:
                half = temporal_indices[:per_half]
            temporal_indices = torch.cat([half, half], dim=0)

        if is_i2v:
            tgt_latent_len = latents.shape[2]
            target_latents = latents
            condition_latents = None
        else:
            tgt_latent_len = latents.shape[2] // 2
            target_latents = latents[:, :, :tgt_latent_len, ...]
            condition_latents = latents[:, :, tgt_latent_len:, ...]
        
        # Deterministic seed per step/batch (use downsampled target shape)
        val_seed = self.global_seed + self.global_step + batch_idx
        if "dtype" in self._generate_noise_params:
            noise = self.pipe.generate_noise(
                target_latents.shape,
                seed=val_seed,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            extra_kwargs = {}
            if "rand_torch_dtype" in self._generate_noise_params:
                extra_kwargs["rand_torch_dtype"] = torch.float32
            if "device" in self._generate_noise_params:
                extra_kwargs["device"] = self.device
            if "torch_dtype" in self._generate_noise_params:
                extra_kwargs["torch_dtype"] = self.pipe.torch_dtype
            noise = self.pipe.generate_noise(
                target_latents.shape,
                seed=val_seed,
                **extra_kwargs,
            )
        noise = noise.to(dtype=self.pipe.torch_dtype, device=self.device)
        if is_i2v:
            noise[:, :, 0, ...] = 0

        # Use multi-step scheduler
        self.pipe.scheduler.set_timesteps(self.val_steps, shift=self.pipe.scheduler.shift, denoising_strength=1.0)
        latents_gen = noise
        if is_i2v:
            latents_gen[:, :, 0, ...] = target_latents[:, :, 0, ...]
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            if is_i2v:
                latents_input = latents_gen.clone()
                latents_input[:, :, 0, ...] = target_latents[:, :, 0, ...]
            else:
                latents_input = torch.cat([latents_gen, condition_latents], dim=2)
            extra_input = self.pipe.prepare_extra_input(latents_input)
            noise_pred = self._get_denoising_model()(
                latents_input,
                timestep=timestep,
                cam_emb=cam_emb,
                cam_intrinsics=cam_intrinsics,
                temporal_indices=temporal_indices,
                **prompt_emb,
                **extra_input,
                **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                t_highfreq_ratio=self.t_highfreq_ratio,
                frame_downsample_to=self.frame_downsample_to,
            )
            if is_i2v:
                latents_gen = self.pipe.scheduler.step(
                    noise_pred,
                    self.pipe.scheduler.timesteps[progress_id],
                    latents_input,
                )
                latents_gen[:, :, 0, ...] = target_latents[:, :, 0, ...]
            else:
                pred_tgt_full = noise_pred[:, :, :tgt_latent_len, ...]
                latents_gen = self.pipe.scheduler.step(
                    pred_tgt_full,
                    self.pipe.scheduler.timesteps[progress_id],
                    latents_input[:, :, :tgt_latent_len, ...]
                )
        
        # Decode video and calculate PSNR (reuse decode logic)
        dummy_timestep = torch.tensor([0], device=self.device, dtype=self.pipe.torch_dtype)
        
        if is_i2v:
            noisy_latents = latents_gen
            origin_latents = target_latents
        else:
            noisy_latents = torch.cat([latents_gen, condition_latents], dim=2)
            origin_latents = torch.cat([target_latents, condition_latents], dim=2)
        psnr_value, combined_frames, metadata = self.decode_video(
            latents_gen,
            noisy_latents,
            tgt_latent_len,
            origin_latents,
            dummy_timestep,
            batch,
            condition_latents=None if is_i2v else condition_latents,
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


    def _init_pipeline(self, model_manager, pipeline_type):
        if pipeline_type == "v2v":
            return WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
        available_names = set(model_manager.model_name)
        if "wan_video_text_encoder" in available_names:
            pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        if "wan_video_image_encoder" in available_names:
            pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        if "wan_video_dit" in available_names:
            pipe.dit = model_manager.fetch_model("wan_video_dit")
        if "wan_video_vae" in available_names:
            pipe.vae = model_manager.fetch_model("wan_video_vae")
        if "wan_video_dit2" in available_names:
            pipe.dit2 = model_manager.fetch_model("wan_video_dit2")
        if "wan_video_motion_controller" in available_names:
            pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        if "wan_video_vace" in available_names:
            pipe.vace = model_manager.fetch_model("wan_video_vace")
        if "wan_video_animate_adapter" in available_names:
            pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")
        existing_names = list(getattr(pipe, "model_names", []))
        tracked = []
        for name in ["text_encoder", "image_encoder", "dit", "dit2", "vae", "motion_controller", "vace", "animate_adapter"]:
            if getattr(pipe, name, None) is not None:
                tracked.append(name)
        pipe.model_names = list(dict.fromkeys(existing_names + tracked))
        return pipe


    def _ensure_pipeline_compat(self):
        if not callable(getattr(self.pipe, "denoising_model", None)):
            self.pipe.denoising_model = lambda: getattr(self.pipe, "dit", None)
        if not callable(getattr(self.pipe, "prepare_extra_input", None)):
            self.pipe.prepare_extra_input = lambda latents=None: {}
        if not hasattr(self.pipe, "decode_video"):
            def _decode_video(latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
                return self.pipe.vae.decode(latents, device=self.pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            self.pipe.decode_video = _decode_video
        if not hasattr(self.pipe, "tensor2video"):
            def _tensor2video(frames):
                frames_np = rearrange(frames, "C T H W -> T H W C")
                frames_np = ((frames_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                return [Image.fromarray(frame) for frame in frames_np]
            self.pipe.tensor2video = _tensor2video
        if not hasattr(self.pipe, "encode_video"):
            def _encode_video(input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
                return self.pipe.vae.encode(input_video, device=self.pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            self.pipe.encode_video = _encode_video
        if not getattr(self.pipe, "model_names", None):
            names = []
            for name in ["text_encoder", "image_encoder", "dit", "dit2", "vae", "motion_controller", "vace", "animate_adapter"]:
                if getattr(self.pipe, name, None) is not None:
                    names.append(name)
            self.pipe.model_names = names


    def _get_denoising_model(self):
        model = None
        denoiser = getattr(self.pipe, "denoising_model", None)
        if callable(denoiser):
            model = denoiser()
        if model is None:
            model = getattr(self.pipe, "dit", None)
        if model is None:
            raise AttributeError("Unable to locate denoising model for the current pipeline.")
        return model


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self._get_denoising_model().train()

    @staticmethod
    def _strip_cam_layer_weights(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        filtered = {
            k: v for k, v in state_dict.items()
            if ".cam_encoder." not in k and ".projector." not in k
        }
        removed = len(state_dict) - len(filtered)
        if removed > 0:
            print(f"Stripping {removed} legacy camera-layer parameters from checkpoint.")
        return filtered



    def decode_video(self, noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch, condition_latents=None):
        """Decode video and calculate PSNR without saving"""
        # Use VideoDecoder to handle the complex decoding logic
        psnr_value, combined_frames, metadata = self.video_decoder.decode_and_create_combined_video(
            noise_pred, noisy_latents, tgt_latent_len, origin_latents, timestep, batch, condition_latents=condition_latents
        )
        
        return psnr_value, combined_frames, metadata
    
    def _compute_downsample_indices(self, total_frames, frame_downsample, is_i2v):
        if frame_downsample >= total_frames:
            raise ValueError(
                f"frame_downsample_to={frame_downsample} must be smaller than sequence length {total_frames}"
            )
        if is_i2v:
            indices = torch.linspace(
                0, total_frames - 1, steps=frame_downsample, dtype=torch.float64
            ).round().long()
            return indices.clamp_(0, total_frames - 1)

        per_half = total_frames // 2
        if per_half == 0:
            raise ValueError("V2V mode expects latents to contain at least two frames (target + condition).")
        effective_steps = min(frame_downsample, per_half)
        base = torch.linspace(
            0, per_half - 1, steps=effective_steps, dtype=torch.float64
        ).round().long()

        base = base.clamp_(0, per_half - 1)
        index_full = torch.cat([base, base + per_half], dim=0)
        return index_full

    def _apply_frame_downsample(self, latents, cam_emb, cam_intrinsics):
        frame_downsample = getattr(self, "frame_downsample_to", 0)
        if not isinstance(frame_downsample, int) or frame_downsample <= 0:
            return latents, cam_emb, cam_intrinsics, None

        total_frames = latents.shape[2]
        is_i2v = self.pipeline_type == "i2v"
        indices = self._compute_downsample_indices(total_frames, frame_downsample, is_i2v)
        indices_device = indices.to(latents.device)

        latents = latents.index_select(2, indices_device)
        if cam_emb is not None and cam_emb.dim() >= 2:
            cam_emb = cam_emb.index_select(1, indices.to(cam_emb.device))
        if cam_intrinsics is not None and cam_intrinsics.dim() >= 2:
            cam_intrinsics = cam_intrinsics.index_select(1, indices.to(cam_intrinsics.device))

        return latents, cam_emb, cam_intrinsics, indices
    
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
            state_dict = self._get_denoising_model().state_dict()
            
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
        trainable_modules = filter(lambda p: p.requires_grad, self._get_denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self._get_denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self._get_denoising_model().state_dict()
        if not (os.path.exists(os.path.join(checkpoint_dir))):
            os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
