"""Lightning module for ReCamMaster training."""

import copy
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import imageio
import lightning as pl
import numpy as np
import torch
from einops import rearrange
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import WanVideoReCamMasterPipeline, ModelManager
from diffsynth.pipelines.wan_video_new import WanVideoPipeline

from src.wandb_module import VideoDecoder


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        latent_path,
        text_encoder_path=None,
        tokenizer_path=None,
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
        val_guidance_scale=None,
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
        self.text_encoder_path = text_encoder_path
        inferred_tokenizer_path = self._infer_tokenizer_path(text_encoder_path)
        self.tokenizer_path = tokenizer_path or inferred_tokenizer_path
        self.val_guidance_scale = None
        if pipeline_type == "i2v" and val_guidance_scale is not None and val_guidance_scale > 0:
            self.val_guidance_scale = float(val_guidance_scale)
        if pipeline_type == "i2v" and not self.text_encoder_path:
            raise ValueError("i2v 模式必须提供 text_encoder_path。")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        models_to_load = [vae_path]
        if text_encoder_path:
            models_to_load.append(text_encoder_path)
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
        
        # Initialize video decoder
        self.video_decoder = VideoDecoder(self.pipe, pipeline_type=self.pipeline_type)

        # Prompt dropout configuration (i2v only)
        self.prompt_dropout_prob = 0.2 if pipeline_type == "i2v" else 0.0
        self.uncond_prompt_context: Optional[torch.Tensor] = None
        needs_uncond = (
            pipeline_type == "i2v"
            and (
                self.prompt_dropout_prob > 0
                or (self.val_guidance_scale is not None and self.val_guidance_scale > 0)
            )
        )
        if needs_uncond:
            self.uncond_prompt_context = self._build_uncond_prompt_context()
            self._release_text_encoder()
        
    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = copy.deepcopy(batch.get("prompt_emb", {}))
        prompt_context = prompt_emb.get("context")
        if torch.is_tensor(prompt_context):
            if prompt_context.dim() == 4:
                prompt_context = prompt_context[:, 0]
            elif prompt_context.dim() == 2:
                prompt_context = prompt_context.unsqueeze(0)
            prompt_context = prompt_context.to(self.device)
            prompt_emb["context"] = prompt_context
        image_emb = {}
        
        is_i2v = self.pipeline_type == "i2v"

        if is_i2v and self.prompt_dropout_prob > 0 and torch.is_tensor(prompt_context):
            if self.uncond_prompt_context is None:
                raise RuntimeError("i2v 模式的 prompt dropout 需要可用的空 prompt embedding。")
            drop_flag = torch.rand((), device=self.device) < self.prompt_dropout_prob
            if drop_flag.item():
                prompt_context = self._get_uncond_context_for_batch(prompt_context.shape[0])
                prompt_emb["context"] = prompt_context

        cam_emb = batch["camera"].to(self.device)
        cam_intrinsics = batch.get("intrinsics")
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.to(self.device)
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
            temporal_indices=temporal_indices,
            fuse_vae_embedding_in_latents=is_i2v,
        )

        # Build per-half indices to match model's internal downsampling (two-halves scheme on target half)
        if is_i2v:
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(),
                training_target.float()
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
        latents = batch["latents"].to(self.device)
        prompt_emb = copy.deepcopy(batch.get("prompt_emb", {}))
        context = prompt_emb.get("context")
        if torch.is_tensor(context):
            if context.dim() == 4:  # (batch, prompt_type, seq, hidden)
                context = context[:, 0]
            elif context.dim() == 2:  # (seq, hidden)
                context = context.unsqueeze(0)
            prompt_emb["context"] = context.to(self.device)
        image_emb =  {}
        for key, value in list(image_emb.items()):
            if torch.is_tensor(value):
                image_emb[key] = value.to(self.device)
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
        
        # Deterministic seed shared across validation samples
        val_seed = self.global_seed
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
        guidance_scale = self.val_guidance_scale if is_i2v else None
        use_cfg = guidance_scale is not None and guidance_scale > 0
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            if is_i2v:
                latents_input = latents_gen.clone()
                latents_input[:, :, 0, ...] = target_latents[:, :, 0, ...]
            else:
                latents_input = torch.cat([latents_gen, condition_latents], dim=2)
            extra_input = self.pipe.prepare_extra_input(latents_input)
            if not use_cfg:
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
                    fuse_vae_embedding_in_latents=is_i2v,
                )
            else:
                batch_size = latents_input.shape[0]
                prompt_cfg = copy.deepcopy(prompt_emb)
                if "context" not in prompt_cfg or not torch.is_tensor(prompt_cfg["context"]):
                    raise RuntimeError("启用 CFG 需要可用的 prompt context。")
                uncond_context = self._get_uncond_context_for_batch(batch_size)
                prompt_cfg["context"] = torch.cat(
                    [prompt_cfg["context"], uncond_context], dim=0
                )
                latents_input_cat = torch.cat([latents_input, latents_input], dim=0)
                cam_emb_cat = torch.cat([cam_emb, cam_emb], dim=0)
                cam_intrinsics_cat = None
                if cam_intrinsics is not None:
                    cam_intrinsics_cat = torch.cat([cam_intrinsics, cam_intrinsics], dim=0)
                extra_input_cat = self._duplicate_condition_dict(extra_input, batch_size)
                image_emb_cat = self._duplicate_condition_dict(image_emb, batch_size)
                noise_pred_cat = self._get_denoising_model()(
                    latents_input_cat,
                    timestep=timestep,
                    cam_emb=cam_emb_cat,
                    cam_intrinsics=cam_intrinsics_cat,
                    temporal_indices=temporal_indices,
                    **prompt_cfg,
                    **extra_input_cat,
                    **image_emb_cat,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                    t_highfreq_ratio=self.t_highfreq_ratio,
                    frame_downsample_to=self.frame_downsample_to,
                    fuse_vae_embedding_in_latents=is_i2v,
                )
                noise_pred_cond, noise_pred_uncond = torch.chunk(noise_pred_cat, 2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
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
        
        # Decode整批视频并计算单样本指标
        sample_results = self.decode_video(
            latents_gen,
            target_latents,
            batch,
            condition_latents=None if is_i2v else condition_latents,
        )

        batch_psnr_values = []
        for sample_result in sample_results:
            psnr_value = float(sample_result.get("psnr", 0.0))
            video_frames = sample_result.get("combined_frames", [])
            metadata = dict(sample_result.get("metadata", {}))
            metadata["batch_idx"] = int(batch_idx)
            batch_psnr_values.append(psnr_value)

            combined_path = os.path.abspath(
                self.save_video_with_naming(video_frames, metadata, video_type="val")
            )

            if not hasattr(self, "_val_psnr_sum"):
                self._val_psnr_sum = 0.0
                self._val_count = 0
            self._val_psnr_sum += psnr_value
            self._val_count += 1

            sample_idx = metadata.get("sample_idx", 0)
            print(
                f"Rank {self.global_rank}: Saved validation video for batch {batch_idx}, sample {sample_idx} at step {self.global_step}: {combined_path}"
            )

        # Restore training timesteps after validation to avoid affecting training_step
        self.pipe.scheduler.set_timesteps(self.train_timesteps, training=True)

        avg_batch_psnr = float(np.mean(batch_psnr_values)) if batch_psnr_values else 0.0
        return {"psnr": avg_batch_psnr}


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
        if getattr(pipe, "prompter", None) is not None and pipe.text_encoder is not None:
            pipe.prompter.fetch_models(pipe.text_encoder)
            tokenizer_path = self.tokenizer_path
            if not tokenizer_path:
                tokenizer_path = self._infer_tokenizer_path(self.text_encoder_path)
            if tokenizer_path:
                pipe.prompter.fetch_tokenizer(tokenizer_path)
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

    def _build_uncond_prompt_context(self) -> torch.Tensor:
        prompter = getattr(self.pipe, "prompter", None)
        text_encoder = getattr(self.pipe, "text_encoder", None)
        if prompter is None or text_encoder is None:
            raise RuntimeError("i2v 训练需要可用的 text encoder 来生成空 prompt embedding。")
        if getattr(prompter, "text_encoder", None) is None:
            prompter.fetch_models(text_encoder)
        empty_prompt = prompter.encode_prompt("", positive=True, device="cpu")
        if not torch.is_tensor(empty_prompt):
            raise TypeError("encode_prompt 应返回 Tensor。")
        if empty_prompt.dim() == 3 and empty_prompt.shape[0] == 1:
            empty_prompt = empty_prompt[0]
        elif empty_prompt.dim() != 2:
            raise ValueError(f"空 prompt embedding 形状异常: {empty_prompt.shape}")
        return empty_prompt.to(dtype=self.pipe.torch_dtype)

    def _release_text_encoder(self) -> None:
        """释放 text encoder 引用，避免占用 GPU 显存。"""
        text_encoder = getattr(self.pipe, "text_encoder", None)
        if text_encoder is None:
            return
        prompter = getattr(self.pipe, "prompter", None)
        if prompter is not None and getattr(prompter, "text_encoder", None) is text_encoder:
            prompter.text_encoder = None
        self.pipe.text_encoder = None
        del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_uncond_context_for_batch(self, batch_size: int) -> torch.Tensor:
        if self.uncond_prompt_context is None:
            raise RuntimeError("启用 CFG 需要在初始化阶段构建空 prompt embedding。")
        context = self.uncond_prompt_context.to(self.device)
        if context.dim() == 2:
            context = context.unsqueeze(0)
        if context.shape[0] == batch_size:
            return context
        if context.shape[0] == 1 and batch_size > 1:
            return context.expand(batch_size, -1, -1).contiguous()
        repeats = (batch_size + context.shape[0] - 1) // context.shape[0]
        return context.repeat(repeats, 1, 1)[:batch_size]

    def _duplicate_condition_dict(self, data, batch_size: int):
        if not isinstance(data, dict) or not data:
            return {} if data is None else copy.deepcopy(data)
        duplicated = copy.deepcopy(data)
        for key, value in duplicated.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
                duplicated[key] = torch.cat([value, value], dim=0)
        return duplicated

    @staticmethod
    def _infer_tokenizer_path(text_encoder_path: Optional[str]) -> Optional[str]:
        if not text_encoder_path:
            return None
        candidate = Path(text_encoder_path).resolve().parent / "google" / "umt5-xxl"
        if candidate.exists():
            return str(candidate)
        return None


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
    def decode_video(self, pred_latents, gt_target_latents, batch, condition_latents=None):
        """Decode整个 batch，并返回每个样本的可视化与指标。"""
        return self.video_decoder.decode_and_create_combined_videos(
            pred_latents,
            gt_target_latents,
            condition_latents,
            batch,
        )
    
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
    
    def save_video_with_naming(self, video_frames, metadata, video_type="val"):
        """Save video locally using metadata-aware naming."""
        scene_id = str(metadata.get("scene_id", "unknown"))
        condition_cam_type = str(metadata.get("condition_cam_type", "unknown"))
        target_cam_type = str(metadata.get("target_cam_type", "unknown"))
        batch_idx = metadata.get("batch_idx")
        sample_idx = metadata.get("sample_idx")

        os.makedirs(self.latent_path, exist_ok=True)
        if video_type == "val":
            suffix_parts = [f"step{self.global_step}"]
            if batch_idx is not None:
                suffix_parts.append(f"B{batch_idx}")
            if sample_idx is not None:
                suffix_parts.append(f"Idx{sample_idx}")
            suffix_parts.append(f"Scene{scene_id}_S{condition_cam_type}_T{target_cam_type}")
            video_name = "_".join(suffix_parts) + ".mp4"
            video_path = os.path.join(self.latent_path, video_name)
        elif video_type == "test":
            video_path = os.path.join(
                self.latent_path,
                f"test_epoch{self.current_epoch}_Scene{scene_id}_S{condition_cam_type}_T{target_cam_type}.mp4",
            )
        else:
            raise ValueError(f"Unknown video_type: {video_type}")
        
        # Compress frames for local storage (optimized for storage)
        local_scale = 1.0   # 保持四宫格完整尺寸
        frame_skip = 2      # Skip every other frame (1/2 frames)
        local_fps = 4       # Reduced FPS from 8 to 4
        local_quality = 4   # Lower quality for smaller file size
        
        compressed_combined_frames = []
        for i, frame in enumerate(video_frames):
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
        imageio.mimsave(
            video_path,
            compressed_combined_frames,
            fps=local_fps,
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None,
            output_params=["-movflags", "faststart"],
            quality=local_quality,
        )
        print(f"Saved {video_type} video: {video_path}")
        
        return video_path

    def on_validation_epoch_start(self):
        # Reset accumulators
        self._val_psnr_sum = 0.0
        self._val_count = 0
        
    def on_validation_epoch_end(self):
        # Log average PSNR across the validation set
        if getattr(self, "_val_count", 0) > 0:
            avg_psnr = self._val_psnr_sum / self._val_count
            self.log("val/psnr", avg_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True, batch_size=self._val_count)
        
        # Persist a lightweight checkpoint for inspection (rank 0 only)
        if getattr(self, "_has_started_training", False) and self.global_rank == 0:
            self.save_validation_checkpoint()

    def save_validation_checkpoint(self):
        """Save current denoiser weights after validation."""
        checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
        checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
        if not checkpoint_dir:
            # Default to training_log/<run>/checkpoints alongside video_debug
            latent_root = Path(self.latent_path).resolve()
            checkpoint_dir = latent_root.parent / "checkpoints"
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_step = int(self.global_step)
            state_dict = self._get_denoising_model().state_dict()
            checkpoint_path = os.path.join(str(checkpoint_dir), f"validation_step{current_step}.ckpt")
            torch.save(state_dict, checkpoint_path)
            print(f"[validation] Saved checkpoint at step {current_step}: {checkpoint_path}")
        except Exception as exc:
            print(f"[validation] Failed to save checkpoint: {exc}")

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
