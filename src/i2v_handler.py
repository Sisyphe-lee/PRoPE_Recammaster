from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from base_handler import (
    BaseInferenceDataset,
    BasePipelineHandler,
    InferenceSample,
    PreparedInference,
    TargetSpec,
    NEGATIVE_PROMPT,
    compute_relative_c2w,
    convert_c2w_convention_I2V,
    c2w_to_w2c,
    invert_se3,
    normalize_joint_translation,
    preprocess_video_frame,
    register_dataset,
    register_pipeline,
    select_pose_sequence,
)

try:
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
except Exception:
    WanVideoPipeline = Any  # type: ignore


@register_dataset("example_i2v")
class ExampleI2VDataset(BaseInferenceDataset):
    """Image-conditioned dataset for Wan2.2 example data."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)

        image_dir = self.dataset_path / options.get("image_dir", "images")
        if not image_dir.exists():
            raise FileNotFoundError(f"Condition image directory not found: {image_dir}")
        self.image_paths = sorted(
            p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found under {image_dir}")

        metadata_path = self.dataset_path / options.get("metadata_filename", "metadata.csv")
        self.text_map: Dict[str, str] = {}
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            for _, row in metadata.iterrows():
                file_name = str(row.get("file_name", "")).strip()
                if not file_name:
                    continue
                stem = Path(file_name).stem
                raw_text = row.get("text", "")
                text = "" if pd.isna(raw_text) else str(raw_text)
                self.text_map[stem] = text

        self.num_frames = int(options.get("num_frames", 81))
        self.height = int(options.get("height", 480))
        self.width = int(options.get("width", 832))
        self.cam_interval = int(options.get("camera_interval", 4))
        self.cam_indices = np.arange(self.num_frames, dtype=np.int64)[:: self.cam_interval]
        if self.cam_indices.size == 0:
            raise ValueError("camera_interval produced empty cam_indices")

        self.preprocess = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _process_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            frame = preprocess_video_frame(img.convert("RGB"), self.height, self.width)
        tensor = self.preprocess(frame)
        video = tensor.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        return video.to(torch.float32)

    def __getitem__(self, index: int) -> InferenceSample:
        image_path = self.image_paths[index]
        video_tensor = self._process_image(image_path)
        stem = image_path.stem
        text = self.text_map.get(stem, "")

        return InferenceSample(
            video=video_tensor,
            text=text,
            cond_data={
                "condition_image_path": str(image_path),
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={
                "source_path": str(image_path),
                "stem": stem,
            },
        )


@register_dataset("sdg_i2v")
class SDGI2VDataset(BaseInferenceDataset):
    """SDG I2V 数据集：支持直接图片或视频首帧作为条件。"""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)
        metadata_path = Path(options.get("metadata_path", options.get("metadata_filename", "metadata.csv")))
        if not metadata_path.is_absolute():
            metadata_path = dataset_path / metadata_path
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        if "video_absolute_path" not in metadata.columns:
            raise ValueError(f"{metadata_path} 缺少列 video_absolute_path")

        self.source_paths = [self._resolve_path(p) for p in metadata["video_absolute_path"].tolist()]
        self.texts = metadata.get("caption", pd.Series([""] * len(metadata))).fillna("").tolist()

        self.num_frames = int(options.get("num_frames", 81))
        self.height = int(options.get("height", 480))
        self.width = int(options.get("width", 832))
        self.cam_interval = int(options.get("camera_interval", 4))
        self.cam_indices = np.arange(self.num_frames, dtype=np.int64)[:: self.cam_interval]
        if self.cam_indices.size == 0:
            raise ValueError("camera_interval produced empty cam_indices")

        self.preprocess = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _resolve_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return self.dataset_path / path

    def __len__(self) -> int:
        return len(self.source_paths)

    def _load_condition_frame(self, path: Path) -> Image.Image:
        if path.suffix.lower() in self.SUPPORTED_EXTS:
            with Image.open(path) as img:
                return preprocess_video_frame(img.convert("RGB"), self.height, self.width)
        reader = imageio.get_reader(str(path))
        try:
            raw = reader.get_data(0)
        finally:
            reader.close()
        return preprocess_video_frame(Image.fromarray(raw), self.height, self.width)

    def __getitem__(self, index: int) -> InferenceSample:
        source_path = self.source_paths[index]
        text = self.texts[index]
        frame = self._load_condition_frame(source_path)
        tensor = self.preprocess(frame)
        video = tensor.unsqueeze(1).repeat(1, self.num_frames, 1, 1).to(torch.float32)

        return InferenceSample(
            video=video,
            text=text,
            cond_data={
                "condition_image_path": str(source_path),
                "condition_image": frame,
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={
                "source_path": str(source_path),
                "stem": source_path.stem,
            },
        )


@register_pipeline("i2v")
class I2VPipelineHandler(BasePipelineHandler):
    """Custom inference loop that mirrors the training/validation i2v flow."""

    def build_inputs(
        self,
        sample: InferenceSample,
        target: TargetSpec,
        *,
        source_video: torch.Tensor,
    ) -> PreparedInference:
        cond_info = sample.cond_data
        cam_indices = np.asarray(cond_info["cam_indices"], dtype=np.int64)
        target_c2ws_sel, matched_inds = select_pose_sequence(target.raw_pose, target.raw_inds, cam_indices)
        target_c2ws = convert_c2w_convention_I2V(target_c2ws_sel)
        ref_w2c = invert_se3(target_c2ws[0])
        tgt_rel_c2w = compute_relative_c2w(ref_w2c, target_c2ws)
        _, tgt_norm = normalize_joint_translation(tgt_rel_c2w, tgt_rel_c2w)
        tgt_rel_w2c = c2w_to_w2c(tgt_norm)
        camera_tensor = torch.from_numpy(tgt_rel_w2c).unsqueeze(0).to(dtype=self.dtype)

        return PreparedInference(
            pipe_kwargs={
                "camera_embedding": camera_tensor,
                "condition_image_path": cond_info["condition_image_path"],
                "condition_image": cond_info.get("condition_image"),
                "num_frames": int(cond_info["num_frames"]),
                "height": int(cond_info["height"]),
                "width": int(cond_info["width"]),
            },
            target_rel_w2c=tgt_rel_w2c,
            cam_indices=cam_indices,
            metadata={
                "target_name": target.name,
                "matched_inds": matched_inds,
            },
        )

    def _load_condition_image(self, path: str, height: int, width: int) -> Image.Image:
        with Image.open(path) as img:
            processed = preprocess_video_frame(img.convert("RGB"), height, width)
        return processed

    def _encode_condition_embeddings(
        self,
        pipe: WanVideoPipeline,
        image: Image.Image,
        *,
        num_frames: int,
        height: int,
        width: int,
        tiled: bool,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        device = pipe.device
        dtype = pipe.torch_dtype

        image_tensor = pipe.preprocess_image(image.resize((width, height))).to(device)
        clip_feature = None
        if pipe.image_encoder is not None:
            clip_feature = pipe.image_encoder.encode_image([image_tensor]).to(dtype=dtype, device=device)

        zeros_tail = torch.zeros(
            image_tensor.shape[1],
            max(num_frames - 1, 0),
            height,
            width,
            device=device,
            dtype=image_tensor.dtype,
        )
        vae_input = torch.cat([image_tensor.transpose(0, 1), zeros_tail], dim=1).unsqueeze(0)
        vae_latents = pipe.vae.encode(
            vae_input.to(dtype=dtype),
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0].to(dtype=dtype, device=device)
        latent_h = height // pipe.vae.upsampling_factor
        latent_w = width // pipe.vae.upsampling_factor
        msk = torch.ones(1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        if num_frames > 1:
            msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]
        y = torch.concat([msk, vae_latents], dim=0).unsqueeze(0)

        single_video = pipe.preprocess_video([image]).to(device)
        first_latents = pipe.vae.encode(
            single_video.to(dtype=dtype),
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=dtype, device=device)

        return {
            "clip_feature": clip_feature,
            "y": y.to(dtype=dtype, device=device),
            "first_latents": first_latents,
        }

    def run_inference(
        self,
        pipe: WanVideoPipeline,
        prepared: PreparedInference,
        prompt_text: str,
        pipe_kwargs: Dict[str, Any],
    ) -> List[Image.Image]:
        cfg_scale = float(pipe_kwargs.get("cfg_scale", self.global_opts.get("cfg_scale", 5.0)))
        num_steps = int(pipe_kwargs.get("num_inference_steps", 25))
        seed = int(pipe_kwargs.get("seed", 0))
        negative_prompt = pipe_kwargs.get("negative_prompt", NEGATIVE_PROMPT)
        tiled = bool(pipe_kwargs.get("tiled", True))
        tile_size = pipe_kwargs.get("tile_size", (34, 34))
        tile_stride = pipe_kwargs.get("tile_stride", (18, 16))

        camera_embedding = pipe_kwargs["camera_embedding"].to(device=pipe.device, dtype=pipe.torch_dtype)
        condition_image = pipe_kwargs.get("condition_image")
        condition_path = pipe_kwargs.get("condition_image_path")
        num_frames = int(pipe_kwargs["num_frames"])
        height = int(pipe_kwargs["height"])
        width = int(pipe_kwargs["width"])

        if condition_image is None:
            if condition_path is None:
                raise ValueError("condition_image_path or condition_image must be provided for i2v inference")
            condition_image = self._load_condition_image(condition_path, height, width)
        embeddings = self._encode_condition_embeddings(
            pipe,
            condition_image,
            num_frames=num_frames,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        clip_feature = embeddings["clip_feature"]
        y = embeddings["y"]
        first_latents = embeddings["first_latents"]

        latent_frames = camera_embedding.shape[1]
        latent_shape = (
            1,
            first_latents.shape[1],
            latent_frames,
            first_latents.shape[-2],
            first_latents.shape[-1],
        )
        latents_gen = pipe.generate_noise(latent_shape, seed=seed, rand_device="cpu")
        latents_gen = latents_gen.to(dtype=pipe.torch_dtype, device=pipe.device)
        latents_gen[:, :, 0:1] = first_latents

        positive_prompt = prompt_text or ""
        context_pos = pipe.prompter.encode_prompt(positive_prompt, positive=True, device=pipe.device).to(dtype=pipe.torch_dtype)
        context_neg = pipe.prompter.encode_prompt(negative_prompt or "", positive=False, device=pipe.device).to(dtype=pipe.torch_dtype)
        pipe.scheduler.set_timesteps(num_steps, shift=pipe.scheduler.shift, denoising_strength=1.0)
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            latents_input = latents_gen.clone()
            latents_input[:, :, 0:1] = first_latents
            prepare_extra_input = getattr(pipe, "prepare_extra_input", lambda latents=None: {})
            extra_input = prepare_extra_input(latents_input)
            dit_kwargs = dict(
                cam_emb=camera_embedding,
                clip_feature=clip_feature,
                y=y,
                temporal_indices=None,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                fuse_vae_embedding_in_latents=True,
                **extra_input,
            )
            noise_pred_pos = pipe.dit(
                latents_input,
                timestep=timestep,
                context=context_pos,
                **dit_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_neg = pipe.dit(
                    latents_input,
                    timestep=timestep,
                    context=context_neg,
                    **dit_kwargs,
                )
                noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos

            latents_gen = pipe.scheduler.step(
                noise_pred,
                pipe.scheduler.timesteps[progress_id],
                latents_input,
            )
            latents_gen[:, :, 0:1] = first_latents

        decoded = pipe.vae.decode(
            latents_gen,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if decoded.dim() == 4:
            decoded = decoded.unsqueeze(0)
        frames = pipe.vae_output_to_video(decoded)
        return frames
