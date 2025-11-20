"""
Video decoding helpers for ReCamMaster validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm


class VideoDecoder:
    """Handles video decoding and processing for validation."""

    def __init__(self, pipe, pipeline_type="v2v"):
        """
        Initialize video decoder.

        Args:
            pipe: Diffusion pipeline that provides encode/decode utilities.
            pipeline_type: Either "v2v" or "i2v", used to resolve condition latents.
        """
        self.pipe = pipe
        self.pipeline_type = pipeline_type

    def decode_and_create_combined_videos(
        self,
        pred_latents,
        gt_target_latents,
        condition_latents,
        batch,
    ):
        """
        解码整个 batch 的潜变量并生成四宫格可视化。

        Args:
            pred_latents: 模型输出的目标潜变量 (B, C, T, H, W)
            gt_target_latents: GT 目标潜变量 (B, C, T, H, W)
            condition_latents: 条件潜变量，V2V 时为视频半段，I2V 时可为空
            batch: 当前 batch，包含元数据

        Returns:
            List[Dict]: 每个样本包含 psnr、combined_frames、metadata。
        """
        num_samples = pred_latents.shape[0]
        results = []

        self.pipe.load_models_to_device(["vae"])
        for sample_idx in range(num_samples):
            pred_sample = pred_latents[sample_idx : sample_idx + 1]
            gt_sample = gt_target_latents[sample_idx : sample_idx + 1]
            cond_sample = self._resolve_condition_latents(
                gt_target_latents, condition_latents, sample_idx
            )

            pred_frames = self._decode_latents(pred_sample)
            gt_frames = self._decode_latents(gt_sample)
            cond_frames = self._decode_condition_frames(cond_sample, gt_frames)

            psnr_value = self._compute_psnr(pred_frames, gt_frames)
            error_frames = self._create_heatmap_error_frames(torch.abs(gt_frames - pred_frames))
            combined_frames = self._combine_frames(cond_frames, gt_frames, pred_frames, error_frames)

            metadata = self._extract_metadata(batch, sample_idx)
            metadata["sample_idx"] = sample_idx

            results.append(
                {
                    "psnr": psnr_value,
                    "combined_frames": combined_frames,
                    "metadata": metadata,
                }
            )

        self.pipe.load_models_to_device([])
        return results

    def _resolve_condition_latents(self, gt_target_latents, provided_condition, sample_idx):
        if provided_condition is not None:
            return provided_condition[sample_idx : sample_idx + 1]
        if self.pipeline_type == "i2v":
            return gt_target_latents[sample_idx : sample_idx + 1, :, :1, ...]
        return None

    def _decode_latents(self, latents):
        latents = latents.to(dtype=self.pipe.torch_dtype)
        return self.pipe.decode_video(latents)[0]

    def _decode_condition_frames(self, cond_sample, gt_frames):
        if cond_sample is None:
            return torch.zeros_like(gt_frames[:, :1, ...])
        cond_frames = self._decode_latents(cond_sample)
        if cond_frames.shape[1] == 1 and gt_frames.shape[1] > 1:
            cond_frames = cond_frames.repeat(1, gt_frames.shape[1], 1, 1)
        return cond_frames

    def _compute_psnr(self, pred_frames, gt_frames):
        pred_norm = (pred_frames.clamp(-1, 1) + 1) / 2
        gt_norm = (gt_frames.clamp(-1, 1) + 1) / 2
        mse = F.mse_loss(pred_norm, gt_norm)
        mse_value = float(mse.item())
        if mse_value == 0.0:
            return 100.0
        return 20.0 * np.log10(1.0 / np.sqrt(mse_value))

    def _frames_to_numpy(self, frames_tensor):
        frames = self.pipe.tensor2video(frames_tensor)
        return [np.array(frame) for frame in frames]

    def _combine_frames(self, cond_frames, gt_frames, pred_frames, error_frames):
        cond_np = self._frames_to_numpy(cond_frames)
        gt_np = self._frames_to_numpy(gt_frames)
        pred_np = self._frames_to_numpy(pred_frames)
        error_np = [np.array(frame) for frame in error_frames]

        combined = []
        for cond_frame, gt_frame, pred_frame, err_frame in zip(cond_np, gt_np, pred_np, error_np):
            top_row = np.concatenate([cond_frame, gt_frame], axis=1)
            bottom_row = np.concatenate([pred_frame, err_frame], axis=1)
            combined.append(np.concatenate([top_row, bottom_row], axis=0))
        return combined

    def _extract_metadata(self, batch, sample_idx):
        try:
            scene_raw = batch.get("scene_id", ["unknown"])
            cond_raw = batch.get("condition_cam_type", ["unknown"])
            tgt_raw = batch.get("target_cam_type", ["unknown"])
            scene_id = (
                scene_raw[sample_idx]
                if isinstance(scene_raw, list) and len(scene_raw) > sample_idx
                else scene_raw
            )
            cond_type = (
                cond_raw[sample_idx]
                if isinstance(cond_raw, list) and len(cond_raw) > sample_idx
                else cond_raw
            )
            tgt_type = (
                tgt_raw[sample_idx]
                if isinstance(tgt_raw, list) and len(tgt_raw) > sample_idx
                else tgt_raw
            )
        except Exception as exc:
            print(f"Error extracting metadata from batch: {exc}")
            scene_id, cond_type, tgt_type = "unknown", "unknown", "unknown"

        return {
            "scene_id": str(scene_id),
            "condition_cam_type": str(cond_type),
            "target_cam_type": str(tgt_type),
        }

    def _create_heatmap_error_frames(self, error_frames_tensor):
        """Convert error tensor to heatmap visualization using colormap."""
        error_frames_list = []
        for t in range(error_frames_tensor.shape[1]):
            frame_error = error_frames_tensor[:, t, :, :].mean(dim=0)
            frame_min, frame_max = frame_error.min(), frame_error.max()
            if frame_max > frame_min:
                frame_norm = (frame_error - frame_min) / (frame_max - frame_min)
            else:
                frame_norm = torch.zeros_like(frame_error)

            frame_np = frame_norm.float().cpu().numpy()
            colored_frame = cm.jet(frame_np)[:, :, :3]
            colored_frame = (colored_frame * 255).astype(np.uint8)

            error_frames_list.append(Image.fromarray(colored_frame))

        return error_frames_list
