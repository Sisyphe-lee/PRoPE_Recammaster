#!/usr/bin/env python
"""
Quick helper to decode cached latents (.wan22.tensors.pth) into a video file.

Example:
    python tools/decode_latents.py \
        --tensor-path /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10/scene999/videos/cam9.mp4.wan22.tensors.pth \
        --vae-path models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
        --output ./test.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import imageio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from diffsynth import ModelManager, WanVideoPipeline  # noqa: E402


def tensor_to_numpy(video_tensor: torch.Tensor):
    """Convert [C, T, H, W] tensor in [-1, 1] to list of uint8 frames."""
    video_np = ((video_tensor.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    video_np = video_np.permute(1, 2, 3, 0).cpu().numpy()  # T H W C
    return [frame for frame in video_np]


def main():
    parser = argparse.ArgumentParser(description="Decode cached latents to video.")
    parser.add_argument("--tensor-path", required=True, help="Path to *.wan22.tensors.pth file.")
    parser.add_argument("--vae-path", required=True, help="Path to Wan VAE weights (Wan2.2_VAE.pth).")
    parser.add_argument("--output", required=True, help="Output video path or directory.")
    parser.add_argument("--device", default="cuda", help="Decode device (cuda or cpu).")
    parser.add_argument("--fps", type=int, default=15, help="FPS when saving mp4.")
    args = parser.parse_args()

    tensor_path = Path(args.tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    if not os.path.isfile(args.vae_path):
        raise FileNotFoundError(f"VAE weights not found: {args.vae_path}")

    print(f"[decode] loading latents from {tensor_path}")
    cache = torch.load(tensor_path, map_location="cpu", weights_only=True)
    latents = cache["latents"]
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    if latents.dim() == 5:  # [1, C, T, H, W]
        latents = latents[0]
    if latents.dim() != 4:
        raise ValueError(f"Unexpected latents shape {latents.shape}, expected [C, T, H, W].")

    print(f"[decode] latents shape: {latents.shape}")

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([args.vae_path])
    device_type = args.device
    if device_type != "cpu" and not torch.cuda.is_available():
        device_type = "cpu"
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device=device_type
    )
    pipe.load_models_to_device(["vae"])
    pipe.vae.to(device=pipe.device, dtype=pipe.torch_dtype)

    latents = latents.unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
    print(latents.shape)
    with torch.inference_mode():
        decoded = pipe.decode_video(latents)[0].cpu()

    frames = tensor_to_numpy(decoded)
    output_path = Path(args.output)
    if output_path.is_dir() or args.output.endswith(os.sep):
        output_path = output_path / f"{tensor_path.stem}_decoded.mp4"
    if output_path.suffix.lower() not in {".mp4", ".gif"}:
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.fps, quality=5)
    print(f"[decode] saved video to {output_path} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
