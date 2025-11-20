#!/usr/bin/env python
"""
Compare the first-frame latents stored in two *.wan22.tensors.pth files.
Usage:
    python compare_latent_first_frame.py <tensor_a> <tensor_b>
"""

import argparse
from pathlib import Path

import torch


def load_first_frame_latent(tensor_path: Path) -> torch.Tensor:
    """Load cached latents and return the first-frame slice."""
    data = torch.load(tensor_path, map_location="cpu")
    latents = data["latents"]
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    if latents.dim() == 5:
        latents = latents.squeeze(0)
    if latents.dim() != 4:
        raise ValueError(f"Unexpected latent shape {tuple(latents.shape)} in {tensor_path}")
    # Layout: (channels, frames, height, width)
    first = latents[:, 0, :, :].contiguous()
    return first


def main():
    parser = argparse.ArgumentParser(description="Compare first-frame latents from cached tensors")
    parser.add_argument("tensor_a", help="Path to camXX.mp4.wan22.tensors.pth")
    parser.add_argument("tensor_b", help="Path to another camXX tensor file")
    args = parser.parse_args()

    lat_a = load_first_frame_latent(Path(args.tensor_a))
    lat_b = load_first_frame_latent(Path(args.tensor_b))

    if lat_a.shape != lat_b.shape:
        raise ValueError(f"Shape mismatch: {lat_a.shape} vs {lat_b.shape}")

    diff = lat_a - lat_b
    l2 = diff.norm().item()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    base_norm = lat_a.norm().item()
    rel = l2 / base_norm if base_norm > 0 else float("inf")

    print(f"Latent shape: {tuple(lat_a.shape)}")
    print(f"L2 difference: {l2:.4f}")
    print(f"Relative L2 (vs tensor_a): {rel:.6f}")
    print(f"Mean abs diff: {mean_abs:.6f}")
    print(f"Max abs diff: {max_abs:.6f}")


if __name__ == "__main__":
    main()
