#!/usr/bin/env python3
"""
Quick checker for (R|t) round-trip from dataset-style 3x4 flatten (12-dim) to 4x4 homogeneous matrix.

Usage:
    python check_cam_roundtrip.py --json /path/to/camera_extrinsics.json \
        --cond 1 --tgt 2 --frames 81 --stride 4 --tol 1e-5 --rotation_only \
        [--no_bf16]

Notes:
- By default we mimic training: cam_emb stored in bfloat16, which may introduce ~1e-3
  rotation error due to bf16 precision. Use --no_bf16 to compare in float32 if you want
  to isolate layout correctness without dtype-induced error.
"""
import argparse
import json
import numpy as np
import torch
from typing import List


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix, dtype=np.float64)


def normalize_c2w(c2w: np.ndarray) -> np.ndarray:
    """Replicate the normalization in dataset.py (axis reordering, sign flip, scale)."""
    c2w = c2w.copy()
    # Reorder columns: [:, [1,2,0,3]]
    c2w = c2w[:, [1, 2, 0, 3]]
    # Flip Y axis
    c2w[:3, 1] *= -1.0
    # Scale translation to meters (this is why we often want rotation-only checks)
    c2w[:3, 3] /= 100.0
    return c2w


def load_traj(cam_json: dict, view_idx: int, cam_idx: List[int]) -> np.ndarray:
    """Load and normalize trajectory for a given camera index.
    Returns array of shape (T, 4, 4) after normalization.
    """
    traj = [parse_matrix(cam_json[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
    # Keep the same transpose as dataset.py (T,4,4) -> (T,4,4); symmetric but we keep it for fidelity
    traj = np.stack(traj, axis=0).transpose(0, 2, 1)
    c2ws = []
    for c2w in traj:
        c2ws.append(normalize_c2w(c2w))
    return np.stack(c2ws, axis=0)  # (T,4,4)


def build_cam_emb(cond_traj: np.ndarray, tgt_traj: np.ndarray, use_bf16: bool) -> torch.Tensor:
    """Build cam_emb tensor of shape (1, 2*T, 12) by concatenating cond then tgt,
    and flattening top 3x4 row-major.
    """
    all_c2w = np.concatenate([cond_traj, tgt_traj], axis=0)  # (2T,4,4)
    top3x4 = all_c2w[:, :3, :]  # (2T,3,4)
    cam_emb = torch.from_numpy(top3x4.reshape(1, top3x4.shape[0], 12))
    if use_bf16:
        cam_emb = cam_emb.to(dtype=torch.bfloat16)
    else:
        cam_emb = cam_emb.to(dtype=torch.float32)
    return cam_emb, all_c2w


def restore_from_cam_emb(cam_emb: torch.Tensor) -> torch.Tensor:
    """Restore to (B,N,4,4) by view(B,N,3,4) and stacking bottom row [0,0,0,1]."""
    B, N, _ = cam_emb.shape
    reshaped = cam_emb.view(B, N, 3, 4).to(dtype=torch.float32)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=reshaped.dtype, device=reshaped.device)
    bottom = bottom.view(1, 1, 1, 4).expand(B, N, 1, 4)
    restored = torch.cat([reshaped, bottom], dim=2)  # (B,N,4,4)
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Path to camera_extrinsics.json')
    parser.add_argument('--cond', type=int, default=1, help='Condition camera index (e.g., 1)')
    parser.add_argument('--tgt', type=int, default=2, help='Target camera index (e.g., 2)')
    parser.add_argument('--frames', type=int, default=81, help='Total frames in sequence (default 81)')
    parser.add_argument('--stride', type=int, default=4, help='Frame stride (default 4 -> 21 samples)')
    parser.add_argument('--tol', type=float, default=1e-5, help='Tolerance for max abs diff')
    parser.add_argument('--show_per_frame', action='store_true', help='Print per-frame diffs')
    parser.add_argument('--rotation_only', action='store_true', default=True, help='Compare only the 3x3 rotation blocks (default: True)')
    parser.add_argument('--no_bf16', action='store_true', help='Do NOT cast cam_emb to bfloat16 (use float32 to avoid precision loss)')
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        cam_json = json.load(f)

    cam_idx = list(range(args.frames))[::args.stride]
    print(f"Sampled frames: len={len(cam_idx)} -> {cam_idx[:5]}... (stride={args.stride})")
    print(f"Checking order: [cond={args.cond}, tgt={args.tgt}] (cond first)")

    cond_traj = load_traj(cam_json, args.cond, cam_idx)  # (T,4,4)
    tgt_traj = load_traj(cam_json, args.tgt, cam_idx)    # (T,4,4)

    cam_emb, all_c2w = build_cam_emb(cond_traj, tgt_traj, use_bf16=not args.no_bf16)
    print(f"cam_emb shape: {tuple(cam_emb.shape)}  (expect (1, {2*len(cam_idx)}, 12)); dtype={cam_emb.dtype}")

    restored = restore_from_cam_emb(cam_emb)  # (1, 2T, 4, 4)
    print(f"restored shape: {tuple(restored.shape)}")

    orig = torch.from_numpy(all_c2w).to(dtype=torch.float32).unsqueeze(0)  # (1,2T,4,4)

    if args.rotation_only:
        diffs = (restored[:, :, :3, :3] - orig[:, :, :3, :3]).abs()
    else:
        diffs = (restored - orig).abs()

    max_diff_overall = float(diffs.max().item())

    if args.show_per_frame:
        # Flatten per (4x4) or (3x3) block accordingly
        per_frame_max = diffs.view(diffs.shape[1], -1).max(dim=1).values.cpu().numpy()
        for i, d in enumerate(per_frame_max):
            tag = 'cond' if i < len(cam_idx) else 'tgt'
            print(f"frame_idx[{i:02d}] ({tag}) max_abs_diff = {d:.6e}")

    label = 'rotation-only' if args.rotation_only else 'full-matrix'
    print(f"Overall max_abs_diff ({label}): {max_diff_overall:.6e}")
    if max_diff_overall <= args.tol:
        print("CHECK PASSED: restored matrix matches original within tolerance.")
    else:
        print("CHECK FAILED: difference exceeds tolerance. Likely due to bf16 precision if dtype=bf16; try --no_bf16 to isolate layout correctness.")


if __name__ == '__main__':
    main()
