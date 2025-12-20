#!/usr/bin/env python
"""
Quick checker to compare MultiCam camera_extrinsics.json with target_traj/*.npz
and verify training vs inference pose transforms stay aligned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MulticamImageConditionDataset  # type: ignore
from src.inference_unified import (  # type: ignore
    center_trajectory,
    convert_c2w_convention_I2V,
    compute_relative_c2w,
    invert_se3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pose alignment between JSON and NPZ targets.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to camera_extrinsics.json")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to target_traj/camXX.npz")
    parser.add_argument("--cam_key", type=str, default="cam03", help="Camera key, e.g., cam03")
    parser.add_argument("--num_frames", type=int, default=21, help="Frames to sample for comparison")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=4,
        help="Stride used to subsample frames (set 0 to use linspace like training).",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=5,
        help="Index (after subsampling) to print for side-by-side comparison.",
    )
    return parser.parse_args()


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix, dtype=np.float32)


def select_indices(total: int, num: int, stride: int) -> np.ndarray:
    if total <= 0 or num <= 0:
        raise ValueError("total and num must be positive.")
    if stride > 0:
        return np.arange(total, dtype=np.int64)[::stride][:num]
    positions = np.linspace(0, total - 1, num=num)
    return np.clip(np.round(positions).astype(np.int64), 0, total - 1)


def load_json_cam(json_path: Path, cam_key: str) -> np.ndarray:
    with open(json_path, "r") as f:
        cam_json = json.load(f)
    keys = sorted(cam_json.keys(), key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0))
    frames: List[np.ndarray] = []
    for k in keys:
        entry = cam_json[k]
        if cam_key not in entry:
            continue
        frames.append(parse_matrix(entry[cam_key]))
    if not frames:
        raise ValueError(f"{cam_key} not found in {json_path}")
    return np.stack(frames, axis=0)


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    npz_path = Path(args.npz_path)

    raw_json = load_json_cam(json_path, args.cam_key)
    npz = np.load(npz_path)
    raw_npz = npz["data"]
    inds_npz = npz["inds"]

    print(f"JSON frames: {raw_json.shape}, NPZ frames: {raw_npz.shape}, inds len: {len(inds_npz)}")
    if raw_json.shape != raw_npz.shape or not np.allclose(raw_json, raw_npz):
        diff = float(np.max(np.abs(raw_json - raw_npz)))
        print(f"[warn] JSON/NPZ raw mismatch, max abs diff={diff}")
    else:
        print("[ok] JSON and NPZ raw matrices are identical.")

    idx = select_indices(raw_json.shape[0], args.num_frames, args.frame_stride)
    sample_idx = min(args.sample_idx, len(idx) - 1)

    # Training-style transform
    reorder = MulticamImageConditionDataset._reorder_c2w_axes
    train_c2w = np.stack([reorder(m.copy()) for m in raw_json], axis=0)
    train_sel = train_c2w[idx]
    train_sel = center_trajectory(train_sel)
    ref_w2c_train = invert_se3(train_sel[0])
    train_rel = compute_relative_c2w(ref_w2c_train, train_sel)

    # Inference-style transform
    inf_c2w = convert_c2w_convention_I2V(raw_npz[idx])
    inf_c2w = center_trajectory(inf_c2w)
    ref_w2c_inf = invert_se3(inf_c2w[0])
    inf_rel = compute_relative_c2w(ref_w2c_inf, inf_c2w)

    max_diff = float(np.max(np.abs(train_rel - inf_rel)))
    mean_diff = float(np.mean(np.abs(train_rel - inf_rel)))
    print(f"Relative pose diff (train vs inference): max={max_diff:.6f}, mean={mean_diff:.6f}")

    print(f"\nFrame idx {idx[sample_idx]} after subsampling:")
    print("train_rel:")
    print(train_rel[sample_idx])
    print("\ninference_rel:")
    print(inf_rel[sample_idx])


if __name__ == "__main__":
    main()
