#!/usr/bin/env python3
"""
Convert camera extrinsics stored in JSON into plain .npz files.

The script simply parses each camera matrix exactly as stored in the JSON and
writes it out without any additional processing. The inference pipeline will
apply its own convention conversions and normalisation.

Usage:
    python scripts/convert_camera_json_to_npz.py \
        --json example_test_data/cameras/camera_extrinsics_ori.json \
        --output_dir example_test_data/target_traj_json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    mat = np.array(matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Parsed matrix has unexpected ndim: {mat.ndim}")
    return mat


def convert(json_path: Path, output_dir: Path) -> None:
    with open(json_path, "r") as f:
        cam_data = json.load(f)

    frame_keys = sorted(cam_data.keys(), key=lambda k: int(k.replace("frame", "")))
    cam_names = sorted(cam_data[frame_keys[0]].keys())

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cam_names:
        matrices: list[np.ndarray] = []
        for frame_key in frame_keys:
            matrices.append(parse_matrix(cam_data[frame_key][cam_name]))
        data = np.stack(matrices, axis=0).astype(np.float32)
        inds = np.arange(len(frame_keys), dtype=np.int64)
        out_path = output_dir / f"{cam_name}.npz"
        np.savez(out_path, data=data, inds=inds)
        print(f"[info] wrote {out_path}")

        restored = np.load(out_path)
        if not np.allclose(restored["data"], data, atol=1e-6):
            raise ValueError(f"Verification failed for {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert camera extrinsics JSON into .npz files.")
    parser.add_argument("--json", type=str, required=True, help="Path to camera_extrinsics JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store generated .npz files.")
    args = parser.parse_args()

    json_path = Path(args.json)
    output_dir = Path(args.output_dir)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    convert(json_path, output_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
