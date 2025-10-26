#!/usr/bin/env python3
"""
Convert a directory of NPZ trajectories into the legacy RayDiffusion JSON format.

Each NPZ is expected to contain:
    - data: (N, 4, 4) array of camera-to-world matrices.
    - inds: (N,) array of frame indices (defaults to range(N) if absent).

The resulting JSON matches the structure consumed by trajectory_viz_utils._load_json_trajectories.
"""

import argparse
import json
import os
from typing import Dict, Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert NPZ trajectory directory to legacy JSON format.')
    parser.add_argument('input_dir', help='Directory containing per-camera NPZ files.')
    parser.add_argument('output_json', help='Path to write the aggregated JSON file.')
    parser.add_argument(
        '--camera-prefix',
        default='cam',
        help='Prefix for camera names when NPZ stems are purely numeric (default: cam).'
    )
    parser.add_argument(
        '--camera-padding',
        type=int,
        default=2,
        help='Number of digits to pad numeric camera indices (default: 2).'
    )
    return parser.parse_args()


def iter_npz_files(npz_dir: str) -> Iterable[str]:
    for entry in sorted(os.listdir(npz_dir)):
        if entry.lower().endswith('.npz'):
            yield os.path.join(npz_dir, entry)


def infer_camera_name(stem: str, prefix: str, padding: int) -> str:
    if stem.isdigit():
        return f'{prefix}{int(stem):0{padding}d}'
    return stem


def matrix_to_legacy_string(c2w: np.ndarray) -> str:
    mat = np.array(c2w, dtype=float, copy=True)
    mat[:3, 3] *= 100.0
    mat[:3, 1] *= -1.0
    mat = mat[:, np.argsort([1, 2, 0, 3])]
    rows = mat.T

    def fmt(val: float) -> str:
        return f'{val:.6g}'

    chunks = [f"[{' '.join(fmt(x) for x in row)}]" for row in rows]
    return ' '.join(chunks) + ' '


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        matrices = data['data']
        indices = data['inds'] if 'inds' in data else np.arange(len(matrices))
    if matrices.shape[0] != indices.shape[0]:
        raise ValueError(f'Mismatch between poses and indices in {npz_path}')
    return matrices, indices.astype(int)


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f'Input directory not found: {args.input_dir}')

    frame_dict: Dict[int, Dict[str, str]] = {}

    for npz_path in iter_npz_files(args.input_dir):
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        cam_name = infer_camera_name(stem, args.camera_prefix, args.camera_padding)
        matrices, indices = load_npz(npz_path)

        for c2w, frame_idx in zip(matrices, indices):
            frames = frame_dict.setdefault(int(frame_idx), {})
            if cam_name in frames:
                raise ValueError(f'Duplicate trajectory data for {cam_name} at frame {frame_idx} in {npz_path}')
            frames[cam_name] = matrix_to_legacy_string(c2w)

    sorted_frames = dict(
        (f'frame{frame_idx}', frame_dict[frame_idx])
        for frame_idx in sorted(frame_dict.keys())
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(sorted_frames, f, indent=4)

    print(f'Wrote {len(sorted_frames)} frames to {args.output_json}')


if __name__ == '__main__':
    main()
