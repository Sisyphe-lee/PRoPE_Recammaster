#!/usr/bin/env python3
"""
诊断脚本：比较旧版 inference_recammaster 与新版 inference_unified 生成的 target pose。

默认流程：
1. 读取 metadata.csv 指定样本的视频路径，并解析其对应的 source 相机 .npz。
2. 按旧脚本逻辑（camera_extrinsics_ori.json）计算 target pose。
3. 按新脚本逻辑（目标目录下的 .npz）计算 target pose。
4. 输出逐元素最大差异，并在存在不一致时列出首个不一致的索引。

用法示例：
    python tools/compare_target_pose.py \
        --dataset-root example_test_data \
        --metadata-file metadata.csv \
        --sample-index 0 \
        --camera-json cameras/camera_extrinsics_ori.json \
        --camera-id cam01 \
        --target-npz target_traj_json/cam01.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 基础数学工具（与推理脚本保持一致）
# ---------------------------------------------------------------------------

def ensure_homogeneous(mats: np.ndarray) -> np.ndarray:
    if mats.ndim != 3:
        raise ValueError(f"Pose array must be 3D, got shape {mats.shape}")
    T, h, w = mats.shape
    if h == 4 and w == 4:
        return mats.astype(np.float32)
    if h == 3 and w == 4:
        last = np.tile(np.array([[0, 0, 0, 1]], dtype=mats.dtype), (T, 1, 1))
        return np.concatenate([mats, last], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported pose shape {mats.shape}, expected (T,4,4) or (T,3,4)")


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation_inv
    result[:3, 3] = translation_inv
    return result


def convert_c2w_convention(c2w: np.ndarray) -> np.ndarray:
    converted = c2w.copy()
    converted = converted[:, [1, 2, 0, 3]]
    converted[:3, 1] *= -1.0
    return converted


def center_trajectory(c2ws: np.ndarray) -> np.ndarray:
    centered = c2ws.copy()
    origin = centered[0, :3, 3].copy()
    if not np.allclose(origin, 0.0):
        centered[:, :3, 3] -= origin
    return centered


def compute_relative_c2w(ref_w2c: np.ndarray, c2ws: np.ndarray) -> np.ndarray:
    rel_w2c = np.einsum("ab,tbc->tac", ref_w2c, c2ws)
    return np.stack([invert_se3(mat) for mat in rel_w2c], axis=0)


def c2w_to_w2c(rel_c2w: np.ndarray) -> np.ndarray:
    return np.stack([invert_se3(T) for T in rel_c2w], axis=0)


def normalize_joint_translation(
    cond_rel_c2w: np.ndarray,
    tgt_rel_c2w: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([tgt_rel_c2w, cond_rel_c2w], axis=0)
    translations = combined[:, :3, 3]
    norms = np.linalg.norm(translations, axis=1)
    max_norm = float(np.max(norms)) if norms.size > 0 else 0.0
    if not np.isfinite(max_norm) or max_norm < eps:
        return cond_rel_c2w.copy(), tgt_rel_c2w.copy()
    scale = max_norm
    cond_scaled = cond_rel_c2w.copy()
    tgt_scaled = tgt_rel_c2w.copy()
    cond_scaled[:, :3, 3] /= scale
    tgt_scaled[:, :3, 3] /= scale
    return cond_scaled, tgt_scaled


def nearest_index(inds: np.ndarray, target: int) -> int:
    return int(np.abs(inds - target).argmin())


def select_pose_sequence(mats: np.ndarray, inds: np.ndarray, desired: np.ndarray) -> np.ndarray:
    selected = []
    for idx in desired:
        matches = np.where(inds == idx)[0]
        choice = int(matches[0]) if matches.size > 0 else nearest_index(inds, idx)
        selected.append(mats[choice])
    return np.stack(selected, axis=0)


# ---------------------------------------------------------------------------
# 数据载入工具
# ---------------------------------------------------------------------------

def resolve_video_path(base_path: Path, file_name: str) -> Path:
    if os.path.isabs(file_name):
        candidate = Path(file_name)
        if candidate.exists():
            return candidate
    direct = Path(file_name)
    if direct.exists():
        return direct.resolve()
    candidate = base_path / file_name
    if candidate.exists():
        return candidate
    candidate = base_path / "videos" / file_name
    if candidate.exists():
        return candidate
    if "videos" in file_name:
        candidate = base_path / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve video path for '{file_name}' under '{base_path}'")


def load_metadata_list(metadata_path: Path) -> list[str]:
    file_names: list[str] = []
    with metadata_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "file_name" not in reader.fieldnames:
            raise KeyError(f"'file_name' column not found in {metadata_path}")
        for row in reader:
            file_names.append(row["file_name"])
    if not file_names:
        raise RuntimeError(f"No entries found in {metadata_path}")
    return file_names


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append([float(val) for val in row.split()])
    mat = np.array(matrix, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected camera matrix shape {mat.shape}")
    return mat


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def compute_conditional_poses(
    src_pose_path: Path,
    cam_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(src_pose_path, allow_pickle=True)
    src_mats = ensure_homogeneous(data["data"])
    src_inds = data["inds"].astype(np.int64)

    selected = select_pose_sequence(src_mats, src_inds, cam_indices)
    selected_c2w = convert_c2w_convention(selected)
    selected_c2w = center_trajectory(selected_c2w)

    ref_c2w = selected_c2w[0]
    ref_w2c = invert_se3(ref_c2w)
    cond_rel_c2w = compute_relative_c2w(ref_w2c, selected_c2w)
    cond_rel_w2c = c2w_to_w2c(cond_rel_c2w)
    return cond_rel_c2w, cond_rel_w2c, ref_w2c


def compute_target_from_json(
    camera_json: Path,
    camera_id: str,
    cam_indices: np.ndarray,
    ref_w2c: np.ndarray,
    cond_rel_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with camera_json.open("r") as f:
        cam_data = json.load(f)

    traj = []
    for idx in cam_indices:
        frame_key = f"frame{idx}"
        if frame_key not in cam_data or camera_id not in cam_data[frame_key]:
            raise KeyError(f"Missing entry for {frame_key}/{camera_id} in {camera_json}")
        traj.append(parse_matrix(cam_data[frame_key][camera_id]))

    traj = np.stack(traj, axis=0).transpose(0, 2, 1)
    traj = convert_c2w_convention(traj)
    traj = center_trajectory(traj)
    tgt_rel_c2w = compute_relative_c2w(ref_w2c, traj)

    cond_joint, tgt_joint = normalize_joint_translation(cond_rel_c2w, tgt_rel_c2w)
    cond_rel_w2c_norm = c2w_to_w2c(cond_joint)
    tgt_rel_w2c = c2w_to_w2c(tgt_joint)
    pose_embedding = np.concatenate([tgt_rel_w2c, cond_rel_w2c_norm], axis=0)
    return pose_embedding, tgt_rel_c2w, tgt_rel_w2c, cond_rel_w2c_norm


def compute_target_from_npz(
    target_npz: Path,
    cam_indices: np.ndarray,
    ref_w2c: np.ndarray,
    cond_rel_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(target_npz, allow_pickle=False) as data:
        if "data" not in data or "inds" not in data:
            raise KeyError(f"{target_npz} must contain 'data' and 'inds'")
        raw_pose = ensure_homogeneous(data["data"])
        raw_inds = data["inds"].astype(np.int64)

    target_w2cs = select_pose_sequence(raw_pose, raw_inds, cam_indices)
    target_c2ws = target_w2cs.transpose(0, 2, 1)
    target_c2ws = convert_c2w_convention(target_c2ws)
    target_c2ws = center_trajectory(target_c2ws)
    tgt_rel_c2w = compute_relative_c2w(ref_w2c, target_c2ws)

    cond_joint, tgt_joint = normalize_joint_translation(cond_rel_c2w, tgt_rel_c2w)
    cond_rel_w2c_norm = c2w_to_w2c(cond_joint)
    tgt_rel_w2c = c2w_to_w2c(tgt_joint)
    pose_embedding = np.concatenate([tgt_rel_w2c, cond_rel_w2c_norm], axis=0)
    return pose_embedding, tgt_rel_c2w, tgt_rel_w2c, cond_rel_w2c_norm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare target pose embeddings between old/new inference pipelines.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="数据集根目录，例如 example_test_data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv", help="metadata 文件名（相对 dataset-root）")
    parser.add_argument("--sample-index", type=int, default=0, help="选择第几个样本（0-based）")
    parser.add_argument("--num-frames", type=int, default=81, help="推理使用的帧数")
    parser.add_argument("--camera-interval", type=int, default=4, help="相机采样间隔")
    parser.add_argument("--camera-json", type=Path, required=True, help="旧脚本使用的 camera_extrinsics JSON")
    parser.add_argument("--camera-id", type=str, required=True, help="目标相机 ID，例如 cam01")
    parser.add_argument("--target-npz", type=Path, required=True, help="新脚本读取的 target pose npz")
    parser.add_argument("--atol", type=float, default=1e-5, help="绝对误差容忍度")
    parser.add_argument("--rtol", type=float, default=1e-5, help="相对误差容忍度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    metadata_path = dataset_root / args.metadata_file
    camera_json = (dataset_root / args.camera_json).resolve() if not args.camera_json.is_absolute() else args.camera_json.resolve()
    target_npz = (dataset_root / args.target_npz).resolve() if not args.target_npz.is_absolute() else args.target_npz.resolve()

    file_list = load_metadata_list(metadata_path)
    if not (0 <= args.sample_index < len(file_list)):
        raise IndexError(f"sample_index {args.sample_index} out of range (dataset size {len(file_list)})")

    video_rel = file_list[args.sample_index]
    video_path = resolve_video_path(dataset_root, video_rel)
    src_pose_path = video_path.with_suffix(".npz")
    if not src_pose_path.exists():
        raise FileNotFoundError(f"Source pose npz not found: {src_pose_path}")

    cam_indices = np.arange(args.num_frames, dtype=np.int64)[:: args.camera_interval]
    if cam_indices.size == 0:
        raise ValueError("camera_interval produced empty cam_indices")

    cond_rel_c2w, cond_rel_w2c, ref_w2c = compute_conditional_poses(src_pose_path, cam_indices)

    pose_old, tgt_c2w_old, tgt_w2c_old, cond_w2c_old = compute_target_from_json(
        camera_json=camera_json,
        camera_id=args.camera_id,
        cam_indices=cam_indices,
        ref_w2c=ref_w2c,
        cond_rel_c2w=cond_rel_c2w,
    )
    pose_new, tgt_c2w_new, tgt_w2c_new, cond_w2c_new = compute_target_from_npz(
        target_npz=target_npz,
        cam_indices=cam_indices,
        ref_w2c=ref_w2c,
        cond_rel_c2w=cond_rel_c2w,
    )

    def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
        if np.allclose(a, b, atol=args.atol, rtol=args.rtol):
            print(f"[OK] {name}: allclose within atol={args.atol}, rtol={args.rtol}")
        else:
            diff = np.abs(a - b)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"[FAIL] {name}: max abs diff {diff.max():.6e} at index {max_idx}")

    print(f"Comparing sample='{video_path.name}', camera='{args.camera_id}'")
    report("target_rel_c2w", tgt_c2w_old, tgt_c2w_new)
    report("target_rel_w2c", tgt_w2c_old, tgt_w2c_new)
    report("cond_rel_w2c_norm", cond_w2c_old, cond_w2c_new)
    report("pose_embedding", pose_old, pose_new)


if __name__ == "__main__":
    main()
