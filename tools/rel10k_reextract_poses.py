#!/usr/bin/env python
"""
从原始 re10k .torch 文件重新提取姿态，并覆盖整理目录下的 extrinsics.npz。

示例：
  python scripts/rel10k_reextract_poses.py \\
    --dataset-root /nas/datasets/relestate10k/train
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Re-extract rel10k poses from raw .torch shards")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="整理后的 rel10k 序列根目录（含 train/test 子目录或直接指向 train）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有 extrinsics.npz（默认仅处理不存在的文件）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行线程数（默认 4，设为 1 禁用并行）。",
    )
    return parser.parse_args()


def load_shard(shard_path: Path) -> List[Dict]:
    """不缓存，避免长时间堆积大文件占用内存。"""
    shard_path = shard_path.resolve()
    items = torch.load(shard_path, map_location="cpu")
    if not isinstance(items, list):
        raise ValueError(f"{shard_path} 内容异常，期望 list。")
    return items


def build_c2w_sequence(cameras: torch.Tensor, frame_indices: List[int]) -> np.ndarray:
    """cameras: [N, 18]，后 12 个为 3x4 w2c，按 frame_indices 取出并求逆为 c2w。"""
    c2w_list: List[np.ndarray] = []
    for idx in frame_indices:
        cam_row = cameras[idx]
        w2c_3x4 = cam_row[6:18].view(3, 4).numpy()
        R = w2c_3x4[:, :3]
        t = w2c_3x4[:, 3]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c).astype(np.float32)
        c2w_list.append(c2w)
    return np.stack(c2w_list, axis=0)


def process_sequence(seq_dir: Path) -> Tuple[bool, str]:
    meta_path = seq_dir / "metadata.json"
    if not meta_path.exists():
        return False, f"缺少 metadata.json: {seq_dir}"
    metadata = json.load(open(meta_path, "r"))
    key = metadata.get("key")
    source_torch = metadata.get("source_torch")
    frame_indices = metadata.get("frame_indices")
    if not key or not source_torch or not isinstance(frame_indices, list):
        return False, f"metadata 字段缺失: {meta_path}"

    shard_path = Path(source_torch)
    if not shard_path.exists():
        return False, f"源 shard 不存在: {shard_path}"
    items = load_shard(shard_path)
    entry = next((item for item in items if item.get("key") == key), None)
    if entry is None:
        return False, f"{shard_path} 中未找到 key={key}"
    cameras = entry.get("cameras")
    if cameras is None:
        return False, f"{key} 缺少 cameras"

    c2w_seq = build_c2w_sequence(cameras, frame_indices)
    out_path = seq_dir / "extrinsics.npz"
    np.savez(out_path, c2w=c2w_seq)
    return True, f"{seq_dir.name}: saved {out_path}"


def main():
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists():
        print(f"[error] dataset root 不存在: {root}", file=sys.stderr)
        sys.exit(1)

    # 支持传入 train/test 根目录，或更上一层（自动下钻）
    seq_dirs = []
    if (root / "metadata.json").exists():
        seq_dirs = [root]
    else:
        for sub in root.iterdir():
            if (sub / "metadata.json").exists():
                seq_dirs.append(sub)
    seq_dirs = sorted(d for d in seq_dirs if d.is_dir())
    if not seq_dirs:
        print(f"[error] 未在 {root} 下找到序列目录（含 metadata.json）。", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0

    def worker(seq_dir: Path):
        extr_path = seq_dir / "extrinsics.npz"
        if extr_path.exists() and not args.overwrite:
            return None, f"skip existing: {seq_dir.name}"
        success, msg = process_sequence(seq_dir)
        return success, msg

    workers = max(int(args.workers), 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, d): d for d in seq_dirs}
        for fut in as_completed(futures):
            success, msg = fut.result()
            if success is None:
                continue
            if success:
                ok += 1
                print("[ok]", msg)
            else:
                fail += 1
                print("[fail]", msg, file=sys.stderr)

    print(f"[done] 完成：成功 {ok}，失败 {fail}（跳过的已存在文件未计数）")


if __name__ == "__main__":
    main()
