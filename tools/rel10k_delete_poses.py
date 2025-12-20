#!/usr/bin/env python
"""
删除整理后的 rel10k 目录下的旧 pose 文件（extrinsics.npz）。

示例：
  python scripts/rel10k_delete_poses.py --dataset-root /nas/datasets/relestate10k/train
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Delete rel10k extrinsics files")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="整理后的 rel10k 序列根目录（含 train/test 子目录或直接指向 train）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将删除的文件，不实际删除。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists():
        print(f"[error] dataset root 不存在: {root}", file=sys.stderr)
        sys.exit(1)

    extr_files = list(root.rglob("extrinsics.npz"))
    if not extr_files:
        print(f"[info] 在 {root} 下未找到 extrinsics.npz，无需删除。")
        return

    print(f"[info] 找到 {len(extr_files)} 个 extrinsics.npz")
    for fp in extr_files:
        print(f"  - {fp}")
        if not args.dry_run:
            try:
                fp.unlink()
            except Exception as exc:
                print(f"[warn] 删除失败 {fp}: {exc}", file=sys.stderr)
    if args.dry_run:
        print("[info] dry-run 模式，未实际删除任何文件。")
    else:
        print("[done] 删除完成。")


if __name__ == "__main__":
    main()
