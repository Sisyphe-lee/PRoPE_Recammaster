#!/usr/bin/env python
"""
Scan Wan tensor caches and统计 image_emb 字段的占比。

Usage 示例：
python tools/check_image_emb.py \
    --root /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train \
    --pattern "f*/scene*/videos/*.tensors.pth"
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import torch


def iter_tensor_paths(root: Path, pattern: str) -> Iterable[Path]:
    """Yield tensor paths under root according to the provided glob pattern."""
    if "**" in pattern or pattern.startswith("**"):
        # Path.glob automatically handles ** when recursive=True
        yield from root.glob(pattern)
        return
    # Allow pattern like "f*/scene*/videos/*.tensors.pth"
    for sub in root.glob(pattern.split(os.sep)[0]):
        base = sub
        parts = pattern.split(os.sep)[1:]
        if not parts:
            yield base
            continue
        glob_pattern = os.path.join(*parts)
        yield from base.rglob(glob_pattern)


def resolve_paths(args) -> List[Path]:
    if args.list_file:
        paths = []
        with open(args.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute() and args.root:
                    path = Path(args.root) / path
                paths.append(path)
        return paths

    root = Path(args.root)
    pattern = args.pattern or "**/*.tensors.pth"
    return list(root.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="统计 image_emb 非空的 tensor 文件数量。")
    parser.add_argument("--root", required=False, default=".", help="包含 tensor 缓存的根目录。")
    parser.add_argument("--pattern", default="**/*.tensors.pth", help="相对于 root 的 glob pattern。")
    parser.add_argument("--list-file", help="包含 tensor 路径的文本文件（每行一个）。")
    parser.add_argument("--limit", type=int, default=0, help="仅检查前 N 个样本（0=全部）。")
    parser.add_argument("--show-examples", type=int, default=10, help="展示若干 image_emb 非空的样本路径。")
    parser.add_argument("--verbose", action="store_true", help="打印进度。")
    args = parser.parse_args()

    if not args.list_file:
        root = Path(args.root)
        if not root.exists():
            raise FileNotFoundError(f"指定根目录不存在: {root}")
        tensor_paths = list(root.glob(args.pattern))
    else:
        tensor_paths = resolve_paths(args)

    total = 0
    empty = 0
    non_empty = 0
    examples = []
    keys_counter = {}

    limit = args.limit if args.limit and args.limit > 0 else None

    for idx, tensor_path in enumerate(tensor_paths):
        if limit is not None and idx >= limit:
            break
        try:
            cache = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            print(f"[warn] 加载失败 {tensor_path}: {exc}")
            continue

        image_emb = cache.get("image_emb") or {}
        total += 1
        if not image_emb:
            empty += 1
        else:
            non_empty += 1
            if len(examples) < args.show_examples:
                examples.append(str(tensor_path))
            key_set = tuple(sorted(image_emb.keys()))
            keys_counter[key_set] = keys_counter.get(key_set, 0) + 1

        if args.verbose and total % 200 == 0:
            print(f"[info] 已处理 {total} 个样本 (non-empty: {non_empty})")

    print("总样本数:", total)
    print("image_emb 为空数量:", empty)
    print("image_emb 非空数量:", non_empty)
    if keys_counter:
        print("非空 image_emb 的键分布:")
        for key_tuple, count in sorted(keys_counter.items(), key=lambda kv: -kv[1]):
            print(f"  keys={list(key_tuple)} -> {count}")
    if examples:
        print("示例非空样本:")
        for path in examples:
            print("  ", path)


if __name__ == "__main__":
    main()
