#!/usr/bin/env python3
"""扫描 metadata 中列出的 latent tensors，列出无法成功 torch.load 的文件。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm


def load_paths(dataset_path: str, metadata_path: str, tensor_suffix: str) -> List[Path]:
    df = pd.read_csv(metadata_path)
    required_cols = {"video_absolute_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata 缺少列: {missing}")

    base = Path(dataset_path)
    paths: List[Path] = []
    for row in df.itertuples(index=False):
        raw_path = getattr(row, "video_absolute_path")
        full_path = Path(raw_path) if os.path.isabs(raw_path) else base / raw_path
        paths.append(Path(str(full_path) + tensor_suffix))
    return paths


def check_files(paths: List[Path]) -> List[Path]:
    corrupted: List[Path] = []
    for path in tqdm(paths, desc="checking", unit="file"):
        if not path.exists():
            corrupted.append(path)
            continue
        try:
            torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] 读取 {path} 失败: {exc}")
            corrupted.append(path)
    return corrupted


def write_report(corrupted: List[Path], output: Path | None):
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        for path in corrupted:
            fp.write(str(path) + "\n")
    print(f"报告已写入 {output} ({len(corrupted)} 条)")


def parse_args():
    parser = argparse.ArgumentParser(description="检查 .tensors/.wan22.tensors.pth 是否损坏")
    parser.add_argument("--dataset_path", required=True, help="MultiCam 数据根目录，用于解析相对路径")
    parser.add_argument("--metadata_path", required=True, help="metadata CSV（需包含 video_absolute_path）")
    parser.add_argument("--tensor_suffix", default=".wan22.tensors.pth", help="latent 文件后缀")
    parser.add_argument("--report_path", default="corrupted_tensors.txt", help="输出结果列表")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = load_paths(args.dataset_path, args.metadata_path, args.tensor_suffix)
    corrupted = check_files(paths)
    print(f"共检查 {len(paths)} 个文件，其中 {len(corrupted)} 个加载失败。")
    if corrupted:
        write_report(corrupted, Path(args.report_path))


if __name__ == "__main__":
    main()
