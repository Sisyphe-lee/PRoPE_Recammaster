#!/usr/bin/env python
"""
Convert the RealEstate10k caption JSON dump into the two-column metadata CSV
used by ReCamMaster.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

JsonValue = Union[str, List["JsonValue"]]


def parse_concatenated_json(path: Path) -> Dict[str, List[JsonValue]]:
    """Support JSON files that contain multiple top-level objects concatenated."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    objs: List[Dict[str, List[JsonValue]]] = []

    while True:
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Top-level JSON block必须是object, 实际: {type(obj)}")
        objs.append(obj)  # type: ignore[arg-type]
        idx = end

    if not objs:
        raise ValueError(f"{path} 为空或不是有效 JSON")

    merged: Dict[str, List[JsonValue]] = {}
    for obj in objs:
        for key, value in obj.items():
            if not isinstance(value, list):
                raise ValueError(f"{key} 的值必须是 list, 实际: {type(value)}")
            if key in merged:
                merged[key].extend(value)
            else:
                merged[key] = list(value)
    return merged


def pick_caption(entry: JsonValue) -> str:
    """选择每条记录的首个非空字符串描述。"""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, list):
        for candidate in entry:
            text = pick_caption(candidate)
            if text:
                return text
    return ""


def iter_rows(
    dataset_root: Path,
    video_filename: str,
    data: Dict[str, List[JsonValue]],
) -> Iterable[Tuple[str, str]]:
    for key in sorted(data):
        entries = data[key]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            caption = pick_caption(entry)
            if not caption:
                continue
            video_path = dataset_root / key / video_filename
            yield str(video_path), caption


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 re10k.json 转成 metadata CSV (video_absolute_path, caption)."
    )
    parser.add_argument(
        "--json-path",
        default="re10k.json",
        type=Path,
        help="输入 caption JSON 路径（默认: re10k.json）",
    )
    parser.add_argument(
        "--dataset-root",
        default="/nas/datasets/relestate10k/train",
        type=Path,
        help="视频根目录，最终路径为 root/<id>/video.mp4",
    )
    parser.add_argument(
        "--output-csv",
        default="metadata/metadata_re10k.csv",
        type=Path,
        help="输出 CSV 路径（默认: metadata/metadata_re10k.csv）",
    )
    parser.add_argument(
        "--video-filename",
        default="video.mp4",
        help="每个序列下的视频文件名（默认: video.mp4）",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    data = parse_concatenated_json(args.json_path)
    rows = list(iter_rows(args.dataset_root, args.video_filename, data))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_absolute_path", "caption"])
        writer.writerows(rows)

    print(f"写入 {len(rows)} 条记录 -> {args.output_csv}")


if __name__ == "__main__":
    main()
