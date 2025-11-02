#!/usr/bin/env python3
"""
Extract the first frame from each video in a directory.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

import cv2


VIDEO_EXTENSIONS: List[str] = [
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
]


def iter_video_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def extract_first_frame(video_path: Path) -> "cv2.Mat":
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError(f"无法读取视频第一帧: {video_path}")

    return frame


def write_frame(frame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"保存图像失败: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取目录内各视频的第一帧。")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="包含视频的目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="保存第一帧图像的目标目录。",
    )
    parser.add_argument(
        "--image_suffix",
        default=".jpg",
        help="输出图像的扩展名，例如 .jpg 或 .png。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志信息。",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=level,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    suffix: str = args.image_suffix if args.image_suffix.startswith(".") else f".{args.image_suffix}"

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    for video_path in iter_video_files(input_dir):
        output_name = video_path.stem + suffix
        output_path = output_dir / output_name
        if output_path.exists() and not args.overwrite:
            logging.info("跳过已存在的文件: %s", output_path)
            continue

        logging.info("处理视频: %s", video_path)
        frame = extract_first_frame(video_path)
        write_frame(frame, output_path)
        logging.info("已保存第一帧至: %s", output_path)

    logging.info("处理完成。")


if __name__ == "__main__":
    main()
