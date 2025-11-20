#!/usr/bin/env python
"""
Compare the first-frame pixels of two MultiCam videos after resize+crop.
Usage:
    python compare_first_frame.py <video_a.mp4> <video_b.mp4>
"""

import argparse
import math
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageChops, ImageStat

TARGET_W, TARGET_H = 832, 480


def preprocess(path: Path) -> Image.Image:
    """Read the first frame, resize with aspect ratio, then center-crop."""
    frame = iio.imread(path, index=0)
    image = Image.fromarray(frame).convert("RGB")
    width, height = image.size
    scale = max(TARGET_W / width, TARGET_H / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), Image.BILINEAR)
    left = max(0, (new_width - TARGET_W) // 2)
    top = max(0, (new_height - TARGET_H) // 2)
    return image.crop((left, top, left + TARGET_W, top + TARGET_H))


def main():
    parser = argparse.ArgumentParser(description="Compare first frames of two cams")
    parser.add_argument("video_a", help="Path to the first camXX.mp4 file")
    parser.add_argument("video_b", help="Path to the second camXX.mp4 file")
    args = parser.parse_args()

    img_a = preprocess(Path(args.video_a))
    img_b = preprocess(Path(args.video_b))

    diff = ImageChops.difference(img_a, img_b)
    stat = ImageStat.Stat(diff)
    sum2 = sum(stat.sum2)
    max_abs = max(max(channel_extrema) for channel_extrema in diff.getextrema())

    print(f"L2 difference: {math.sqrt(sum2):.2f}")
    print(f"Max abs pixel diff: {max_abs}")


if __name__ == "__main__":
    main()
