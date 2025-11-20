#!/usr/bin/env python
"""
Extract a specific frame from a video and save it as an image.

Usage example:
    python scripts/extract_frame.py \
        --video /path/to/video.mp4 \
        --output /tmp/video_frame3.png \
        --frame-index 2

Frame index is zero-based, so 2 corresponds to the 3rd frame.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a single frame from a video.")
    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--output", required=True, help="Path to save the extracted frame image.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=2,
        help="Zero-based index of the frame to extract (default: 2 -> third frame).",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Optional Pillow format override (e.g., PNG/JPEG). Defaults to output suffix.",
    )
    return parser.parse_args()


def extract_frame(video_path: Path, frame_index: int) -> Image.Image:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(frame_index)
    except IndexError as exc:
        total = getattr(reader, "count_frames", lambda: None)()
        msg = (
            f"Video '{video_path}' does not have frame index {frame_index}. "
            f"Total frames: {total if total is not None else 'unknown'}."
        )
        raise ValueError(msg) from exc
    finally:
        reader.close()
    return Image.fromarray(frame)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_image = extract_frame(video_path, args.frame_index)
    frame_image.save(output_path, format=args.format)
    print(f"[extract_frame] Saved frame {args.frame_index} from {video_path} to {output_path}")


if __name__ == "__main__":
    main()
