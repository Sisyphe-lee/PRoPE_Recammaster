#!/usr/bin/env python3
"""
Filter a metadata CSV by dropping videos that lack corresponding pose files.

Example:
    python scripts/filter_missing_pose.py \
        --input-csv metadata/output_sdg.csv \
        --pose-root /nas/datasets/vipe_wild_sdg_1m/pose \
        --output-csv metadata/output_sdg_with_pose.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FilterStats:
    total: int
    kept: int
    dropped: int


def build_pose_path(video_path: str, pose_root: Path) -> Path:
    stem = Path(video_path).stem
    return pose_root / f"{stem}.npz"


def write_missing_log(missing: Iterable[tuple[str, Path]], log_path: Path) -> None:
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_absolute_path", "pose_path"])
        for video_path, pose_path in missing:
            writer.writerow([video_path, str(pose_path)])


def filter_csv(
    input_csv: Path,
    pose_root: Path,
    output_csv: Path,
    missing_log: Path | None = None,
) -> FilterStats:
    input_csv = input_csv.expanduser()
    pose_root = pose_root.expanduser()
    output_csv = output_csv.expanduser()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not pose_root.is_dir():
        raise FileNotFoundError(f"Pose root not found: {pose_root}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    missing_entries: list[tuple[str, Path]] = []

    with input_csv.open("r", newline="") as f_in, output_csv.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")
        if "video_absolute_path" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'video_absolute_path' column")

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            video_path = row["video_absolute_path"]
            pose_path = build_pose_path(video_path, pose_root)
            if pose_path.is_file():
                writer.writerow(row)
                kept += 1
            else:
                missing_entries.append((video_path, pose_path))

    if missing_log is not None:
        missing_log = missing_log.expanduser()
        missing_log.parent.mkdir(parents=True, exist_ok=True)
        write_missing_log(missing_entries, missing_log)

    return FilterStats(total=total, kept=kept, dropped=total - kept)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop rows without pose files from a metadata CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("metadata/output_sdg.csv"),
        help="Path to the input metadata CSV.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("/nas/datasets/vipe_wild_sdg_1m/pose"),
        help="Directory containing pose .npz files named after the video stem.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write the filtered CSV. "
        "Defaults to <input-stem>_with_pose.csv next to the input file.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=None,
        help="Optional path to record rows dropped due to missing pose files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = args.input_csv.with_name(f"{args.input_csv.stem}_with_pose{args.input_csv.suffix}")

    stats = filter_csv(
        input_csv=args.input_csv,
        pose_root=args.pose_root,
        output_csv=output_csv,
        missing_log=args.missing_log,
    )

    print(f"[info] total rows={stats.total}, kept={stats.kept}, dropped={stats.dropped}, output={output_csv}")
    if args.missing_log is not None:
        print(f"[info] missing list written to {args.missing_log}")


if __name__ == "__main__":  # pragma: no cover
    main()
