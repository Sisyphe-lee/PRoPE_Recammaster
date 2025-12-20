"""
Filter SDG samples by camera motion and write a new metadata CSV.

Criteria:
- Max rotation (relative to first frame) exceeds --rotation-threshold-deg.
- OR max translation norm exceeds --translation-threshold.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SDG samples by pose change.")
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Input SDG metadata CSV (e.g., metadata/output_sdg_with_pose.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for filtered samples.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root. If omitted, inferred from each video path (parent of 'rgb').",
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=5.0,
        help="Rotation threshold (degrees).",
    )
    parser.add_argument(
        "--translation-threshold",
        type=float,
        default=0.2,
        help="Translation norm threshold (in the same units as the pose data).",
    )
    return parser.parse_args()


def base_id_from_path(video_path: Path) -> str:
    name = video_path.name
    return re.sub(r"\.mp4.*$", "", name)


def load_pose_npz(video_path: Path, dataset_root: Path) -> np.ndarray:
    base_id = base_id_from_path(video_path)
    pose_path = dataset_root / "pose" / f"{base_id}.npz"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose file: {pose_path}")
    pose_npz = np.load(pose_path)
    poses = pose_npz.get("data")
    if poses is None:
        raise KeyError(f"'data' key not found in {pose_path}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Pose shape expected [N,4,4], got {poses.shape} ({pose_path})")
    return poses.astype(np.float64)


def relative_metrics(poses: np.ndarray) -> Tuple[float, float]:
    """
    Compute max rotation (deg) and max translation norm relative to the first frame.
    Rotation uses axis-angle magnitude: acos((trace(R)-1)/2).
    """
    if poses.shape[0] == 0:
        return 0.0, 0.0
    T0_inv = np.linalg.inv(poses[0])
    max_rot_deg = 0.0
    max_trans = 0.0
    for i in range(poses.shape[0]):
        rel = T0_inv @ poses[i]
        R = rel[:3, :3]
        t = rel[:3, 3]
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(trace)))
        norm_t = float(np.linalg.norm(t))
        if angle > max_rot_deg:
            max_rot_deg = angle
        if norm_t > max_trans:
            max_trans = norm_t
    return max_rot_deg, max_trans


def infer_dataset_root(video_path: Path, user_root: Path = None) -> Path:
    if user_root:
        return user_root
    parent = video_path.parent
    if parent.name == "rgb":
        return parent.parent
    return parent


def main():
    args = parse_args()
    df = pd.read_csv(args.metadata)
    required_cols = {"video_absolute_path"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"Missing required columns in metadata: {missing}")

    kept_rows = []
    skipped = 0
    for idx, row in df.iterrows():
        video_path = Path(row["video_absolute_path"])
        try:
            dataset_root = infer_dataset_root(video_path, args.dataset_root)
            poses = load_pose_npz(video_path, dataset_root)
            max_rot_deg, max_trans = relative_metrics(poses)
            if max_rot_deg > args.rotation_threshold_deg or max_trans > args.translation_threshold:
                row_dict = row.to_dict()
                row_dict["max_rotation_deg"] = max_rot_deg
                row_dict["max_translation_norm"] = max_trans
                kept_rows.append(row_dict)
        except Exception as exc:
            skipped += 1
            print(f"[warn] skip {video_path}: {exc}", file=sys.stderr)
            continue

    if not kept_rows:
        print("No samples matched the thresholds; output CSV will be empty.")
        filtered_df = pd.DataFrame(columns=list(df.columns) + ["max_rotation_deg", "max_translation_norm"])
    else:
        filtered_df = pd.DataFrame(kept_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output, index=False)
    print(f"Filtered {len(kept_rows)} samples (skipped {skipped}); saved to {args.output}")


if __name__ == "__main__":
    main()
