import argparse
from pathlib import Path

import numpy as np


def load_poses(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到文件: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    pose_key = "data" if "data" in data else data.files[0]
    poses = np.asarray(data[pose_key])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"期望形状 [N,4,4]，实际为 {poses.shape} (key={pose_key})")
    inds = data["inds"] if "inds" in data else None
    return poses, inds


def print_poses(poses: np.ndarray, inds=None, max_frames: int | None = None):
    np.set_printoptions(precision=4, suppress=True)
    total = poses.shape[0]
    limit = total if max_frames is None else min(max_frames, total)
    print(f"总帧数: {total}，打印前 {limit} 帧")
    if inds is not None:
        inds = np.asarray(inds)
        if inds.shape[0] != total:
            print(f"[warn] inds 长度 {inds.shape[0]} 与姿态帧数 {total} 不一致，忽略 inds。")
            inds = None
    for idx in range(limit):
        frame_id = f"{idx}"
        if inds is not None:
            frame_id += f" (ind={inds[idx]})"
        print(f"\nFrame {frame_id}:")
        print(poses[idx])


def main():
    parser = argparse.ArgumentParser(description="打印相机位姿 npz 文件 (c2w/w2c)")
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="evaluation/example_eval/20251103_165044/pose/1_cam05.npz",
        help="相机位姿 npz 路径 (默认当前目录示例文件)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="仅打印前 N 帧 (默认打印全部)",
    )
    args = parser.parse_args()

    poses, inds = load_poses(Path(args.npz_path))
    print(f"文件: {Path(args.npz_path).resolve()}")
    print_poses(poses, inds=inds, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
