"""
Filter SDG samples by keeping small translation and rotation while retaining at least a target ratio.
默认行为：根据翻译和最大旋转角度的分位数筛选，保留至少 60% 样本。
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.dataset import (  # noqa: E402
    Camera,
    DatasetSpec,
    SdgImageConditionDataset,
    compute_relative_c2w,
    invert_SE3_np,
    normalize_translation,
    resolve_tensor_path,
)

TARGET_FRAMES = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SDG by small translation and rotation.")
    parser.add_argument("--metadata", type=Path, required=True, help="SDG metadata CSV (e.g., metadata/output_sdg_pose_filtered.csv).")
    parser.add_argument("--dataset-root", type=Path, required=True, help="SDG dataset root (contains rgb/pose/intrinsics).")
    parser.add_argument("--tensor-suffix", type=str, default=".wan22.tensors.pth", help="Tensor suffix for latent files.")
    parser.add_argument("--target-keep", type=float, default=0.6, help="目标保留率（0-1），小运动优先。")
    parser.add_argument("--quantile-step", type=float, default=5.0, help="当保留率不足时，每次降低阈值的分位步长（百分位）。")
    parser.add_argument("--sample-limit", type=int, default=0, help="可选：限制处理的样本数（0=全部）。")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子。")
    parser.add_argument("--fx-max", type=float, default=1200.0, help="内参 fx/fy 上限（0 关闭检查）。")
    parser.add_argument("--cx-dev-max", type=float, default=120.0, help="cx 偏离中心的最大像素（0 关闭检查）。")
    parser.add_argument("--cy-dev-max", type=float, default=80.0, help="cy 偏离中心的最大像素（0 关闭检查）。")
    parser.add_argument("--det-min", type=float, default=0.9, help="旋转矩阵行列式下限（0 关闭检查）。")
    parser.add_argument("--jump-trans-thr", type=float, default=0.2, help="相邻帧平移跳变阈值（0 关闭检查）。")
    parser.add_argument("--jump-rot-deg", type=float, default=5.0, help="相邻帧旋转跳变阈值，单位度（0 关闭检查）。")
    parser.add_argument("--output", type=Path, required=True, help="输出 CSV 路径。")
    return parser.parse_args()


def rotation_to_euler(rot_mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sy = np.sqrt(rot_mats[:, 0, 0] ** 2 + rot_mats[:, 1, 0] ** 2)
    yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    pitch = np.arctan2(-rot_mats[:, 2, 0], sy + 1e-8)
    roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
    return yaw, pitch, roll


def compute_pose_metrics(c2ws: np.ndarray, get_relative_pose_fn) -> Dict[str, float]:
    cam_params = [Camera(c2w) for c2w in c2ws]
    ref_cam = cam_params[0]
    rel_c2w = compute_relative_c2w(cam_params, ref_cam, get_relative_pose_fn)
    rel_c2w_norm, _, _ = normalize_translation(rel_c2w, rel_c2w)
    rel_w2c = np.stack([invert_SE3_np(T) for T in rel_c2w_norm], axis=0)

    translations = rel_w2c[:, :3, 3]
    t_norms = np.linalg.norm(translations, axis=1)
    t_max = float(np.max(t_norms)) if t_norms.size else 0.0

    rot = rel_w2c[:, :3, :3]
    det_vals = np.linalg.det(rot)
    det_min = float(np.min(det_vals)) if det_vals.size else 1.0
    det_max = float(np.max(det_vals)) if det_vals.size else 1.0
    yaw, pitch, roll = rotation_to_euler(rot)
    yaw_deg = np.abs(yaw) * 180.0 / math.pi
    pitch_deg = np.abs(pitch) * 180.0 / math.pi
    roll_deg = np.abs(roll) * 180.0 / math.pi
    rot_max = float(np.max([np.max(yaw_deg), np.max(pitch_deg), np.max(roll_deg)]) if rot.size else 0.0)

    # 帧间跳变统计
    jump_trans = 0.0
    jump_rot = 0.0
    if rel_w2c.shape[0] > 1:
        diff_t = np.diff(rel_w2c[:, :3, 3], axis=0)
        jump_trans = float(np.max(np.linalg.norm(diff_t, axis=1)))
        Rs = rel_w2c[:, :3, :3]
        rel_R = np.einsum("bij,bjk->bik", Rs[1:], np.transpose(Rs[:-1], (0, 2, 1)))
        angles = np.arccos(np.clip((np.trace(rel_R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
        jump_rot = float(np.max(angles) * 180.0 / math.pi)

    return {
        "translation_max": t_max,
        "rotation_max": rot_max,
        "yaw_max": float(np.max(yaw_deg)) if yaw_deg.size else 0.0,
        "pitch_max": float(np.max(pitch_deg)) if pitch_deg.size else 0.0,
        "roll_max": float(np.max(roll_deg)) if roll_deg.size else 0.0,
        "jump_translation_max": jump_trans,
        "jump_rotation_max": jump_rot,
        "det_min": det_min,
        "det_max": det_max,
    }


def resolve_paths_with_metadata(meta_path: Path, dataset_root: Path, tensor_suffixes: Tuple[str, ...]) -> List[Tuple[str, Dict]]:
    df = pd.read_csv(meta_path)
    if "video_absolute_path" not in df.columns:
        raise ValueError(f"{meta_path} 缺少 video_absolute_path 列")
    rows = []
    for row in df.to_dict(orient="records"):
        video_path = row["video_absolute_path"]
        tensor_path = resolve_tensor_path(video_path, dataset_root=dataset_root.as_posix(), tensor_suffixes=tensor_suffixes)
        if tensor_path and Path(tensor_path).exists():
            rows.append((tensor_path, row))
        else:
            print(f"[warn] 未找到 tensor: {video_path}")
    return rows


def choose_thresholds(trans_vals: np.ndarray, rot_vals: np.ndarray, target_keep: float, step: float) -> Tuple[float, float, float, float]:
    q = max(0.0, min(100.0, target_keep * 100.0))
    while True:
        trans_thr = float(np.percentile(trans_vals, q)) if trans_vals.size else 0.0
        rot_thr = float(np.percentile(rot_vals, q)) if rot_vals.size else 0.0
        keep_mask = (trans_vals <= trans_thr) & (rot_vals <= rot_thr)
        keep_ratio = float(np.mean(keep_mask)) if keep_mask.size else 0.0
        if keep_ratio >= target_keep or q >= 100.0:
            return trans_thr, rot_thr, keep_ratio, q
        q = min(100.0, q + step)


def main():
    args = parse_args()
    tensor_suffixes = (args.tensor_suffix,)

    spec = DatasetSpec(name="sdg", root=args.dataset_root.as_posix(), metadata_path=args.metadata.as_posix())
    paired = resolve_paths_with_metadata(args.metadata, Path(spec.root), tensor_suffixes)
    if not paired:
        print("[warn] 无可用样本，退出")
        return

    if args.sample_limit and args.sample_limit > 0 and len(paired) > args.sample_limit:
        rng = np.random.default_rng(args.seed)
        paired = rng.choice(paired, size=args.sample_limit, replace=False).tolist()

    paths = [p for p, _ in paired]
    meta_map = {p: m for p, m in paired}
    dataset = SdgImageConditionDataset(
        steps_per_epoch=1,
        paths=paths,
        fixed_length=len(paths),
        seed=args.seed,
        dataset_root=spec.root,
    )

    records = []
    trans_vals = []
    rot_vals = []
    errors = 0
    for path in paths:
        try:
            c2ws = dataset._load_camera_sequence(path, TARGET_FRAMES)
            intr = dataset._load_intrinsics_for_sample(path, TARGET_FRAMES).numpy()
            stats = compute_pose_metrics(c2ws, dataset.get_relative_pose)
            base = meta_map.get(path, {})
            rec = dict(base)
            # 内参统计（平均值）
            fx_mean = float(np.mean(intr[:, 0, 0])) if intr.size else 0.0
            fy_mean = float(np.mean(intr[:, 1, 1])) if intr.size else 0.0
            cx_mean = float(np.mean(intr[:, 0, 2])) if intr.size else 0.0
            cy_mean = float(np.mean(intr[:, 1, 2])) if intr.size else 0.0
            rec.update(
                {
                    "tensor_path": path,
                    **stats,
                    "fx_mean": fx_mean,
                    "fy_mean": fy_mean,
                    "cx_mean": cx_mean,
                    "cy_mean": cy_mean,
                }
            )
            records.append(rec)
            trans_vals.append(stats["translation_max"])
            rot_vals.append(stats["rotation_max"])
        except Exception as exc:
            errors += 1
            print(f"[warn] 处理样本失败 {path}: {exc}")

    if not records:
        print("[warn] 没有成功的样本，退出")
        return

    trans_arr = np.asarray(trans_vals, dtype=np.float32)
    rot_arr = np.asarray(rot_vals, dtype=np.float32)
    trans_thr, rot_thr, keep_ratio, used_quantile = choose_thresholds(
        trans_arr, rot_arr, target_keep=args.target_keep, step=args.quantile_step
    )

    kept = []
    target_w, target_h = dataset.image_size
    cx_center = float(target_w) / 2.0
    cy_center = float(target_h) / 2.0
    for rec in records:
        if not (rec["translation_max"] <= trans_thr and rec["rotation_max"] <= rot_thr):
            continue
        if args.det_min > 0 and rec["det_min"] < args.det_min:
            continue
        if args.fx_max > 0 and (rec["fx_mean"] > args.fx_max or rec["fy_mean"] > args.fx_max):
            continue
        if args.cx_dev_max > 0 and abs(rec["cx_mean"] - cx_center) > args.cx_dev_max:
            continue
        if args.cy_dev_max > 0 and abs(rec["cy_mean"] - cy_center) > args.cy_dev_max:
            continue
        if args.jump_trans_thr > 0 and rec["jump_translation_max"] > args.jump_trans_thr:
            continue
        if args.jump_rot_deg > 0 and rec["jump_rotation_max"] > args.jump_rot_deg:
            continue
        kept.append(rec)

    df_out = pd.DataFrame(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print(
        f"[sdg] 总计 {len(records)}，保留 {len(kept)} ({len(kept)/len(records):.2%})，"
        f"trans_thr={trans_thr:.4f}，rot_thr={rot_thr:.4f}，quantile={used_quantile:.1f}，错误 {errors}。输出 -> {args.output}"
    )


if __name__ == "__main__":
    main()
