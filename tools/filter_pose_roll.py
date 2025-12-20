"""
Filter SDG/Rel10k samples to boost roll dominance and roll magnitude.

规则（对每个数据集单独计算）：
1) 先统计每个样本的最大 yaw/pitch/roll（度，取绝对值）和最大平移。
2) 计算 roll_thr = max(roll_min_deg, percentile(roll_max, roll_quantile)).
3) 保留：roll_max >= roll_thr
   或 roll_max >= dom_ratio * max(pitch_max, yaw_max) 且 roll_max >= roll_min_deg。

输出 CSV 包含原始 metadata 列及统计值。
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, REPO_ROOT.as_posix())

from src.dataset import (  # noqa: E402
    Camera,
    DatasetSpec,
    RelEstate10kImageConditionDataset,
    SdgImageConditionDataset,
    compute_relative_c2w,
    invert_SE3_np,
    normalize_translation,
    resolve_tensor_path,
)

TARGET_FRAMES = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SDG/Rel10k by roll dominance.")
    parser.add_argument("--sdg-metadata", type=Path, required=True, help="metadata/output_sdg_pose_filtered.csv")
    parser.add_argument("--sdg-root", type=Path, required=True, help="Dataset root for SDG (contains rgb/pose/intrinsics).")
    parser.add_argument("--rel10k-metadata", type=Path, required=True, help="metadata/metadata_re10k.csv")
    parser.add_argument("--rel10k-root", type=Path, required=True, help="RelEstate10k root (train split).")
    parser.add_argument("--tensor-suffix", type=str, default=".wan22.tensors.pth", help="Tensor suffix to resolve latents.")
    parser.add_argument("--sdg-roll-min-deg", type=float, default=5.0, help="SDG: hard lower bound for roll threshold.")
    parser.add_argument("--sdg-roll-quantile", type=float, default=70.0, help="SDG: quantile (0-100) for roll threshold.")
    parser.add_argument("--rel-roll-min-deg", type=float, default=4.0, help="Rel10k: hard lower bound for roll threshold.")
    parser.add_argument("--rel-roll-quantile", type=float, default=65.0, help="Rel10k: quantile (0-100) for roll threshold.")
    parser.add_argument("--dom-ratio", type=float, default=0.8, help="Roll must reach dom_ratio * max(pitch,yaw) to count as dominant.")
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.4,
        help="若初始筛选后保留率低于该值，自动降低 roll_quantile（每次减 5）放宽阈值，直至达到或无更低 quantile。",
    )
    parser.add_argument(
        "--target-roll-ratio",
        type=float,
        default=0.45,
        help="可选：控制最终 roll 主轴占比（0-1）。若 roll 占比高于该值，则下采样 roll-dominant 样本以接近目标。",
    )
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional cap per dataset (0=all).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling when sample-limit>0.")
    parser.add_argument("--output-sdg", type=Path, required=True, help="Output CSV for filtered SDG.")
    parser.add_argument("--output-rel10k", type=Path, required=True, help="Output CSV for filtered Rel10k.")
    return parser.parse_args()


def rotation_to_euler(rot_mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sy = np.sqrt(rot_mats[:, 0, 0] ** 2 + rot_mats[:, 1, 0] ** 2)
    yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    pitch = np.arctan2(-rot_mats[:, 2, 0], sy + 1e-8)
    roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
    return yaw, pitch, roll


def compute_pose_metrics(c2ws: np.ndarray, get_relative_pose_fn):
    cam_params = [Camera(c2w) for c2w in c2ws]
    ref_cam = cam_params[0]
    rel_c2w = compute_relative_c2w(cam_params, ref_cam, get_relative_pose_fn)
    rel_c2w_norm, _, _ = normalize_translation(rel_c2w, rel_c2w)
    rel_w2c = np.stack([invert_SE3_np(T) for T in rel_c2w_norm], axis=0)

    translations = rel_w2c[:, :3, 3]
    t_norms = np.linalg.norm(translations, axis=1)
    t_max = float(np.max(t_norms)) if t_norms.size else 0.0

    rot = rel_w2c[:, :3, :3]
    yaw, pitch, roll = rotation_to_euler(rot)
    yaw_deg = np.abs(yaw) * 180.0 / math.pi
    pitch_deg = np.abs(pitch) * 180.0 / math.pi
    roll_deg = np.abs(roll) * 180.0 / math.pi
    yaw_max = float(np.max(yaw_deg)) if yaw_deg.size else 0.0
    pitch_max = float(np.max(pitch_deg)) if pitch_deg.size else 0.0
    roll_max = float(np.max(roll_deg)) if roll_deg.size else 0.0
    axis_idx = int(np.argmax([yaw_max, pitch_max, roll_max]))
    axis_label = ["yaw", "pitch", "roll"][axis_idx]

    return {
        "translation_max": t_max,
        "yaw_max": yaw_max,
        "pitch_max": pitch_max,
        "roll_max": roll_max,
        "dominant_axis": axis_label,
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


def process_dataset(
    name: str,
    dataset_cls,
    spec: DatasetSpec,
    meta_path: Path,
    output_path: Path,
    tensor_suffixes: Tuple[str, ...],
    roll_min_deg: float,
    roll_quantile: float,
    dom_ratio: float,
    target_roll_ratio: float,
    min_keep_ratio: float,
    sample_limit: int,
    seed: int,
):
    paired = resolve_paths_with_metadata(meta_path, Path(spec.root), tensor_suffixes)
    if not paired:
        print(f"[warn] {name} 无可用样本，跳过")
        return

    if sample_limit and sample_limit > 0 and len(paired) > sample_limit:
        rng = np.random.default_rng(seed)
        paired = rng.choice(paired, size=sample_limit, replace=False).tolist()

    paths = [p for p, _ in paired]
    meta_map = {p: m for p, m in paired}
    dataset = dataset_cls(
        steps_per_epoch=1,
        paths=paths,
        fixed_length=len(paths),
        seed=seed,
        dataset_root=spec.root,
    )

    records = []
    roll_vals = []
    errors = 0
    for path in paths:
        try:
            c2ws = dataset._load_camera_sequence(path, TARGET_FRAMES)
            stats = compute_pose_metrics(c2ws, dataset.get_relative_pose)
            roll_vals.append(stats["roll_max"])
            base = meta_map.get(path, {})
            rec = dict(base)
            rec.update({"tensor_path": path, **stats})
            records.append(rec)
        except Exception as exc:
            errors += 1
            print(f"[warn] 处理 {name} 样本失败 {path}: {exc}")

    if not records:
        print(f"[warn] {name} 没有成功的样本，跳过写入")
        return

    roll_arr = np.asarray(roll_vals, dtype=np.float32)
    base_quantile = roll_quantile
    roll_thr = max(roll_min_deg, float(np.percentile(roll_arr, base_quantile))) if roll_arr.size else roll_min_deg

    def apply_keep(threshold: float):
        kept_local = []
        roll_dom_local = []
        non_roll_dom_local = []
        for rec in records:
            roll_max = rec["roll_max"]
            pitch_max = rec["pitch_max"]
            yaw_max = rec["yaw_max"]
            cond_keep = roll_max >= threshold or (roll_max >= dom_ratio * max(pitch_max, yaw_max) and roll_max >= roll_min_deg)
            if cond_keep:
                kept_local.append(rec)
                if rec["dominant_axis"] == "roll":
                    roll_dom_local.append(rec)
                else:
                    non_roll_dom_local.append(rec)
        return kept_local, roll_dom_local, non_roll_dom_local

    kept, roll_dom, non_roll_dom = apply_keep(roll_thr)

    # 若保留率低于要求，逐步降低 quantile（放宽阈值）
    q = base_quantile
    while kept and len(kept) / len(records) < min_keep_ratio and q > 0:
        q = max(0.0, q - 5.0)
        roll_thr = max(roll_min_deg, float(np.percentile(roll_arr, q)))
        kept, roll_dom, non_roll_dom = apply_keep(roll_thr)

    # 如果 roll 占比过高，按 target_roll_ratio 下采样 roll-dominant
    rng = np.random.default_rng(seed + 123)
    if kept and 0.0 < target_roll_ratio < 1.0 and len(roll_dom) > 0:
        if len(non_roll_dom) == 0:
            # 仅有 roll-dominant 样本时不做下采样，避免全空
            pass
        else:
            desired_roll = int(target_roll_ratio / max(1e-6, 1 - target_roll_ratio) * len(non_roll_dom))
            desired_roll = max(1, desired_roll)
            if desired_roll < len(roll_dom):
                roll_dom = rng.choice(roll_dom, size=desired_roll, replace=False).tolist()
            kept = roll_dom + non_roll_dom

    df_out = pd.DataFrame(kept)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    kept_ratio = len(kept) / max(len(records), 1)
    print(
        f"[{name}] 总计 {len(records)}，保留 {len(kept)} ({kept_ratio:.2%})，阈值 roll_thr={roll_thr:.2f} 度，错误 {errors}。输出 -> {output_path}"
    )


def main():
    args = parse_args()
    tensor_suffixes = (args.tensor_suffix,)

    sdg_spec = DatasetSpec(name="sdg", root=args.sdg_root.as_posix(), metadata_path=args.sdg_metadata.as_posix())
    rel_spec = DatasetSpec(name="rel10k", root=args.rel10k_root.as_posix(), metadata_path=args.rel10k_metadata.as_posix())

    process_dataset(
        name="sdg",
        dataset_cls=SdgImageConditionDataset,
        spec=sdg_spec,
        meta_path=args.sdg_metadata,
        output_path=args.output_sdg,
        tensor_suffixes=tensor_suffixes,
        roll_min_deg=args.sdg_roll_min_deg,
        roll_quantile=args.sdg_roll_quantile,
        dom_ratio=args.dom_ratio,
        target_roll_ratio=args.target_roll_ratio,
        min_keep_ratio=args.min_keep_ratio,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )

    process_dataset(
        name="rel10k",
        dataset_cls=RelEstate10kImageConditionDataset,
        spec=rel_spec,
        meta_path=args.rel10k_metadata,
        output_path=args.output_rel10k,
        tensor_suffixes=tensor_suffixes,
        roll_min_deg=args.rel_roll_min_deg,
        roll_quantile=args.rel_roll_quantile,
        dom_ratio=args.dom_ratio,
        target_roll_ratio=args.target_roll_ratio,
        min_keep_ratio=args.min_keep_ratio,
        sample_limit=args.sample_limit,
        seed=args.seed + 17,
    )


if __name__ == "__main__":
    main()
