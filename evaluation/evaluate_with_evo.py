#!/usr/bin/env python3
"""
Evaluate camera pose trajectories by aligning predictions to ground truth and
reporting ATE/ARE and RPE metrics. Handles degenerate trajectories such as
pure rotations or 1D translations by falling back to orientation-based
alignment and robust scale estimation.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

_EPS = 1e-12
_CSV_FIELDS = [
    "name",
    "frames",
    "rank_gt",
    "scale",
    "pre_scale",
    "rotation_source",
    "ATE_RMSE",
    "RTE_RMSE",
    "RRE_RMSE",
    "path_length_gt",
    "path_length_pred",
]


@dataclass
class TrajectoryResult:
    name: str
    frames: int
    rank_gt: int
    scale: float
    pre_scale: float
    rotation_source: str
    ate_rmse: float
    rte_rmse: float
    rre_rmse: float
    path_length_gt: float
    path_length_pred: float


def _normalize_pose(mat: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = mat[:3, :3]

    col_norm = float(np.linalg.norm(mat[:3, 3]))
    row_norm = float(np.linalg.norm(mat[3, :3]))
    if col_norm > 1e-8 or abs(mat[3, 3] - 1.0) > 1e-8:
        pose[:3, 3] = mat[:3, 3]
    elif row_norm > 1e-8:
        pose[:3, 3] = mat[3, :3]
    else:
        pose[:3, 3] = 0.0
    return pose


def _load_pose_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        mats = data["data"].astype(np.float64)
        if "idx" in data:
            stamps = data["idx"].astype(np.float64)
        elif "inds" in data:
            stamps = data["inds"].astype(np.float64)
        else:
            raise KeyError(f"{path} missing idx/inds key")
    poses = np.stack([_normalize_pose(mat) for mat in mats], axis=0)
    return poses, stamps


def _intersect_on_stamps(
    gt_poses: np.ndarray,
    gt_stamps: np.ndarray,
    pr_poses: np.ndarray,
    pr_stamps: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_stamps = np.asarray(gt_stamps, dtype=np.float64).reshape(-1)
    pr_stamps = np.asarray(pr_stamps, dtype=np.float64).reshape(-1)

    gt_order = np.argsort(gt_stamps)
    pr_order = np.argsort(pr_stamps)

    matched_gt: List[int] = []
    matched_pr: List[int] = []
    matched_stamp: List[float] = []
    i = 0
    j = 0
    while i < gt_order.size and j < pr_order.size:
        gt_idx = int(gt_order[i])
        pr_idx = int(pr_order[j])
        sgt = float(gt_stamps[gt_idx])
        spr = float(pr_stamps[pr_idx])
        diff = spr - sgt
        if abs(diff) <= tol:
            matched_gt.append(gt_idx)
            matched_pr.append(pr_idx)
            matched_stamp.append(0.5 * (sgt + spr))
            i += 1
            j += 1
        elif diff > tol:
            i += 1
        else:
            j += 1

    if not matched_gt:
        raise ValueError("no overlapping frame ids between GT and predictions")
    poses_gt = gt_poses[np.array(matched_gt, dtype=int)]
    poses_pr = pr_poses[np.array(matched_pr, dtype=int)]
    stamps = np.array(matched_stamp, dtype=np.float64)
    return poses_gt, poses_pr, stamps


def _covariance_rank(points: np.ndarray) -> int:
    if points.shape[0] < 2:
        return 0
    centered = points - points.mean(axis=0, keepdims=True)
    if not np.any(np.isfinite(centered)):
        return 0
    _, singular_vals, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_vals.size == 0:
        return 0
    tol = 1e-8 * singular_vals[0]
    return int(np.sum(singular_vals > tol))


def _orientation_alignment(rot_gt: np.ndarray, rot_pr: np.ndarray) -> np.ndarray:
    accum = np.zeros((3, 3), dtype=np.float64)
    for Rg, Rp in zip(rot_gt, rot_pr):
        accum += Rg @ Rp.T
    if np.linalg.norm(accum) < _EPS:
        return np.eye(3, dtype=np.float64)
    U, _, Vt = np.linalg.svd(accum)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


def _rot_align_vec_to_vec(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src / (np.linalg.norm(src) + _EPS)
    dst = dst / (np.linalg.norm(dst) + _EPS)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180 deg rotation around any axis orthogonal to src
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(src, axis)
        v = v / (np.linalg.norm(v) + _EPS)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3, dtype=np.float64) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1 - c) / (s**2 + _EPS))


def _positive_scale(Pgt_c: np.ndarray, Ppr_c: np.ndarray) -> float:
    denom = float(np.sum(Ppr_c**2))
    if denom < _EPS:
        return 1.0
    num = float(np.sum(Ppr_c * Pgt_c))
    if num <= 0.0:
        alt = float(np.sum(Pgt_c**2))
        if alt <= 0.0:
            return 1.0
        return np.sqrt(max(alt / denom, 0.0))
    return num / denom


def _robust_scale(Pgt: np.ndarray, Ppr: np.ndarray) -> float:
    if Pgt.shape[0] < 2:
        return 1.0
    diffs_gt = np.linalg.norm(np.diff(Pgt, axis=0), axis=1)
    diffs_pr = np.linalg.norm(np.diff(Ppr, axis=0), axis=1)
    mask = (diffs_gt > 1e-9) & (diffs_pr > 1e-9)
    ratios = diffs_gt[mask] / diffs_pr[mask] if np.any(mask) else np.empty(0)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size >= 3:
        return float(np.median(ratios))
    Pgt_c = Pgt - Pgt.mean(axis=0, keepdims=True)
    Ppr_c = Ppr - Ppr.mean(axis=0, keepdims=True)
    return _positive_scale(Pgt_c, Ppr_c)


def _robust_scale_1d(gt_vals: np.ndarray, pr_vals: np.ndarray) -> float:
    diffs_gt = np.diff(gt_vals)
    diffs_pr = np.diff(pr_vals)
    mask = (np.abs(diffs_gt) > 1e-9) & (np.abs(diffs_pr) > 1e-9)
    ratios = diffs_gt[mask] / diffs_pr[mask] if np.any(mask) else np.empty(0)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size >= 3:
        scale = float(np.median(np.abs(ratios)))
    else:
        std_pr = float(np.std(pr_vals))
        std_gt = float(np.std(gt_vals))
        scale = std_gt / std_pr if std_pr > 1e-12 else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        return 1.0
    return scale


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(diffs))


def _align_trajectories(
    poses_gt: np.ndarray, poses_pr: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    prep = poses_pr.copy()

    Pgt = poses_gt[:, :3, 3]
    Rgt = poses_gt[:, :3, :3]

    pre_scale = 1.0
    path_gt = _path_length(Pgt)
    path_pr = _path_length(prep[:, :3, 3])
    if path_gt > 1e-9 and path_pr > 1e-9:
        pre_scale = path_gt / path_pr
        prep[:, :3, 3] *= pre_scale

    Ppr = prep[:, :3, 3]
    Rpr = prep[:, :3, :3]

    rank_gt = _covariance_rank(Pgt)

    mu_gt = Pgt.mean(axis=0)
    mu_pr = Ppr.mean(axis=0)
    src = Ppr - mu_pr
    dst = Pgt - mu_gt
    cov = dst.T @ src / max(Pgt.shape[0], 1)
    cov_norm = float(np.linalg.norm(cov))

    use_position_rotation = cov_norm > 1e-9 and rank_gt >= 2

    if use_position_rotation:
        U, singular_vals, Vt = np.linalg.svd(cov)
        diag = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            diag[-1, -1] = -1.0
        R = U @ diag @ Vt
        Ppr_rot = (R @ Ppr.T).T
        R_aligned = np.einsum("ij,njk->nik", R, Rpr)
        scale = _robust_scale(Pgt, Ppr_rot)
        t = mu_gt - scale * Ppr_rot.mean(axis=0)
        rotation_source = "positions"
    elif rank_gt == 1:
        axis_gt = _principal_axis(Pgt)
        axis_pr = _principal_axis(Ppr)
        R_axis = _rot_align_vec_to_vec(axis_pr, axis_gt)

        Ppr_rot = (R_axis @ Ppr.T).T
        R_aligned = np.einsum("ij,njk->nik", R_axis, Rpr)

        proj_gt = (Pgt - mu_gt) @ axis_gt
        proj_pr = (Ppr_rot - Ppr_rot.mean(axis=0)) @ axis_gt
        scale = _robust_scale_1d(proj_gt, proj_pr)

        t = mu_gt - scale * Ppr_rot.mean(axis=0)
        R = R_axis
        singular_vals = np.zeros(3, dtype=np.float64)
        rotation_source = "axis"
    else:
        R = _orientation_alignment(Rgt, Rpr)
        singular_vals = np.zeros(3, dtype=np.float64)
        Ppr_rot = (R @ Ppr.T).T
        R_aligned = np.einsum("ij,njk->nik", R, Rpr)
        scale = 1.0
        t = mu_gt - Ppr_rot.mean(axis=0)
        rotation_source = "orientations"

    aligned = poses_pr.copy()
    aligned[:, :3, :3] = R_aligned
    aligned[:, :3, 3] = scale * Ppr_rot + t

    info = {
        "scale": float(scale * pre_scale),
        "rank_gt": int(rank_gt),
        "rotation_source": rotation_source,
        "cov_norm": cov_norm,
        "singular_values": singular_vals.tolist(),
        "pre_scale": float(pre_scale),
        "path_length_gt": path_gt,
        "path_length_pred": path_pr,
        "scale_local": float(scale),
        "rotation_matrix": R.tolist(),
        "translation": t.tolist(),
    }
    return aligned, info


def _rotation_angle(R: np.ndarray) -> float:
    trace = float((np.trace(R) - 1.0) * 0.5)
    trace = max(-1.0, min(1.0, trace))
    return float(np.arccos(trace))


def _rmse(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))


def _compute_metrics(
    poses_gt: np.ndarray,
    poses_est: np.ndarray,
    skip_translation: bool = False,
) -> Dict[str, float]:
    Pgt = poses_gt[:, :3, 3]
    Pest = poses_est[:, :3, 3]
    if skip_translation:
        ate_rmse = float("nan")
    else:
        trans_errors = np.linalg.norm(Pgt - Pest, axis=1)
        ate_rmse = _rmse(trans_errors)

    if poses_gt.shape[0] < 2:
        return {
            "ate_rmse": ate_rmse,
            "rte_rmse": float("nan"),
            "rre_rmse": float("nan"),
        }

    rpe_trans: List[float] = []
    rpe_rot: List[float] = []
    inv_gt_rel: List[np.ndarray] = []
    for Ta, Tb in zip(poses_gt[:-1], poses_gt[1:]):
        rel_gt = np.linalg.inv(Ta) @ Tb
        inv_gt_rel.append(np.linalg.inv(rel_gt))
    rel_est: List[np.ndarray] = [
        np.linalg.inv(Ta) @ Tb for Ta, Tb in zip(poses_est[:-1], poses_est[1:])
    ]
    for inv_rel_gt, rel_est_mat in zip(inv_gt_rel, rel_est):
        err = inv_rel_gt @ rel_est_mat
        if not skip_translation:
            rpe_trans.append(float(np.linalg.norm(err[:3, 3])))
        rpe_rot.append(float(np.rad2deg(_rotation_angle(err[:3, :3]))))

    return {
        "ate_rmse": ate_rmse,
        "rte_rmse": float("nan") if skip_translation else _rmse(rpe_trans),
        "rre_rmse": _rmse(rpe_rot),
    }


def _zero_origin(poses: np.ndarray) -> np.ndarray:
    """Translate the trajectory so the first frame sits at the origin."""
    if poses.shape[0] == 0:
        return poses
    shifted = poses.copy()
    shifted[:, :3, 3] -= poses[0, :3, 3]
    return shifted


def _evaluate_pair(
    name: str,
    gt_path: Path,
    pr_path: Path,
    stamp_tol: float,
    static_translation_threshold: float,
) -> TrajectoryResult:
    poses_gt, idx_gt = _load_pose_file(gt_path)
    poses_pr, idx_pr = _load_pose_file(pr_path)
    poses_gt, poses_pr, stamps = _intersect_on_stamps(
        poses_gt, idx_gt, poses_pr, idx_pr, stamp_tol
    )

    poses_gt = _zero_origin(poses_gt)
    poses_pr = _zero_origin(poses_pr)

    aligned, info = _align_trajectories(poses_gt, poses_pr)
    skip_translation = (
        static_translation_threshold > 0.0
        and info.get("path_length_gt", 0.0) <= static_translation_threshold
    )
    metrics = _compute_metrics(poses_gt, aligned, skip_translation=skip_translation)

    return TrajectoryResult(
        name=name,
        frames=int(len(stamps)),
        rank_gt=int(info["rank_gt"]),
        scale=float(info["scale"]),
        pre_scale=float(info.get("pre_scale", 1.0)),
        rotation_source=str(info["rotation_source"]),
        ate_rmse=float(metrics["ate_rmse"]),
        rte_rmse=float(metrics["rte_rmse"]),
        rre_rmse=float(metrics["rre_rmse"]),
        path_length_gt=float(info.get("path_length_gt", float("nan"))),
        path_length_pred=float(info.get("path_length_pred", float("nan"))),
    )


def _mean_ignore_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _print_summary(results: Sequence[TrajectoryResult]) -> None:
    print("\n===== Per-file metrics =====")
    for res in results:
        print(
            f"{res.name}: frames={res.frames:4d} rank={res.rank_gt} "
            f"scale={res.scale:.6f} (pre={res.pre_scale:.6f}) "
            f"(R:{res.rotation_source}) | "
            f"ATE={res.ate_rmse:.6f} | RTE={res.rte_rmse:.6f} | "
            f"RRE={res.rre_rmse:.6f}"
        )

    print("\n===== Overall mean (ignoring NaN) =====")
    print(f"ATE translational RMSE: {_mean_ignore_nan(r.ate_rmse for r in results):.6f}")
    print(f"RTE translational RMSE: {_mean_ignore_nan(r.rte_rmse for r in results):.6f}")
    print(f"RRE rotational RMSE:    {_mean_ignore_nan(r.rre_rmse for r in results):.6f}")


def _write_csv(results: Sequence[TrajectoryResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "name": res.name,
                    "frames": res.frames,
                    "rank_gt": res.rank_gt,
                    "scale": res.scale,
                    "pre_scale": res.pre_scale,
                    "rotation_source": res.rotation_source,
                    "ATE_RMSE": res.ate_rmse,
                    "RTE_RMSE": res.rte_rmse,
                    "RRE_RMSE": res.rre_rmse,
                    "path_length_gt": res.path_length_gt,
                    "path_length_pred": res.path_length_pred,
                }
            )
        writer.writerow(
            {
                "name": "__MEAN__",
                "frames": "",
                "rank_gt": "",
                "scale": "",
                "pre_scale": "",
                "rotation_source": "",
                "ATE_RMSE": _mean_ignore_nan(r.ate_rmse for r in results),
                "RTE_RMSE": _mean_ignore_nan(r.rte_rmse for r in results),
                "RRE_RMSE": _mean_ignore_nan(r.rre_rmse for r in results),
                "path_length_gt": _mean_ignore_nan(r.path_length_gt for r in results),
                "path_length_pred": _mean_ignore_nan(r.path_length_pred for r in results),
            }
        )


def _select_top_errors(results: Sequence[TrajectoryResult], topk: int) -> List[TrajectoryResult]:
    topk = max(int(topk), 0)
    finite = [r for r in results if np.isfinite(r.ate_rmse)]
    if topk == 0:
        return []
    return sorted(finite, key=lambda r: r.ate_rmse, reverse=True)[:topk]


def _write_top_errors(top_results: Sequence[TrajectoryResult], csv_path: Path) -> None:
    if not top_results:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for res in top_results:
            writer.writerow(
                {
                    "name": res.name,
                    "frames": res.frames,
                    "rank_gt": res.rank_gt,
                    "scale": res.scale,
                    "pre_scale": res.pre_scale,
                    "rotation_source": res.rotation_source,
                    "ATE_RMSE": res.ate_rmse,
                    "RTE_RMSE": res.rte_rmse,
                    "RRE_RMSE": res.rre_rmse,
                    "path_length_gt": res.path_length_gt,
                    "path_length_pred": res.path_length_pred,
                }
            )
    print(f"Top-{len(top_results)} ATE误差结果已写入 {csv_path}")


def _find_video(name: str, pred_dir: Path, gt_dir: Path) -> Path | None:
    candidates = [
        pred_dir / f"{name}.mp4",
        pred_dir.parent / f"{name}.mp4",
        gt_dir / f"{name}.mp4",
        gt_dir.parent / f"{name}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _copy_top_videos(
    top_results: Sequence[TrajectoryResult],
    pred_dir: Path,
    gt_dir: Path,
    out_dir: Path,
) -> None:
    if not top_results:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for res in top_results:
        src = _find_video(res.name, pred_dir, gt_dir)
        if src is None:
            print(f"Warning: 未找到 {res.name} 对应的 mp4，跳过。")
            continue
        dst = out_dir / src.name
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:  # pragma: no cover - 防御性日志
            print(f"Warning: 复制 {src} 到 {dst} 失败: {exc}")
    print(f"Top-{copied} 视频已复制到 {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate GT vs predicted pose trajectories with degeneracy-safe alignment."
    )
    parser.add_argument("gt_dir", type=Path, help="Directory containing ground-truth .npz files")
    parser.add_argument("pred_dir", type=Path, help="Directory containing predicted .npz files")
    parser.add_argument(
        "--stamp-tol",
        type=float,
        default=1e-3,
        help="Timestamp match tolerance for aligning GT/pred frames (default: 1e-3).",
    )
    parser.add_argument(
        "--static-translation-threshold",
        type=float,
        default=-1.0,
        help=(
            "If >0, skip translation metrics when GT path length is <= threshold "
            "(use rotation error only)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (directory will be created if needed)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Write the top-K highest ATE errors to CSV (default: 30)",
    )
    parser.add_argument(
        "--topk-out",
        type=Path,
        default=Path(__file__).resolve().parent / "top30_errors.csv",
        help="CSV path for the top-K ATE errors (default: evaluation/top30_errors.csv)",
    )
    parser.add_argument(
        "--topk-video-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "top30error",
        help="Directory to copy the top-K error videos (default: evaluation/top30error)",
    )
    parser.add_argument(
        "--no-copy-topk-videos",
        action="store_false",
        dest="copy_topk_videos",
        help="Disable copying the top-K error videos.",
    )
    parser.set_defaults(copy_topk_videos=True)
    args = parser.parse_args()

    if not args.gt_dir.is_dir() or not args.pred_dir.is_dir():
        print("Error: both gt_dir and pred_dir must be directories containing .npz files.")
        return 1

    gt_files = {f.name: f for f in args.gt_dir.glob("*.npz")}
    pr_files = {f.name: f for f in args.pred_dir.glob("*.npz")}
    common_names = sorted(gt_files.keys() & pr_files.keys())
    if not common_names:
        print("Error: no matching .npz filenames between the two directories.")
        return 1

    results: List[TrajectoryResult] = []
    for fname in common_names:
        name = Path(fname).stem
        try:
            res = _evaluate_pair(
                name,
                gt_files[fname],
                pr_files[fname],
                args.stamp_tol,
                args.static_translation_threshold,
            )
            results.append(res)
        except Exception as exc:
            print(f"Warning: could not process {fname}: {exc}")

    if not results:
        print("Error: evaluation produced no results.")
        return 1

    _print_summary(results)
    top_results = _select_top_errors(results, args.topk)
    _write_top_errors(top_results, args.topk_out)
    if args.copy_topk_videos:
        _copy_top_videos(top_results, args.pred_dir, args.gt_dir, args.topk_video_dir)
    if args.csv:
        _write_csv(results, args.csv)
        print(f"Wrote CSV to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
