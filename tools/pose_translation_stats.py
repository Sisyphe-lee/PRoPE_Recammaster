import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_pose_files(root: Path, video_root: Path | None = None, limit: int | None = None) -> List[Path]:
    if limit is not None and limit > 0:
        collected: List[Path] = []
        for p in root.glob("*.npz"):
            if p.is_file():
                if video_root is None or (video_root / f"{p.stem}.mp4").exists():
                    collected.append(p)
            if len(collected) >= limit:
                break
        return sorted(collected)
    return sorted(
        p
        for p in root.rglob("*.npz")
        if p.is_file() and (video_root is None or (video_root / f"{p.stem}.mp4").exists())
    )


def _select_pose_key(data: np.lib.npyio.NpzFile) -> str:
    if "data" in data:
        return "data"
    for key in data.files:
        if key != "inds":
            return key
    return data.files[0]


def load_max_translation_norm(path: Path) -> float:
    data = np.load(path, allow_pickle=False)
    key = _select_pose_key(data)
    poses = np.asarray(data[key])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"期望形状 [N,4,4]，实际为 {poses.shape} ({path})")
    translations = poses[:, :3, 3].astype(np.float64)
    if translations.size == 0:
        return 0.0
    norms = np.linalg.norm(translations, axis=1)
    return float(norms.max())


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(values.min(initial=0.0)),
        "max": float(values.max(initial=0.0)),
        "mean": float(values.mean() if values.size > 0 else 0.0),
        "p5": float(np.percentile(values, 5) if values.size > 0 else 0.0),
        "p10": float(np.percentile(values, 10) if values.size > 0 else 0.0),
        "p25": float(np.percentile(values, 25) if values.size > 0 else 0.0),
        "median": float(np.percentile(values, 50) if values.size > 0 else 0.0),
        "p75": float(np.percentile(values, 75) if values.size > 0 else 0.0),
        "p90": float(np.percentile(values, 90) if values.size > 0 else 0.0),
        "p99": float(np.percentile(values, 99) if values.size > 0 else 0.0),
    }


def _nearest_path(values: List[Tuple[float, Path]], target: float) -> Tuple[float, Path]:
    closest = min(values, key=lambda x: abs(x[0] - target))
    return closest


def plot_histogram(values: np.ndarray, output: Path, bins: int = 50, percentiles: dict | None = None) -> None:
    """
    用 PIL 绘制简单直方图以避免 matplotlib 的后台依赖。
    """
    counts, bin_edges = np.histogram(values, bins=bins)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 600
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 60, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # 坐标轴
    x0, y0 = margin_left, margin_top
    x1, y1 = margin_left, margin_top + plot_h
    x2 = margin_left + plot_w
    draw.line((x0, y1, x2, y1), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)

    max_count = counts.max() if counts.size > 0 else 0
    for i, count in enumerate(counts):
        if max_count <= 0:
            bar_h = 0
        else:
            bar_h = int(round((count / max_count) * plot_h))
        bar_x0 = x0 + int(i * plot_w / counts.size)
        bar_x1 = x0 + int((i + 1) * plot_w / counts.size) - 1
        bar_y0 = y1 - bar_h
        draw.rectangle((bar_x0, bar_y0, bar_x1, y1), fill="steelblue", outline="black")

    # 轴标签
    min_val = bin_edges[0] if bin_edges.size > 0 else 0.0
    max_val = bin_edges[-1] if bin_edges.size > 0 else 0.0
    label_y = y1 + 8
    # Add more ticks in [0,1] to highlight small translations
    tick_positions = [min_val, 0.1, 0.25, 0.5, 1.0, max_val]
    if percentiles:
        tick_positions.extend([percentiles.get(k) for k in ("p10", "p25", "median", "p75", "p90") if percentiles.get(k) is not None])
    tick_positions = [t for t in tick_positions if min_val <= t <= max_val]
    for tval in tick_positions:
        frac = 0 if max_val == min_val else (tval - min_val) / (max_val - min_val)
        tx = x0 + int(frac * plot_w)
        draw.line((tx, y1, tx, y1 + 5), fill="black", width=1)
        draw.text((tx, label_y), f"{tval:.4f}", fill="black", font=font, anchor="la")

    draw.text((x0, margin_top - 20), f"n={values.size}", fill="black", font=font)
    draw.text((width // 2, height - margin_bottom // 2), "Max translation norm", fill="black", font=font, anchor="mm")
    draw.text((10, height // 2), "Count", fill="black", font=font)

    img.save(output)


def main():
    parser = argparse.ArgumentParser(description="统计相机位姿文件的最大平移分布，并绘制直方图")
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("/nas/datasets/vipe_wild_sdg_1m/pose"),
        help="包含 pose npz 的目录",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="可选，对应 mp4 目录；提供后仅统计存在匹配视频的 pose。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="仅采样前 N 个文件（按文件名排序），0 表示全部",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/pose_translation_hist.png"),
        help="直方图输出路径",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="直方图分箱数量",
    )
    parser.add_argument(
        "--sample-output-dir",
        type=Path,
        default=Path("data_check"),
        help="将采样的 mp4 拷贝到此目录，文件名追加 _pXX 后缀（需提供 --video-root）。",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="每个分位区间随机拷贝的样本数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="采样随机种子。",
    )
    args = parser.parse_args()

    pose_files = find_pose_files(
        args.pose_root,
        video_root=args.video_root,
        limit=None if args.limit == 0 else args.limit,
    )
    if not pose_files:
        raise FileNotFoundError(f"目录下未找到 npz 文件: {args.pose_root}")
    print(f"扫描文件数: {len(pose_files)} (pose_root={args.pose_root}, video_root={args.video_root})")

    values: List[Tuple[float, Path]] = []
    errors: List[Tuple[Path, Exception]] = []
    for path in pose_files:
        try:
            max_norm = load_max_translation_norm(path)
            values.append((max_norm, path))
        except Exception as exc:
            errors.append((path, exc))

    max_norms = np.array([v for v, _ in values], dtype=np.float64)
    stats = summarize(max_norms)
    print("统计:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")

    # Nearest paths to key percentiles
    for label in ("p5", "p10", "p25", "median", "p75", "p90", "p99"):
        target = stats[label]
        norm, path = _nearest_path(values, target)
        print(f"{label}≈{target:.6f} -> {norm:.6f} | {path}")

    topk = sorted(values, key=lambda x: x[0], reverse=True)[:5]
    print("最大位移 Top 5:")
    for norm, path in topk:
        print(f"  {norm:.6f} | {path}")

    # 直方图：忽略 p99 以外的数据，以减少长尾影响
    filtered = max_norms[max_norms <= stats["p99"]] if max_norms.size > 0 else max_norms
    plot_histogram(filtered, args.output, bins=args.bins, percentiles=stats)
    print(f"直方图已保存: {args.output.resolve()} (仅统计 <= p99 部分)")

    if errors:
        print(f"[warn] 读取失败 {len(errors)} 个文件，示例: {errors[0][0]} -> {errors[0][1]}")

    # Optional sampling and copying videos by percentile buckets
    if args.video_root is not None:
        buckets = [
            ("p05", 0.0, stats["p5"]),
            ("p10", stats["p5"], stats["p10"]),
            ("p25", stats["p10"], stats["p25"]),
            ("p50", stats["p25"], stats["median"]),
            ("p75", stats["median"], stats["p75"]),
            ("p90", stats["p75"], stats["p90"]),
            ("p99", stats["p90"], stats["p99"]),
            ("p100", stats["p99"], stats["max"]),
            # absolute ranges
            ("abs_0_1e-4", 0.0, 1e-4),
            ("abs_1e-4_1e-3", 1e-4, 1e-3),
            ("abs_1e-3_1e-2", 1e-3, 1e-2),
            ("abs_1e-2_1e-1", 1e-2, 1e-1),
            ("abs_1e-1_5e-1", 1e-1, 5e-1),
            ("abs_5e-1_1", 5e-1, 1.0),
            ("abs_1_5", 1.0, 5.0),
            ("abs_5_10", 5.0, 10.0),
            ("abs_10_40", 10.0, 40.0),
        ]
        rng = np.random.default_rng(args.seed)
        args.sample_output_dir.mkdir(parents=True, exist_ok=True)
        for label, low, high in buckets:
            candidates = [(n, p) for n, p in values if (n >= low and n <= high)]
            if not candidates:
                print(f"[sample] 区间 {label} 无可用样本（{low:.6f}-{high:.6f}）")
                continue
            count = min(args.sample_count, len(candidates))
            indices = rng.choice(len(candidates), size=count, replace=False)
            print(f"[sample] 区间 {label} ({low:.6f}-{high:.6f}) 可用 {len(candidates)}，采样 {count}")
            bucket_dir = args.sample_output_dir / label
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for idx in indices:
                norm, pose_path = candidates[int(idx)]
                stem = pose_path.stem
                video_path = args.video_root / f"{stem}.mp4"
                if not video_path.exists():
                    print(f"[sample][warn] 缺少视频 {video_path}，跳过")
                    continue
                dst = bucket_dir / f"{label}_{stem}.mp4"
                shutil.copy2(video_path, dst)
                # print(f"[sample] {label} {norm:.6f} | {video_path} -> {dst}")


if __name__ == "__main__":
    main()
