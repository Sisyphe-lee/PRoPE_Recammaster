"""
Streamlit 可视化：基于训练对齐逻辑统计各数据集的位姿分布。
固定 21 帧，角度取绝对值并转换为度数；仅统计轨迹内的最大值。

运行示例：
streamlit run tools/streamlit_pose_stats.py -- --config specs.json --sample-limit 10000 --tensor-suffix .wan22.tensors.pth
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 确保 repo 根目录与 DiffSynth 依赖在 sys.path 内
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "DiffSynth-Studio"
for p in (REPO_ROOT, THIRD_PARTY):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

from src.dataset import (
    Camera,
    DatasetSpec,
    MulticamImageConditionDataset,
    RelEstate10kImageConditionDataset,
    SdgImageConditionDataset,
    compute_relative_c2w,
    invert_SE3_np,
    normalize_translation,
    _gather_paths_for_spec,
)

TARGET_FRAMES = 21
DEFAULT_SPEC_TEXT = json.dumps(
    [
        {"name": "multicam", "root": "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train", "metadata_path": "metadata/output_recam.csv", "weight": 1.0},
        {"name": "sdg", "root": "/nas/datasets/vipe_wild_sdg_1m", "metadata_path": "metadata/output_sdg_with_pose.csv", "weight": 0.5},
        {"name": "rel10k", "root": "/nas/datasets/relestate10k/train", "metadata_path": "metadata/metadata_re10k.csv", "weight": 0.5},
    ],
    indent=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 pose 分布（21 帧，角度度数取绝对值）。")
    parser.add_argument("--config", type=Path, default=None, help="可选：DatasetSpec JSON 文件路径，用于预填侧边栏。")
    parser.add_argument("--tensor-suffix", type=str, default=".wan22.tensors.pth", help="覆盖默认 latent 后缀。")
    parser.add_argument("--sample-limit", type=int, default=10000, help="每个数据集采样数量上限（0=全部）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（仅用于路径截取时的稳定顺序）。")
    return parser.parse_args()


def _load_spec_text(config_path: Optional[Path]) -> str:
    if config_path and config_path.exists():
        try:
            return config_path.read_text(encoding="utf-8")
        except Exception:
            return DEFAULT_SPEC_TEXT
    return DEFAULT_SPEC_TEXT


def _parse_specs(text: str) -> List[DatasetSpec]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("DatasetSpec JSON 需为列表。")
    specs = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("DatasetSpec 列表元素需为对象。")
        specs.append(
            DatasetSpec(
                name=item.get("name"),
                root=item.get("root"),
                metadata_path=item.get("metadata_path"),
                weight=item.get("weight", 1.0),
            )
        )
    return specs


def _rotation_to_euler(rot_mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与 tools/visualize_batch_cameras 保持一致：返回 yaw/pitch/roll（弧度）。"""
    sy = np.sqrt(rot_mats[:, 0, 0] ** 2 + rot_mats[:, 1, 0] ** 2)
    yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    pitch = np.arctan2(-rot_mats[:, 2, 0], sy + 1e-8)
    roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
    return yaw, pitch, roll


def _dataset_registry() -> Dict[str, type]:
    return {
        "multicam": MulticamImageConditionDataset,
        "sdg": SdgImageConditionDataset,
        "rel10k": RelEstate10kImageConditionDataset,
    }


def _tensor_suffixes(tensor_suffix: Optional[str]) -> Tuple[str, ...]:
    if tensor_suffix and tensor_suffix.strip():
        return (tensor_suffix.strip(),)
    return (".wan22.tensors.pth",)


def compute_pose_metrics(
    c2ws: np.ndarray,
    get_relative_pose_fn,
    intr_loader=None,
    sample_path: Optional[str] = None,
):
    cam_params = [Camera(c2w) for c2w in c2ws]
    ref_cam = cam_params[0]
    rel_c2w = compute_relative_c2w(cam_params, ref_cam, get_relative_pose_fn)
    rel_c2w_norm, _, baseline = normalize_translation(rel_c2w, rel_c2w)
    rel_w2c = np.stack([invert_SE3_np(T) for T in rel_c2w_norm], axis=0)

    translations = rel_w2c[:, :3, 3]
    t_norms = np.linalg.norm(translations, axis=1)
    t_max = float(np.max(t_norms)) if t_norms.size else 0.0

    rot = rel_w2c[:, :3, :3]
    yaw, pitch, roll = _rotation_to_euler(rot)
    yaw_deg = np.abs(yaw) * 180.0 / math.pi
    pitch_deg = np.abs(pitch) * 180.0 / math.pi
    roll_deg = np.abs(roll) * 180.0 / math.pi
    yaw_max = float(np.max(yaw_deg)) if yaw_deg.size else 0.0
    pitch_max = float(np.max(pitch_deg)) if pitch_deg.size else 0.0
    roll_max = float(np.max(roll_deg)) if roll_deg.size else 0.0
    axis_idx = int(np.argmax([yaw_max, pitch_max, roll_max]))
    axis_label = ["yaw", "pitch", "roll"][axis_idx]

    fx_mean = fy_mean = cx_mean = cy_mean = 0.0
    if intr_loader is not None and sample_path is not None:
        try:
            intr = intr_loader(sample_path, c2ws.shape[0]).numpy()
            fx_mean = float(np.mean(intr[..., 0, 0]))
            fy_mean = float(np.mean(intr[..., 1, 1]))
            cx_mean = float(np.mean(intr[..., 0, 2]))
            cy_mean = float(np.mean(intr[..., 1, 2]))
        except Exception:
            pass

    return {
        "translation_max": t_max,
        "baseline": float(baseline),
        "yaw_max": yaw_max,
        "pitch_max": pitch_max,
        "roll_max": roll_max,
        "dominant_axis": axis_label,
        "fx_mean": fx_mean,
        "fy_mean": fy_mean,
        "cx_mean": cx_mean,
        "cy_mean": cy_mean,
    }


def collect_pose_stats(
    specs: List[DatasetSpec],
    tensor_suffixes: Tuple[str, ...],
    sample_limit: int,
    seed: int,
):
    registry = _dataset_registry()
    results: Dict[str, Dict[str, List[float]]] = {}
    for spec in specs:
        ds_cls = registry.get(spec.name)
        if ds_cls is None:
            continue
        paths = _gather_paths_for_spec(spec, tensor_suffixes, skip_exists_check=True)
        if not paths:
            continue
        paths = sorted(paths)
        if sample_limit and sample_limit > 0:
            np.random.seed(seed)
            paths = list(np.random.permutation(paths))[:sample_limit]

        dataset = ds_cls(
            steps_per_epoch=1,
            paths=paths,
            fixed_length=len(paths),
            seed=seed,
            dataset_root=spec.root,
        )

        trans_max: List[float] = []
        yaw_max: List[float] = []
        pitch_max: List[float] = []
        roll_max: List[float] = []
        dominant_axes: List[str] = []
        baselines: List[float] = []
        fx_vals: List[float] = []
        fy_vals: List[float] = []
        cx_vals: List[float] = []
        cy_vals: List[float] = []
        errors = 0

        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        for idx, path in enumerate(paths):
            try:
                c2ws = dataset._load_camera_sequence(path, TARGET_FRAMES)
                stats = compute_pose_metrics(
                    c2ws,
                    dataset.get_relative_pose,
                    intr_loader=dataset._load_intrinsics_for_sample,
                    sample_path=path,
                )
                trans_max.append(stats["translation_max"])
                yaw_max.append(stats["yaw_max"])
                pitch_max.append(stats["pitch_max"])
                roll_max.append(stats["roll_max"])
                dominant_axes.append(stats["dominant_axis"])
                baselines.append(stats["baseline"])
                fx_vals.append(stats["fx_mean"])
                fy_vals.append(stats["fy_mean"])
                cx_vals.append(stats["cx_mean"])
                cy_vals.append(stats["cy_mean"])
            except Exception as exc:
                errors += 1
                print(f"[warn] 处理 {spec.name} 样本失败 {path}: {exc}")
            progress = (idx + 1) / max(len(paths), 1)
            progress_bar.progress(progress)
            progress_text.text(f"{spec.name}: {idx+1}/{len(paths)}")
        progress_bar.empty()
        progress_text.empty()

        results[spec.name] = {
            "translation_max": trans_max,
            "yaw_max": yaw_max,
            "pitch_max": pitch_max,
            "roll_max": roll_max,
            "dominant_axes": dominant_axes,
            "baseline": baselines,
            "fx_mean": fx_vals,
            "fy_mean": fy_vals,
            "cx_mean": cx_vals,
            "cy_mean": cy_vals,
            "sampled": len(paths),
            "errors": errors,
        }
    return results


def _summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _hist_df(vals: List[float], bins: int = 40) -> pd.DataFrame:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return pd.DataFrame(columns=["bin", "count"])
    hist, edges = np.histogram(arr, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame({"bin": centers, "count": hist})


def _filter_angles(vals: List[float], threshold: float = 2.0) -> List[float]:
    """过滤掉小角度（默认 <2 度）的值，避免平移主导时干扰分布。"""
    return [v for v in vals if v >= threshold]


def render_results(stats: Dict[str, Dict[str, List[float]]]):
    if not stats:
        st.warning("未得到统计结果，请检查配置。")
        return
    for name, info in stats.items():
        st.header(f"{name}（样本 {info['sampled']}，失败 {info['errors']}）")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Translation max 分布")
            st.json(_summarize(info["translation_max"]))
            df_hist = _hist_df(info["translation_max"])
            if df_hist.empty:
                st.info("无平移统计数据。")
            else:
                st.bar_chart(df_hist, x="bin", y="count")
        with col2:
            st.subheader("最大旋转轴占比")
            if info["dominant_axes"]:
                vc = pd.Series(info["dominant_axes"]).value_counts(normalize=True)
                df_axis = pd.DataFrame({"axis": vc.index, "ratio": vc.values})
                st.bar_chart(df_axis, x="axis", y="ratio")
            else:
                st.info("无可用数据。")
        with st.expander("平移归一化基准 (normalize_translation 的 max_norm)", expanded=False):
            st.json(_summarize(info.get("baseline", [])))
            df_hist = _hist_df(info.get("baseline", []))
            if not df_hist.empty:
                st.bar_chart(df_hist, x="bin", y="count")

        st.subheader("每轴最大旋转角度分布（度）")
        cols = st.columns(3)
        for col, axis_name, vals in zip(
            cols,
            ["yaw", "pitch", "roll"],
            [info["yaw_max"], info["pitch_max"], info["roll_max"]],
        ):
            with col:
                st.markdown(f"**{axis_name}**")
                filtered = _filter_angles(vals)
                if not filtered:
                    st.info("过滤后无数据（全部 <2 度）。")
                else:
                    st.json(_summarize(filtered))
                    df_hist = _hist_df(filtered)
                    st.bar_chart(df_hist, x="bin", y="count")

        st.subheader("内参分布（序列内均值）")
        cols_intr = st.columns(4)
        intr_items = [
            ("fx_mean", info.get("fx_mean", [])),
            ("fy_mean", info.get("fy_mean", [])),
            ("cx_mean", info.get("cx_mean", [])),
            ("cy_mean", info.get("cy_mean", [])),
        ]
        for col, (label, vals) in zip(cols_intr, intr_items):
            with col:
                st.markdown(f"**{label}**")
                st.json(_summarize(vals))
                df_hist = _hist_df(vals)
                if not df_hist.empty:
                    st.bar_chart(df_hist, x="bin", y="count")


def main():
    args = parse_args()
    spec_text_default = _load_spec_text(args.config)

    st.set_page_config(page_title="Pose Stats (21 frames)", layout="wide")
    st.title("ReCamMaster Pose 分布统计")

    st.sidebar.header("数据集配置")
    spec_text = st.sidebar.text_area("DatasetSpec JSON 列表", value=spec_text_default, height=220)
    tensor_suffix = st.sidebar.text_input("tensor_suffix", value=args.tensor_suffix)
    sample_limit = st.sidebar.number_input("每个数据集采样上限 (0=全部)", min_value=0, value=int(args.sample_limit))
    seed = st.sidebar.number_input("seed", min_value=0, value=int(args.seed))
    st.sidebar.write(f"target_frames 固定 {TARGET_FRAMES}")
    run_button = st.sidebar.button("开始统计", type="primary")

    try:
        specs = _parse_specs(spec_text)
    except Exception as exc:
        st.error(f"解析 DatasetSpec 失败: {exc}")
        return

    st.subheader("DatasetSpec 概览")
    st.dataframe(
        pd.DataFrame(
            [
                {"name": s.name, "root": s.root, "metadata_path": s.metadata_path or "", "weight": s.weight}
                for s in specs
            ]
        )
    )

    if not run_button:
        st.info("点击“开始统计”运行扫描。")
        return

    tensor_suffixes = _tensor_suffixes(tensor_suffix)
    with st.spinner("扫描数据集中..."):
        stats = collect_pose_stats(
            specs=specs,
            tensor_suffixes=tensor_suffixes,
            sample_limit=int(sample_limit),
            seed=int(seed),
        )
    render_results(stats)


if __name__ == "__main__":
    main()
