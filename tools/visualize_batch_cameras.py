"""
Streamlit 可视化：采样真实 batch，查看相机参数分布、样本数量与权重、归一化因子。
运行示例：
streamlit run tools/visualize_batch_cameras.py -- --config specs.json --pipeline-type i2v --batch-size 2 --num-batches 5
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

# 确保 repo 根目录与 DiffSynth 依赖在 sys.path 内，避免在 streamlit 下找不到 src
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "DiffSynth-Studio"
for p in (REPO_ROOT, THIRD_PARTY):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

from src.dataset import (
    BaseImageConditionDataset,
    Camera,
    DatasetSpec,
    MixedImageConditionDataset,
    MulticamImageConditionDataset,
    RelEstate10kImageConditionDataset,
    SdgImageConditionDataset,
    TensorDataset,
    _gather_paths_for_spec,
    compute_relative_c2w,
    create_datasets,
    normalize_translation,
)

# Sidebar 预填的 DatasetSpec 示例
DEFAULT_SPEC_TEXT = json.dumps(
    [
        {"name": "multicam", "root": "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train", "metadata_path": "metadata/metadata_all.csv", "weight": 1.0},
        {"name": "sdg", "root": "/nas/datasets/vipe_wild_sdg_1m", "metadata_path": "metadata/output_sdg_with_pose.csv", "weight": 0.5},
        {"name": "rel10k", "root": "/nas/datasets/relestate10k/train ", "metadata_path": "metadata/metadata_re10k.csv", "weight": 0.5},
    ],
    indent=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 batch 相机分布、样本数量与归一化因子")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选：包含 DatasetSpec 列表的 JSON 文件路径（用于预填充侧边栏）。",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["i2v", "v2v"],
        default="i2v",
        help="训练/推理管线类型，影响默认 tensor 后缀。",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="DataLoader batch 大小（可视化用，避免过大占用显存/内存）。")
    parser.add_argument("--num-batches", type=int, default=5, help="采样多少个 batch 进行统计。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers。")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="构建数据集时使用的 steps_per_epoch。")
    parser.add_argument("--val-size", type=int, default=8, help="从每个数据集中划分到验证集的样本数上限。")
    parser.add_argument("--tensor-suffix", type=str, default="", help="覆盖默认 latent 后缀（留空使用 i2v/.wan22 或 v2v/.tensors）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保持与训练一致）。")
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


def _debug_wrapper(dataset: torch.utils.data.Dataset, name_hint: str) -> torch.utils.data.Dataset:
    class DebugDataset(torch.utils.data.Dataset):
        def __init__(self, base: torch.utils.data.Dataset, hint: str):
            self.base = base
            self.hint = hint

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            sample = self.base[index]
            sample = dict(sample)
            sample["_debug_index"] = index
            if "dataset_name" not in sample:
                sample["dataset_name"] = getattr(self.base, "dataset_name", self.hint)
            return sample

    return DebugDataset(dataset, name_hint)


def _build_dataset_map(train_dataset: torch.utils.data.Dataset) -> Dict[str, torch.utils.data.Dataset]:
    if isinstance(train_dataset, MixedImageConditionDataset):
        mapping: Dict[str, torch.utils.data.Dataset] = {}
        for ds in train_dataset.datasets:
            name = getattr(ds, "dataset_name", "unknown")
            if name not in mapping:
                mapping[name] = ds
        return mapping
    name = getattr(train_dataset, "dataset_name", "v2v")
    return {name: train_dataset}


def collate_for_visualization(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "camera",
        "intrinsics",
        "path",
        "scene_id",
        "condition_cam_type",
        "target_cam_type",
        "dataset_name",
        "_debug_index",
    ]
    output: Dict[str, Any] = {}
    for key in keys:
        vals = [item[key] for item in batch if key in item]
        if not vals:
            continue
        first = vals[0]
        if torch.is_tensor(first):
            output[key] = torch.stack(vals, dim=0)
        else:
            output[key] = vals
    return output


def _rotation_to_euler(rot_mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rot_mats: [F, 3, 3] w2c 旋转矩阵，返回 yaw/pitch/roll（弧度）。
    """
    sy = np.sqrt(rot_mats[:, 0, 0] ** 2 + rot_mats[:, 1, 0] ** 2)
    yaw = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    pitch = np.arctan2(-rot_mats[:, 2, 0], sy + 1e-8)
    roll = np.arctan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2])
    return yaw, pitch, roll


def _summarize_translation(norms: np.ndarray) -> Dict[str, float]:
    if norms.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p90": float(np.percentile(norms, 90)),
    }


def _infer_cond_idx(seed: int, index: int, tgt_idx: int) -> int:
    random.seed(seed + index + 1000)
    cond_idx = random.randint(1, 10)
    while cond_idx == tgt_idx:
        cond_idx = random.randint(1, 10)
    return cond_idx


def _baseline_v2v(
    dataset: TensorDataset,
    sample_path: str,
    sample_index: int,
) -> Dict[str, Any]:
    match = dataset_path_regex(sample_path)
    tgt_idx = int(match.group(1)) if match else 0
    cond_idx = _infer_cond_idx(dataset.seed, sample_index, tgt_idx)

    base_path = str(Path(sample_path).parent.parent)
    camera_json = Path(base_path) / "cameras" / "camera_extrinsics.json"
    with open(camera_json, "r") as file:
        cam_data = json.load(file)

    cam_idx = list(range(81))[::4]
    multiview_c2ws: List[List[np.ndarray]] = []
    # 与训练一致：先 cond 再 tgt，ref = cond[0]
    for view_idx in [cond_idx, tgt_idx]:
        traj = [dataset.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.0
            c2ws.append(c2w)
        multiview_c2ws.append(c2ws)

    cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
    tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
    ref_cam = cond_cam_params[0]

    cond_rel_c2w = compute_relative_c2w(cond_cam_params, ref_cam, dataset.get_relative_pose)
    tgt_rel_c2w = compute_relative_c2w(tgt_cam_params, ref_cam, dataset.get_relative_pose)
    cond_rel_norm, tgt_rel_norm, baseline = normalize_translation(cond_rel_c2w, tgt_rel_c2w)

    cond_norms_raw = np.linalg.norm(cond_rel_c2w[:, :3, 3], axis=1)
    tgt_norms_raw = np.linalg.norm(tgt_rel_c2w[:, :3, 3], axis=1)
    cond_norms = np.linalg.norm(cond_rel_norm[:, :3, 3], axis=1)
    tgt_norms = np.linalg.norm(tgt_rel_norm[:, :3, 3], axis=1)

    return {
        "baseline": float(baseline),
        "tgt_norms_raw": tgt_norms_raw,
        "cond_norms_raw": cond_norms_raw,
        "tgt_norms": tgt_norms,
        "cond_norms": cond_norms,
    }


def _baseline_i2v(
    dataset: BaseImageConditionDataset,
    sample_path: str,
    num_frames: int,
) -> Dict[str, Any]:
    c2ws = dataset._load_camera_sequence(sample_path, num_frames)
    cam_params = [Camera(c2w) for c2w in c2ws]
    ref_cam = cam_params[0]
    rel_c2w = compute_relative_c2w(cam_params, ref_cam, dataset.get_relative_pose)
    rel_c2w_norm, _, baseline = normalize_translation(rel_c2w, rel_c2w)
    raw_norms = np.linalg.norm(rel_c2w[:, :3, 3], axis=1)
    normed = np.linalg.norm(rel_c2w_norm[:, :3, 3], axis=1)
    return {"baseline": float(baseline), "tgt_norms_raw": raw_norms, "tgt_norms": normed}


def dataset_path_regex(path: str):
    import re

    return re.search(r"cam(\d+)", path)


def compute_baseline_info(
    dataset_map: Dict[str, torch.utils.data.Dataset],
    dataset_name: str,
    sample_path: str,
    sample_index: int,
    num_frames: int,
) -> Dict[str, Any]:
    ds = dataset_map.get(dataset_name)
    if ds is None:
        return {}
    if isinstance(ds, TensorDataset):
        return _baseline_v2v(ds, sample_path, sample_index)
    if isinstance(ds, BaseImageConditionDataset):
        return _baseline_i2v(ds, sample_path, num_frames)
    return {}

def _dataset_registry() -> Dict[str, type]:
    return {
        "multicam": MulticamImageConditionDataset,
        "sdg": SdgImageConditionDataset,
        "rel10k": RelEstate10kImageConditionDataset,
    }


def collect_full_translation_norms(
    specs: List[DatasetSpec],
    pipeline_type: str,
    target_frames: int,
    tensor_suffixes: Tuple[str, ...],
    seed: int,
    sample_limit: int = 10000,
    progress_cb=None,
) -> Dict[str, Dict[str, Any]]:
    """
    扫描每个数据集的全部样本（或采样上限），按 dataset.py 的归一化逻辑计算 translation norm 分布。
    仅支持 i2v 多数据集；不加载 latents，只读相机文件以节省内存。
    """
    results: Dict[str, Dict[str, Any]] = {}
    if pipeline_type != "i2v":
        return results
    registry = _dataset_registry()

    for spec in specs:
        ds_cls = registry.get(spec.name)
        if ds_cls is None:
            continue
        paths = _gather_paths_for_spec(spec, tensor_suffixes)
        if sample_limit and sample_limit > 0:
            paths = paths[:sample_limit]
        if not paths:
            continue

        ds = ds_cls(
            steps_per_epoch=1,
            paths=paths,
            fixed_length=len(paths),
            seed=seed,
            dataset_root=spec.root,
        )
        all_norms: List[np.ndarray] = []
        baselines: List[float] = []
        for idx, p in enumerate(paths):
            try:
                c2ws = ds._load_camera_sequence(p, target_frames)
                cam_params = [Camera(c2w) for c2w in c2ws]
                ref_cam = cam_params[0]
                rel_c2w = compute_relative_c2w(cam_params, ref_cam, ds.get_relative_pose)
                rel_c2w_norm, _, baseline = normalize_translation(rel_c2w, rel_c2w)
                norms = np.linalg.norm(rel_c2w_norm[:, :3, 3], axis=1)
                # 按轨迹（视频）记录一个值：该轨迹归一化后的最大平移范数
                all_norms.append(np.asarray([norms.max() if norms.size > 0 else 0.0], dtype=np.float32))
                baselines.append(float(baseline))
            except Exception as exc:
                print(f"[warn] 处理 {spec.name} 样本失败 {p}: {exc}")
            if progress_cb:
                progress_cb(spec.name, idx + 1, len(paths))
        if all_norms:
            stacked = np.concatenate(all_norms)
            results[spec.name] = {
                "norms": stacked,
                "baselines": baselines,
                "sample_count": len(paths),
                "frame_count": stacked.size,
            }
    return results


def summarize_batches(
    batches: List[Dict[str, Any]],
    dataset_map: Dict[str, torch.utils.data.Dataset],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []

    for batch_id, batch in enumerate(batches):
        cameras = batch.get("camera")
        paths = batch.get("path", [])
        dataset_names = batch.get("dataset_name", [])
        indices = batch.get("_debug_index", [])

        if cameras is None or not torch.is_tensor(cameras):
            continue
        cams_np = cameras.cpu().numpy()
        if cams_np.ndim == 3:
            cams_np = cams_np[None, ...]

        for i in range(cams_np.shape[0]):
            cam = cams_np[i]  # [F,3,4]
            num_frames = cam.shape[0]
            translations = cam[:, :, 3]
            norms = np.linalg.norm(translations, axis=1)
            rot = cam[:, :, :3]
            yaw, pitch, roll = _rotation_to_euler(rot)

            dataset_name = dataset_names[i] if i < len(dataset_names) else "unknown"
            sample_path = paths[i] if i < len(paths) else "n/a"
            sample_index = int(indices[i]) if i < len(indices) else -1
            baseline_info = compute_baseline_info(dataset_map, dataset_name, sample_path, sample_index, num_frames)

            sample_rows.append(
                {
                    "batch_id": batch_id,
                    "dataset_name": dataset_name,
                    "path": sample_path,
                    "frames": num_frames,
                    "t_norm_min": float(norms.min(initial=0.0)),
                    "t_norm_max": float(norms.max(initial=0.0)),
                    "t_norm_mean": float(norms.mean() if norms.size > 0 else 0.0),
                    "t_norm_median": float(np.median(norms) if norms.size > 0 else 0.0),
                    "baseline": float(baseline_info.get("baseline", 0.0)),
                }
            )

            for f_idx in range(num_frames):
                frame_rows.append(
                    {
                        "batch_id": batch_id,
                        "dataset_name": dataset_name,
                        "frame": f_idx,
                        "t_norm": float(norms[f_idx]),
                        "yaw": float(yaw[f_idx]),
                        "pitch": float(pitch[f_idx]),
                        "roll": float(roll[f_idx]),
                        "baseline": float(baseline_info.get("baseline", 0.0)),
                        "source_path": sample_path,
                    }
                )
    return pd.DataFrame(sample_rows), pd.DataFrame(frame_rows)


def collect_batches(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_batches: int,
    num_workers: int,
) -> List[Dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_for_visualization,
    )
    batches: List[Dict[str, Any]] = []
    for idx, batch in enumerate(loader):
        if idx >= num_batches:
            break
        batches.append(batch)
    return batches


def summarize_dataset_specs(specs: Iterable[DatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "name": spec.name,
                "root": spec.root,
                "metadata_path": spec.metadata_path or "",
                "weight": spec.weight,
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset_sizes(train_dataset: torch.utils.data.Dataset) -> pd.DataFrame:
    rows = []
    if isinstance(train_dataset, MixedImageConditionDataset):
        for ds in train_dataset.datasets:
            rows.append(
                {
                    "dataset_name": getattr(ds, "dataset_name", "unknown"),
                    "samples": len(getattr(ds, "path", [])),
                    "steps_per_epoch": getattr(ds, "steps_per_epoch", 0),
                }
            )
    else:
        rows.append(
            {
                "dataset_name": getattr(train_dataset, "dataset_name", "train"),
                "samples": len(getattr(train_dataset, "path", [])),
                "steps_per_epoch": getattr(train_dataset, "steps_per_epoch", 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    spec_text_default = _load_spec_text(args.config)

    st.set_page_config(page_title="Batch Camera Visualizer", layout="wide")
    st.title("ReCamMaster Batch 相机分布可视化")

    st.sidebar.header("数据集配置")
    spec_text = st.sidebar.text_area("DatasetSpec JSON 列表", value=spec_text_default, height=220)
    pipeline_type = st.sidebar.selectbox("pipeline_type", ["i2v", "v2v"], index=0 if args.pipeline_type == "i2v" else 1)
    tensor_suffix = st.sidebar.text_input("tensor_suffix 覆盖（空则使用默认）", value=args.tensor_suffix)
    steps_per_epoch = st.sidebar.number_input("steps_per_epoch", min_value=1, value=args.steps_per_epoch)
    val_size = st.sidebar.number_input("val_size", min_value=0, value=args.val_size)
    batch_size = st.sidebar.number_input("batch_size", min_value=1, value=args.batch_size)
    num_batches = st.sidebar.number_input("num_batches", min_value=1, value=args.num_batches)
    num_workers = st.sidebar.number_input("num_workers", min_value=0, value=args.num_workers)
    seed = st.sidebar.number_input("seed", min_value=0, value=args.seed)
    target_frames = st.sidebar.number_input("target_frames (全量扫描用)", min_value=1, value=21)
    sample_limit = st.sidebar.number_input("全量扫描采样上限 (0=全部)", min_value=0, value=0)
    run_button = st.sidebar.button("运行采样", type="primary")
    full_scan_button = st.sidebar.button("全量扫描 translation 范数", type="secondary")

    try:
        specs = _parse_specs(spec_text)
    except Exception as exc:  # pragma: no cover - streamlit 交互
        st.error(f"解析 DatasetSpec 失败: {exc}")
        return

    if not run_button and not full_scan_button:
        st.info("在侧边栏确认配置后点击“运行采样”或“全量扫描”。")
        return

    st.subheader("DatasetSpec 概览")
    st.dataframe(summarize_dataset_specs(specs))

    tensor_suffix_arg = tensor_suffix if tensor_suffix.strip() else None
    tensor_suffixes_default = (
        (tensor_suffix_arg,) if tensor_suffix_arg else (".wan22.tensors.pth",) if pipeline_type == "i2v" else (".tensors.pth",)
    )

    if run_button:
        with st.spinner("构建数据集..."):
            try:
                train_dataset, _ = create_datasets(
                    dataset_specs=specs,
                    val_size=val_size,
                    steps_per_epoch=steps_per_epoch,
                    seed=seed,
                    pipeline_type=pipeline_type,
                    tensor_suffix=tensor_suffix_arg,
                )
            except Exception as exc:  # pragma: no cover - streamlit 交互
                st.error(f"构建数据集失败: {exc}")
                return

        st.subheader("数据集规模")
        st.dataframe(summarize_dataset_sizes(train_dataset))

        debug_dataset = _debug_wrapper(train_dataset, pipeline_type)
        dataset_map = _build_dataset_map(train_dataset)

        with st.spinner("采样 batch 并统计..."):
            batches = collect_batches(
                dataset=debug_dataset,
                batch_size=batch_size,
                num_batches=num_batches,
                num_workers=num_workers,
            )
        if not batches:
            st.warning("未采样到任何 batch。请检查 steps_per_epoch 与路径配置。")
            return

        sample_df, frame_df = summarize_batches(batches, dataset_map)
        st.subheader("样本粒度统计（每条样本）")
        st.dataframe(sample_df)

        st.subheader("相机平移范数分布（帧级）")
        if not frame_df.empty:
            st.bar_chart(frame_df, x="frame", y="t_norm", color="dataset_name")
        else:
            st.info("帧级数据为空，无法绘制直方图。")

        st.subheader("归一化基线（baseline/max_norm）分布")
        if "baseline" in sample_df.columns and not sample_df.empty:
            st.bar_chart(sample_df, x="path", y="baseline", color="dataset_name")
        else:
            st.info("缺少 baseline 信息。")

        st.subheader("旋转角度（yaw/pitch/roll）统计")
        if not frame_df.empty:
            st.line_chart(frame_df, x="frame", y=["yaw", "pitch", "roll"], color="dataset_name")
        else:
            st.info("帧级数据为空，无法展示旋转角度。")

    if full_scan_button:
        if pipeline_type != "i2v":
            st.warning("全量扫描当前仅支持 i2v 多数据集。")
        else:
            st.subheader("全量 translation 范数分布（按数据集）")
            progress = st.progress(0.0, text="扫描中...")

            def cb(name: str, done: int, total: int):
                progress.progress(min(1.0, done / max(total, 1)), text=f"{name}: {done}/{total}")

            stats = collect_full_translation_norms(
                specs=specs,
                pipeline_type=pipeline_type,
                target_frames=int(target_frames),
                tensor_suffixes=tensor_suffixes_default,
                seed=seed,
                sample_limit=int(sample_limit),
                progress_cb=cb,
            )
            progress.empty()
            if not stats:
                st.info("未得到统计结果，请检查路径/后缀配置。")
            else:
                cols = st.columns(max(1, len(stats)))
                for col, (name, info) in zip(cols, stats.items()):
                    norms = info["norms"]
                    hist, bins = np.histogram(norms, bins=50)
                    centers = 0.5 * (bins[:-1] + bins[1:])
                    df_hist = pd.DataFrame({"bin": centers, "count": hist})
                    with col:
                        st.write(f"{name}: 样本 {info['sample_count']}，帧 {info['frame_count']}")
                        st.bar_chart(df_hist, x="bin", y="count")
                        summary = _summarize_translation(norms)
                        st.json(summary)


if __name__ == "__main__":
    main()
