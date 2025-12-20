from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
import torchvision.transforms.functional as TF

from base_handler import (
    BaseInferenceDataset,
    BasePipelineHandler,
    InferenceSample,
    PreparedInference,
    TargetSpec,
    center_trajectory,
    compute_relative_c2w,
    convert_c2w_convention,
    c2w_to_w2c,
    ensure_homogeneous,
    invert_se3,
    normalize_joint_translation,
    preprocess_video_frame,
    register_dataset,
    register_pipeline,
    select_pose_sequence,
)


@register_dataset("example")
class ExampleDataset(BaseInferenceDataset):
    """Dataset used by the original src/inference_recammaster.py script."""

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)
        metadata_filename = options.get("metadata_filename", "metadata.csv")
        metadata_path = self.dataset_path / metadata_filename
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        self.video_paths = [self._resolve_video_path(p) for p in metadata["file_name"].tolist()]
        self.texts = metadata.get("text", pd.Series([""] * len(metadata))).fillna("").tolist()

        self.num_frames = int(options.get("num_frames", 81))
        self.max_num_frames = int(options.get("max_num_frames", 81))
        self.frame_interval = int(options.get("frame_interval", 1))
        self.height = int(options.get("height", 480))
        self.width = int(options.get("width", 832))
        self.cam_interval = int(options.get("camera_interval", 4))
        self.cam_indices = np.arange(self.num_frames, dtype=np.int64)[:: self.cam_interval]
        if self.cam_indices.size == 0:
            raise ValueError("camera_interval produced empty cam_indices")

        self.preprocess = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _resolve_video_path(self, file_name: str) -> str:
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
        if os.path.exists(file_name):
            return os.path.abspath(file_name)
        candidate = self.dataset_path / file_name
        if candidate.exists():
            return str(candidate)
        candidate = self.dataset_path / "videos" / file_name
        if candidate.exists():
            return str(candidate)
        if not os.path.isabs(file_name) and "videos" in file_name:
            candidate = self.dataset_path / file_name
            if candidate.exists():
                return str(candidate)
        return str(self.dataset_path / "videos" / file_name)

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, path: str) -> torch.Tensor:
        max_start = self.max_num_frames - (self.num_frames - 1) * self.frame_interval
        if max_start <= 0:
            raise ValueError("max_num_frames too small to sample requested num_frames")
        start_frame = torch.randint(0, max_start, (1,)).item()
        reader = imageio.get_reader(path)
        frames: List[torch.Tensor] = []
        try:
            for frame_idx in range(self.num_frames):
                raw = reader.get_data(start_frame + frame_idx * self.frame_interval)
                pil_img = Image.fromarray(raw)
                processed = self.preprocess(preprocess_video_frame(pil_img, self.height, self.width))
                frames.append(processed)
        finally:
            reader.close()
        stacked = torch.stack(frames, dim=0)
        return stacked.permute(1, 0, 2, 3)

    def _load_source_c2ws(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Source pose file not found: {path}")
        data = np.load(path, allow_pickle=True)
        if "data" not in data or "inds" not in data:
            raise ValueError(f"Invalid pose file {path}, expected keys 'data' and 'inds'")
        mats = ensure_homogeneous(data["data"].astype(np.float32))
        inds = data["inds"].astype(np.int64)
        return mats, inds

    def __getitem__(self, index: int) -> InferenceSample:
        video_path = Path(self.video_paths[index])
        video_tensor = self._load_video(str(video_path)).to(torch.float32)

        src_pose_path = video_path.with_suffix(".npz")
        src_c2ws, src_inds = self._load_source_c2ws(src_pose_path)
        src_c2ws_sel, _ = select_pose_sequence(src_c2ws, src_inds, self.cam_indices)
        src_c2ws_sel = convert_c2w_convention(src_c2ws_sel)
        src_c2ws_sel = center_trajectory(src_c2ws_sel)
        ref_c2w = src_c2ws_sel[0]
        ref_w2c = invert_se3(ref_c2w)
        cond_rel_c2w = compute_relative_c2w(ref_w2c, src_c2ws_sel)
        cond_rel_w2c = c2w_to_w2c(cond_rel_c2w)

        return InferenceSample(
            video=video_tensor,
            text=self.texts[index],
            cond_data={
                "cond_rel_c2w": cond_rel_c2w.astype(np.float32),
                "cond_rel_w2c": cond_rel_w2c.astype(np.float32),
                "ref_w2c": ref_w2c.astype(np.float32),
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={"source_path": str(video_path), "stem": video_path.stem},
        )


@register_dataset("sdg_v2v")
class SDGV2VDataset(BaseInferenceDataset):
    """V2V 推理用的 SDG 数据集（视频和相机姿态分目录存放）。"""

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)
        metadata_path = Path(options.get("metadata_path", options.get("metadata_filename", "metadata.csv")))
        if not metadata_path.is_absolute():
            metadata_path = dataset_path / metadata_path
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        if "video_absolute_path" not in metadata.columns:
            raise ValueError(f"{metadata_path} 缺少列 video_absolute_path")
        self.video_paths = [self._resolve_video_path(p) for p in metadata["video_absolute_path"].tolist()]
        self.texts = metadata.get("caption", pd.Series([""] * len(metadata))).fillna("").tolist()

        pose_dir_opt = options.get("pose_dir")
        self.pose_dir = Path(pose_dir_opt) if pose_dir_opt else dataset_path / "pose"
        if not self.pose_dir.exists():
            raise FileNotFoundError(f"Pose 目录不存在: {self.pose_dir}")

        self.num_frames = int(options.get("num_frames", 81))
        self.max_num_frames = int(options.get("max_num_frames", 81))
        self.frame_interval = int(options.get("frame_interval", 1))
        self.height = int(options.get("height", 480))
        self.width = int(options.get("width", 832))
        self.cam_interval = int(options.get("camera_interval", 4))
        self.cam_indices = np.arange(self.num_frames, dtype=np.int64)[:: self.cam_interval]
        if self.cam_indices.size == 0:
            raise ValueError("camera_interval produced empty cam_indices")

        self.preprocess = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _resolve_video_path(self, file_name: str) -> str:
        if os.path.isabs(file_name):
            return file_name
        candidate = self.dataset_path / file_name
        return str(candidate)

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, path: str) -> torch.Tensor:
        reader = imageio.get_reader(path)
        frames: List[torch.Tensor] = []
        try:
            for frame_idx in range(self.num_frames):
                raw = reader.get_data(frame_idx * self.frame_interval)
                pil_img = Image.fromarray(raw)
                processed = self.preprocess(preprocess_video_frame(pil_img, self.height, self.width))
                frames.append(processed)
        finally:
            reader.close()
        stacked = torch.stack(frames, dim=0)
        return stacked.permute(1, 0, 2, 3)

    def _load_pose(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        pose_path = self.pose_dir / f"{stem}.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose 文件不存在: {pose_path}")
        with np.load(pose_path, allow_pickle=False) as data:
            if "data" not in data or "inds" not in data:
                raise ValueError(f"{pose_path} 缺少 data/inds")
            mats = ensure_homogeneous(data["data"].astype(np.float32))
            inds = data["inds"].astype(np.int64)
        return mats, inds

    def __getitem__(self, index: int) -> InferenceSample:
        video_path = Path(self.video_paths[index])
        video_tensor = self._load_video(str(video_path)).to(torch.float32)

        pose_all, pose_inds = self._load_pose(video_path.stem)
        pose_sel, matched_inds = select_pose_sequence(pose_all, pose_inds, self.cam_indices)
        pose_sel = center_trajectory(pose_sel)
        ref_w2c = invert_se3(pose_sel[0])
        cond_rel_c2w = compute_relative_c2w(ref_w2c, pose_sel)
        cond_rel_w2c = c2w_to_w2c(cond_rel_c2w)

        return InferenceSample(
            video=video_tensor,
            text=self.texts[index],
            cond_data={
                "cond_rel_c2w": cond_rel_c2w.astype(np.float32),
                "cond_rel_w2c": cond_rel_w2c.astype(np.float32),
                "ref_w2c": ref_w2c.astype(np.float32),
                "cam_indices": self.cam_indices.copy(),
                "pose_path": str(self.pose_dir / f"{video_path.stem}.npz"),
            },
            metadata={
                "source_path": str(video_path),
                "stem": video_path.stem,
                "matched_inds": matched_inds.tolist(),
            },
        )


@register_dataset("pointodyssey")
class PointOdysseyDataset(BaseInferenceDataset):
    """Dataset used by evaluation/render_pointodyssey.py."""

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)
        split = options.get("split", "test")
        split_dir = self.dataset_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.num_frames = int(options.get("num_frames", 81))
        self.height = int(options.get("height", 480))
        self.width = int(options.get("width", 832))
        self.cam_interval = int(options.get("camera_interval", 4))
        self.cam_indices = np.arange(self.num_frames, dtype=np.int64)[:: self.cam_interval] if self.cam_interval > 0 else np.arange(self.num_frames, dtype=np.int64)
        if self.cam_indices.size == 0:
            raise ValueError("camera_interval produced empty cam_indices")
        max_samples = options.get("max_samples")
        self.samples: List[Dict[str, Path]] = []

        video_paths = sorted(split_dir.glob("*.mp4"))
        for video_path in video_paths:
            stem = video_path.stem
            anno_path = split_dir / stem / "anno.npz"
            if not anno_path.exists():
                continue
            try:
                frame_count = self._count_video_frames(video_path)
            except Exception:
                continue
            if frame_count < self.num_frames:
                continue
            self.samples.append({"video_path": video_path, "anno_path": anno_path, "stem": stem})
            if max_samples is not None and len(self.samples) >= int(max_samples):
                break

        self.preprocess = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _count_video_frames(self, path: Path) -> int:
        reader = imageio.get_reader(str(path))
        try:
            return reader.count_frames()
        finally:
            reader.close()

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_and_resize(self, image: Image.Image) -> Image.Image:
        orig_width, orig_height = image.size
        scale = max(self.width / orig_width, self.height / orig_height)
        resized_height = max(int(round(orig_height * scale)), 1)
        resized_width = max(int(round(orig_width * scale)), 1)
        resized = TF.resize(image, (resized_height, resized_width), interpolation=InterpolationMode.BILINEAR, antialias=True)
        top = max((resized_height - self.height) // 2, 0)
        left = max((resized_width - self.width) // 2, 0)
        return TF.crop(resized, top, left, self.height, self.width)

    def _load_video(self, path: Path) -> torch.Tensor:
        reader = imageio.get_reader(str(path))
        frames: List[torch.Tensor] = []
        try:
            for frame_id in range(self.num_frames):
                frame = reader.get_data(frame_id)
                pil_img = Image.fromarray(frame)
                processed = self.preprocess(preprocess_video_frame(pil_img, self.height, self.width))
                frames.append(processed)
        finally:
            reader.close()
        stacked = torch.stack(frames, dim=0)
        return stacked.permute(1, 0, 2, 3)

    def _load_cond_camera(self, anno_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(anno_path, allow_pickle=False) as data:
            extrinsics = data["extrinsics"].astype(np.float32)
        if extrinsics.shape[0] < self.num_frames:
            raise ValueError(f"{anno_path} has insufficient frames: {extrinsics.shape[0]}")
        cond_w2cs = extrinsics[: self.num_frames][self.cam_indices]
        cond_c2ws = np.stack([invert_se3(mat) for mat in cond_w2cs], axis=0)
        cond_c2ws = convert_c2w_convention(cond_c2ws)
        cond_c2ws = center_trajectory(cond_c2ws)
        ref_c2w = cond_c2ws[0]
        ref_w2c = invert_se3(ref_c2w)
        cond_rel_c2w = compute_relative_c2w(ref_w2c, cond_c2ws)
        cond_rel_w2c = c2w_to_w2c(cond_rel_c2w)
        return cond_rel_c2w, cond_rel_w2c, ref_w2c

    def __getitem__(self, index: int) -> InferenceSample:
        info = self.samples[index]
        video_tensor = self._load_video(info["video_path"]).to(torch.float32)
        cond_rel_c2w, cond_rel_w2c, ref_w2c = self._load_cond_camera(info["anno_path"])

        return InferenceSample(
            video=video_tensor,
            text="",
            cond_data={
                "cond_rel_c2w": cond_rel_c2w.astype(np.float32),
                "cond_rel_w2c": cond_rel_w2c.astype(np.float32),
                "ref_w2c": ref_w2c.astype(np.float32),
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={
                "source_path": str(info["video_path"]),
                "stem": info["stem"],
            },
        )


@register_pipeline("v2v")
class V2VPipelineHandler(BasePipelineHandler):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        global_opts: Dict[str, Any],
    ):
        self.device = device
        self.dtype = dtype
        self.global_opts = global_opts

    def build_inputs(
        self,
        sample: InferenceSample,
        target: TargetSpec,
        *,
        source_video: torch.Tensor,
    ) -> PreparedInference:
        cond_rel_c2w = np.asarray(sample.cond_data["cond_rel_c2w"], dtype=np.float32)
        cond_rel_w2c = np.asarray(sample.cond_data["cond_rel_w2c"], dtype=np.float32)
        ref_w2c = np.asarray(sample.cond_data["ref_w2c"], dtype=np.float32)
        cam_indices = np.asarray(sample.cond_data["cam_indices"], dtype=np.int64)

        target_w2cs_sel, matched_inds = select_pose_sequence(target.raw_pose, target.raw_inds, cam_indices)
        target_w2cs_sel = target_w2cs_sel.astype(np.float32)
        target_c2ws = target_w2cs_sel.transpose(0, 2, 1)
        target_c2ws = convert_c2w_convention(target_c2ws)
        target_c2ws = center_trajectory(target_c2ws)
        tgt_rel_c2w = compute_relative_c2w(ref_w2c, target_c2ws)

        cond_norm, tgt_norm = normalize_joint_translation(cond_rel_c2w, tgt_rel_c2w)
        cond_rel_w2c_norm = c2w_to_w2c(cond_norm)
        tgt_rel_w2c = c2w_to_w2c(tgt_norm)

        pose_embedding = np.concatenate([tgt_rel_w2c, cond_rel_w2c_norm], axis=0).astype(np.float32)
        if self.global_opts.get("debug_pose", False):
            print(f"[debug] target {target.name} first translation {tgt_rel_w2c[0, :3, 3]}")
        target_camera = torch.from_numpy(pose_embedding).unsqueeze(0).to(dtype=self.dtype)

        return PreparedInference(
            pipe_kwargs={
                "target_camera": target_camera,
                "source_video": source_video,
            },
            target_rel_w2c=tgt_rel_w2c,
            cam_indices=cam_indices,
            metadata={
                "target_name": target.name,
                "matched_inds": matched_inds,
            },
        )
