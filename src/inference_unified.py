"""
Unified inference pipeline supporting multiple datasets and pipeline modes.

Currently mirrors the behaviour of:
  * src/inference_recammaster.py (example dataset, v2v)
  * evaluation/render_pointodyssey.py (PointOdyssey dataset, v2v)

Future datasets or inference modes (e.g. i2v) can extend the registry-based
factory hooks defined in this module.
"""

from __future__ import annotations

import abc
import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, v2
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video  # noqa: E402
from diffsynth.pipelines.wan_video_new import WanVideoPipeline  # noqa: E402

WAN_MODEL_ROOT = PROJECT_ROOT / "models" / "Wan-AI"
WAN21_MODEL_DIR = WAN_MODEL_ROOT / "Wan2.1-T2V-1.3B"
WAN22_MODEL_DIR = WAN_MODEL_ROOT / "Wan2.2-TI2V-5B"


# ---------------------------------------------------------------------------
# Common data containers
# ---------------------------------------------------------------------------

@dataclass
class InferenceSample:
    video: torch.Tensor
    text: str
    cond_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSpec:
    name: str
    raw_pose: np.ndarray  # Expect shape (T, 4, 4)
    raw_inds: np.ndarray  # Frame indices associated with raw_pose
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedInference:
    pipe_kwargs: Dict[str, Any]
    target_rel_w2c: np.ndarray
    cam_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset base class and registry
# ---------------------------------------------------------------------------

class BaseInferenceDataset(Dataset, metaclass=abc.ABCMeta):
    """Return per-sample video tensors and conditioning metadata."""

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        self.options = options
        self.dataset_path = dataset_path

    @abc.abstractmethod
    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> InferenceSample:  # pragma: no cover - abstract
        raise NotImplementedError


DATASET_REGISTRY: Dict[str, type[BaseInferenceDataset]] = {}


def register_dataset(name: str) -> Callable[[type[BaseInferenceDataset]], type[BaseInferenceDataset]]:
    def decorator(cls: type[BaseInferenceDataset]) -> type[BaseInferenceDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered")
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


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
        # Fallback: assume dataset_path/videos/file_name
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
        stacked = torch.stack(frames, dim=0)  # (T, C, H, W)
        return stacked.permute(1, 0, 2, 3)  # (C, T, H, W)

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
        video_tensor = self._load_video(str(video_path))
        video_tensor = video_tensor.to(torch.float32)

        src_pose_path = video_path.with_suffix(".npz")
        src_c2ws, src_inds = self._load_source_c2ws(src_pose_path)
        src_c2ws_sel, _ = select_pose_sequence(src_c2ws, src_inds, self.cam_indices)
        src_c2ws_sel = convert_c2w_convention(src_c2ws_sel)
        src_c2ws_sel = center_trajectory(src_c2ws_sel)
        ref_c2w = src_c2ws_sel[0]
        ref_w2c = invert_se3(ref_c2w)
        cond_rel_c2w = compute_relative_c2w(ref_w2c, src_c2ws_sel)
        cond_rel_w2c = c2w_to_w2c(cond_rel_c2w)

        sample = InferenceSample(
            video=video_tensor,
            text=self.texts[index],
            cond_data={
                "cond_rel_c2w": cond_rel_c2w.astype(np.float32),
                "cond_rel_w2c": cond_rel_w2c.astype(np.float32),
                "ref_w2c": ref_w2c.astype(np.float32),
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={
                "source_path": str(video_path),
                "stem": video_path.stem,
            },
        )
        return sample


@register_dataset("example_i2v")
class ExampleI2VDataset(BaseInferenceDataset):
    """Image-conditioned dataset for Wan2.2 example data."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, *, dataset_path: Path, options: Dict[str, Any]) -> None:
        super().__init__(dataset_path=dataset_path, options=options)

        image_dir = self.dataset_path / options.get("image_dir", "images")
        if not image_dir.exists():
            raise FileNotFoundError(f"Condition image directory not found: {image_dir}")
        self.image_paths = sorted(
            p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found under {image_dir}")

        metadata_path = self.dataset_path / options.get("metadata_filename", "metadata.csv")
        self.text_map: Dict[str, str] = {}
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            for _, row in metadata.iterrows():
                file_name = str(row.get("file_name", "")).strip()
                if not file_name:
                    continue
                stem = Path(file_name).stem
                raw_text = row.get("text", "")
                text = "" if pd.isna(raw_text) else str(raw_text)
                self.text_map[stem] = text

        self.num_frames = int(options.get("num_frames", 81))
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

    def __len__(self) -> int:
        return len(self.image_paths)

    def _process_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            frame = preprocess_video_frame(img.convert("RGB"), self.height, self.width)
        tensor = self.preprocess(frame)  # (C, H, W)
        video = tensor.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        return video.to(torch.float32)

    def __getitem__(self, index: int) -> InferenceSample:
        image_path = self.image_paths[index]
        video_tensor = self._process_image(image_path)
        stem = image_path.stem
        text = self.text_map.get(stem, "")

        return InferenceSample(
            video=video_tensor,
            text=text,
            cond_data={
                "condition_image_path": str(image_path),
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "cam_indices": self.cam_indices.copy(),
            },
            metadata={
                "source_path": str(image_path),
                "stem": stem,
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

# ---------------------------------------------------------------------------
# Target pose helpers
# ---------------------------------------------------------------------------

def load_target_pose_directory(target_dir: Path) -> List[TargetSpec]:
    """Load every .npz under target_dir into TargetSpec entries."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Target pose directory not found: {target_dir}")
    specs: List[TargetSpec] = []
    for npz_path in sorted(target_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "data" not in data or "inds" not in data:
                raise ValueError(f"Invalid target pose file {npz_path}: expect keys 'data' and 'inds'")
            raw_pose = data["data"].astype(np.float32)
            raw_inds = data["inds"].astype(np.int64)
        specs.append(
            TargetSpec(
                name=npz_path.stem,
                raw_pose=raw_pose,
                raw_inds=raw_inds,
                metadata={"path": str(npz_path)},
            )
        )
    if not specs:
        raise RuntimeError(f"No .npz pose files found in {target_dir}")
    return specs


def load_dit_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], rank: int) -> None:
    model_state = model.state_dict()
    compatible_state: Dict[str, torch.Tensor] = {}
    skipped_for_shape: List[str] = []
    unexpected_keys: List[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_for_shape.append(key)
            continue
        compatible_state[key] = value

    load_msg = model.load_state_dict(compatible_state, strict=True)
    missing_keys, still_unexpected = load_msg

    if rank == 0:
        if skipped_for_shape:
            print(f"[warning] skipped {len(skipped_for_shape)} keys with mismatched shapes: {skipped_for_shape[:5]}{'...' if len(skipped_for_shape) > 5 else ''}")
        if unexpected_keys or still_unexpected:
            total_unexpected = set(unexpected_keys).union(still_unexpected)
            if total_unexpected:
                print(f"[warning] ignored unexpected keys: {list(total_unexpected)[:5]}{'...' if len(total_unexpected) > 5 else ''}")
        if missing_keys:
            print(f"[warning] missing {len(missing_keys)} keys when loading weights: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")


# ---------------------------------------------------------------------------
# Pose utilities
# ---------------------------------------------------------------------------

def ensure_homogeneous(mats: np.ndarray) -> np.ndarray:
    if mats.ndim != 3:
        raise ValueError(f"Pose array must be 3D, got {mats.shape}")
    T, h, w = mats.shape
    if h == 4 and w == 4:
        return mats
    if h == 3 and w == 4:
        last = np.tile(np.array([[0, 0, 0, 1]], dtype=mats.dtype), (T, 1, 1))
        return np.concatenate([mats, last], axis=1)
    raise ValueError(f"Unsupported pose shape {mats.shape}, expected (T,4,4) or (T,3,4)")


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_inv
    result[:3, 3] = translation_inv
    return result.astype(np.float32)


def convert_c2w_convention(c2w: np.ndarray) -> np.ndarray:
    """
    调整坐标轴顺序并翻转 Y 轴以匹配训练时的约定。

    兼容输入形状：
      * (4, 4) 单帧矩阵
      * (T, 4, 4) 多帧序列
    """
    converted = c2w.copy()
    if converted.ndim == 2:
        converted = converted[:, [1, 2, 0, 3]]
        converted[:3, 1] *= -1.0
    elif converted.ndim == 3:
        converted = converted[:, :, [1, 2, 0, 3]]
        converted[:, :3, 1] *= -1.0
    else:
        raise ValueError(f"convert_c2w_convention expects 2D or 3D array, got shape {converted.shape}")
    return converted


def center_trajectory(c2ws: np.ndarray) -> np.ndarray:
    if c2ws.shape[0] == 0:
        return c2ws
    centered = c2ws.copy()
    origin = centered[0, :3, 3].copy()
    if np.allclose(origin, 0):
        return centered
    centered[:, :3, 3] -= origin
    return centered


def compute_relative_c2w(ref_w2c: np.ndarray, c2ws: np.ndarray) -> np.ndarray:
    rel_w2c = [ref_w2c @ c2w for c2w in c2ws]
    return np.stack([invert_se3(mat) for mat in rel_w2c], axis=0)


def c2w_to_w2c(rel_c2w: np.ndarray) -> np.ndarray:
    return np.stack([invert_se3(T) for T in rel_c2w], axis=0)


def normalize_joint_translation(cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([cond_rel_c2w, tgt_rel_c2w], axis=0)
    translations = combined[:, :3, 3]
    norms = np.linalg.norm(translations, axis=1)
    max_norm = float(np.max(norms)) if norms.size > 0 else 0.0
    if not np.isfinite(max_norm) or max_norm < eps:
        return cond_rel_c2w.copy(), tgt_rel_c2w.copy()
    scale = max_norm
    cond_scaled = cond_rel_c2w.copy()
    tgt_scaled = tgt_rel_c2w.copy()
    cond_scaled[:, :3, 3] /= scale
    tgt_scaled[:, :3, 3] /= scale
    return cond_scaled, tgt_scaled


def nearest_index(inds: np.ndarray, target: int) -> int:
    return int(np.abs(inds - target).argmin())


def select_pose_sequence(mats: np.ndarray, inds: np.ndarray, desired: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    selected = []
    chosen_inds = []
    for idx in desired:
        matches = np.where(inds == idx)[0]
        choice = int(matches[0]) if matches.size > 0 else nearest_index(inds, idx)
        selected.append(mats[choice])
        chosen_inds.append(inds[choice])
    return np.stack(selected, axis=0), np.asarray(chosen_inds, dtype=np.int64)


def preprocess_video_frame(image: Image.Image, height: int, width: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized_h = max(int(round(src_h * scale)), 1)
    resized_w = max(int(round(src_w * scale)), 1)
    resized = TF.resize(image, (resized_h, resized_w), interpolation=InterpolationMode.BILINEAR, antialias=True)
    top = max((resized_h - height) // 2, 0)
    left = max((resized_w - width) // 2, 0)
    cropped = TF.crop(resized, top, left, height, width)
    return cropped


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed_environment() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return distributed, world_size, rank, local_rank, device


def broadcast_output_directory(base_dir: Path, distributed: bool, rank: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if distributed:
        payload = [timestamp if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        timestamp = payload[0]
    output_dir = base_dir / timestamp
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    return output_dir


def cleanup_distributed_environment(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Placeholder pipeline handler registry
# ---------------------------------------------------------------------------

class BasePipelineHandler(abc.ABC):
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

    @abc.abstractmethod
    def build_inputs(
        self,
        sample: InferenceSample,
        target: TargetSpec,
        *,
        source_video: torch.Tensor,
    ) -> PreparedInference:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_inference(
        self,
        pipe: Any,
        prepared: PreparedInference,
        prompt_text: str,
        pipe_kwargs: Dict[str, Any],
    ) -> Any:
        """Default inference simply forwards to the pipeline callable."""
        return pipe(**pipe_kwargs)

PIPELINE_REGISTRY: Dict[str, type[BasePipelineHandler]] = {}


def register_pipeline(name: str) -> Callable[[type[BasePipelineHandler]], type[BasePipelineHandler]]:
    def decorator(cls: type[BasePipelineHandler]) -> type[BasePipelineHandler]:
        if name in PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline '{name}' already registered")
        PIPELINE_REGISTRY[name] = cls
        return cls

    return decorator


NEGATIVE_PROMPT = (
    "人物肢体不完整，动作诡异，肢体模糊，"
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _first_existing_path(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _load_v2v_pipeline(device_str: str) -> WanVideoReCamMasterPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        [
            str(WAN21_MODEL_DIR / "diffusion_pytorch_model.safetensors"),
            str(WAN21_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(WAN21_MODEL_DIR / "Wan2.1_VAE.pth"),
        ]
    )
    return WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device_str)


def _load_i2v_pipeline(device_str: str) -> WanVideoPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    text_encoder_path = _first_existing_path(
        [
            WAN22_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
            WAN21_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
        ]
    )
    if text_encoder_path is None:
        raise FileNotFoundError("Cannot locate Wan text encoder weights under Wan2.1/2.2 directories.")

    vae_path = WAN22_MODEL_DIR / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing Wan2.2 VAE weights: {vae_path}")

    diffusion_paths = sorted(WAN22_MODEL_DIR.glob("diffusion_pytorch_model-*.safetensors"))
    if not diffusion_paths:
        fallback = WAN22_MODEL_DIR / "diffusion_pytorch_model.safetensors"
        if fallback.exists():
            diffusion_paths = [fallback]
        else:
            raise FileNotFoundError(f"Missing Wan2.2 diffusion weights under {WAN22_MODEL_DIR}")

    models_to_load: List[Any] = [str(text_encoder_path), str(vae_path)]
    if len(diffusion_paths) == 1:
        models_to_load.append(str(diffusion_paths[0]))
    else:
        models_to_load.append([str(p) for p in diffusion_paths])
    model_manager.load_models(models_to_load)

    pipe = WanVideoPipeline(device=device_str, torch_dtype=torch.bfloat16)
    available_names = set(model_manager.model_name)
    if "wan_video_text_encoder" in available_names:
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    if "wan_video_image_encoder" in available_names:
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
    if "wan_video_dit" in available_names:
        pipe.dit = model_manager.fetch_model("wan_video_dit")
    if "wan_video_dit2" in available_names:
        pipe.dit2 = model_manager.fetch_model("wan_video_dit2")
    if "wan_video_vae" in available_names:
        pipe.vae = model_manager.fetch_model("wan_video_vae")
    if "wan_video_motion_controller" in available_names:
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
    if "wan_video_vace" in available_names:
        pipe.vace = model_manager.fetch_model("wan_video_vace")
    if "wan_video_animate_adapter" in available_names:
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

    tokenizer_path = _first_existing_path(
        [
            WAN22_MODEL_DIR / "google" / "umt5-xxl",
            WAN21_MODEL_DIR / "google" / "umt5-xxl",
        ]
    )
    if pipe.text_encoder is not None:
        pipe.prompter.fetch_models(pipe.text_encoder)
    if tokenizer_path is None:
        raise FileNotFoundError("Cannot locate Wan tokenizer directory under Wan2.1/2.2 models.")
    pipe.prompter.fetch_tokenizer(str(tokenizer_path))

    existing_names = list(getattr(pipe, "model_names", []))
    tracked = []
    for name in ["text_encoder", "image_encoder", "dit", "dit2", "vae", "motion_controller", "vace", "animate_adapter"]:
        if getattr(pipe, name, None) is not None:
            tracked.append(name)
    pipe.model_names = list(dict.fromkeys(existing_names + tracked))
    return pipe


def initialize_inference_pipeline(pipeline_kind: str, device_str: str) -> Any:
    if pipeline_kind == "v2v":
        return _load_v2v_pipeline(device_str)
    if pipeline_kind == "i2v":
        return _load_i2v_pipeline(device_str)
    raise ValueError(f"Unsupported pipeline_kind='{pipeline_kind}'")


def load_checkpoint_file(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    path_str = str(path)
    if path_str.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore

        return load_file(path_str)
    try:

        return torch.load(path_str, map_location="cpu")
    except RuntimeError as err:
        if "PytorchStreamReader" not in str(err):
            raise
        with open(path, "rb") as fh:
            magic = fh.read(4)
        if magic == b"SAFE":
            from safetensors.torch import load_file  # type: ignore

            return load_file(path_str)
        raise


@register_pipeline("v2v")
class V2VPipelineHandler(BasePipelineHandler):
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


@register_pipeline("i2v")
class I2VPipelineHandler(BasePipelineHandler):
    """Custom inference loop that mirrors the training/validation i2v flow."""

    def build_inputs(
        self,
        sample: InferenceSample,
        target: TargetSpec,
        *,
        source_video: torch.Tensor,
    ) -> PreparedInference:
        cond_info = sample.cond_data
        cam_indices = np.asarray(cond_info["cam_indices"], dtype=np.int64)
        target_w2cs_sel, matched_inds = select_pose_sequence(target.raw_pose, target.raw_inds, cam_indices)

        target_c2ws = target_w2cs_sel.transpose(0, 2, 1)

        target_c2ws = convert_c2w_convention(target_c2ws)
        target_c2ws = center_trajectory(target_c2ws)
        ref_w2c = invert_se3(target_c2ws[0])
        tgt_rel_c2w = compute_relative_c2w(ref_w2c, target_c2ws)
        _, tgt_norm = normalize_joint_translation(tgt_rel_c2w, tgt_rel_c2w)
        tgt_rel_w2c = c2w_to_w2c(tgt_norm)
        camera_tensor = torch.from_numpy(tgt_rel_w2c).unsqueeze(0).to(dtype=self.dtype)

        return PreparedInference(
            pipe_kwargs={
                "camera_embedding": camera_tensor,
                "condition_image_path": cond_info["condition_image_path"],
                "num_frames": int(cond_info["num_frames"]),
                "height": int(cond_info["height"]),
                "width": int(cond_info["width"]),
            },
            target_rel_w2c=tgt_rel_w2c,
            cam_indices=cam_indices,
            metadata={
                "target_name": target.name,
                "matched_inds": matched_inds,
            },
        )

    def _load_condition_image(self, path: str, height: int, width: int) -> Image.Image:
        with Image.open(path) as img:
            processed = preprocess_video_frame(img.convert("RGB"), height, width)
        return processed

    def _encode_condition_embeddings(
        self,
        pipe: WanVideoPipeline,
        image: Image.Image,
        *,
        num_frames: int,
        height: int,
        width: int,
        tiled: bool,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        device = pipe.device
        dtype = pipe.torch_dtype

        image_tensor = pipe.preprocess_image(image.resize((width, height))).to(device)
        clip_feature = None
        if pipe.image_encoder is not None:
            clip_feature = pipe.image_encoder.encode_image([image_tensor]).to(dtype=dtype, device=device)

        zeros_tail = torch.zeros(
            image_tensor.shape[1],
            max(num_frames - 1, 0),
            height,
            width,
            device=device,
            dtype=image_tensor.dtype,
        )
        vae_input = torch.cat([image_tensor.transpose(0, 1), zeros_tail], dim=1).unsqueeze(0)
        vae_latents = pipe.vae.encode(
            vae_input.to(dtype=dtype),
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0].to(dtype=dtype, device=device)
        latent_h = height // pipe.vae.upsampling_factor
        latent_w = width // pipe.vae.upsampling_factor
        msk = torch.ones(1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        if num_frames > 1:
            msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]
        y = torch.concat([msk, vae_latents], dim=0).unsqueeze(0)

        single_video = pipe.preprocess_video([image]).to(device)
        first_latents = pipe.vae.encode(
            single_video.to(dtype=dtype),
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=dtype, device=device)

        return {
            "clip_feature": clip_feature,
            "y": y.to(dtype=dtype, device=device),
            "first_latents": first_latents,
        }

    def run_inference(
        self,
        pipe: WanVideoPipeline,
        prepared: PreparedInference,
        prompt_text: str,
        pipe_kwargs: Dict[str, Any],
    ) -> Sequence[Image.Image]:
        cfg_scale = float(pipe_kwargs.get("cfg_scale", self.global_opts.get("cfg_scale", 5.0)))
        num_steps = int(pipe_kwargs.get("num_inference_steps", 25))
        seed = int(pipe_kwargs.get("seed", 0))
        negative_prompt = pipe_kwargs.get("negative_prompt", NEGATIVE_PROMPT)
        tiled = bool(pipe_kwargs.get("tiled", True))
        tile_size = pipe_kwargs.get("tile_size", (34, 34))
        tile_stride = pipe_kwargs.get("tile_stride", (18, 16))

        camera_embedding = pipe_kwargs["camera_embedding"].to(device=pipe.device, dtype=pipe.torch_dtype)
        condition_path = pipe_kwargs["condition_image_path"]
        num_frames = int(pipe_kwargs["num_frames"])
        height = int(pipe_kwargs["height"])
        width = int(pipe_kwargs["width"])

        condition_image = self._load_condition_image(condition_path, height, width)
        embeddings = self._encode_condition_embeddings(
            pipe,
            condition_image,
            num_frames=num_frames,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        clip_feature = embeddings["clip_feature"]
        y = embeddings["y"]
        first_latents = embeddings["first_latents"]

        latent_frames = camera_embedding.shape[1]
        latent_shape = (
            1,
            first_latents.shape[1],
            latent_frames,
            first_latents.shape[-2],
            first_latents.shape[-1],
        )
        latents_gen = pipe.generate_noise(latent_shape, seed=seed, rand_device="cpu")
        latents_gen = latents_gen.to(dtype=pipe.torch_dtype, device=pipe.device)
        latents_gen[:, :, 0:1] = first_latents

        positive_prompt = prompt_text or ""
        context_pos = pipe.prompter.encode_prompt(positive_prompt, positive=True, device=pipe.device).to(dtype=pipe.torch_dtype)
        context_neg = pipe.prompter.encode_prompt(negative_prompt or "", positive=False, device=pipe.device).to(dtype=pipe.torch_dtype)
        pipe.scheduler.set_timesteps(num_steps, shift=pipe.scheduler.shift, denoising_strength=1.0)
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            latents_input = latents_gen.clone()
            latents_input[:, :, 0:1] = first_latents
            prepare_extra_input = getattr(pipe, "prepare_extra_input", lambda latents=None: {})
            extra_input = prepare_extra_input(latents_input)
            dit_kwargs = dict(
                cam_emb=camera_embedding,
                clip_feature=clip_feature,
                y=y,
                temporal_indices=None,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                fuse_vae_embedding_in_latents=True,
                **extra_input,
            )
            noise_pred_pos = pipe.dit(
                latents_input,
                timestep=timestep,
                context=context_pos,
                **dit_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_neg = pipe.dit(
                    latents_input,
                    timestep=timestep,
                    context=context_neg,
                    **dit_kwargs,
                )
                noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos

            latents_gen = pipe.scheduler.step(
                noise_pred,
                pipe.scheduler.timesteps[progress_id],
                latents_input,
            )
            latents_gen[:, :, 0:1] = first_latents

        decoded = pipe.vae.decode(
            latents_gen,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if decoded.dim() == 4:
            decoded = decoded.unsqueeze(0)
        frames = pipe.vae_output_to_video(decoded)
        return frames

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_kv_options(options: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            raise ValueError(f"Option '{item}' must be in key=value format")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ReCamMaster inference")
    parser.add_argument("--dataset_kind", type=str, required=True, choices=["example", "example_i2v", "pointodyssey"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--dataset_option", action="append", default=[], help="Additional dataset options (key=value)")
    parser.add_argument("--target_pose_dir", type=str, required=True, help="Directory containing target pose .npz files")
    parser.add_argument("--pipeline_kind", type=str, default="v2v", choices=["v2v", "i2v"], help="Inference pipeline mode")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint to load")
    parser.add_argument("--output_dir", type=str, default="evaluation/example_eval", help="Directory to save outputs")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--frame_downsample_to", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Samples per batch (currently only 1 supported)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_options = parse_kv_options(args.dataset_option)
    dataset_path = Path(args.dataset_path)
    target_pose_dir = Path(args.target_pose_dir)

    distributed, world_size, rank, _, device = setup_distributed_environment()

    if args.debug and rank == 0:
        print("Debug mode is enabled.")
        import debugpy  # type: ignore

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")
    elif args.debug and rank != 0:
        print(f"[rank {rank}] Debug mode requested but only rank 0 enters debug session.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_cls = DATASET_REGISTRY.get(args.dataset_kind)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset_kind={args.dataset_kind}")
    dataset = dataset_cls(dataset_path=dataset_path, options=dataset_options)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; please check dataset_path and options.")

    target_specs = load_target_pose_directory(target_pose_dir)

    if args.batch_size != 1:
        raise NotImplementedError("Only batch_size=1 is supported currently.")

    enforced_downsample = 0
    if args.frame_downsample_to != 0 and rank == 0:
        print(f"[info] frame_downsample_to={args.frame_downsample_to} ignored; enforcing 0 for inference stability.")

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    def collate_single(batch: List[InferenceSample]) -> InferenceSample:
        if len(batch) != 1:
            raise ValueError("batch_size other than 1 is not supported")
        return batch[0]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_single,
    )

    if sampler is not None:
        sampler.set_epoch(0)

    device_str = device.type if device.type == "cpu" else f"cuda:{device.index}"
    pipe = initialize_inference_pipeline(args.pipeline_kind, device_str)

    if rank == 0:
        print(f"Using device: {device_str} | world_size={world_size}")
        print(f"Loading checkpoint from: {args.ckpt_path}")

    state_dict = load_checkpoint_file(args.ckpt_path)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "module" in state_dict:
        state_dict = state_dict["module"]

    prefixes_to_remove = ["model.", "module.", "pipe.dit.", "dit."]
    cleaned_state: Dict[str, torch.Tensor] = {}
    dropped_keys: List[str] = []
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        if ".cam_encoder." in new_key or ".projector." in new_key:
            dropped_keys.append(new_key)
            continue
        cleaned_state[new_key] = value
    if rank == 0 and dropped_keys:
        print(f"[info] dropping {len(dropped_keys)} camera/projector keys: {dropped_keys[:5]}{'...' if len(dropped_keys) > 5 else ''}")

    load_dit_state_dict(pipe.dit, cleaned_state, rank)

    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    pipe.eval()

    handler_cls = PIPELINE_REGISTRY.get(args.pipeline_kind)
    if handler_cls is None:
        raise ValueError(f"No inference handler registered for pipeline_kind={args.pipeline_kind}")
    handler = handler_cls(device=device, dtype=torch.bfloat16, global_opts={"cfg_scale": args.cfg_scale, "debug_pose": args.debug})

    base_output_dir = Path(args.output_dir)
    output_dir = broadcast_output_directory(base_output_dir, distributed, rank)

    if rank == 0:
        print(f"[info] Saving outputs to: {output_dir}")

    for sample_idx, sample in enumerate(dataloader):
        source_video = sample.video.unsqueeze(0).to(device)
        prompt_text = sample.text
        sample_stem = sample.metadata.get("stem", f"sample_{sample_idx:04d}")

        for target in target_specs:
            output_stem = f"{sample_stem}_{target.name}"
            video_path = output_dir / f"{output_stem}.mp4"
            pose_path = video_path.with_suffix(".npz")
            
            # if target.name != 'cam01':
            #     continue

            if video_path.exists():
                if rank == 0:
                    print(f"[info] Skip existing output: {video_path.name}")
                continue

            prepared = handler.build_inputs(sample, target, source_video=source_video)

            pipe_kwargs = dict(
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
                cfg_scale=args.cfg_scale,
                frame_downsample_to=enforced_downsample,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                tiled=True,
            )
            pipe_kwargs.update(prepared.pipe_kwargs)

            with torch.no_grad():
                video = handler.run_inference(pipe, prepared, prompt_text, pipe_kwargs)

            save_video(video, str(video_path), fps=30, quality=5)
            np.savez(
                pose_path,
                data=prepared.target_rel_w2c.astype(np.float32),
                inds=prepared.cam_indices.astype(np.int64),
            )

        if rank == 0:
            print(f"[info] Completed sample {sample_idx + 1}/{len(dataloader)} -> {sample_stem}")

    cleanup_distributed_environment(distributed)


if __name__ == "__main__":
    main()
