from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

__all__ = [
    "BaseInferenceDataset",
    "BasePipelineHandler",
    "InferenceSample",
    "TargetSpec",
    "PreparedInference",
    "DATASET_REGISTRY",
    "PIPELINE_REGISTRY",
    "register_dataset",
    "register_pipeline",
    "NEGATIVE_PROMPT",
    "load_checkpoint_file",
    "load_dit_state_dict",
    "ensure_homogeneous",
    "invert_se3",
    "convert_c2w_convention",
    "convert_c2w_convention_I2V",
    "center_trajectory",
    "compute_relative_c2w",
    "c2w_to_w2c",
    "normalize_joint_translation",
    "nearest_index",
    "select_pose_sequence",
    "preprocess_video_frame",
]


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

    def apply_checkpoint(self, pipe: Any, ckpt_path: str | Path | None, rank: int) -> None:
        """Load and apply checkpoint weights to the pipeline if provided."""
        if ckpt_path is None:
            return
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_dir():
            if rank == 0:
                print(f"[info] Provided ckpt_path='{ckpt_path}' is a directory; skip overriding base weights.")
            return
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
        if rank == 0:
            print(f"Loading checkpoint from: {ckpt_path}")
        ckpt_state = load_checkpoint_file(ckpt_path)
        if isinstance(ckpt_state, dict) and "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]
        if isinstance(ckpt_state, dict) and "module" in ckpt_state:
            ckpt_state = ckpt_state["module"]

        prefixes_to_remove = ["model.", "module.", "pipe.dit.", "dit."]
        cleaned_state: Dict[str, torch.Tensor] = {}
        dropped_keys: List[str] = []
        for key, value in ckpt_state.items():
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

        if not hasattr(pipe, "dit") or pipe.dit is None:
            raise AttributeError("Pipeline does not contain 'dit' module for checkpoint loading.")
        load_dit_state_dict(pipe.dit, cleaned_state, rank)


PIPELINE_REGISTRY: Dict[str, type[BasePipelineHandler]] = {}


def register_pipeline(name: str) -> Callable[[type[BasePipelineHandler]], type[BasePipelineHandler]]:
    def decorator(cls: type[BasePipelineHandler]) -> type[BasePipelineHandler]:
        if name in PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline '{name}' already registered")
        PIPELINE_REGISTRY[name] = cls
        return cls

    return decorator


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


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


def load_dit_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], rank: int) -> None:
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
    """调整坐标轴顺序并翻转 Y 轴以匹配训练时的约定。"""
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


def convert_c2w_convention_I2V(c2w: np.ndarray) -> np.ndarray:
    """
    调整坐标轴顺序并翻转 Y 轴以匹配训练时的约定。

    输入可为 w2c（平移在最后一行）或已转置的 c2w（平移在最后一列），内部会自动判断是否需要转置。
    """
    converted = c2w.transpose(0, 2, 1)
    converted = converted[:, :, [1, 2, 0, 3]]
    converted = converted[:, [1, 2, 0, 3], :]
    converted[:, :3, 1] *= -1.0
    converted[:, 1, :3] *= -1.0
    converted[:, 1, 3] *= -1.0
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


def normalize_joint_translation(cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1) -> Tuple[np.ndarray, np.ndarray]:
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
