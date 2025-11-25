"""
Dataset classes for ReCamMaster training
"""

import bisect
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


DATASET_ALIASES = {
    "multicam": "multicam",
    "multi_cam": "multicam",
    "multi": "multicam",
    "rel10k": "rel10k",
    "re10k": "rel10k",
    "relestate10k": "rel10k",
    "realestate10k": "rel10k",
}


def normalize_dataset_name(name: str) -> str:
    if not name:
        raise ValueError("dataset_type 不能为空。")
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"未知的数据集类型 '{name}'. 支持: {sorted(set(DATASET_ALIASES.values()))}")
    return DATASET_ALIASES[key]


@dataclass
class DatasetSpec:
    name: str
    root: str
    metadata_path: Optional[str]
    weight: float = 1.0

    def __post_init__(self):
        self.name = normalize_dataset_name(self.name)
        self.root = os.path.abspath(self.root)
        if self.metadata_path:
            self.metadata_path = os.path.abspath(self.metadata_path)
        self.weight = float(max(self.weight, 0.0))


DEFAULT_IMAGE_WIDTH = 832.0
DEFAULT_IMAGE_HEIGHT = 480.0
DEFAULT_SENSOR_WIDTH_MM = 23.76
DEFAULT_SENSOR_HEIGHT_MM = 23.76
DEFAULT_FOCAL_MM = 18.0

DATASET_ID_PATTERN = re.compile(r"/train/([^/]+)/")
FOCAL_PATTERN = re.compile(r"f(?P<focal>\d+(?:\.\d+)?)", re.IGNORECASE)


def _select_temporal_indices(total_frames: int, target_frames: int) -> np.ndarray:
    """
    Evenly sample indices from [0, total_frames) to match the latent frame count.
    """
    if target_frames <= 0:
        raise ValueError("target_frames 必须大于 0。")
    if total_frames <= 0:
        raise ValueError("total_frames 必须大于 0。")
    if target_frames >= total_frames:
        return np.arange(total_frames, dtype=np.int64)
    positions = np.linspace(0, total_frames - 1, num=target_frames)
    return np.clip(np.round(positions).astype(np.int64), 0, total_frames - 1)


def infer_dataset_id(path: str) -> str:
    """
    Infer dataset identifier (e.g., f18_aperture10) from a tensor/video path.
    """
    match = DATASET_ID_PATTERN.search(path)
    if match:
        return match.group(1)
    parts = re.split(r"[\\/]+", path)
    for part in parts:
        if part.startswith("f") and "aperture" in part:
            return part
    return "unknown"


def _safe_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_intrinsics_for_dataset(
    dataset_id: str,
    image_width: float = DEFAULT_IMAGE_WIDTH,
    image_height: float = DEFAULT_IMAGE_HEIGHT,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    sensor_height_mm: float = DEFAULT_SENSOR_HEIGHT_MM,
    default_focal_mm: float = DEFAULT_FOCAL_MM,
) -> np.ndarray:
    """
    Build a 3x3 camera intrinsic matrix for a dataset identifier derived from folder name.
    """
    focal_match = FOCAL_PATTERN.search(dataset_id)
    focal_mm = _safe_float(focal_match.group("focal") if focal_match else None, default_focal_mm)
    fx = focal_mm * (image_width / sensor_width_mm)
    fy = focal_mm * (image_height / sensor_height_mm)
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _normalize_prompt_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure prompt tensor carries batch, sequence, hidden dims for downstream reshape ops.
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 1:
        tensor = tensor.view(1, tensor.shape[0], 1)
    return tensor


def _ensure_prompt_context(prompt_emb: Optional[object]) -> Dict[str, torch.Tensor]:
    """
    Normalize prompt embeddings so downstream code always receives a dict with 'context'.
    """
    if prompt_emb is None:
        return {}
    if isinstance(prompt_emb, dict):
        ctx = prompt_emb.get("context")
        if torch.is_tensor(ctx):
            prompt_emb["context"] = _normalize_prompt_tensor(ctx)
        return prompt_emb
    if torch.is_tensor(prompt_emb):
        return {"context": _normalize_prompt_tensor(prompt_emb)}
    return {}


def resolve_tensor_path(
    video_path: str,
    dataset_root: Optional[str] = None,
    tensor_suffixes: Optional[Tuple[str, ...]] = None,
) -> str:
    """
    Resolve absolute tensor path from a video path and optional dataset root.
    """
    tensor_suffixes = tensor_suffixes or (".tensors.pth",)
    candidate = video_path
    if not os.path.isabs(candidate):
        if dataset_root is not None:
            candidate = os.path.join(dataset_root, candidate.lstrip("/"))
        candidate = os.path.abspath(candidate)
    possible_bases = [candidate]
    root, ext = os.path.splitext(candidate)
    if ext:
        possible_bases.append(root)
        second_root, second_ext = os.path.splitext(root)
        if second_ext:
            possible_bases.append(second_root)
    seen = set()
    ordered_bases = []
    for base in possible_bases:
        if base not in seen:
            ordered_bases.append(base)
            seen.add(base)
    for base in ordered_bases:
        for suffix in tensor_suffixes:
            if base.endswith(suffix):
                tensor_path = base
            else:
                tensor_path = base + suffix
            if os.path.exists(tensor_path):
                return tensor_path
    return ""


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class BaseCameraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        steps_per_epoch: int,
        paths: Optional[List[str]],
        fixed_length: Optional[int],
        seed: int,
        dataset_root: Optional[str] = None,
        image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
    ):
        path_list = list(paths or [])
        if len(path_list) == 0:
            raise ValueError("metadata 中未找到可用的 tensor 缓存文件。")
        self.path = path_list
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.fixed_length = fixed_length
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.sensor_size_mm = sensor_size_mm
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}

    def _sample_path_index(self, index: int) -> int:
        torch.manual_seed(self.seed + index)
        data_id = torch.randint(0, len(self.path), (1,), dtype=torch.long)[0].item()
        data_id = (data_id + index) % len(self.path)
        return data_id

    @staticmethod
    def parse_matrix(matrix_str: str) -> np.ndarray:
        rows = matrix_str.strip().split("] [")
        matrix = []
        for row in rows:
            row = row.replace("[", "").replace("]", "")
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    @staticmethod
    def get_relative_pose(cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def _get_intrinsics_tensor(self, data_path: str, repeat: int) -> torch.Tensor:
        dataset_id = infer_dataset_id(data_path)
        if dataset_id not in self._intrinsics_cache:
            intrinsics_np = compute_intrinsics_for_dataset(
                dataset_id,
                image_width=self.image_size[0],
                image_height=self.image_size[1],
                sensor_width_mm=self.sensor_size_mm[0],
                sensor_height_mm=self.sensor_size_mm[1],
            )
            self._intrinsics_cache[dataset_id] = torch.from_numpy(intrinsics_np)
        Ks = self._intrinsics_cache[dataset_id]
        expanded = Ks.unsqueeze(0).repeat(repeat, 1, 1)
        return expanded.to(torch.float32)

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return self.steps_per_epoch


# ------------------
# Pose utilities (moved outside __getitem__)
# ------------------

def invert_SE3_np(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix (numpy, float32 output)."""
    T = T.astype(np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rinv
    out[:3, 3] = tinv
    return out.astype(np.float32)


def compute_relative_c2w(cam_params, ref_cam: Camera, get_relative_pose_fn) -> np.ndarray:
    """
    Compute relative c2w poses (cam_i <- ref) for a list of cameras.
    - ref_cam: reference Camera (e.g., cond[0])
    - get_relative_pose_fn: function that accepts [ref_cam, cam_i] and returns an array
      whose second element [1] is cam_ref<-cam_i (relative pose in w2c form composed with ref).
    Returns: [N, 4, 4] numpy array of relative c2w.
    """
    rel_c2w_list = []
    for i in range(len(cam_params)):
        relative_pose_matrix = get_relative_pose_fn([ref_cam, cam_params[i]])  # [I, cam_ref<-cam_i]
        cam_ref_from_cami = relative_pose_matrix[1]
        cami_from_ref = invert_SE3_np(cam_ref_from_cami)  # c2w relative
        rel_c2w_list.append(cami_from_ref)
    return np.stack(rel_c2w_list, axis=0)


def normalize_translation_baseline(cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1e-3):
    """
    Normalize only the translation components of relative c2w trajectories using a shared baseline.
    Baseline strategy:
      1) Use L2 norm of cond last frame translation relative to ref.
      2) If invalid or too small, use median of cond translation norms across frames.
      3) If still invalid, fall back to 1.0.
    Operates on copies and returns (cond_rel_c2w_norm, tgt_rel_c2w_norm, baseline).
    """
    cond_rel = cond_rel_c2w.copy()
    tgt_rel = tgt_rel_c2w.copy()

    baseline = float(np.linalg.norm(cond_rel[-1][:3, 3], ord=2))
    if not np.isfinite(baseline):
        baseline = 0.0
    if baseline <= eps:
        cond_dists = np.linalg.norm(cond_rel[:, :3, 3], axis=1)
        valid = cond_dists > eps
        if np.any(valid):
            baseline = float(np.median(cond_dists[valid]))
        else:
            baseline = 1.0

    cond_rel[:, :3, 3] = cond_rel[:, :3, 3] / baseline
    tgt_rel[:, :3, 3] = tgt_rel[:, :3, 3] / baseline
    return cond_rel, tgt_rel, baseline


def normalize_translation(cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1e-2):
    """
    Normalize translations by the maximum L2 norm across BOTH cond and tgt relative trajectories.
    - Compute max_norm = max(||t_i||) over all frames i from both cond_rel_c2w and tgt_rel_c2w.
    - If max_norm < eps (default 0.01), do NOT normalize and return copies unchanged.
    - Otherwise divide both translations by max_norm.

    Returns: (cond_rel_norm, tgt_rel_norm, max_norm)
    Interface matches normalize_translation_baseline for easy drop-in replacement.
    """
    cond_rel = cond_rel_c2w.copy()
    tgt_rel = tgt_rel_c2w.copy()

    # Collect translation norms
    cond_norms = np.linalg.norm(cond_rel[:, :3, 3], axis=1)
    tgt_norms = np.linalg.norm(tgt_rel[:, :3, 3], axis=1)

    max_norm = float(np.max([np.max(cond_norms) if cond_norms.size > 0 else 0.0,
                              np.max(tgt_norms) if tgt_norms.size > 0 else 0.0]))
    if not np.isfinite(max_norm):
        max_norm = 0.0

    if max_norm < eps:
        # No normalization
        return cond_rel, tgt_rel, max_norm

    cond_rel[:, :3, 3] = cond_rel[:, :3, 3] / max_norm
    tgt_rel[:, :3, 3] = tgt_rel[:, :3, 3] / max_norm
    return cond_rel, tgt_rel, max_norm


## TODO: Multicam dataset class
### class MulticamDataset():

## TODO: rel10k dataset class
### class Relestate10kDataset():



class TensorDataset(BaseCameraDataset):
    def __init__(
        self,
        steps_per_epoch,
        paths=None,
        fixed_length=None,
        seed=42,
        dataset_root: Optional[str] = None,
        image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
    ):
        paths = paths or []
        print(len(paths), "tensors cached in metadata.")
        super().__init__(
            steps_per_epoch=steps_per_epoch,
            paths=paths,
            fixed_length=fixed_length,
            seed=seed,
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )

    def __getitem__(self, index):
        # Return: 
        # data['latents']: torch.Size([16, 21*2, 60, 104])
        # data['camera']: torch.Size([21, 3, 4])
        # data['prompt_emb']['context'][0]: torch.Size([512, 4096])
        while True:
            try:
                data = {}
                data_id = self._sample_path_index(index)
                path_tgt = self.path[data_id]
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                # Use deterministic random selection for condition camera
                random.seed(self.seed + index + 1000)  # Different seed offset for condition selection
                cond_idx = random.randint(1, 10)
                while cond_idx == tgt_idx:
                    cond_idx = random.randint(1, 10)
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                data['latents'] = torch.cat((data_tgt['latents'],data_cond['latents']),dim=1)
                data['prompt_emb'] = _ensure_prompt_context(data_tgt.get('prompt_emb'))
                data['image_emb'] = {}

                # Extract scene information from path
                scene_match = re.search(r'scene(\d+)', path_tgt.lower())
                scene_id = scene_match.group(1) if scene_match else 'unknown'
                
                # Store camera type information
                data['scene_id'] = scene_id
                data['condition_cam_type'] = f"cam{cond_idx:02d}"
                data['target_cam_type'] = f"cam{tgt_idx:02d}"
                data['path'] = path_tgt  # Keep original path for reference

                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")              
                with open(tgt_camera_path, 'r') as file:
                    cam_data = json.load(file)
                # Build c2w trajectories for cond and tgt with axis normalization (no fixed scaling)
                multiview_c2ws = []
                cam_idx = list(range(81))[::4]
                for view_idx in [cond_idx, tgt_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        # Removed fixed /100 scaling; scale will be determined by baseline normalization
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                # Make Camera objects for relative pose utility
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]

                # Reference is cond[0] in world coordinates
                ref_cam = cond_cam_params[0]

                # 1) Compute relative c2w for both trajectories
                cond_rel_c2w = compute_relative_c2w(cond_cam_params, ref_cam, self.get_relative_pose)
                tgt_rel_c2w = compute_relative_c2w(tgt_cam_params, ref_cam, self.get_relative_pose)

                # 2) Normalize translation with shared baseline
                cond_rel_c2w, tgt_rel_c2w, _baseline = normalize_translation(cond_rel_c2w, tgt_rel_c2w)

                # 3) Invert to obtain relative w2c
                tgt_rel_w2c = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w], axis=0)
                cond_rel_w2c = np.stack([invert_SE3_np(T) for T in cond_rel_c2w], axis=0)

                # Concatenate tgt first then cond to align with latents
                all_w2c = np.concatenate([tgt_rel_w2c, tgt_rel_w2c], axis=0)
                camera_tensor = torch.from_numpy(all_w2c).to(torch.float32)
                data['camera'] = camera_tensor
                data['intrinsics'] = self._get_intrinsics_tensor(path_tgt, repeat=camera_tensor.shape[0])
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                # Use deterministic fallback for reproducibility
                index = (index + 1) % len(self.path)
        return data
    

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return self.steps_per_epoch

class BaseImageConditionDataset(BaseCameraDataset):
    """Shared logic for Wan2.2 image-conditioned datasets."""

    dataset_name = "base_i2v"

    def _load_camera_sequence(self, sample_path: str, num_frames: int) -> np.ndarray:
        raise NotImplementedError

    def _load_intrinsics_for_sample(self, sample_path: str, num_frames: int) -> torch.Tensor:
        return self._get_intrinsics_tensor(sample_path, repeat=num_frames)

    def _resolve_ids(self, sample_path: str) -> Tuple[str, str, str]:
        scene_match = re.search(r"scene(\d+)", sample_path.lower())
        scene_id = scene_match.group(1) if scene_match else "unknown"
        return scene_id, "cam00_img", "cam00"

    def _build_camera_tensor(self, sample_path: str, num_frames: int) -> torch.Tensor:
        c2ws = self._load_camera_sequence(sample_path, num_frames)
        c2ws = np.asarray(c2ws, dtype=np.float32)
        if c2ws.ndim != 3 or c2ws.shape[0] != num_frames or c2ws.shape[1:] != (4, 4):
            raise ValueError(f"Camera序列形状异常: expected [{num_frames}, 4, 4], got {c2ws.shape}")
        cam_params = [Camera(c2w) for c2w in c2ws]
        ref_cam = cam_params[0]
        tgt_rel_c2w = compute_relative_c2w(cam_params, ref_cam, self.get_relative_pose)
        tgt_rel_c2w_norm, _, _ = normalize_translation(tgt_rel_c2w, tgt_rel_c2w)
        tgt_rel_w2c = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w_norm], axis=0)
        return torch.from_numpy(tgt_rel_w2c).to(torch.float32)

    def __getitem__(self, index):
        while True:
            try:
                data_id = self._sample_path_index(index)
                sample_path = self.path[data_id]
                # TODO: 将样本统一切换为 cam10 版本以便做对比实验
                sample_path = re.sub(r"cam\d{2}", "cam10", sample_path)
                ##  TODO:load raw22 from .wan22.tensors.pth raw21 from .tensors.pth
                
                raw = torch.load(sample_path, weights_only=True, map_location="cpu")
                ## TODO: prompt_raw from raw22, others from raw21
                prompt_raw = raw.get("prompt_emb")
                wan22_path = re.sub(r"(?:\\.wan22)?\\.tensors\\.pth$", ".wan22.tensors.pth", sample_path)
                if os.path.exists(wan22_path):
                    try:
                        raw22 = torch.load(wan22_path, weights_only=True, map_location="cpu")
                        prompt_raw = raw22.get("prompt_emb", prompt_raw)
                    except Exception as exc:
                        print(f"[warn] 加载 wan22 prompt 失败 {wan22_path}: {exc}")

                latents = raw["latents"]
                if isinstance(latents, (list, tuple)):
                    latents = latents[0]
                if latents.dim() == 5:
                    latents = latents.squeeze(0)

                num_frames = latents.shape[1]
                camera_tensor = self._build_camera_tensor(sample_path, num_frames)
                intrinsics = self._load_intrinsics_for_sample(sample_path, camera_tensor.shape[0])
                scene_id, cond_cam_type, target_cam_type = self._resolve_ids(sample_path)

                data = {
                    "latents": latents,
                    "prompt_emb": _ensure_prompt_context(prompt_raw),
                    # 训练/验证不依赖 image_emb；始终返回空字典以避免不同样本键不一致导致 collate 报错
                    "image_emb": {},
                    "path": sample_path,
                    "camera": camera_tensor,
                    "intrinsics": intrinsics,
                    "scene_id": scene_id,
                    "condition_cam_type": cond_cam_type,
                    "target_cam_type": target_cam_type,
                    "dataset_name": self.dataset_name,
                }
                return data
            except Exception as exc:
                print(f"ERROR WHEN LOADING {self.dataset_name.upper()} SAMPLE: {exc}")
                index = (index + 1) % max(1, len(self.path))


class MulticamImageConditionDataset(BaseImageConditionDataset):
    dataset_name = "multicam"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._camera_json_cache: Dict[str, Dict] = {}

    @staticmethod
    def _frame_order_key(key: str) -> int:
        match = re.search(r"(\d+)", key or "")
        return int(match.group(1)) if match else 0

    def _load_camera_json(self, camera_path: Path) -> Dict:
        camera_path = camera_path.as_posix()
        if camera_path not in self._camera_json_cache:
            with open(camera_path, "r") as file:
                self._camera_json_cache[camera_path] = json.load(file)
        return self._camera_json_cache[camera_path]

    @staticmethod
    def _reorder_c2w_axes(c2w: np.ndarray) -> np.ndarray:
        adjusted = c2w.transpose(1, 0)
        adjusted = adjusted[:, [1, 2, 0, 3]]
        adjusted[:3, 1] *= -1.0
        return adjusted

    def _load_camera_sequence(self, sample_path: str, num_frames: int) -> np.ndarray:
        scene_dir = Path(sample_path).parent.parent
        camera_path = scene_dir / "cameras" / "camera_extrinsics.json"
        cam_data = self._load_camera_json(camera_path)

        cam_match = re.search(r"cam(\d+)", sample_path)
        if not cam_match:
            raise ValueError(f"未能解析摄像机编号: {sample_path}")
        cam_idx = int(cam_match.group(1))
        cam_key = f"cam{cam_idx:02d}"

        ordered_keys = sorted(cam_data.keys(), key=self._frame_order_key)
        frames = []
        for key in ordered_keys:
            frame_entry = cam_data[key]
            if cam_key not in frame_entry:
                raise KeyError(f"{camera_path} 中缺少 {cam_key} 数据")
            c2w = self.parse_matrix(frame_entry[cam_key])
            frames.append(self._reorder_c2w_axes(np.asarray(c2w, dtype=np.float32)))
        if not frames:
            raise ValueError(f"{camera_path} 不包含任何帧数据。")
        indices = _select_temporal_indices(len(frames), num_frames)
        selected = [frames[idx] for idx in indices]
        return np.stack(selected, axis=0)

    def _resolve_ids(self, sample_path: str) -> Tuple[str, str, str]:
        scene_match = re.search(r"scene(\d+)", sample_path.lower())
        scene_id = scene_match.group(1) if scene_match else "unknown"
        cam_match = re.search(r"cam(\d+)", sample_path)
        cam_idx = int(cam_match.group(1)) if cam_match else 0
        return scene_id, f"cam{cam_idx:02d}_img", f"cam{cam_idx:02d}"


class RelEstate10kImageConditionDataset(BaseImageConditionDataset):
    dataset_name = "rel10k"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rel10k_intr_cache: Dict[str, np.ndarray] = {}
        self._rel10k_extr_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _sequence_dir(sample_path: str) -> Path:
        return Path(sample_path).parent

    def _load_camera_sequence(self, sample_path: str, num_frames: int) -> np.ndarray:
        seq_dir = self._sequence_dir(sample_path)
        extr_path = seq_dir / "extrinsics.npz"
        extr_key = extr_path.as_posix()
        if extr_key not in self._rel10k_extr_cache:
            if not extr_path.exists():
                raise FileNotFoundError(f"缺少 extrinsics 文件: {extr_path}")
            self._rel10k_extr_cache[extr_key] = np.load(extr_path)["c2w"].astype(np.float32)
        c2w_all = self._rel10k_extr_cache[extr_key]
        indices = _select_temporal_indices(c2w_all.shape[0], num_frames)
        return c2w_all[indices]

    def _load_intrinsics_for_sample(self, sample_path: str, num_frames: int) -> torch.Tensor:
        seq_dir = self._sequence_dir(sample_path)
        intr_path = seq_dir / "intrinsics.npz"
        intr_key = intr_path.as_posix()
        if intr_key not in self._rel10k_intr_cache:
            if not intr_path.exists():
                raise FileNotFoundError(f"缺少 intrinsics 文件: {intr_path}")
            self._rel10k_intr_cache[intr_key] = np.load(intr_path)["K"].astype(np.float32)
        Ks = self._rel10k_intr_cache[intr_key]
        indices = _select_temporal_indices(Ks.shape[0], num_frames)
        return torch.from_numpy(Ks[indices]).to(torch.float32)

    def _resolve_ids(self, sample_path: str) -> Tuple[str, str, str]:
        scene_id = self._sequence_dir(sample_path).name
        return scene_id, "cam00_img", "cam00"


class MixedImageConditionDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that samples from multiple I2V datasets according to weights.
    """

    def __init__(
        self,
        datasets: List[BaseImageConditionDataset],
        weights: List[float],
        steps_per_epoch: int,
        seed: int,
    ):
        if not datasets:
            raise ValueError("MixedImageConditionDataset 需要至少一个子数据集。")
        filtered = [(ds, max(float(w), 0.0)) for ds, w in zip(datasets, weights)]
        filtered = [(ds, w) for ds, w in filtered if w > 0]
        if not filtered:
            filtered = [(datasets[0], 1.0)]
        self.datasets = [ds for ds, _ in filtered]
        raw_weights = np.array([w for _, w in filtered], dtype=np.float64)
        probs = raw_weights / raw_weights.sum()
        self._cumulative = probs.cumsum().tolist()
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.seed = seed
        self._subset_lengths = [max(len(ds), 1) for ds in self.datasets]

    def __len__(self):
        return self.steps_per_epoch

    def _choose_subset(self, index: int) -> int:
        if len(self.datasets) == 1:
            return 0
        torch.manual_seed(self.seed + index)
        rand_val = float(torch.rand(1).item())
        idx = bisect.bisect_left(self._cumulative, rand_val)
        if idx >= len(self.datasets):
            idx = len(self.datasets) - 1
        return idx

    def __getitem__(self, index):
        subset_idx = self._choose_subset(index)
        subset = self.datasets[subset_idx]
        subset_len = self._subset_lengths[subset_idx]
        local_idx = index % subset_len
        return subset[local_idx]


class InterleavedDataset(torch.utils.data.Dataset):
    """
    Deterministic round-robin sampler to balance validation samples across datasets.
    """

    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        if len(datasets) < 2:
            raise ValueError("InterleavedDataset 需要至少两个子数据集。")
        self.datasets = datasets
        self._subset_lengths = [max(len(ds), 1) for ds in datasets]
        self._num_subsets = len(datasets)
        self._per_subset_cycle = max(self._subset_lengths)
        self._length = self._num_subsets * self._per_subset_cycle

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        subset_idx = index % self._num_subsets
        cycle_idx = index // self._num_subsets
        subset = self.datasets[subset_idx]
        subset_len = self._subset_lengths[subset_idx]
        local_idx = cycle_idx % subset_len
        return subset[local_idx]


def _collect_paths_from_metadata(
    metadata_path: str,
    tensor_suffixes: Tuple[str, ...],
    dataset_root: Optional[str] = None,
) -> List[str]:
    metadata = pd.read_csv(metadata_path)
    if "video_absolute_path" not in metadata.columns:
        raise ValueError(f"Required column 'video_absolute_path' not found in {metadata_path}")
    all_paths = []
    missing = []
    for p in metadata["video_absolute_path"]:
        tp = resolve_tensor_path(p, dataset_root=dataset_root, tensor_suffixes=tensor_suffixes)
        if tp and os.path.exists(tp):
            all_paths.append(tp)
        else:
            if len(missing) < 20:
                print(f"Warning: missing tensor file: {tp or p}")
            missing.append(tp or p)
    if missing:
        print(f"Warning: {len(missing)} tensor files listed in {metadata_path} were not found on disk.")
    return sorted(all_paths)


def _resolve_rel10k_split_root(root: str) -> Path:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {root_path}")
    candidate = root_path / "train"
    if root_path.is_dir() and candidate.is_dir() and not (root_path / "video.mp4").exists():
        return candidate
    return root_path


def _collect_rel10k_paths_from_fs(root: str, tensor_suffixes: Tuple[str, ...]) -> List[str]:
    split_root = _resolve_rel10k_split_root(root)
    if (split_root / "video.mp4").exists():
        seq_dirs = [split_root]
    else:
        seq_dirs = [d for d in sorted(split_root.iterdir()) if d.is_dir()]
    tensor_files: List[str] = []
    for seq_dir in seq_dirs:
        for suffix in tensor_suffixes:
            candidate = seq_dir / ("video.mp4" + suffix)
            if candidate.exists():
                tensor_files.append(candidate.as_posix())
                break
    if not tensor_files:
        raise ValueError(f"在 {split_root} 下未找到任何 {tensor_suffixes} 文件。")
    return sorted(tensor_files)


def _gather_paths_for_spec(spec: DatasetSpec, tensor_suffixes: Tuple[str, ...]) -> List[str]:
    if spec.name == "multicam":
        if not spec.metadata_path:
            raise ValueError("MultiCam 数据集需要提供对应的 metadata CSV。")
        return _collect_paths_from_metadata(spec.metadata_path, tensor_suffixes, dataset_root=spec.root)
    if spec.name == "rel10k":
        if spec.metadata_path:
            return _collect_paths_from_metadata(spec.metadata_path, tensor_suffixes, dataset_root=spec.root)
        return _collect_rel10k_paths_from_fs(spec.root, tensor_suffixes)
    raise ValueError(f"Unsupported dataset type '{spec.name}'.")


def _split_train_val(paths: List[str], val_size: int) -> Tuple[List[str], List[str]]:
    if not paths:
        return [], []
    sorted_paths = sorted(paths)
    max_val = min(val_size, len(sorted_paths))
    if len(sorted_paths) - max_val <= 0:
        max_val = max(0, len(sorted_paths) - 1)
    val_paths = sorted_paths[-max_val:] if max_val > 0 else []
    train_paths = sorted_paths[:-max_val] if max_val > 0 else sorted_paths
    if not train_paths:
        train_paths = sorted_paths
        val_paths = []
    return train_paths, val_paths



def create_datasets(
    dataset_specs: List[DatasetSpec],
    val_size: int,
    steps_per_epoch: int,
    seed: int = 42,
    image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
    pipeline_type: str = "v2v",
    tensor_suffix: Optional[str] = None,
):
    """
    Create training and validation datasets for ReCamMaster.

    Args:
        dataset_specs: 数据集配置列表（支持多数据源混采）。
        val_size: 每个数据集用于验证的样本上限。
        steps_per_epoch: 训练阶段的步数。
        seed: 全局随机种子。
        pipeline_type: 'v2v' 或 'i2v'。
        tensor_suffix: 可选，覆盖默认的 latent 后缀（例如 '.tensors.pth' 或 '.wan22.tensors.pth'）。
    """
    if not dataset_specs:
        raise ValueError("create_datasets 需要至少一个 dataset_spec。")

    if tensor_suffix:
        tensor_suffixes = (tensor_suffix,)
    else:
        tensor_suffixes = (".wan22.tensors.pth",) if pipeline_type == "i2v" else (".tensors.pth",)

    if pipeline_type == "v2v":
        if len(dataset_specs) != 1:
            raise ValueError("当前 v2v 训练仅支持单一数据集。")
        spec = dataset_specs[0]
        all_paths = _gather_paths_for_spec(spec, tensor_suffixes)
        train_paths, val_paths = _split_train_val(all_paths, val_size)
        print(f"[v2v] Dataset split -> train: {len(train_paths)}  val: {len(val_paths)}")
        train_dataset = TensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
            dataset_root=spec.root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        if not val_paths:
            val_paths = train_paths[: max(1, min(len(train_paths), val_size))]
        val_steps = max(len(val_paths), 1)
        val_dataset = TensorDataset(
            steps_per_epoch=val_steps,
            paths=val_paths,
            fixed_length=val_steps,
            seed=seed + 10000,
            dataset_root=spec.root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        return train_dataset, val_dataset

    # i2v pipeline with optional multi-dataset mixing
    dataset_registry = {
        "multicam": MulticamImageConditionDataset,
        "rel10k": RelEstate10kImageConditionDataset,
    }
    train_subsets: List[BaseImageConditionDataset] = []
    val_subsets: List[BaseImageConditionDataset] = []
    weights: List[float] = []
    fallback_info: Optional[Tuple[type, List[str], int, DatasetSpec]] = None

    for idx, spec in enumerate(dataset_specs):
        if spec.weight <= 0:
            print(f"[info] 数据集 {spec.name} 的采样权重为 0，跳过训练。")
            continue
        dataset_cls = dataset_registry.get(spec.name)
        if dataset_cls is None:
            raise ValueError(f"未实现的数据集 '{spec.name}' 的 i2v 支持。")
        all_paths = _gather_paths_for_spec(spec, tensor_suffixes)
        if not all_paths:
            print(f"[warn] 数据集 {spec.name} 没有可用样本，跳过。")
            continue
        train_paths, val_paths = _split_train_val(all_paths, val_size)
        if not train_paths:
            print(f"[warn] 数据集 {spec.name} 训练集为空，跳过。")
            continue
        subset_seed = seed + idx * 997
        train_subset = dataset_cls(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=subset_seed,
            dataset_root=spec.root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        train_subsets.append(train_subset)
        weights.append(max(spec.weight, 0.0))
        fallback_info = fallback_info or (dataset_cls, train_paths, subset_seed, spec)

        if val_paths:
            val_subset = dataset_cls(
                steps_per_epoch=len(val_paths),
                paths=val_paths,
                fixed_length=len(val_paths),
                seed=subset_seed + 10000,
                dataset_root=spec.root,
                image_size=image_size,
                sensor_size_mm=sensor_size_mm,
            )
            val_subsets.append(val_subset)

    if not train_subsets:
        raise ValueError("未能构建任何训练数据集，请检查 dataset_type / 权重配置。")

    if len(train_subsets) == 1 and weights[0] > 0:
        train_dataset = train_subsets[0]
    else:
        train_dataset = MixedImageConditionDataset(train_subsets, weights, steps_per_epoch, seed)

    valid_val_subsets = [ds for ds in val_subsets if len(ds) > 0]
    if not valid_val_subsets:
        if fallback_info is None:
            fallback_dataset = train_subsets[0]
            val_dataset = fallback_dataset
        else:
            dataset_cls, fallback_paths, subset_seed, spec = fallback_info
            fallback_count = min(len(fallback_paths), max(1, val_size))
            fallback_subset = dataset_cls(
                steps_per_epoch=fallback_count,
                paths=fallback_paths[:fallback_count],
                fixed_length=fallback_count,
                seed=subset_seed + 20000,
                dataset_root=spec.root,
                image_size=image_size,
                sensor_size_mm=sensor_size_mm,
            )
            valid_val_subsets = [fallback_subset]

    if len(valid_val_subsets) == 1:
        val_dataset = valid_val_subsets[0]
    else:
        val_dataset = InterleavedDataset(valid_val_subsets)

    return train_dataset, val_dataset
