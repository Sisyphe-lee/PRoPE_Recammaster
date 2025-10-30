"""
Dataset classes for ReCamMaster training
"""

import os
import re
import torch
import numpy as np
import random
import json
import pandas as pd
from typing import Optional, Tuple, Dict, List


DEFAULT_IMAGE_WIDTH = 832.0
DEFAULT_IMAGE_HEIGHT = 480.0
DEFAULT_SENSOR_WIDTH_MM = 23.76
DEFAULT_SENSOR_HEIGHT_MM = 23.76
DEFAULT_FOCAL_MM = 18.0

DATASET_ID_PATTERN = re.compile(r"/train/([^/]+)/")
FOCAL_PATTERN = re.compile(r"f(?P<focal>\d+(?:\.\d+)?)", re.IGNORECASE)


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


class TensorDataset(torch.utils.data.Dataset):
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
        self.path = paths or []
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.fixed_length = fixed_length
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.sensor_size_mm = sensor_size_mm
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
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
        # Keep intrinsics as float32 for numerical stability
        return expanded.to(torch.float32)

    def __getitem__(self, index):
        # Return: 
        # data['latents']: torch.Size([16, 21*2, 60, 104])
        # data['camera']: torch.Size([21, 3, 4])
        # data['prompt_emb']['context'][0]: torch.Size([512, 4096])
        while True:
            try:
                data = {}
                # Use deterministic random selection based on seed and index
                torch.manual_seed(self.seed + index)
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
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
                data['prompt_emb'] = data_tgt['prompt_emb']
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


class ImageConditionTensorDataset(TensorDataset):
    """
    Dataset variant for Wan2.2 image-to-video fine-tuning.
    Uses the same metadata caching and sampling logic as TensorDataset,
    but prepares target latents with a single image condition taken from
    the first frame of the sequence.
    """

    def __getitem__(self, index):
        while True:
            try:
                torch.manual_seed(self.seed + index)
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path)
                sample_path = self.path[data_id]
                raw = torch.load(sample_path, weights_only=True, map_location="cpu")

                latents = raw["latents"]
                if isinstance(latents, (list, tuple)):
                    latents = latents[0]
                if latents.dim() == 5:
                    latents = latents.squeeze(0)

                data = {
                    "latents": latents,
                    "prompt_emb": raw.get("prompt_emb", {}),
                    "image_emb": raw.get("image_emb", {}),
                    "path": sample_path,
                }

                scene_match = re.search(r'scene(\d+)', sample_path.lower())
                scene_id = scene_match.group(1) if scene_match else 'unknown'

                cam_match = re.search(r'cam(\d+)', sample_path)
                cam_idx = int(cam_match.group(1)) if cam_match else 0

                base_path = sample_path.rsplit('/', 2)[0]
                camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(camera_path, 'r') as file:
                    cam_data = json.load(file)

                cam_idx_list = list(range(81))[::4]
                traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in cam_idx_list]
                traj = np.stack(traj).transpose(0, 2, 1)

                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.0
                    c2ws.append(c2w)

                cam_params = [Camera(cam_param) for cam_param in c2ws]
                ref_cam = cam_params[0]
                tgt_rel_c2w = compute_relative_c2w(cam_params, ref_cam, self.get_relative_pose)
                tgt_rel_c2w_norm, _, _ = normalize_translation(tgt_rel_c2w, tgt_rel_c2w)

                tgt_rel_w2c = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w_norm], axis=0)
                camera_tensor = torch.from_numpy(tgt_rel_w2c).to(torch.float32)

                data["camera"] = camera_tensor
                data["intrinsics"] = self._get_intrinsics_tensor(sample_path, repeat=camera_tensor.shape[0])
                data["scene_id"] = scene_id
                data["condition_cam_type"] = f"cam{cam_idx:02d}_img"
                data["target_cam_type"] = f"cam{cam_idx:02d}"

                return data
            except Exception as e:
                print(f"ERROR WHEN LOADING I2V SAMPLE: {e}")
                index = (index + 1) % len(self.path)





def create_datasets(
    metadata_path,
    val_size,
    steps_per_epoch,
    use_validation_dataset=False,
    num_val_scenes=3,
    cameras_per_scene=10,
    seed=42,
    dataset_root: Optional[str] = None,
    image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
    pipeline_type: str = "v2v",
):
    """
    Create training and validation datasets
    
    Args:
        metadata_path: Path to metadata CSV file
        val_size: Number of samples for simple validation (when use_validation_dataset=False)
        steps_per_epoch: Number of steps per epoch for training
        use_validation_dataset: Whether to use ValidationDataset (True) or simple split (False)
        num_val_scenes: Number of scenes for ValidationDataset
        cameras_per_scene: Number of cameras per scene for ValidationDataset
        seed: Random seed for ValidationDataset
        pipeline_type: Selects which latent tensors to load (v2v=16ch, i2v=48ch)
    
    Returns:
        train_dataset, val_dataset
    """
    tensor_suffixes = (".wan22.tensors.pth",) if pipeline_type == "i2v" else (".tensors.pth",)
    # Load metadata and get all tensor file paths
    metadata = pd.read_csv(metadata_path)
    if "video_absolute_path" not in metadata.columns:
        raise ValueError(f"Required column 'video_absolute_path' not found in {metadata_path}")
    if "dataset" not in metadata.columns:
        metadata["dataset"] = metadata["video_absolute_path"].apply(infer_dataset_id)
    
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
        print(f"Warning: {len(missing)} tensor files listed in metadata were not found on disk.")
    
    print(f"Total available tensor files: {len(all_paths)}")
    val_size = min(val_size, len(all_paths))
    val_paths = all_paths[:val_size]
    train_paths = all_paths[val_size:]
    print(f"Dataset split -> train: {len(train_paths)}  val: {len(val_paths)}  (val_size={val_size})")
    if pipeline_type == "i2v":
        val_dataset = ImageConditionTensorDataset(
            steps_per_epoch=val_size,
            paths=val_paths,
            fixed_length=val_size,
            seed=seed + 10000,
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        train_dataset = ImageConditionTensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
    else:
        train_dataset = TensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        val_dataset = TensorDataset(
            steps_per_epoch=val_size,
            paths=val_paths,
            fixed_length=val_size,
            seed=seed + 10000,  # Different seed for validation dataset
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        
    return train_dataset, val_dataset
