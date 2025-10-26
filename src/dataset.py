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
from typing import Optional, Tuple, Dict, Callable


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
    parts = re.split(r"[\\/]", path)
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


def resolve_tensor_path(video_path: str, dataset_root: Optional[str] = None) -> str:
    """
    Resolve absolute tensor path from a video path and optional dataset root.
    """
    candidate = video_path
    if not os.path.isabs(candidate):
        if dataset_root is not None:
            candidate = os.path.join(dataset_root, candidate.lstrip("/"))
        candidate = os.path.abspath(candidate)
    tensor_path = candidate + ".tensors.pth"
    if os.path.exists(tensor_path):
        return tensor_path
    return ""


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def invert_SE3_np(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix (numpy, float32 output).
    Args:
        T: [4,4] homogeneous transform.
    Returns:
        [4,4] inverse transform.
    """
    T = T.astype(np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rinv
    out[:3, 3] = tinv
    return out.astype(np.float32)


def compute_relative_c2w(
    cam_params: list,
    ref_cam: Camera,
    get_relative_pose_fn: Callable[[list], np.ndarray],
) -> np.ndarray:
    """Compute relative c2w poses (cam_i <- ref) for a list of cameras.
    We first compute cam_ref<-cam_i via get_relative_pose, then invert to get cam_i<-ref (c2w relative).
    Args:
        cam_params: list of Camera
        ref_cam: reference Camera (e.g., cond[0])
        get_relative_pose_fn: function that maps [ref_cam, cam_i] -> poses with index 1 being cam_ref<-cam_i
    Returns:
        Array [N,4,4] of relative c2w.
    """
    rel_c2w_list = []
    for cam_i in cam_params:
        relative_pose_matrix = get_relative_pose_fn([ref_cam, cam_i])
        cam_ref_from_cami = relative_pose_matrix[1]
        cami_from_ref = invert_SE3_np(cam_ref_from_cami)  # c2w relative
        rel_c2w_list.append(cami_from_ref)
    return np.stack(rel_c2w_list, axis=0)


def normalize_translations_by_max_norm(a_rel_c2w: np.ndarray, b_rel_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize the translation components of two pose sequences by their joint max L2 norm.
    Args:
        a_rel_c2w: [Na,4,4]
        b_rel_c2w: [Nb,4,4]
    Returns:
        (a_normed, b_normed, max_norm)
    """
    a = a_rel_c2w.astype(np.float32).copy()
    b = b_rel_c2w.astype(np.float32).copy()
    all_trans = np.concatenate([a[:, :3, 3], b[:, :3, 3]], axis=0)
    norms = np.linalg.norm(all_trans, axis=1)
    max_norm = float(np.max(norms)) if norms.size > 0 else 1.0
    if max_norm < 1e-8:
        max_norm = 1.0
    a[:, :3, 3] /= max_norm
    b[:, :3, 3] /= max_norm
    return a, b, max_norm


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
        return expanded.to(torch.bfloat16)

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
                # Build c2w trajectories for cond and tgt with axis/scale normalization
                multiview_c2ws = []
                cam_idx = list(range(81))[::4]
                for view_idx in [cond_idx, tgt_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 100
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                # Make Camera objects for relative pose utility
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]

                # Reference is cond[0] in world coordinates
                ref_cam = cond_cam_params[0]

                # 1) Compute relative c2w for both trajectories using shared helpers
                cond_rel_c2w = compute_relative_c2w(cond_cam_params, ref_cam, self.get_relative_pose)
                tgt_rel_c2w = compute_relative_c2w(tgt_cam_params, ref_cam, self.get_relative_pose)

                # 2) Normalize all translations across both trajectories by the max norm
                tgt_rel_c2w, cond_rel_c2w, _ = normalize_translations_by_max_norm(tgt_rel_c2w, tgt_rel_c2w)

                # 3) Invert to obtain relative w2c
                tgt_rel_w2c = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w], axis=0)
                cond_rel_w2c = np.stack([invert_SE3_np(T) for T in cond_rel_c2w], axis=0)

                # Concatenate tgt first then cond to align with latents
                all_w2c = np.concatenate([tgt_rel_w2c, tgt_rel_c2w], axis=0)
                camera_tensor = torch.from_numpy(all_w2c).to(torch.bfloat16)
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


class ValidationDataset(torch.utils.data.Dataset):
    """
    Validation dataset that randomly selects 3 scenes and creates all possible 
    cond-target camera combinations (10x10 = 100 per scene, 300 total)
    """
    def __init__(
        self,
        all_paths,
        num_val_scenes=3,
        cameras_per_scene=10,
        seed=42,
        image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
    ):
        """
        Args:
            all_paths: List of all tensor file paths
            num_val_scenes: Number of scenes to use for validation (default: 3)
            cameras_per_scene: Number of cameras per scene (default: 10)
            seed: Random seed for reproducible scene selection
        """
        self.cameras_per_scene = cameras_per_scene
        self.num_val_scenes = num_val_scenes
        self.image_size = image_size
        self.sensor_size_mm = sensor_size_mm
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}
        
        # Set random seed for reproducible validation set
        random.seed(seed)
        np.random.seed(seed)
        
        # Extract all unique scenes from paths
        scene_to_paths = {}
        for path in all_paths:
            match = re.search(r'scene(\d+)', path)
            if match:
                scene_id = int(match.group(1))
                if scene_id not in scene_to_paths:
                    scene_to_paths[scene_id] = []
                scene_to_paths[scene_id].append(path)
        
        # Filter scenes that have at least 10 cameras
        valid_scenes = {}
        for scene_id, paths in scene_to_paths.items():
            if len(paths) >= cameras_per_scene:
                valid_scenes[scene_id] = paths
        
        print(f"Found {len(valid_scenes)} scenes with at least {cameras_per_scene} cameras")
        
        # Randomly select validation scenes
        available_scenes = list(valid_scenes.keys())
        self.val_scenes = random.sample(available_scenes, min(num_val_scenes, len(available_scenes)))
        print(f"Selected validation scenes: {self.val_scenes}")
        
        # Create all cond-target combinations for validation
        self.val_combinations = []
        for scene_id in self.val_scenes:
            scene_paths = valid_scenes[scene_id]
            # Take first 10 cameras for this scene
            scene_cameras = scene_paths[:cameras_per_scene]
            
            # Create all possible cond-target pairs (10x10 = 100 combinations)
            for cond_path in scene_cameras:
                for tgt_path in scene_cameras:
                    if cond_path != tgt_path:  # Skip same camera pairs
                        self.val_combinations.append({
                            'cond_path': cond_path,
                            'tgt_path': tgt_path,
                            'scene_id': scene_id
                        })
        
        print(f"Created {len(self.val_combinations)} validation combinations from {len(self.val_scenes)} scenes")
        
        # Store remaining scenes for training
        self.train_scenes = [scene_id for scene_id in available_scenes if scene_id not in self.val_scenes]
        print(f"Remaining {len(self.train_scenes)} scenes available for training")
    
    def get_training_paths(self, all_paths):
        """Get all paths from training scenes"""
        train_paths = []
        for scene_id in self.train_scenes:
            # Find all paths for this scene
            scene_paths = [path for path in all_paths if f'scene{scene_id}' in path]
            train_paths.extend(scene_paths)
        return train_paths
    
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
        return expanded.to(torch.bfloat16)

    def __getitem__(self, index):
        """Get a validation sample"""
        combination = self.val_combinations[index]
        cond_path = combination['cond_path']
        tgt_path = combination['tgt_path']
        scene_id = combination['scene_id']
        
        try:
            data = {}
            
            # Load target and condition data
            data_tgt = torch.load(tgt_path, weights_only=True, map_location="cpu")
            data_cond = torch.load(cond_path, weights_only=True, map_location="cpu")
            
            # Combine latents
            data['latents'] = torch.cat((data_tgt['latents'], data_cond['latents']), dim=1)
            data['prompt_emb'] = data_tgt['prompt_emb']
            data['image_emb'] = {}
            
            # Extract camera information from paths
            cond_match = re.search(r'cam(\d+)', cond_path)
            tgt_match = re.search(r'cam(\d+)', tgt_path)
            cond_cam_idx = int(cond_match.group(1)) if cond_match else 1
            tgt_cam_idx = int(tgt_match.group(1)) if tgt_match else 2
            
            # Store metadata
            data['scene_id'] = str(scene_id)
            data['condition_cam_type'] = f"cam{cond_cam_idx:02d}"
            data['target_cam_type'] = f"cam{tgt_cam_idx:02d}"
            data['path'] = tgt_path
            
            # Load camera trajectories
            base_path = tgt_path.rsplit('/', 2)[0]
            tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
            
            with open(tgt_camera_path, 'r') as file:
                cam_data = json.load(file)
            
            # Build c2w trajectories with axis/scale normalization (cond first, then tgt)
            multiview_c2ws = []
            cam_idx = list(range(81))[::4]
            for view_idx in [cond_cam_idx, tgt_cam_idx]:
                traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                traj = np.stack(traj).transpose(0, 2, 1)
                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.
                    c2w[:3, 3] /= 100
                    c2ws.append(c2w)
                multiview_c2ws.append(c2ws)
            # Reference is cond[0]
            cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
            tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
            ref_cam = cond_cam_params[0]

            # 1) Relative c2w for both trajectories using shared helpers
            cond_rel_c2w = compute_relative_c2w(cond_cam_params, ref_cam, self.get_relative_pose)
            tgt_rel_c2w = compute_relative_c2w(tgt_cam_params, ref_cam, self.get_relative_pose)

            # 2) Normalize translations across both trajectories using max norm
            tgt_rel_c2w, cond_rel_c2w, _ = normalize_translations_by_max_norm(tgt_rel_c2w, tgt_rel_c2w)

            # 3) Invert to obtain relative w2c
            tgt_w2c_rel = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w], axis=0)
            cond_w2c_rel = np.stack([invert_SE3_np(T) for T in cond_rel_c2w], axis=0)

            # Concatenate tgt first then cond to align with latents order
            all_w2c = np.concatenate([tgt_w2c_rel, tgt_w2c_rel], axis=0)
            camera_tensor = torch.from_numpy(all_w2c).to(torch.bfloat16)
            data['camera'] = camera_tensor
            data['intrinsics'] = self._get_intrinsics_tensor(tgt_path, repeat=camera_tensor.shape[0])
            
            return data
            
        except Exception as e: 
            print(f"ERROR WHEN LOADING VALIDATION SAMPLE: {e}")
            # Return a deterministic sample if loading fails (use index % length for reproducibility)
            fallback_idx = index % len(self.val_combinations)
            return self.__getitem__(fallback_idx)
    
    def __len__(self):
        return len(self.val_combinations)


def create_datasets(
    metadata_path,
    val_size,
    steps_per_epoch,
    use_validation_dataset=True,
    num_val_scenes=3,
    cameras_per_scene=10,
    seed=42,
    dataset_root: Optional[str] = None,
    image_size: Tuple[float, float] = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    sensor_size_mm: Tuple[float, float] = (DEFAULT_SENSOR_WIDTH_MM, DEFAULT_SENSOR_HEIGHT_MM),
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
    
    Returns:
        train_dataset, val_dataset
    """
    # Load metadata and get all tensor file paths
    metadata = pd.read_csv(metadata_path)
    if "video_absolute_path" not in metadata.columns:
        raise ValueError(f"Required column 'video_absolute_path' not found in {metadata_path}")
    if "dataset" not in metadata.columns:
        metadata["dataset"] = metadata["video_absolute_path"].apply(infer_dataset_id)
    
    all_paths = []
    missing = []
    for p in metadata["video_absolute_path"]:
        tp = resolve_tensor_path(p, dataset_root=dataset_root)
        if tp and os.path.exists(tp):
            all_paths.append(tp)
        else:
            if len(missing) < 20:
                print(f"Warning: missing tensor file: {tp or p}")
            missing.append(tp or p)
    if missing:
        print(f"Warning: {len(missing)} tensor files listed in metadata were not found on disk.")
    
    print(f"Total available tensor files: {len(all_paths)}")
    
    if use_validation_dataset:
        # Use ValidationDataset for better validation coverage
        print("Using ValidationDataset")

        val_dataset = ValidationDataset(
            all_paths=all_paths,
            num_val_scenes=num_val_scenes,
            cameras_per_scene=cameras_per_scene,
            seed=seed,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        
        # Get training paths from remaining scenes
        train_paths = val_dataset.get_training_paths(all_paths)
        
        print(f"Dataset split -> train: {len(train_paths)}  val: {len(val_dataset)}")
        
        train_dataset = TensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
            dataset_root=dataset_root,
            image_size=image_size,
            sensor_size_mm=sensor_size_mm,
        )
        
        return train_dataset, val_dataset
    
    else:
        # Use simple split based on metadata order
        val_size = min(val_size, len(all_paths))
        val_paths = all_paths[:val_size]
        # train_paths = all_paths[val_size:]
        train_paths = all_paths[val_size:]
        print(f"Dataset split -> train: {len(train_paths)}  val: {len(val_paths)}  (val_size={val_size})")
        
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
