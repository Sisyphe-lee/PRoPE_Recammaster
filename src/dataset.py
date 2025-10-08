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


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, steps_per_epoch, paths=None, fixed_length=None, seed=42):
        self.path = paths
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        # Set random seed for this dataset
        random.seed(seed)
        np.random.seed(seed)
        self.fixed_length = fixed_length

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

                def invert_SE3_np(T):
                    """Invert a 4x4 SE(3) matrix."""
                    T = T.astype(np.float64)
                    R = T[:3, :3]
                    t = T[:3, 3]
                    Rinv = R.T
                    tinv = -Rinv @ t
                    out = np.eye(4, dtype=np.float64)
                    out[:3, :3] = Rinv
                    out[:3, 3] = tinv
                    return out.astype(np.float32)

                def compute_relative_c2w(cam_params):
                    """
                    Compute relative c2w poses (cam_i <- ref) for a list of cameras
                    using ref_cam = cond[0]. We first compute cam_ref<-cam_i via
                    get_relative_pose, then invert to get cam_i<-ref (c2w relative).
                    Returns an array of shape [N, 4, 4].
                    """
                    rel_c2w_list = []
                    for i in range(len(cam_params)):
                        # get_relative_pose returns [I, cam_ref<-cam_i]
                        relative_pose_matrix = self.get_relative_pose([ref_cam, cam_params[i]])
                        cam_ref_from_cami = relative_pose_matrix[1]
                        cami_from_ref = invert_SE3_np(cam_ref_from_cami)  # c2w relative
                        rel_c2w_list.append(cami_from_ref)
                    return np.stack(rel_c2w_list, axis=0)

                # 1) Compute relative c2w for both trajectories
                cond_rel_c2w = compute_relative_c2w(cond_cam_params)
                tgt_rel_c2w = compute_relative_c2w(tgt_cam_params)

                # 2) Normalize all translations across both trajectories by the max norm
                all_rel_c2w = np.concatenate([tgt_rel_c2w, cond_rel_c2w], axis=0)
                translations = all_rel_c2w[:, :3, 3]
                norms = np.linalg.norm(translations, axis=1)
                max_norm = np.max(norms) if norms.size > 0 else 1.0
                if max_norm < 1e-8:
                    max_norm = 1.0
                # Apply normalization back to each trajectory
                tgt_rel_c2w[:, :3, 3] = tgt_rel_c2w[:, :3, 3] / max_norm
                cond_rel_c2w[:, :3, 3] = cond_rel_c2w[:, :3, 3] / max_norm

                # 3) Invert to obtain relative w2c
                tgt_rel_w2c = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w], axis=0)
                cond_rel_w2c = np.stack([invert_SE3_np(T) for T in cond_rel_c2w], axis=0)

                # Concatenate tgt first then cond to align with latents
                all_w2c = np.concatenate([tgt_rel_w2c, cond_rel_w2c], axis=0)
                data['camera'] = torch.from_numpy(all_w2c).to(torch.bfloat16)
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
    def __init__(self, all_paths, num_val_scenes=3, cameras_per_scene=10, seed=42):
        """
        Args:
            all_paths: List of all tensor file paths
            num_val_scenes: Number of scenes to use for validation (default: 3)
            cameras_per_scene: Number of cameras per scene (default: 10)
            seed: Random seed for reproducible scene selection
        """
        self.cameras_per_scene = cameras_per_scene
        self.num_val_scenes = num_val_scenes
        
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

            # Stable SE(3) inverse
            def invert_SE3_np(T):
                T = T.astype(np.float64)
                R = T[:3, :3]
                t = T[:3, 3]
                Rinv = R.T
                tinv = -Rinv @ t
                out = np.eye(4, dtype=np.float64)
                out[:3, :3] = Rinv
                out[:3, 3] = tinv
                return out.astype(np.float32)

            # Compute relative c2w: cam_i <- ref via get_relative_pose then inverse
            def compute_relative_c2w(cam_params):
                rel_c2w_list = []
                for i in range(len(cam_params)):
                    relative_pose_matrix = self.get_relative_pose([ref_cam, cam_params[i]])
                    cam_ref_from_cami = relative_pose_matrix[1]
                    cami_from_ref = invert_SE3_np(cam_ref_from_cami)  # c2w relative
                    rel_c2w_list.append(cami_from_ref)
                return np.stack(rel_c2w_list, axis=0)

            # 1) Relative c2w for both trajectories
            cond_rel_c2w = compute_relative_c2w(cond_cam_params)
            tgt_rel_c2w = compute_relative_c2w(tgt_cam_params)

            # 2) Normalize translations across both trajectories using max norm
            all_rel_c2w = np.concatenate([tgt_rel_c2w, cond_rel_c2w], axis=0)
            translations = all_rel_c2w[:, :3, 3]
            norms = np.linalg.norm(translations, axis=1)
            max_norm = np.max(norms) if norms.size > 0 else 1.0
            if max_norm < 1e-8:
                max_norm = 1.0
            tgt_rel_c2w[:, :3, 3] = tgt_rel_c2w[:, :3, 3] / max_norm
            cond_rel_c2w[:, :3, 3] = cond_rel_c2w[:, :3, 3] / max_norm

            # 3) Invert to obtain relative w2c
            tgt_w2c_rel = np.stack([invert_SE3_np(T) for T in tgt_rel_c2w], axis=0)
            cond_w2c_rel = np.stack([invert_SE3_np(T) for T in cond_rel_c2w], axis=0)

            # Concatenate tgt first then cond to align with latents order
            all_w2c = np.concatenate([tgt_w2c_rel, cond_w2c_rel], axis=0)
            data['camera'] = torch.from_numpy(all_w2c).to(torch.bfloat16)
            
            return data
            
        except Exception as e: 
            print(f"ERROR WHEN LOADING VALIDATION SAMPLE: {e}")
            # Return a deterministic sample if loading fails (use index % length for reproducibility)
            fallback_idx = index % len(self.val_combinations)
            return self.__getitem__(fallback_idx)
    
    def __len__(self):
        return len(self.val_combinations)


def create_datasets(metadata_path, val_size, steps_per_epoch, use_validation_dataset=True, num_val_scenes=3, cameras_per_scene=10, seed=42):
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
    
    all_paths = []
    for p in metadata["video_absolute_path"]:
        tp = p + ".tensors.pth"
        if os.path.exists(tp):
            all_paths.append(tp)
        else:
            print(f"Warning: missing tensor file: {tp}")
    
    print(f"Total available tensor files: {len(all_paths)}")
    
    if use_validation_dataset:
        # Use ValidationDataset for better validation coverage
        print("Using ValidationDataset")

        val_dataset = ValidationDataset(
            all_paths=all_paths,
            num_val_scenes=num_val_scenes,
            cameras_per_scene=cameras_per_scene,
            seed=seed
        )
        
        # Get training paths from remaining scenes
        train_paths = val_dataset.get_training_paths(all_paths)
        
        print(f"Dataset split -> train: {len(train_paths)}  val: {len(val_dataset)}")
        
        train_dataset = TensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
        )
        
        return train_dataset, val_dataset
    
    else:
        # Use simple split based on metadata order
        val_size = min(val_size, len(all_paths))
        val_paths = all_paths[:val_size]
        # train_paths = all_paths[val_size:]
        train_paths = all_paths
        print(f"Dataset split -> train: {len(train_paths)}  val: {len(val_paths)}  (val_size={val_size})")
        
        train_dataset = TensorDataset(
            steps_per_epoch=steps_per_epoch,
            paths=train_paths,
            fixed_length=None,
            seed=seed,
        )
        val_dataset = TensorDataset(
            steps_per_epoch=val_size,
            paths=val_paths,
            fixed_length=val_size,
            seed=seed + 10000,  # Different seed for validation dataset
        )
        
        return train_dataset, val_dataset
