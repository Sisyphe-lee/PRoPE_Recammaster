import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video, VideoData
import os, imageio, argparse
from datetime import datetime
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

class TextVideoCameraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, args, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.base_path = base_path
        raw_paths = metadata["file_name"].to_list()
        self.path = [self._resolve_video_path(p) for p in raw_paths]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.args = args
        self.cam_type = self.args.cam_type
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def _resolve_video_path(self, file_name: str) -> str:
        # If absolute and exists, return
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
        # Candidate 1: relative as-is from CWD
        if os.path.exists(file_name):
            return file_name
        # Candidate 2: base_path + file_name
        cand2 = os.path.join(self.base_path, file_name)
        if os.path.exists(cand2):
            return cand2
        # Candidate 3: base_path/videos + file_name
        cand3 = os.path.join(self.base_path, "videos", file_name)
        if os.path.exists(cand3):
            return cand3
        # Candidate 4: if file_name already contains "videos/...", try base_path + that
        if not os.path.isabs(file_name) and ("videos" in file_name):
            cand4 = os.path.join(self.base_path, file_name)
            if os.path.exists(cand4):
                return cand4
        # Fallback: return base_path/videos/file_name (may error later but is most likely)
        return cand3
        
    
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames


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

        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def _convert_c2w_convention(self, c2w: np.ndarray) -> np.ndarray:
        """Match the convention used for tgt camera JSON: reorder axes, flip Y, scale translation.
        Input: 4x4 c2w
        Output: 4x4 c2w after convention conversion
        """
        c2w = c2w.copy()
        c2w = c2w[:, [1, 2, 0, 3]]  # reorder axes
        c2w[:3, 1] *= -1.            # flip Y
        c2w[:3, 3] /= 100.           # scale translation
        return c2w

    def _to_homogeneous(self, mats: np.ndarray) -> np.ndarray:
        """Ensure poses are T x 4 x 4 homogeneous matrices."""
        if mats.ndim != 3:
            raise ValueError(f"Pose array must be 3D, got shape {mats.shape}")
        T, h, w = mats.shape
        if h == 4 and w == 4:
            return mats
        if h == 3 and w == 4:
            last = np.tile(np.array([[0, 0, 0, 1]], dtype=mats.dtype), (T, 1, 1))
            return np.concatenate([mats, last], axis=1)
        raise ValueError(f"Unsupported pose shape {mats.shape}, expected (T,4,4) or (T,3,4)")

    def _load_source_c2ws_from_npz(self, npz_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load source camera c2w sequence and frame inds from an .npz file.
        Returns: (c2w: (T,4,4) float32, inds: (T,) int64)
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Source pose file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        if 'data' not in data or 'inds' not in data:
            raise ValueError(f"Invalid pose file {npz_path}, expected keys 'data' and 'inds', got {data.files}")
        mats = data['data'].astype(np.float32)
        inds = data['inds']
        mats = self._to_homogeneous(mats)  # (T,4,4)
        return mats, inds

    def _nearest_index(self, inds: np.ndarray, target: int) -> int:
        """Find index i such that inds[i] is closest to target."""
        j = (np.abs(inds - target)).argmin()
        return int(j)

    def _invert_se3(self, matrix: np.ndarray) -> np.ndarray:
        """Invert a 4x4 SE(3) matrix in a numerically stable way."""
        matrix = matrix.astype(np.float64)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        rotation_inv = rotation.T
        translation_inv = -rotation_inv @ translation
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation_inv
        result[:3, 3] = translation_inv
        return result.astype(np.float32)

    def _compute_relative_c2w(self, ref_cam: Camera, cams: list[Camera]) -> np.ndarray:
        """Compute cam_i<-ref (c2w) for each camera in cams."""
        rel_c2w_list = []
        for cam in cams:
            relative_pose = self.get_relative_pose([ref_cam, cam])
            cam_ref_from_cam = relative_pose[1]
            cam_from_ref = self._invert_se3(cam_ref_from_cam)
            rel_c2w_list.append(cam_from_ref)
        return np.stack(rel_c2w_list, axis=0)

    def _normalize_pairwise_distance(self, rel_c2w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize a trajectory so that the maximum pairwise translation distance
        between any two cameras is <= 1.
        """
        translations = rel_c2w[:, :3, 3]
        if translations.shape[0] <= 1:
            return rel_c2w
        diffs = translations[None, :, :] - translations[:, None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        max_dist = float(np.max(dists))
        if not np.isfinite(max_dist) or max_dist < eps:
            return rel_c2w
        scaled = rel_c2w.copy()
        scaled[:, :3, 3] /= max_dist
        return scaled

    def _normalize_joint_translation(self, cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
        """
        Jointly normalize condition and target trajectories so that the maximum
        translation norm across both trajectories is <= 1.
        """
        combined = np.concatenate([tgt_rel_c2w, cond_rel_c2w], axis=0)
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

    def _c2w_to_w2c(self, rel_c2w: np.ndarray) -> np.ndarray:
        """Invert a trajectory of c2w matrices to w2c."""
        return np.stack([self._invert_se3(T) for T in rel_c2w], axis=0)

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        num_frames = video.shape[1]
        assert num_frames == 81
        data = {"text": text, "video": video, "path": path}

        cam_idx = list(range(num_frames))[::4]

        # Use parameterized camera extrinsics filename instead of hard-coding
        tgt_camera_path = os.path.join(
            self.args.dataset_path,
            "cameras",
            getattr(self.args, "camera_extrinsics_filename", "camera_extrinsics_ori.json")
        )
        with open(tgt_camera_path, 'r') as file:
            cam_data = json.load(file)

        videos_dir = os.path.dirname(path)
        src_fname = os.path.basename(path)
        src_stem, _ = os.path.splitext(src_fname)
        src_npz_path = os.path.join(videos_dir, f"{src_stem}.npz")
        src_c2ws, src_inds = self._load_source_c2ws_from_npz(src_npz_path)

        src_c2ws_sampled = []
        for t in cam_idx:
            matches = np.where(src_inds == t)[0]
            if len(matches) > 0:
                j = int(matches[0])
            else:
                j = self._nearest_index(src_inds, t)
            c2w = src_c2ws[j]
            c2w = self._convert_c2w_convention(c2w)
            src_c2ws_sampled.append(c2w)
        src_cam_params = [Camera(c2w) for c2w in src_c2ws_sampled]
        cond_ref_cam = src_cam_params[0]
        cond_rel_c2w = self._compute_relative_c2w(cond_ref_cam, src_cam_params)
        cond_rel_c2w = self._normalize_pairwise_distance(cond_rel_c2w)

        camera_list = []
        for cam_type in range(1, 11):
            traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cam_type):02d}"]) for idx in cam_idx]
            traj = np.stack(traj).transpose(0, 2, 1)
            c2ws = []
            for c2w in traj:
                c2w = self._convert_c2w_convention(c2w)
                c2ws.append(c2w)
            tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
            tgt_rel_c2w = self._compute_relative_c2w(cond_ref_cam, tgt_cam_params)
            tgt_rel_c2w = self._normalize_pairwise_distance(tgt_rel_c2w)
            cond_joint, tgt_joint = self._normalize_joint_translation(cond_rel_c2w, tgt_rel_c2w)
            cond_rel_w2c = self._c2w_to_w2c(cond_joint)
            tgt_rel_w2c = self._c2w_to_w2c(tgt_joint)
            all_w2c = np.concatenate([tgt_rel_w2c, cond_rel_w2c], axis=0).astype(np.float32)
            pose_embedding = torch.from_numpy(all_w2c).to(torch.bfloat16)
            camera_list.append(pose_embedding)
        
        data['camera'] = camera_list
        return data
    

    def __len__(self):
        return len(self.path)

def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./example_test_data",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/ReCamMaster/checkpoints/step20000.ckpt",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cam_type",
        type=str,
        default=1,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--frame_downsample_to",
        type=int,
        default=5,
        help="Temporal latent downsample count (0 表示不降采样)"
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        default="recammaster",
        choices=["wan21", "recammaster"],
        help="Type of checkpoint to load: 'wan21' for original Wan2.1 model, 'recammaster' for ReCamMaster fine-tuned model"
    )
    parser.add_argument(
        "--enable_cam_layers",
        action="store_true",
        help="Enable camera encoder and projector layers injection (only for ReCamMaster)"
    )
    parser.add_argument(
        "--camera_extrinsics_filename",
        type=str,
        default="camera_extrinsics_ori.json",
        help="Filename of the target camera extrinsics JSON under {dataset_path}/cameras/"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    print(f"Loading checkpoint from: {args.ckpt_path}")
    print(f"Checkpoint type: {args.ckpt_type}")
    
    if args.ckpt_type == "wan21":
        # Support both safetensors and torch formats for Wan2.1 DiT checkpoints
        state_dict = None
        if str(args.ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.ckpt_path)
        else:
            # torch-based checkpoint
            raw = torch.load(args.ckpt_path, map_location="cpu")
            # Unwrap common containers
            if isinstance(raw, dict) and 'state_dict' in raw:
                raw = raw['state_dict']
            if isinstance(raw, dict) and 'module' in raw:
                raw = raw['module']
            # Strip common prefixes
            prefixes = ['model.', 'module.', 'pipe.dit.', 'dit.']
            state_dict = {}
            for k, v in raw.items():
                kk = k
                for p in prefixes:
                    if kk.startswith(p):
                        kk = kk[len(p):]
                        break
                # Filter out any camera-layer weights accidentally present
                if '.cam_encoder.' in kk or '.projector.' in kk:
                    continue
                state_dict[kk] = v
        print("Loading Wan2.1 DiT weights...")
        pipe.dit.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if "module" in state_dict:
            state_dict = state_dict["module"]
        prefixes_to_remove = ['model.', 'module.', 'pipe.dit.']
        has_prefix = any(any(key.startswith(p) for p in prefixes_to_remove) for key in state_dict.keys())
        if has_prefix:
            print("Prefix detected in checkpoint keys. Attempting to strip them.")
            new_state_dict = {}
            for k, v in state_dict.items():
                for p in prefixes_to_remove:
                    if k.startswith(p):
                        k = k[len(p):]
                        break
                new_state_dict[k] = v
            state_dict = new_state_dict
            print("Prefixes stripped. Using the new state_dict.")
        contains_cam_layers = any('.cam_encoder.' in k or '.projector.' in k for k in state_dict.keys())
        if contains_cam_layers:
            print("Checkpoint contains camera layers; registering modules before loading...")
            dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            for block in pipe.dit.blocks:
                if not hasattr(block, 'cam_encoder'):
                    block.cam_encoder = nn.Linear(12, dim)
                if not hasattr(block, 'projector'):
                    block.projector = nn.Linear(dim, dim, bias=True)
                block.enable_cam_layers = True
        pipe.dit.load_state_dict(state_dict, strict=True)
        if args.enable_cam_layers and not contains_cam_layers:
            print("Enabling camera layers (not present in checkpoint); registering with identity/zero init...")
            dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            for block in pipe.dit.blocks:
                block.cam_encoder = nn.Linear(12, dim)
                block.projector = nn.Linear(dim, dim)
                with torch.no_grad():
                    block.cam_encoder.weight.zero_()
                    block.cam_encoder.bias.zero_()
                    block.projector.weight.copy_(torch.eye(dim))
                    block.projector.bias.zero_()
                block.enable_cam_layers = True
    
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./result", timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = TextVideoCameraDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["text"]
        source_video = batch["video"]
        camera_list = batch["camera"]
        source_path = batch["path"][0]
        cam_fname = os.path.basename(source_path)  # original video filename

        for cam_type_id, target_camera in enumerate(camera_list, start=1):
            cam_output_dir = os.path.join(output_dir, f"cam_type{cam_type_id}")
            if not os.path.exists(cam_output_dir):
                os.makedirs(cam_output_dir)

            video = pipe(
                prompt=target_text,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                source_video=source_video,
                target_camera=target_camera,
                cfg_scale=args.cfg_scale,
                frame_downsample_to=args.frame_downsample_to,
                num_inference_steps=50,
                seed=0, tiled=True
            )
            filename = cam_fname
            save_video(video, os.path.join(cam_output_dir, filename), fps=30, quality=5)
