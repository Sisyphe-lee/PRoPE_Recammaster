import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video, VideoData
import torch, os, imageio, argparse
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
        self.path = [os.path.join(file_name) for file_name in metadata["file_name"]]
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
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
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


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        num_frames = video.shape[1]
        assert num_frames == 81
        data = {"text": text, "video": video, "path": path}

        # load camera: derive target camera json path and condition cam id from video path
        # Example path: /.../scene3294/videos/cam10.mp4
        # -> cond_cam_type_id = 10
        # -> tgt_camera_path = /.../scene3294/cameras/camera_extrinsics.json
        videos_dir = os.path.dirname(path)
        scene_dir = os.path.dirname(videos_dir)
        tgt_camera_path = os.path.join(scene_dir, "cameras", "camera_extrinsics.json")
        with open(tgt_camera_path, 'r') as file:
            cam_data = json.load(file)

        cam_idx = list(range(num_frames))[::4]

        # 1) Use the cam id parsed from file name as the condition video pose
        cam_fname = os.path.basename(path)  # e.g., cam10.mp4
        cam_stem = os.path.splitext(cam_fname)[0]  # cam10
        cond_cam_type_id = int(cam_stem[3:]) if cam_stem.startswith("cam") and cam_stem[3:].isdigit() else 6
        cond_traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cond_cam_type_id):02d}"]) for idx in cam_idx]
        cond_traj = np.stack(cond_traj).transpose(0, 2, 1)
        cond_c2ws = []
        for c2w in cond_traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            cond_c2ws.append(c2w)
        cond_cam_params = [Camera(cam_param) for cam_param in cond_c2ws]
        ref_cam = cond_cam_params[0]
        # 2) Compute condition poses relative to its own first frame
        cond_relative_poses = []
        for i in range(len(cond_cam_params)):
            cond_relative_pose = self.get_relative_pose([ref_cam, cond_cam_params[i]])
            cond_relative_poses.append(torch.as_tensor(cond_relative_pose)[:,:3,:][1])
        cond_pose_embedding = torch.stack(cond_relative_poses, dim=0)  # 21x3x4
        cond_pose_embedding = rearrange(cond_pose_embedding, 'b c d -> b (c d)')  # 21x12

        # 3) For each target camera, compute relative poses using condition's first frame as reference
        camera_list = []
        for cam_type_id in range(1, 11):
            traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cam_type_id):02d}"]) for idx in cam_idx]
            traj = np.stack(traj).transpose(0, 2, 1)
            c2ws = []
            for c2w in traj:
                c2w = c2w[:, [1, 2, 0, 3]]
                c2w[:3, 1] *= -1.
                c2w[:3, 3] /= 100
                c2ws.append(c2w)
            tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]

            tgt_relative_poses = []
            for i in range(len(tgt_cam_params)):
                # Use condition's first frame as the reference instead of target's first frame
                relative_pose = self.get_relative_pose([ref_cam, tgt_cam_params[i]])
                tgt_relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
            tgt_pose_embedding = torch.stack(tgt_relative_poses, dim=0)  # 21x3x4
            tgt_pose_embedding = rearrange(tgt_pose_embedding, 'b c d -> b (c d)')  # 21x12

            # 4) Concatenate condition pose and target pose along the feature dimension
            combined_embedding = torch.cat([cond_pose_embedding, tgt_pose_embedding], dim=0)  # 42x12
            camera_list.append(combined_embedding.to(torch.bfloat16))

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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # 1. Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. Initialize additional modules introduced in ReCamMaster
    dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    # 3. Load ReCamMaster checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    # Check if any keys have a common prefix
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Common prefixes to check for and remove
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

    # 3.1 Load ReCamMaster checkpoint
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    # Create timestamped output directory under ./result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./result", timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. Prepare test data (source video, target camera, target trajectory)
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

    # 5. Inference
    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["text"]
        source_video = batch["video"]
        camera_list = batch["camera"]  # list of 10 pose embeddings (batched tensors)
        # Derive naming components from dataset path
        source_path = batch["path"][0]
        videos_dir = os.path.dirname(source_path)
        scene_dir = os.path.dirname(videos_dir)
        scene_name = os.path.basename(scene_dir)  # e.g., scene3294
        cam_fname = os.path.basename(source_path)  # e.g., cam10.mp4
        cam_stem = os.path.splitext(cam_fname)[0]  # cam10
        cond_cam_type_id = int(cam_stem[3:]) if cam_stem.startswith("cam") and cam_stem[3:].isdigit() else 6

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
                num_inference_steps=50,
                seed=0, tiled=True
            )
            filename = f"{scene_name}_cam{cond_cam_type_id}_camtype{cam_type_id}.mp4"
            save_video(video, os.path.join(cam_output_dir, filename), fps=30, quality=5)