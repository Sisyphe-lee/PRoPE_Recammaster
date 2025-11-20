#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video


def setup_distributed_environment() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        if distributed and local_rank >= available_devices:
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} 超过可用GPU数量 {available_devices}，"
                "请检查 --nproc_per_node 或 CUDA_VISIBLE_DEVICES 设置"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return distributed, world_size, rank, local_rank, device


def broadcast_output_directory(
    base_dir: Path,
    distributed: bool,
    rank: int,
    resume_timestamp: str | None = None,
) -> Path:
    if resume_timestamp is not None:
        if distributed:
            payload = [resume_timestamp if rank == 0 else None, None]
            if rank == 0:
                resume_dir = base_dir / resume_timestamp
                payload[1] = resume_dir.exists()
            dist.broadcast_object_list(payload, src=0)
            timestamp = payload[0]
            exists = bool(payload[1])
        else:
            timestamp = resume_timestamp
            exists = (base_dir / timestamp).exists()

        out_dir = base_dir / timestamp
        if not exists:
            raise FileNotFoundError(f"指定的时间戳目录不存在: {out_dir}")
        if distributed:
            dist.barrier()
        return out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if distributed:
        payload = [timestamp if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        timestamp = payload[0]
    out_dir = base_dir / timestamp
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    return out_dir


def cleanup_distributed_environment(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation_inv.astype(np.float32)
    result[:3, 3] = translation_inv.astype(np.float32)
    return result


def _compute_relative_c2w(ref_w2c: np.ndarray, c2ws: np.ndarray) -> np.ndarray:
    return np.stack([ref_w2c @ c2w for c2w in c2ws], axis=0)


def _center_c2w(c2ws: np.ndarray) -> np.ndarray:
    if c2ws.shape[0] == 0:
        return c2ws
    centered = c2ws.copy()
    origin = centered[0, :3, 3].copy()
    if np.allclose(origin, 0):
        return centered
    centered[:, :3, 3] -= origin
    return centered


def _normalize_joint_translation(
    cond_rel_c2w: np.ndarray, tgt_rel_c2w: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([cond_rel_c2w, tgt_rel_c2w], axis=0)
    translations = combined[:, :3, 3]
    norms = np.linalg.norm(translations, axis=1)
    max_norm = float(np.max(norms)) if norms.size > 0 else 0.0
    if not np.isfinite(max_norm) or max_norm < eps:
        return cond_rel_c2w.copy(), tgt_rel_c2w.copy()
    cond_scaled = cond_rel_c2w.copy()
    tgt_scaled = tgt_rel_c2w.copy()
    cond_scaled[:, :3, 3] /= max_norm
    tgt_scaled[:, :3, 3] /= max_norm
    return cond_scaled, tgt_scaled


def _c2w_to_w2c(rel_c2w: np.ndarray) -> np.ndarray:
    return np.stack([_invert_se3(mat) for mat in rel_c2w], axis=0)


def _nearest_index(inds: np.ndarray, target: int) -> int:
    return int(np.abs(inds - target).argmin())


def _convert_c2w_convention(c2w: np.ndarray) -> np.ndarray:
    """Align camera convention with training/inference pipeline."""
    converted = c2w.copy()
    converted = converted[:, [1, 2, 0, 3]]
    converted[:3, 1] *= -1.0
    return converted


def _load_target_pose(
    path: Path, cam_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        mats = data["data"].astype(np.float32)
        inds = data["inds"].astype(np.int64)
    selected = []
    for idx in cam_indices:
        matches = np.where(inds == idx)[0]
        choice = int(matches[0]) if matches.size > 0 else _nearest_index(inds, idx)
        selected.append(mats[choice])
    selected = np.stack(selected, axis=0)
    return selected, inds


def _count_video_frames(path: Path) -> int:
    reader = imageio.get_reader(str(path))
    try:
        frame_count = reader.count_frames()
    finally:
        reader.close()
    return frame_count


def _crop_and_resize(image, height: int, width: int):
    orig_width, orig_height = image.size
    scale = max(width / orig_width, height / orig_height)
    resized_height = max(int(round(orig_height * scale)), 1)
    resized_width = max(int(round(orig_width * scale)), 1)
    resized = TF.resize(image, (resized_height, resized_width), interpolation=InterpolationMode.BILINEAR, antialias=True)
    top = max((resized_height - height) // 2, 0)
    left = max((resized_width - width) // 2, 0)
    cropped = TF.crop(resized, top, left, height, width)
    return cropped


def _load_video_frames(
    path: Path,
    num_frames: int,
    height: int,
    width: int,
    frame_process: v2.Compose,
) -> torch.Tensor:
    reader = imageio.get_reader(str(path))
    frames: List[torch.Tensor] = []
    try:
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            pil_img = torchvision.transforms.functional.to_pil_image(frame)
            processed_img = _crop_and_resize(pil_img, height, width)
            processed = frame_process(processed_img)
            frames.append(processed)
    finally:
        reader.close()
    stacked = torch.stack(frames, dim=0)
    stacked = stacked.permute(1, 0, 2, 3)
    return stacked


@dataclass
class CondCameraData:
    rel_c2w: np.ndarray
    rel_w2c: np.ndarray
    ref_w2c: np.ndarray


class PointOdysseyV2VDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        num_frames: int,
        height: int,
        width: int,
        cam_interval: int,
        max_samples: int | None = None,
    ) -> None:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.dataset_root = split_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.cam_indices = list(range(num_frames))[::cam_interval] if cam_interval > 0 else list(range(num_frames))
        self.samples: List[Dict[str, Path]] = []

        video_paths = sorted(split_dir.glob("*.mp4"))
        for video_path in video_paths:
            stem = video_path.stem
            anno_path = split_dir / stem / "anno.npz"
            if not anno_path.exists():
                continue
            try:
                frame_count = _count_video_frames(video_path)
            except Exception:
                continue
            if frame_count < num_frames:
                continue
            self.samples.append(
                {
                    "video_path": video_path,
                    "anno_path": anno_path,
                    "stem": stem,
                }
            )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        self.frame_process = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_cond_camera(self, anno_path: Path) -> CondCameraData:
        with np.load(anno_path, allow_pickle=False) as data:
            extrinsics = data["extrinsics"].astype(np.float32)
        if extrinsics.shape[0] < self.num_frames:
            raise ValueError(f"{anno_path} has insufficient frames: {extrinsics.shape[0]}")
        cond_w2cs = extrinsics[: self.num_frames][self.cam_indices]
        cond_c2ws = np.stack([_invert_se3(mat) for mat in cond_w2cs], axis=0)
        cond_c2ws = np.stack([_convert_c2w_convention(mat) for mat in cond_c2ws], axis=0)
        cond_c2ws = _center_c2w(cond_c2ws)
        ref_c2w = cond_c2ws[0]
        ref_w2c = _invert_se3(ref_c2w)
        rel_c2w = _compute_relative_c2w(ref_w2c, cond_c2ws)
        rel_w2c = _c2w_to_w2c(rel_c2w)
        return CondCameraData(rel_c2w=rel_c2w, rel_w2c=rel_w2c, ref_w2c=ref_w2c)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        info = self.samples[idx]
        video_tensor = _load_video_frames(
            info["video_path"],
            self.num_frames,
            self.height,
            self.width,
            self.frame_process,
        )
        cond = self._load_cond_camera(info["anno_path"])
        item: Dict[str, torch.Tensor | str] = {
            "video": video_tensor,
            "path": str(info["video_path"]),
            "stem": info["stem"],
            "cond_rel_c2w": torch.from_numpy(cond.rel_c2w),
            "cond_rel_w2c": torch.from_numpy(cond.rel_w2c),
            "ref_w2c": torch.from_numpy(cond.ref_w2c),
            "cam_indices": torch.tensor(self.cam_indices, dtype=torch.int64),
            "text": "",
        }
        return item


def build_pose_embedding(
    cond_rel_c2w: np.ndarray,
    ref_w2c: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_c2ws = np.stack([_invert_se3(mat) for mat in target_pose], axis=0)
    target_c2ws = np.stack([_convert_c2w_convention(mat) for mat in target_c2ws], axis=0)
    target_c2ws = _center_c2w(target_c2ws)
    tgt_rel_c2w = _compute_relative_c2w(ref_w2c, target_c2ws)
    cond_scaled, tgt_scaled = _normalize_joint_translation(cond_rel_c2w, tgt_rel_c2w)
    cond_rel_w2c = _c2w_to_w2c(cond_scaled)
    tgt_rel_w2c = _c2w_to_w2c(tgt_scaled)
    pose_embedding = np.concatenate([tgt_rel_w2c, cond_rel_w2c], axis=0).astype(np.float32)
    return pose_embedding, tgt_rel_w2c.astype(np.float32)


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

    load_msg = model.load_state_dict(compatible_state, strict=False)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PointOdyssey evaluation trajectories.")
    parser.add_argument("--dataset_root", type=str, required=True, help="PointOdyssey根目录，例如 /nas/datasets/PointOdyssey")
    parser.add_argument("--split", type=str, default="train", help="使用的数据集划分，默认test")
    parser.add_argument("--target_pose_dir", type=str, required=True, help="包含若干target trajectory .npz的目录")
    parser.add_argument("--ckpt_path", type=str, required=True, help="ReCamMaster v2v checkpoint路径")
    parser.add_argument("--output_root", type=str, default="/nas/users/lcy/v2v_eval", help="输出根目录，脚本会自动创建时间戳子目录")
    parser.add_argument("--timestamp", type=str, default=None, help="继续写入已有的时间戳子目录，格式如 20240101_120000")
    parser.add_argument("--num_frames", type=int, default=81, help="每段视频使用的帧数")
    parser.add_argument("--camera_interval", type=int, default=4, help="相机采样间隔（cam indices 步长）")
    parser.add_argument("--height", type=int, default=480, help="输出高度")
    parser.add_argument("--width", type=int, default=832, help="输出宽度")
    parser.add_argument("--max_samples", type=int, default=None, help="仅处理前N个视频用于调试")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="DataLoader worker数量")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="classifier-free guidance尺度")
    parser.add_argument("--frame_downsample_to", type=int, default=0, help="latent时间下采样比例（0 表示不降采样）")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="扩散推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--pipeline_type", type=str, default="v2v", choices=["v2v"], help="目前仅支持v2v推理")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, world_size, rank, _, device = setup_distributed_environment()

    if rank == 0:
        print(f"[info] world_size={world_size}, device={device}")

    frame_downsample_to = args.frame_downsample_to
    if frame_downsample_to and frame_downsample_to > 0:
        if rank == 0:
            print("[warning] 当前 DiffSynth WanVideo 管线在 frame_downsample_to > 0 时会造成 RoPE 维度不一致，已自动改为 0。")
        frame_downsample_to = 0
    
    if args.debug:
        print("Debug mode is enabled.") 
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    target_pose_paths = sorted(Path(args.target_pose_dir).glob("*.npz"))
    if not target_pose_paths:
        raise FileNotFoundError(f"未在{args.target_pose_dir}找到任何target trajectory npz文件")

    dataset = PointOdysseyV2VDataset(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        cam_interval=args.camera_interval,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，可能是没有满足帧数要求的视频")

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

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ]
    )
    device_str = device.type if device.type == "cpu" else f"cuda:{device.index}"
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device_str)

    if rank == 0:
        print(f"[info] 加载checkpoint: {args.ckpt_path}")

    if str(args.ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(args.ckpt_path)
    else:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
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

    output_root = Path(args.output_root)
    output_dir = broadcast_output_directory(
        output_root,
        distributed,
        rank,
        resume_timestamp=args.timestamp,
    )
    if rank == 0:
        if args.timestamp:
            print(f"[info] 继续写入目录: {output_dir}")
        else:
            print(f"[info] 输出目录: {output_dir}")

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    target_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}

    if sampler is not None:
        sampler.set_epoch(0)

    for batch_idx, batch in enumerate(dataloader):
        source_video = batch["video"].to(device)
        source_path = batch["path"][0]
        source_stem = batch["stem"][0]
        cond_rel_c2w = batch["cond_rel_c2w"][0].numpy()
        ref_w2c = batch["ref_w2c"][0].numpy()
        cam_indices = batch["cam_indices"][0].numpy()

        for target_path in target_pose_paths:
            if target_path not in target_cache:
                pose_data, pose_inds = _load_target_pose(target_path, cam_indices)
                target_cache[target_path] = (pose_data, pose_inds)
            else:
                pose_data, pose_inds = target_cache[target_path]

            pose_embedding_np, target_rel_w2c = build_pose_embedding(
                cond_rel_c2w=cond_rel_c2w,
                ref_w2c=ref_w2c,
                target_pose=pose_data,
            )
            target_camera = (
                torch.from_numpy(pose_embedding_np)
                .unsqueeze(0)
                .to(device=device, dtype=torch.bfloat16)
            )

            output_name = f"{source_stem}_{target_path.stem}"
            video_path = output_dir / f"{output_name}.mp4"
            pose_path = output_dir / f"{output_name}.npz"
            if video_path.exists():
                print(f"[info][rank {rank}] 已存在输出，跳过: {video_path.name}")
                continue

            with torch.no_grad():
                video = pipe(
                    prompt=batch["text"][0],
                    negative_prompt=negative_prompt,
                    source_video=source_video,
                    target_camera=target_camera,
                    cfg_scale=args.cfg_scale,
                    frame_downsample_to=frame_downsample_to,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    tiled=True,
                )

            save_video(video, str(video_path), fps=30, quality=5)
            np.savez(
                pose_path,
                data=target_rel_w2c,
                inds=cam_indices,
                source=str(source_path),
                target_pose=str(target_path),
                target_inds=pose_inds,
            )

        if rank == 0:
            print(f"[info] 完成样本 {batch_idx + 1}/{len(dataloader)} -> {source_stem}")

    cleanup_distributed_environment(distributed)


if __name__ == "__main__":
    main()
