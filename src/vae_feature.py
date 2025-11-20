import argparse
import inspect
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import functional as tvF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import ModelManager, WanVideoPipeline, WanVideoReCamMasterPipeline


ALLOWED_DATASET_TYPES = ("multicam", "re10k")


def resize_with_aspect(image: Image.Image, target_width: int, target_height: int) -> Tuple[Image.Image, float]:
    """
    Resize image while preserving aspect ratio so that both sides are >= target.
    Returns resized image and the applied scale factor.
    """
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("Invalid source image size for resizing.")
    scale = max(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = tvF.resize(
        image,
        size=(new_height, new_width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized, scale


def compute_center_crop_offsets(
    scaled_width: int, scaled_height: int, target_width: int, target_height: int
) -> Tuple[int, int]:
    """
    Compute offsets (left, top) used by torchvision's CenterCrop.
    """
    crop_left = max(0, int(round((scaled_width - target_width) / 2.0)))
    crop_top = max(0, int(round((scaled_height - target_height) / 2.0)))
    return crop_left, crop_top


def compute_rescaled_intrinsics(
    camera_params: Sequence[float],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
    """
    Convert normalized intrinsics (fx_norm, fy_norm, cx_norm, cy_norm) to pixel units
    after applying resize+center-crop described in TextVideoDataset.
    """
    fx_norm, fy_norm, cx_norm, cy_norm = camera_params[:4]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    crop_left, crop_top = compute_center_crop_offsets(resized_width, resized_height, target_width, target_height)
    fx = float(fx_norm) * source_width * scale
    fy = float(fy_norm) * source_height * scale
    cx = float(cx_norm) * source_width * scale - crop_left
    cy = float(cy_norm) * source_height * scale - crop_top
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, scale, (float(crop_left), float(crop_top)), (resized_width, resized_height)


@dataclass
class Re10kSampleInfo:
    key: str
    torch_path: Path
    output_dir: Path
    video_path: Path
    tensor_path: Path
    split: str


## TODO: 添加rel10k的新的数据集
### __getitem__需要返回：text, video, intrinsics, extrinsics, metadata, path.
### 需要先解析rel10k的数据格式，TODO.md有详写

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        raw_paths = metadata["video_absolute_path"].tolist()
        self.path = [
            p if os.path.isabs(p) else os.path.join(base_path, p)
            for p in raw_paths
        ]
        self.text = metadata["caption"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        image, _ = resize_with_aspect(image, self.width, self.height)
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


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        while True:
            try:
                text = self.text[data_id]
                path = self.path[data_id]
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)



class RealEstate10KDataset(torch.utils.data.Dataset):
    """
    Dataset that reads raw RealEstate10K .torch shards and reorganizes them into the
    unified format expected by ReCamMaster training.
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        tensor_suffix: str,
        index_path: Optional[str] = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        skip_existing: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.tensor_suffix = tensor_suffix
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.skip_existing = skip_existing
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.index_path = Path(index_path) if index_path is not None else self.dataset_path / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"未找到 RealEstate10k 索引文件: {self.index_path}")

        with open(self.index_path, "r") as f:
            index_data = json.load(f)
        if not isinstance(index_data, dict):
            raise ValueError(f"索引文件 {self.index_path} 格式异常，需为键值对映射。")

        split_name = self.dataset_path.name
        self.entries: List[Re10kSampleInfo] = []
        skipped = 0
        for key in sorted(index_data.keys()):
            torch_file = self.dataset_path / index_data[key]
            if not torch_file.exists():
                print(f"[re10k] 缺少源数据 {torch_file}, 跳过 {key}")
                continue
            seq_dir = self.output_path / key
            video_path = seq_dir / "video.mp4"
            tensor_path = Path(str(video_path) + tensor_suffix)
            if skip_existing and tensor_path.exists():
                skipped += 1
                continue
            self.entries.append(
                Re10kSampleInfo(
                    key=key,
                    torch_path=torch_file,
                    output_dir=seq_dir,
                    video_path=video_path,
                    tensor_path=tensor_path,
                    split=split_name,
                )
            )
        print(f"[re10k] 待处理序列: {len(self.entries)} (已跳过 {skipped} 个已有缓存)")
        if not self.entries:
            raise ValueError("没有待处理的 RealEstate10k 序列，确认 --no_resume 设置或输出目录是否为空。")

        self._cached_file_path: Optional[Path] = None
        self._cached_items: Optional[List[Dict]] = None

    def __len__(self):
        return len(self.entries)

    def _load_shard(self, shard_path: Path) -> List[Dict]:
        if shard_path == self._cached_file_path and self._cached_items is not None:
            return self._cached_items
        items = torch.load(shard_path, map_location="cpu")
        if not isinstance(items, list):
            raise ValueError(f"Shard {shard_path} 内容异常，期望 list.")
        self._cached_file_path = shard_path
        self._cached_items = items
        return items

    def _select_indices(self, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            raise ValueError("视频帧数必须大于 0")
        if total_frames == 1:
            return np.zeros((self.num_frames,), dtype=np.int64)
        positions = np.linspace(0, total_frames - 1, num=self.num_frames)
        indices = np.clip(np.round(positions).astype(np.int64), 0, total_frames - 1)
        return indices

    def _decode_frame(self, tensor: torch.Tensor) -> Image.Image:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("image tensor must be torch.Tensor")
        return Image.open(io.BytesIO(tensor.numpy().tobytes())).convert("RGB")

    def __getitem__(self, idx):
        if len(self.entries) == 0:
            raise IndexError("空的 RealEstate10k 数据集。")
        cursor = idx % len(self.entries)
        attempts = 0
        while attempts < len(self.entries):
            info = self.entries[cursor]
            try:
                samples = self._load_shard(info.torch_path)
                target = next((item for item in samples if item.get("key") == info.key), None)
                if target is None:
                    raise KeyError(f"在 {info.torch_path} 中未找到 key={info.key}")
                images: List[torch.Tensor] = target["images"]
                cameras: torch.Tensor = target["cameras"]
                timestamps: torch.Tensor = target["timestamps"]
                if len(images) == 0:
                    raise ValueError(f"{info.key} 没有可用帧。")
                frame_indices = self._select_indices(len(images))

                processed_frames: List[torch.Tensor] = []
                frames_uint8: List[np.ndarray] = []
                first_frame_np: Optional[np.ndarray] = None

                source_image = self._decode_frame(images[frame_indices[0]])
                source_width, source_height = source_image.size
                K, scale, crop_offsets, scaled_size = compute_rescaled_intrinsics(
                    cameras[frame_indices[0]], source_width, source_height, self.width, self.height
                )
                Ks = np.repeat(K[None, ...], len(frame_indices), axis=0)
                extrinsics = []

                for order, frame_idx in enumerate(frame_indices):
                    pil_img = source_image if order == 0 else self._decode_frame(images[frame_idx])
                    resized, _ = resize_with_aspect(pil_img, self.width, self.height)
                    if first_frame_np is None:
                        first_frame_np = np.array(resized)
                    frames_uint8.append(np.array(resized))
                    tensor_frame = self.frame_process(resized)
                    processed_frames.append(tensor_frame)

                    cam_row = cameras[frame_idx]
                    R = cam_row[6:15].view(3, 3).numpy()
                    t = cam_row[15:18].numpy()
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :3] = R
                    w2c[:3, 3] = t
                    c2w = np.linalg.inv(w2c).astype(np.float32)
                    extrinsics.append(c2w)

                video_tensor = torch.stack(processed_frames, dim=0)  # [T, C, H, W]
                video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]

                timestamps_np = timestamps.numpy()[frame_indices]
                if len(timestamps) > 1:
                    diffs = np.diff(timestamps.numpy()).astype(np.float64)
                    median_diff = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
                    fps = 1e6 / median_diff if median_diff > 0 else 24.0
                else:
                    fps = 24.0

                metadata = {
                    "key": info.key,
                    "source_url": target.get("url", ""),
                    "source_torch": str(info.torch_path),
                    "split": info.split,
                    "num_frames_raw": len(images),
                    "num_frames_sampled": int(len(frame_indices)),
                    "frame_indices": frame_indices.tolist(),
                    "timestamps": timestamps_np.astype(np.int64).tolist(),
                    "fps": fps,
                    "original_size": [source_width, source_height],
                    "scaled_size": list(scaled_size),
                    "target_size": [self.width, self.height],
                    "scale": scale,
                    "center_crop_offset": {"x": crop_offsets[0], "y": crop_offsets[1]},
                }

                sample = {
                    "text": "",
                    "video": video_tensor,
                    "path": str(info.video_path),
                    "first_frame": first_frame_np,
                    "frames_uint8": torch.from_numpy(np.stack(frames_uint8)),
                    "intrinsics": torch.from_numpy(Ks),
                    "extrinsics": torch.from_numpy(np.stack(extrinsics)),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                }
                return sample
            except Exception as exc:
                print(f"[re10k] 读取 {info.key} 失败: {exc}")
                cursor = (cursor + 1) % len(self.entries)
                attempts += 1
        raise RuntimeError("无法从 RealEstate10k 数据集中成功读取样本。")


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        pipeline_type="v2v",
        tensor_suffix=".tensors.pth",
        dataset_type: str = "multicam",
        overwrite_cached: bool = False,
    ):
        super().__init__()
        self.tensor_suffix = tensor_suffix
        self.pipeline_type = pipeline_type
        if dataset_type not in ALLOWED_DATASET_TYPES:
            raise ValueError(f"dataset_type 必须在 {ALLOWED_DATASET_TYPES}，收到 {dataset_type}.")
        self.dataset_type = dataset_type
        self.overwrite_cached = overwrite_cached

        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided for latent extraction.")
        if vae_path is None:
            raise ValueError("vae_path must be provided for latent extraction.")

        model_path = []
        if text_encoder_path is not None:
            model_path.append(text_encoder_path)
        if vae_path is not None:
            model_path.append(vae_path)
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        if len(model_path) == 0:
            raise ValueError("At least VAE path must be provided for latent extraction.")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        if pipeline_type == "v2v":
            self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        elif pipeline_type == "i2v":
            self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        else:
            raise ValueError(f"Unsupported pipeline_type: {pipeline_type}")

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
    
    def test_step(self, batch, batch_idx):
        text = batch["text"][0] if isinstance(batch["text"], list) else batch["text"]
        path_value = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
        video = batch["video"]

        self.pipe.device = self.device
        if video is not None:
            tensor_path = path_value + self.tensor_suffix
            Path(tensor_path).parent.mkdir(parents=True, exist_ok=True)
            skip_existing = (not self.overwrite_cached) and os.path.exists(tensor_path)
            if skip_existing:
                print(f"File {tensor_path} already exists, skipping.")
                return

            prompt_emb = self.pipe.encode_prompt(text)

            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents_encoded = self.pipe.encode_video(video, **self.tiler_kwargs)
            if isinstance(latents_encoded, (list, tuple)):
                latents = latents_encoded[0]
            else:
                latents = latents_encoded
            if latents.dim() == 5:
                latents = latents[0]

            if "first_frame" in batch:
                first_frame_np = batch["first_frame"][0].cpu().numpy()
                first_frame_img = Image.fromarray(first_frame_np)
                _, _, num_frames, height, width = video.shape
                image_emb = self._encode_condition_image(first_frame_img, num_frames, height, width)
            else:
                image_emb = {}

            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, tensor_path)
            if self.dataset_type == "re10k":
                self._write_re10k_assets(batch, path_value, tensor_path)

    def _encode_condition_image(self, image: Image.Image, num_frames: int, height: int, width: int) -> Dict[str, torch.Tensor]:
        if self.pipeline_type != "i2v":
            return self.pipe.encode_image(image, num_frames, height, width)

        encode_fn = getattr(self.pipe, "encode_image", None)
        if encode_fn is None:
            return self._encode_image_without_clip(image, num_frames, height, width)

        sig = inspect.signature(encode_fn)
        tiled = self.tiler_kwargs.get("tiled", False)
        tile_size = self.tiler_kwargs.get("tile_size", (34, 34))
        tile_stride = self.tiler_kwargs.get("tile_stride", (18, 16))
        try:
            if "end_image" in sig.parameters:
                return encode_fn(
                    image,
                    None,
                    num_frames,
                    height,
                    width,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            return encode_fn(image, num_frames, height, width)
        except AttributeError as exc:
            # image encoder is likely missing; fall back to VAE-only path
            print(f"[warn] encode_image fallback triggered due to: {exc}")
            return self._encode_image_without_clip(image, num_frames, height, width)

    def _encode_image_without_clip(self, image: Image.Image, num_frames: int, height: int, width: int) -> Dict[str, torch.Tensor]:
        self.pipe.load_models_to_device(["vae"])
        resized = image.resize((width, height))
        image_tensor = self.pipe.preprocess_image(resized).to(self.pipe.device)

        transposed = image_tensor.transpose(0, 1)
        if num_frames > 1:
            zeros = torch.zeros(3, num_frames - 1, height, width, device=image_tensor.device)
            vae_input = torch.concat([transposed, zeros], dim=1)
        else:
            vae_input = transposed
        tiled = self.tiler_kwargs.get("tiled", False)
        tile_size = self.tiler_kwargs.get("tile_size", (34, 34))
        tile_stride = self.tiler_kwargs.get("tile_stride", (18, 16))
        y = self.pipe.vae.encode(
            [vae_input.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)],
            device=self.pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        latent_h, latent_w = y.shape[-2], y.shape[-1]
        msk = torch.ones(1, num_frames, latent_h, latent_w, device=self.pipe.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, :1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]

        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        return {"y": y}

    def _write_re10k_assets(self, batch, video_path_str: str, tensor_path: str):
        video_path = Path(video_path_str)
        sequence_dir = video_path.parent
        sequence_dir.mkdir(parents=True, exist_ok=True)

        frames_tensor = batch.get("frames_uint8")
        if frames_tensor is None:
            raise ValueError("re10k 模式需要 frames_uint8 以写入视频。")
        frames_np = frames_tensor[0].cpu().numpy()
        fps = 24.0

        metadata_field = batch.get("metadata")
        metadata_str = metadata_field[0] if isinstance(metadata_field, list) else metadata_field
        metadata = json.loads(metadata_str) if metadata_str else {}
        fps = float(metadata.get("fps", fps))

        fps_value = fps if fps > 0 else 24.0
        imageio.mimsave(video_path, frames_np.astype(np.uint8), fps=fps_value)

        intrinsics = batch.get("intrinsics")
        if intrinsics is not None:
            intr_path = sequence_dir / "intrinsics.npz"
            np.savez(intr_path, K=intrinsics[0].cpu().numpy())
        else:
            intr_path = sequence_dir / "intrinsics.npz"

        extrinsics = batch.get("extrinsics")
        if extrinsics is not None:
            extr_path = sequence_dir / "extrinsics.npz"
            np.savez(extr_path, c2w=extrinsics[0].cpu().numpy())
        else:
            extr_path = sequence_dir / "extrinsics.npz"

        metadata.update(
            {
                "video_path": str(video_path),
                "tensor_path": tensor_path,
                "intrinsics_path": str(intr_path),
                "extrinsics_path": str(extr_path),
            }
        )
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[re10k] 保存 {sequence_dir.name}: video={video_path.name}, tensor={tensor_path}")


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


## TODO: 添加rel10k的相关参数：dataset_kind, re10k_output_path等参数
def parse_args():
    parser = argparse.ArgumentParser(description="Extract VAE features")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="multicam",
        choices=ALLOWED_DATASET_TYPES,
        help="数据组织：multicam 使用 metadata CSV；re10k 读取原始 RealEstate10k .torch 并整理。",
    )
    parser.add_argument(
        "--re10k_output_path",
        type=str,
        default=None,
        help="若 dataset_type=re10k，则此路径用于写入整理后的序列根目录。",
    )
    parser.add_argument(
        "--re10k_index_path",
        type=str,
        default=None,
        help="可选，指定 re10k 的 index.json 路径；默认 dataset_path/index.json。",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--tensor_suffix",
        type=str,
        default=None,
        help="Suffix for cached latent tensors (e.g., .tensors.pth or .wan22.tensors.pth).",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default="v2v",
        choices=["v2v", "i2v"],
        help="Pipeline type used for latent extraction or training.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
        help="Metadata filename relative to dataset_path (ignored when --metadata_path is provided).",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional absolute metadata CSV path. Overrides --metadata_file_name when set.",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="不跳过已有缓存，重新生成（会覆盖同名 latent / 资产）。",
    )
    args = parser.parse_args()
    return args


def resolve_tensor_suffix(args):
    if args.tensor_suffix:
        return args.tensor_suffix
    return ".tensors.pth" if args.pipeline_type == "v2v" else f".{args.pipeline_type}.tensors.pth"

def data_process(args):

    if args.debug:
        print("Debug mode is enabled.") 
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    
    tensor_suffix = resolve_tensor_suffix(args)
    dataset_type = args.dataset_type

    if dataset_type == "multicam":
        metadata_path = args.metadata_path
        if metadata_path is None:
            metadata_path = os.path.join(args.dataset_path, args.metadata_file_name)
        dataset = TextVideoDataset(
            args.dataset_path,
            metadata_path,
            max_num_frames=args.num_frames,
            frame_interval=1,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            is_i2v=args.pipeline_type == "i2v",
        )
    else:
        if args.re10k_output_path is None:
            raise ValueError("dataset_type=re10k 时必须提供 --re10k_output_path。")
        dataset = RealEstate10KDataset(
            dataset_path=args.dataset_path,
            output_path=args.re10k_output_path,
            tensor_suffix=tensor_suffix,
            index_path=args.re10k_index_path,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            skip_existing=not args.no_resume,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        pipeline_type=args.pipeline_type,
        tensor_suffix=tensor_suffix,
        dataset_type=dataset_type,
        overwrite_cached=args.no_resume,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    data_process(args)
