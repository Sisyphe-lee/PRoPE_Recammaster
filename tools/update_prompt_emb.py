#!/usr/bin/env python3
"""多卡并行更新 MultiCam latent 缓存里的 prompt_emb。"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import lightning as pl
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import ModelManager  # noqa: E402
from diffsynth.prompters import WanPrompter  # noqa: E402


class WanPromptEncoder:
    def __init__(self, text_encoder_path: str, tokenizer_path: Optional[str], device: str):
        if text_encoder_path is None:
            raise ValueError("必须提供 text_encoder_path。")

        self.device = torch.device(device)
        manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        manager.load_models([text_encoder_path])
        fetched = manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if fetched is None:
            raise RuntimeError(f"无法从 {text_encoder_path} 加载 wan_video_text_encoder。")
        text_encoder, weight_path = fetched
        tokenizer_dir = Path(tokenizer_path) if tokenizer_path else Path(weight_path).parent / "google" / "umt5-xxl"
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"找不到 tokenizer 目录: {tokenizer_dir}")

        self.text_encoder = text_encoder.to(self.device).eval()
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(str(tokenizer_dir))

    @torch.inference_mode()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        if not isinstance(prompts, list):
            prompts = [prompts]
        return self.prompter.encode_prompt(prompts, positive=True, device=self.device)


class PromptDataset(Dataset):
    def __init__(self, dataset_path: str, metadata_path: str, tensor_suffix: str, resume_set: set[str]):
        df = pd.read_csv(metadata_path)
        if "video_absolute_path" not in df.columns:
            raise ValueError("metadata 需要包含 video_absolute_path 列。")
        if "caption" not in df.columns:
            raise ValueError("metadata 需要包含 caption 列。")

        base = Path(dataset_path)
        samples: List[tuple[str, str]] = []
        for row in df.itertuples(index=False):
            raw_path = getattr(row, "video_absolute_path")
            raw_caption = getattr(row, "caption")
            caption = "" if pd.isna(raw_caption) else str(raw_caption)
            full_path = Path(raw_path) if os.path.isabs(raw_path) else base / raw_path
            tensor_path = str(Path(str(full_path) + tensor_suffix))
            if tensor_path in resume_set:
                continue
            samples.append((caption, tensor_path))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caption, tensor_path = self.samples[idx]
        return {"text": caption, "tensor_path": tensor_path}


def collate_batch(batch):
    return {
        "text": [item["text"] for item in batch],
        "tensor_path": [item["tensor_path"] for item in batch],
    }


class FileLock:
    def __init__(self, target: Path):
        self.lock_path = Path(str(target) + ".lock")
        self.fd: Optional[int] = None

    def acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        fcntl.lockf(self.fd, fcntl.LOCK_EX)
        return self

    def release(self):
        if self.fd is None:
            return
        try:
            fcntl.lockf(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()


def _normalize_state_path(state_path: Optional[str]) -> Optional[Path]:
    if not state_path:
        return None
    return Path(state_path)


def _load_resume_state(state_path: Optional[Path]) -> set[str]:
    if state_path is None or not state_path.exists():
        return set()
    with open(state_path, "r", encoding="utf-8") as fp:
        return {line.strip() for line in fp if line.strip()}


class PromptEmbeddingModule(pl.LightningModule):
    def __init__(
        self,
        *,
        text_encoder_path: str,
        tokenizer_path: Optional[str],
        overwrite: bool,
        state_path: Optional[Path],
    ):
        super().__init__()
        self.text_encoder_path = text_encoder_path
        self.tokenizer_path = tokenizer_path
        self.overwrite = overwrite
        self.state_path = state_path

        self.encoder: Optional[WanPromptEncoder] = None
        self.updated = 0
        self.skipped = 0
        self.missing = 0
        self.completed_paths: List[str] = []

    def on_test_start(self):
        device_str = str(self.device)
        self.encoder = WanPromptEncoder(
            text_encoder_path=self.text_encoder_path,
            tokenizer_path=self.tokenizer_path,
            device=device_str,
        )

    def _mark_completed(self, tensor_path: str):
        self.completed_paths.append(tensor_path)

    def test_step(self, batch, batch_idx):
        if self.encoder is None:
            raise RuntimeError("文本编码器尚未初始化。")

        texts: List[str] = batch["text"]
        tensor_paths: List[str] = batch["tensor_path"]

        pending_indices: List[int] = []
        pending_payloads: List[tuple[str, Path, dict, FileLock]] = []

        for idx, (text, tensor_path) in enumerate(zip(texts, tensor_paths)):
            path_obj = Path(tensor_path)
            if not path_obj.exists():
                self.missing += 1
                continue

            lock = FileLock(path_obj).acquire()
            try:
                data = torch.load(path_obj, map_location="cpu")
            except Exception as exc:
                print(f"[error] 读取 {path_obj} 失败: {exc}")
                lock.release()
                raise

            if not self.overwrite and "prompt_emb" in data:
                self.skipped += 1
                self._mark_completed(tensor_path)
                lock.release()
                continue

            pending_indices.append(idx)
            pending_payloads.append((text, path_obj, data, lock))

        if not pending_payloads:
            return

        prompts = [item[0] for item in pending_payloads]
        prompt_emb_batch = self.encoder.encode(prompts)
        prompt_emb_batch = prompt_emb_batch.detach().to("cpu")

        for emb, (text, path_obj, data, lock) in zip(prompt_emb_batch, pending_payloads):
            tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
            try:
                # 与原始 latent 导出保持一致，保存 dict 且保留 batch 维
                emb_to_store = emb.unsqueeze(0) if emb.dim() == 2 else emb
                data["prompt_emb"] = {"context": emb_to_store.to(dtype=torch.bfloat16, copy=False)}
                torch.save(data, tmp_path)
                os.replace(tmp_path, path_obj)
                self.updated += 1
                self._mark_completed(str(path_obj))
            finally:
                if tmp_path.exists():
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                lock.release()

    def _gather_paths(self) -> Optional[List[str]]:
        if dist.is_available() and dist.is_initialized():
            gathered: List[List[str]] | None
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.completed_paths, gathered if self.global_rank == 0 else None, dst=0)
            if self.global_rank == 0:
                merged: List[str] = []
                for chunk in gathered:
                    if chunk:
                        merged.extend(chunk)
                return merged
            return None
        return list(self.completed_paths)

    def _reduce_stats(self) -> Optional[Sequence[int]]:
        stats = torch.tensor([self.updated, self.skipped, self.missing], device=self.device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.reduce(stats, dst=0, op=dist.ReduceOp.SUM)
            if self.global_rank == 0:
                return stats.tolist()
            return None
        return stats.tolist()

    def on_test_end(self):
        merged_paths = self._gather_paths()
        if merged_paths and self.state_path is not None and self.global_rank == 0:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            unique_paths = list(dict.fromkeys(merged_paths))
            with open(self.state_path, "a", encoding="utf-8") as fp:
                for path in unique_paths:
                    fp.write(path + "\n")

        reduced = self._reduce_stats()
        if reduced is not None:
            updated, skipped, missing = reduced
            print(f"完成：更新 {updated} 个，跳过 {skipped} 个，未找到 {missing} 个缓存。")


def update_prompt_embeddings(args):
    state_path = _normalize_state_path(args.state_path)
    if state_path and args.reset_state and state_path.exists():
        state_path.unlink()
    resume_set = _load_resume_state(state_path)

    dataset = PromptDataset(
        dataset_path=args.dataset_path,
        metadata_path=args.metadata_path,
        tensor_suffix=args.tensor_suffix,
        resume_set=resume_set,
    )
    if len(dataset) == 0:
        print("没有需要处理的样本：resume 记录已覆盖全部条目。")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        persistent_workers=args.dataloader_num_workers > 0,
        collate_fn=collate_batch,
    )

    module = PromptEmbeddingModule(
        text_encoder_path=args.text_encoder_path,
        tokenizer_path=args.tokenizer_path,
        overwrite=args.overwrite,
        state_path=state_path,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = -1  # 使用所有可见 GPU（由 CUDA_VISIBLE_DEVICES 控制）
        if torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "16-mixed"
        strategy = "ddp_find_unused_parameters_false"
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32"
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=not args.disable_progress_bar,
    )
    trainer.test(module, dataloaders=dataloader)


def parse_args():
    default_model_root = PROJECT_ROOT / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    parser = argparse.ArgumentParser(description="多卡并行重新生成 prompt_emb")
    parser.add_argument("--dataset_path", required=True, help="MultiCam 数据根目录")
    parser.add_argument("--metadata_path", required=True, help="metadata CSV，需包含 video_absolute_path 与 caption")
    parser.add_argument("--tensor_suffix", default=".tensors.pth", help="latent 文件的后缀")
    parser.add_argument("--text_encoder_path", default=str(default_model_root / "models_t5_umt5-xxl-enc-bf16.pth"))
    parser.add_argument("--tokenizer_path", default=None, help="可选 tokenizer 目录，默认使用 text encoder 平级的 google/umt5-xxl")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 prompt_emb")
    parser.add_argument(
        "--state_path",
        default=str(PROJECT_ROOT / "models" / "prompt_emb_resume.log"),
        help="记录已完成样本的文件，留空可禁用",
    )
    parser.add_argument("--reset_state", action="store_true", help="运行前删除状态文件")
    parser.add_argument("--batch_size", type=int, default=8, help="每次一起处理的 caption 数量")
    parser.add_argument("--dataloader_num_workers", type=int, default=32)
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        print("Debug mode is enabled.")
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")
    update_prompt_embeddings(args)


if __name__ == "__main__":
    main()
