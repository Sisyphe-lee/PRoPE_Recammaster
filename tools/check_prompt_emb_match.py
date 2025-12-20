#!/usr/bin/env python3
"""验证保存的 prompt embedding 是否与给定 caption 编码一致。

用法示例：
python tools/check_prompt_emb_match.py \
  --caption "A cat walking in the street" \
  --tensor-path /nas/datasets/.../cam01.mp4.wan22.tensors.pth \
  --text-encoder-path models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

from diffsynth import ModelManager  # noqa: E402
from diffsynth.prompters import WanPrompter  # noqa: E402
from src.dataset import _ensure_prompt_context  # noqa: E402


def _infer_text_encoder_path() -> Optional[Path]:
    env_path = os.getenv("TEXT_ENCODER_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        PROJECT_ROOT / "models" / "Wan-AI" / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth",
        PROJECT_ROOT / "models" / "Wan-AI" / "Wan2.1-T2V-1.3B" / "models_t5_umt5-xxl-enc-bf16.pth",
        PROJECT_ROOT / "models" / "Wan-AI" / "Wan2.1-T2V-14B" / "models_t5_umt5-xxl-enc-bf16.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _infer_tokenizer_path(text_encoder_path: Path) -> Optional[Path]:
    if text_encoder_path is None:
        return None
    cand = text_encoder_path.parent / "google" / "umt5-xxl"
    return cand if cand.exists() else None


class WanPromptEncoder:
    def __init__(self, text_encoder_path: Path, tokenizer_path: Optional[Path], device: str):
        if text_encoder_path is None or not Path(text_encoder_path).exists():
            raise FileNotFoundError(f"找不到 text encoder 权重: {text_encoder_path}")

        self.device = torch.device(device)
        manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        manager.load_models([str(text_encoder_path)])
        fetched = manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if fetched is None:
            raise RuntimeError(f"无法从 {text_encoder_path} 加载 wan_video_text_encoder。")
        text_encoder, weight_path = fetched

        tokenizer_dir = tokenizer_path or _infer_tokenizer_path(Path(weight_path))
        if tokenizer_dir is None or not tokenizer_dir.exists():
            raise FileNotFoundError(f"找不到 tokenizer 目录: {tokenizer_dir}")

        self.text_encoder = text_encoder.to(self.device).eval()
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(str(tokenizer_dir))
        self.tokenizer_dir = tokenizer_dir

    @torch.inference_mode()
    def encode(self, prompts: list[str]) -> torch.Tensor:
        if not isinstance(prompts, list):
            prompts = [prompts]
        return self.prompter.encode_prompt(prompts, positive=True, device=self.device)


def _flatten_prompt(prompt: torch.Tensor) -> torch.Tensor:
    if prompt.dim() == 2:
        prompt = prompt.unsqueeze(0)
    return prompt.reshape(prompt.shape[0], -1)


def compare_prompt_embedding(args: argparse.Namespace) -> None:
    tensor_path = Path(args.tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"未找到 tensor 文件: {tensor_path}")

    text_encoder_path = Path(args.text_encoder_path) if args.text_encoder_path else _infer_text_encoder_path()
    if text_encoder_path is None:
        raise FileNotFoundError("未能推断 text encoder 路径，请通过 --text-encoder-path 指定。")

    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else _infer_tokenizer_path(text_encoder_path)
    encoder = WanPromptEncoder(text_encoder_path=text_encoder_path, tokenizer_path=tokenizer_path, device=args.device)

    data = torch.load(tensor_path, map_location="cpu")
    stored = _ensure_prompt_context(data.get("prompt_emb")).get("context")
    if stored is None:
        raise RuntimeError(f"{tensor_path} 中未找到 prompt_emb/context 字段。")

    encoded = encoder.encode([args.caption]).to(dtype=stored.dtype, device=stored.device)

    stored_f = _flatten_prompt(stored).float()
    encoded_f = _flatten_prompt(encoded).float()

    if stored_f.shape != encoded_f.shape:
        raise RuntimeError(f"编码结果形状不匹配: stored {stored_f.shape} vs encoded {encoded_f.shape}")

    diff = stored_f - encoded_f
    mae = float(diff.abs().mean())
    max_abs = float(diff.abs().max())
    mse = float((diff ** 2).mean())
    cosine = float(torch.nn.functional.cosine_similarity(stored_f, encoded_f, dim=1).mean())

    print("=== 配置 ===")
    print(f"text_encoder_path: {text_encoder_path}")
    print(f"tokenizer_path   : {encoder.tokenizer_dir}")
    print(f"tensor_path      : {tensor_path}")
    print(f"device           : {args.device}")
    print(f"caption          : {args.caption}")
    print("\n=== 形状与类型 ===")
    print(f"stored prompt    : shape={tuple(stored.shape)}, dtype={stored.dtype}")
    print(f"re-encoded prompt: shape={tuple(encoded.shape)}, dtype={encoded.dtype}")
    print("\n=== 差异指标 ===")
    print(f"mean |diff|      : {mae:.6f}")
    print(f"max  |diff|      : {max_abs:.6f}")
    print(f"MSE              : {mse:.6f}")
    print(f"cosine similarity: {cosine:.6f}")

    verdict = "一致" if max_abs <= args.tolerance else "不一致"
    print(f"\n结论：{verdict}（阈值 tolerance={args.tolerance}）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 caption 编码是否与保存的 prompt embedding 匹配。")
    parser.add_argument("--caption", required=True, help="要验证的 caption 文本。")
    parser.add_argument("--tensor-path", required=True, help="对应的 .tensors.pth 或 .wan22.tensors.pth 文件路径。")
    parser.add_argument(
        "--text-encoder-path",
        default=None,
        help="可选，自定义 text encoder 权重路径（默认自动从 models/Wan-AI/... 推断）。",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="可选，自定义 tokenizer 目录（默认使用 <text_encoder_dir>/google/umt5-xxl）。",
    )
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, help="推理设备，默认 cuda:0 可用则使用。")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="判定一致性的最大绝对误差阈值。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_prompt_embedding(args)
