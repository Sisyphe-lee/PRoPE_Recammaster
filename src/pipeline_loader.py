from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Sequence
import torch

from diffsynth import ModelManager, WanVideoReCamMasterPipeline
from diffsynth.pipelines.wan_video_new import WanVideoPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))

WAN_MODEL_ROOT = PROJECT_ROOT / "models" / "Wan-AI"
WAN21_MODEL_DIR = WAN_MODEL_ROOT / "Wan2.1-T2V-1.3B"
WAN22_MODEL_DIR = WAN_MODEL_ROOT / "Wan2.2-TI2V-5B"


def _first_existing_path(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def load_v2v_pipeline(device_str: str) -> WanVideoReCamMasterPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        [
            str(WAN21_MODEL_DIR / "diffusion_pytorch_model.safetensors"),
            str(WAN21_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(WAN21_MODEL_DIR / "Wan2.1_VAE.pth"),
        ]
    )
    return WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device_str)


def load_i2v_pipeline(device_str: str, ckpt_type: str = "wan22") -> WanVideoPipeline:
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    text_encoder_path = _first_existing_path(
        [
            WAN22_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
            WAN21_MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
        ]
    )
    if text_encoder_path is None:
        raise FileNotFoundError("Cannot locate Wan text encoder weights under Wan2.1/2.2 directories.")

    if ckpt_type == "wan21":
        vae_root = WAN21_MODEL_DIR
        vae_filename = "Wan2.1_VAE.pth"
    else:
        vae_root = WAN22_MODEL_DIR
        vae_filename = "Wan2.2_VAE.pth"

    vae_path = vae_root / vae_filename
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing VAE weights for {ckpt_type}: {vae_path}")

    if ckpt_type == "wan21":
        diffusion_paths = sorted(WAN21_MODEL_DIR.glob("diffusion_pytorch_model-*.safetensors"))
        if not diffusion_paths:
            fallback = WAN21_MODEL_DIR / "diffusion_pytorch_model.safetensors"
            if fallback.exists():
                diffusion_paths = [fallback]
            else:
                raise FileNotFoundError(f"Missing Wan2.1 diffusion weights under {WAN21_MODEL_DIR}")
    else:
        diffusion_paths = sorted(WAN22_MODEL_DIR.glob("diffusion_pytorch_model-*.safetensors"))
        if not diffusion_paths:
            fallback = WAN22_MODEL_DIR / "diffusion_pytorch_model.safetensors"
            if fallback.exists():
                diffusion_paths = [fallback]
            else:
                raise FileNotFoundError(f"Missing Wan2.2 diffusion weights under {WAN22_MODEL_DIR}")

    models_to_load: List[Any] = [str(text_encoder_path), str(vae_path)]
    if len(diffusion_paths) == 1:
        models_to_load.append(str(diffusion_paths[0]))
    else:
        models_to_load.append([str(p) for p in diffusion_paths])
    model_manager.load_models(models_to_load)

    pipe = WanVideoPipeline(device=device_str, torch_dtype=torch.bfloat16)
    available_names = set(model_manager.model_name)
    if "wan_video_text_encoder" in available_names:
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    if "wan_video_image_encoder" in available_names:
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
    if "wan_video_dit" in available_names:
        pipe.dit = model_manager.fetch_model("wan_video_dit")
    if "wan_video_dit2" in available_names:
        pipe.dit2 = model_manager.fetch_model("wan_video_dit2")
    if "wan_video_vae" in available_names:
        pipe.vae = model_manager.fetch_model("wan_video_vae")
    if "wan_video_motion_controller" in available_names:
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
    if "wan_video_vace" in available_names:
        pipe.vace = model_manager.fetch_model("wan_video_vace")
    if "wan_video_animate_adapter" in available_names:
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

    tokenizer_path = _first_existing_path(
        [
            WAN22_MODEL_DIR / "google" / "umt5-xxl",
            WAN21_MODEL_DIR / "google" / "umt5-xxl",
        ]
    )
    if pipe.text_encoder is not None:
        pipe.prompter.fetch_models(pipe.text_encoder)
    if tokenizer_path is None:
        raise FileNotFoundError("Cannot locate Wan tokenizer directory under Wan2.1/2.2 models.")
    pipe.prompter.fetch_tokenizer(str(tokenizer_path))

    existing_names = list(getattr(pipe, "model_names", []))
    tracked = []
    for name in ["text_encoder", "image_encoder", "dit", "dit2", "vae", "motion_controller", "vace", "animate_adapter"]:
        if getattr(pipe, name, None) is not None:
            tracked.append(name)
    pipe.model_names = list(dict.fromkeys(existing_names + tracked))
    return pipe


def initialize_inference_pipeline(pipeline_kind: str, device_str: str, ckpt_type: str = "wan22") -> Any:
    if pipeline_kind == "v2v":
        return load_v2v_pipeline(device_str)
    if pipeline_kind == "i2v":
        return load_i2v_pipeline(device_str, ckpt_type=ckpt_type)
    raise ValueError(f"Unsupported pipeline_kind='{pipeline_kind}'")
