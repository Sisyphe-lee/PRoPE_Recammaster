"""
Unified inference pipeline supporting multiple datasets and pipeline modes.

Currently mirrors the behaviour of:
  * src/inference_recammaster.py (example dataset, v2v)
  * evaluation/render_pointodyssey.py (PointOdyssey dataset, v2v)

Future datasets or inference modes (e.g. i2v) can extend the registry-based
factory hooks defined in this module.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THIRD_PARTY_DIFFSYNTH = PROJECT_ROOT / "third_party" / "DiffSynth-Studio"
if str(THIRD_PARTY_DIFFSYNTH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIFFSYNTH))
sys.modules.setdefault("inference_unified", sys.modules[__name__])
sys.modules.setdefault("src.inference_unified", sys.modules[__name__])

from diffsynth import save_video  # noqa: E402
from base_handler import DATASET_REGISTRY, PIPELINE_REGISTRY, InferenceSample, TargetSpec, NEGATIVE_PROMPT, ensure_homogeneous
from pipeline_loader import initialize_inference_pipeline
# Import handlers to populate registries
import v2v_handler  # noqa: F401
import i2v_handler  # noqa: F401

# ---------------------------------------------------------------------------
# Target pose helpers
# ---------------------------------------------------------------------------

def load_target_pose_directory(target_dir: Path) -> List[TargetSpec]:
    """Load every .npz under target_dir into TargetSpec entries."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Target pose directory not found: {target_dir}")
    specs: List[TargetSpec] = []
    for npz_path in sorted(target_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "data" not in data or "inds" not in data:
                raise ValueError(f"Invalid target pose file {npz_path}: expect keys 'data' and 'inds'")
            raw_pose = data["data"].astype(np.float32)
            raw_inds = data["inds"].astype(np.int64)
        specs.append(
            TargetSpec(
                name=npz_path.stem,
                raw_pose=raw_pose,
                raw_inds=raw_inds,
                metadata={"path": str(npz_path)},
            )
        )
    if not specs:
        raise RuntimeError(f"No .npz pose files found in {target_dir}")
    return specs


def load_dit_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], rank: int) -> None:
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

    load_msg = model.load_state_dict(compatible_state, strict=True)
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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed_environment() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return distributed, world_size, rank, local_rank, device


def broadcast_output_directory(base_dir: Path, distributed: bool, rank: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if distributed:
        payload = [timestamp if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        timestamp = payload[0]
    output_dir = base_dir / timestamp
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    return output_dir


def cleanup_distributed_environment(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()




# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_kv_options(options: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            raise ValueError(f"Option '{item}' must be in key=value format")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ReCamMaster inference")
    parser.add_argument("--dataset_kind", type=str, required=True, choices=["example", "example_i2v", "pointodyssey", "sdg_v2v", "sdg_i2v"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--dataset_option", action="append", default=[], help="Additional dataset options (key=value)")
    parser.add_argument("--target_pose_dir", type=str, required=True, help="Directory containing target pose .npz files")
    parser.add_argument("--pipeline_kind", type=str, default="v2v", choices=["v2v", "i2v"], help="Inference pipeline mode")
    parser.add_argument(
        "--i2v_ckpt_type",
        type=str,
        default="wan22",
        choices=["wan21", "wan22"],
        help="i2v 模式下选择 wan21 或 wan22 权重/模型。",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint that overrides the base pipeline weights",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation/example_eval", help="Directory to save outputs")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--frame_downsample_to", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Samples per batch (currently only 1 supported)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_options = parse_kv_options(args.dataset_option)
    dataset_path = Path(args.dataset_path)
    target_pose_dir = Path(args.target_pose_dir)

    distributed, world_size, rank, _, device = setup_distributed_environment()

    if args.debug and rank == 0:
        print("Debug mode is enabled.")
        import debugpy  # type: ignore

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")
    elif args.debug and rank != 0:
        print(f"[rank {rank}] Debug mode requested but only rank 0 enters debug session.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_cls = DATASET_REGISTRY.get(args.dataset_kind)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset_kind={args.dataset_kind}")
    dataset = dataset_cls(dataset_path=dataset_path, options=dataset_options)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; please check dataset_path and options.")

    target_specs = load_target_pose_directory(target_pose_dir)

    if args.batch_size != 1:
        raise NotImplementedError("Only batch_size=1 is supported currently.")

    enforced_downsample = 0
    if args.frame_downsample_to != 0 and rank == 0:
        print(f"[info] frame_downsample_to={args.frame_downsample_to} ignored; enforcing 0 for inference stability.")

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

    def collate_single(batch: List[InferenceSample]) -> InferenceSample:
        if len(batch) != 1:
            raise ValueError("batch_size other than 1 is not supported")
        return batch[0]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_single,
    )

    if sampler is not None:
        sampler.set_epoch(0)

    device_str = device.type if device.type == "cpu" else f"cuda:{device.index}"
    ckpt_type = args.i2v_ckpt_type if args.pipeline_kind == "i2v" else "wan21"
    pipe = initialize_inference_pipeline(args.pipeline_kind, device_str, ckpt_type=ckpt_type)

    if rank == 0:
        print(f"Using device: {device_str} | world_size={world_size}")

    handler_cls = PIPELINE_REGISTRY.get(args.pipeline_kind)
    if handler_cls is None:
        raise ValueError(f"No inference handler registered for pipeline_kind={args.pipeline_kind}")
    handler = handler_cls(device=device, dtype=torch.bfloat16, global_opts={"cfg_scale": args.cfg_scale, "debug_pose": args.debug})
    handler.apply_checkpoint(pipe, args.ckpt_path, rank)

    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    pipe.eval()

    base_output_dir = Path(args.output_dir)
    output_dir = broadcast_output_directory(base_output_dir, distributed, rank)

    if rank == 0:
        print(f"[info] Saving outputs to: {output_dir}")

    for sample_idx, sample in enumerate(dataloader):
        source_video = sample.video.unsqueeze(0).to(device)
        prompt_text = sample.text
        sample_stem = sample.metadata.get("stem", f"sample_{sample_idx:04d}")

        for target in target_specs:
            output_stem = f"{sample_stem}_{target.name}"
            video_path = output_dir / f"{output_stem}.mp4"
            pose_path = video_path.with_suffix(".npz")
            
            # if target.name != 'cam01':
            #     continue

            if video_path.exists():
                if rank == 0:
                    print(f"[info] Skip existing output: {video_path.name}")
                continue

            prepared = handler.build_inputs(sample, target, source_video=source_video)

            pipe_kwargs = dict(
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
                cfg_scale=args.cfg_scale,
                frame_downsample_to=enforced_downsample,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                tiled=True,
            )
            pipe_kwargs.update(prepared.pipe_kwargs)

            with torch.no_grad():
                video = handler.run_inference(pipe, prepared, prompt_text, pipe_kwargs)

            save_video(video, str(video_path), fps=30, quality=5)
            np.savez(
                pose_path,
                data=prepared.target_rel_w2c.astype(np.float32),
                inds=prepared.cam_indices.astype(np.int64),
            )

        if rank == 0:
            print(f"[info] Completed sample {sample_idx + 1}/{len(dataloader)} -> {sample_stem}")

    cleanup_distributed_environment(distributed)


if __name__ == "__main__":
    main()
