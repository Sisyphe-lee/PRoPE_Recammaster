"""
Train ReCamMaster With PRoPE Attention
"""
import argparse
import os
import random
import sys
from datetime import datetime
from typing import List, Optional

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist

from src.dataset import DatasetSpec, create_datasets
from src.lightning_trainer import LightningModelForTrain


def set_global_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=False,
        choices=["train"],
        help="Task. Only `train` is supported.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset. 可使用逗号分隔以匹配多个 dataset_type。",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default="v2v",
        choices=["v2v", "i2v"],
        help="Training mode: 'v2v' (wan2.1 T2V 1.5B ) or 'i2v' (Wan2.2 TI2V 5B).",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="multicam",
        help="数据集类型，可用逗号分隔（例如 multicam,re10k）。",
    )
    parser.add_argument(
        "--dataset_weights",
        type=str,
        default=None,
        help="逗号分隔的采样权重，需与 dataset_type 数量一致（默认等权）。",
    )

    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text_encoder.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to Wan tokenizer directory. Defaults to <text_encoder_dir>/google/umt5-xxl when omitted.",
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
        "--dataloader_num_workers",
        type=int,
        default=12,
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
        "--use_wandb",
        default=True,
        action="store_true",
        help="Whether to use WandB logger.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ReCamMaster",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="train",
        help="WandB run name.",
    )
    # Replaced metadata_file_name with metadata_path
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=False,
        default=None,
        help="Absolute path to the metadata CSV file (multicam 模式必填，re10k 可为空).",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=42, 
        help="Number of samples to use for validation (taken from the beginning of metadata).",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--global_seed",
        type=int,
        default=42,
        help="Global random seed for all random operations (training, validation, data loading, etc.)"
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=5,
        help="Number of inference steps used during validation sampling (default: 5)"
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=50,
        help="Number of training steps between validation runs (default: 50)"
    )
    parser.add_argument(
        "--val_check_interval_batches",
        type=int,
        default=None,
        help="Number of batches between validation runs (overrides val_check_interval if set)"
    )
    parser.add_argument(
        "--val_guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale for validation sampling. Default: 1.0 (v2v) / 5.0 (i2v)."
    )
    parser.add_argument(
        "--t_highfreq_ratio",
        type=float,
        default=0.0,
        help="Temporal low-frequency masking ratio for self-attention (passed via **kwargs to blocks)"
    )
    parser.add_argument(
        "--frame_downsample_to",
        type=int,
        default=0,
        help="Per-half frames to sample (two-halves scheme). Use 0 to disable downsampling (default: 0)"
    )
    parser.add_argument(
        "--use_real_temporal_indices",
        action="store_true",
        default=False,
        help="Use real temporal indices for RoPE instead of continuous indices (default: False)"
    )
    parser.add_argument(
        "-P", "--use_physical_index",
        action="store_true",
        default=False,
        help="Duplicate first-half temporal indices to second-half so tgt and cond do not share timestamps (applies regardless of downsampling)"
    )

    parser.add_argument(
        "--distributed_timeout_seconds",
        type=int,
        default=1800,
        help="Timeout in seconds for torch.distributed.init_process_group (default: 1800)"
    )

    args = parser.parse_args()
    return args


def _parse_dataset_types_arg(raw: str) -> List[str]:
    if not raw:
        return ["multicam"]
    types = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return types or ["multicam"]


def _expand_argument(value: Optional[str], count: int, arg_name: str, allow_empty: bool = False) -> List[Optional[str]]:
    if value is None:
        if allow_empty:
            return [None] * count
        if count == 1:
            raise ValueError(f"{arg_name} 必须提供。")
        raise ValueError(f"{arg_name} 需要提供 {count} 个值，用逗号分隔。")
    parts = [item.strip() for item in value.split(",")]
    if len(parts) == 1 and count > 1:
        parts = parts * count
    if len(parts) != count:
        raise ValueError(f"{arg_name} 的数量 ({len(parts)}) 与 dataset_type ({count}) 不一致。")
    results = []
    for part in parts:
        if allow_empty and part.lower() in {"", "none", "null"}:
            results.append(None)
        else:
            results.append(part)
    return results


def _parse_weights(value: Optional[str], count: int) -> List[float]:
    if not value:
        return [1.0] * count
    parts = [item.strip() for item in value.split(",")]
    if len(parts) == 1 and count > 1:
        parts = parts * count
    if len(parts) != count:
        raise ValueError(f"--dataset_weights 的数量 ({len(parts)}) 与 dataset_type ({count}) 不一致。")
    weights = []
    for part in parts:
        try:
            weights.append(max(float(part), 0.0))
        except ValueError as exc:
            raise ValueError(f"无法解析 dataset weight '{part}'") from exc
    return weights


def train(args):
    # Set global seed for reproducibility
    set_global_seed(args.global_seed)
    print(f"Global seed set to: {args.global_seed}")
    
    if args.debug:
        print("Debug mode is enabled.") 
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    dataset_types = _parse_dataset_types_arg(args.dataset_type)
    dataset_paths = _expand_argument(args.dataset_path, len(dataset_types), "--dataset_path")
    metadata_paths = _expand_argument(args.metadata_path, len(dataset_types), "--metadata_path", allow_empty=True)
    dataset_weights = _parse_weights(args.dataset_weights, len(dataset_types))

    dataset_specs: List[DatasetSpec] = []
    for dtype, root, meta, weight in zip(dataset_types, dataset_paths, metadata_paths, dataset_weights):
        if dtype == "multicam" and not meta:
            raise ValueError("MultiCam 数据集需要提供对应的 metadata CSV。")
        if root is None:
            raise ValueError(f"数据集 '{dtype}' 需要提供有效的 --dataset_path。")
        dataset_specs.append(
            DatasetSpec(
                name=dtype,
                root=root,
                metadata_path=meta,
                weight=weight,
            )
        )

    train_dataset, val_dataset = create_datasets(
        dataset_specs=dataset_specs,
        val_size=args.val_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.global_seed,
        image_size=(args.width, args.height),
        pipeline_type=args.pipeline_type,
    )

    def worker_init_fn(worker_id):
        """Initialize worker with deterministic seed"""
        worker_seed = args.global_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.global_seed)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn
    )
    


    time_str = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime('%m-%d-%H%M%S'))
    folder_name = f"{time_str}_{args.wandb_name}"
    latent_path = os.path.join("./training_log", folder_name, "video_debug")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        os.makedirs(latent_path, exist_ok=True)
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        text_encoder_path=args.text_encoder_path,
        tokenizer_path=args.tokenizer_path,
        latent_path=latent_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        global_seed=args.global_seed,
        val_steps=args.val_steps,
        t_highfreq_ratio=getattr(args, 't_highfreq_ratio', 0.0),
        frame_downsample_to=getattr(args, 'frame_downsample_to', 5),
        use_real_temporal_indices=getattr(args, 'use_real_temporal_indices', False),
        use_physical_index=getattr(args, 'use_physical_index', False),
        pipeline_type=getattr(args, 'pipeline_type', 'v2v'),
        val_guidance_scale=getattr(args, 'val_guidance_scale', None),
    )
    
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_name = f"{time_str}_{args.wandb_name}"
        run_dir = os.path.join("./training_log", wandb_name)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            os.makedirs(run_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            ## project=args.wandb_project+wandb_name
            project=f"{args.wandb_project}_{args.wandb_name[:5]}",
            name=wandb_name,
            id=wandb_name,
            config=vars(args),
            save_dir=run_dir,
        )
        logger = [wandb_logger]
    else:
        logger = None
        run_dir = args.output_path
    # Ensure distributed init uses a generous timeout to avoid premature aborts on long steps
    os.environ["TORCH_DIST_INIT_TIMEOUT"] = str(getattr(args, "distributed_timeout_seconds", 1800))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval_batches if args.val_check_interval_batches is not None else args.val_check_interval,  # Use batch-based or step-based validation frequency
        limit_val_batches=len(val_dataset),
        num_sanity_val_steps=0,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=-1,
                dirpath=os.path.join(run_dir, "checkpoints"),
                filename="{epoch}-{step}"
            ),
        ],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=0.05,
    )
    # Run an initial validation at step 0 for debugging/baseline
    # trainer.validate(model, val_dataloader)
    
    # Fit the model
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    try:
        train(args)
    except Exception as e:
        print(f"Fatal error encountered: {e}", flush=True)
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as _:
            pass
        sys.exit(1)
