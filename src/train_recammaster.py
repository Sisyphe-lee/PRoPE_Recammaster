"""
Train ReCamMaster With PRoPE Attention
"""
import argparse
import os
import random
import sys
from datetime import datetime

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist

from src.dataset import create_datasets
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
        help="The path of the Dataset.",
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
        help="Training mode: 'v2v' (ReCamMaster) or 'i2v' (Wan2.2 image-to-video).",
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
        required=True,
        help="Absolute path to the metadata CSV file.",
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
    # Create datasets using the new create_datasets function

    train_dataset, val_dataset = create_datasets(
        metadata_path=args.metadata_path,
        val_size=args.val_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.global_seed,
        dataset_root=args.dataset_path,
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
    latent_path = os.path.join("./wandb", folder_name, "video_debug")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        os.makedirs(latent_path, exist_ok=True)
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
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
    )
    
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_name = f"{time_str}_{args.wandb_name}"
        run_dir = os.path.join("./wandb", wandb_name)
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
    trainer.validate(model, val_dataloader)
    
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
