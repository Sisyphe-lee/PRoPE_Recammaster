#!/bin/bash
set -x
# ReCamMaster Inference Script
# Similar to train.sh but for inference

# For clarity, define paths as variables
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="./results"
PYTHON_BIN="${PYTHON_BIN:-python}"
FRAME_DOWNSAMPLE_TO="${FRAME_DOWNSAMPLE_TO:-5}"
PIPELINE_TYPE="${PIPELINE_TYPE:-v2v}"
# Set the path to the checkpoint you want to use for inference
# For Wan2.1 original model:
WAN21_CHECKPOINT_PATH="./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
# For Wan2.2 original model:
WAN22_CHECKPOINT_PATH="./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors"
WAN21_RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt"
if [ "$PIPELINE_TYPE" = "i2v" ]; then
    CHECKPOINT_PATH="$WAN22_CHECKPOINT_PATH"
else
    if [ -n "$WAN21_RESUME_CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$WAN21_RESUME_CHECKPOINT_PATH"
    else
        CHECKPOINT_PATH="$WAN21_CHECKPOINT_PATH"
    fi
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "$PYTHON_BIN" src/inference_recammaster.py \
    --dataset_path "example_test_data" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --pipeline_type "$PIPELINE_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --cfg_scale 1.0 \
    --frame_downsample_to 0 \
    --dataloader_num_workers 1 \
    --camera_extrinsics_filename "camera_extrinsics_ori.json" \
    # --debug
