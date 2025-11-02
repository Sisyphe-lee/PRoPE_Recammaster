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

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a _recam_visible_gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#_recam_visible_gpu_array[@]}
else
    NUM_GPUS="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)"
fi

if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs, launching distributed inference."
    MASTER_PORT="${MASTER_PORT:-29500}"
    LAUNCH_CMD=("$PYTHON_BIN" "-m" "torch.distributed.run" "--standalone" "--nproc_per_node=$NUM_GPUS" "--master_port" "$MASTER_PORT")
else
    echo "Detected single GPU, running inference on one process."
    LAUNCH_CMD=("$PYTHON_BIN")
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "${LAUNCH_CMD[@]}" src/inference_recammaster.py \
    --dataset_path "example_test_data" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --pipeline_type "$PIPELINE_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --cfg_scale 1.0 \
    --frame_downsample_to 0 \
    --dataloader_num_workers 1 \
    --camera_extrinsics_filename "camera_extrinsics_ori.json"
