#!/bin/bash
set -x
# ReCamMaster Inference Script

# For clarity, define paths as variables
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="./results"
PYTHON_BIN="${PYTHON_BIN:-python}"
FRAME_DOWNSAMPLE_TO="${FRAME_DOWNSAMPLE_TO:-5}"
PIPELINE_TYPE="${PIPELINE_TYPE:-v2v}"
DATASET_PATH="${DATASET_PATH:-example_test_data}"
GPU_IDS="${GPU_IDS:-}"
GPU_IDS="${GPU_IDS// /}"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_ARGS=("$@")

if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
    NUM_GPUS="${#GPU_ARR[@]}"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "[INFO] 使用指定 GPU: ${GPU_IDS} (共 $NUM_GPUS 张)"
elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "[INFO] 使用多卡推理：$NUM_GPUS 张 GPU (默认按 0..N-1)"
else
    echo "[INFO] 使用单卡推理"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
else
    LAUNCHER=("$PYTHON_BIN")
fi

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
    "${LAUNCHER[@]}" src/inference_recammaster.py \
    --dataset_path "$DATASET_PATH" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --pipeline_type "$PIPELINE_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --cfg_scale 1.0 \
    --frame_downsample_to "$FRAME_DOWNSAMPLE_TO" \
    --dataloader_num_workers 1 \
    --camera_extrinsics_filename "camera_extrinsics_ori.json" \
    "${EXTRA_ARGS[@]}"
