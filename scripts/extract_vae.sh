#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0,1}"

DATASET_PATH=${1:-/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f24_aperture5}
METADATA_PATH="/data1/lcy/projects/ReCamMaster/metadata/metadata_f24_aperture5.csv"

PIPELINE_TYPE=${PIPELINE_TYPE:-v2v}

if [[ "$PIPELINE_TYPE" == "i2v" ]]; then
  MODEL_BASE_PATH=${MODEL_BASE_PATH:-models/Wan-AI/Wan2.2-TI2V-5B}
  DEFAULT_TEXT_ENCODER_PATH="$MODEL_BASE_PATH/models_t5_umt5-xxl-enc-bf16.pth"
  DEFAULT_VAE_PATH="$MODEL_BASE_PATH/Wan2.2_VAE.pth"
  DEFAULT_TENSOR_SUFFIX=".wan22.tensors.pth"
else
  MODEL_BASE_PATH=${MODEL_BASE_PATH:-models/Wan-AI/Wan2.1-T2V-1.3B}
  DEFAULT_TEXT_ENCODER_PATH="$MODEL_BASE_PATH/models_t5_umt5-xxl-enc-bf16.pth"
  DEFAULT_VAE_PATH="$MODEL_BASE_PATH/Wan2.1_VAE.pth"
  DEFAULT_TENSOR_SUFFIX=".tensors.pth"
fi

TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-$DEFAULT_TEXT_ENCODER_PATH}
VAE_PATH=${VAE_PATH:-$DEFAULT_VAE_PATH}
TENSOR_SUFFIX=${TENSOR_SUFFIX:-$DEFAULT_TENSOR_SUFFIX}

echo "Running VAE feature extraction on ${DATASET_PATH} (metadata: ${METADATA_PATH})"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u -m src.vae_feature \
  --task data_process \
  --dataset_path "${DATASET_PATH}" \
  --metadata_path "${METADATA_PATH}" \
  --output_path ./models \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --vae_path "${VAE_PATH}" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --dataloader_num_workers 2 \
  --pipeline_type "${PIPELINE_TYPE}" \
  --tensor_suffix "${TENSOR_SUFFIX}"
