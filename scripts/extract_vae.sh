#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=6,7}"

DATASET_PATH=${1:-/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f24_aperture5}
METADATA_PATH="/data1/lcy/projects/ReCamMaster/metadata/metadata_f24_aperture5.csv"

echo "Running VAE feature extraction on ${DATASET_PATH} (metadata: ${METADATA_PATH})"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u -m src.vae_feature \
  --task data_process \
  --dataset_path "${DATASET_PATH}" \
  --metadata_path "${METADATA_PATH}" \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --dataloader_num_workers 2
