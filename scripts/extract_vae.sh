#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0,1}"

usage() {
  cat <<'EOF'
用法: scripts/extract_vae.sh [选项] [DATASET_PATH]

常用示例：
  # MultiCam：读取 metadata CSV
  scripts/extract_vae.sh -s /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f24_aperture5 \
    -m metadata/metadata_f24_aperture5.csv -y v2v

  # RealEstate10k：直接读取 .torch，输出整理后的资产
  scripts/extract_vae.sh -t re10k -s /nas/datasets/re10k/train \
    --re10k-output /nas/datasets/relestate10k/train -y i2v

可选参数：
  -s, --dataset-path PATH       输入数据根目录（默认: MultiCam f24）
  -m, --metadata-path PATH      MultiCam 模式所需的 metadata CSV（默认: metadata/metadata_f24_aperture5.csv）
  -t, --dataset-type TYPE       multicam 或 re10k（默认: multicam）
      --re10k-output PATH       re10k 模式整理后写入的根目录（train/test 子目录）
      --re10k-index  PATH       可选 index.json 覆盖路径
  -y, --pipeline-type TYPE      v2v / i2v（默认: v2v）
      --no-resume               不跳过已存在的 latent，强制重新生成
  -h, --help                    显示本帮助

也可通过环境变量覆盖 TEXT_ENCODER_PATH / VAE_PATH / MODEL_BASE_PATH / TENSOR_SUFFIX / DATASET_TYPE 等。
EOF
  exit 1
}

DATASET_PATH="/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f24_aperture5"
METADATA_PATH="/data1/lcy/projects/ReCamMaster/metadata/metadata_f24_aperture5.csv"
DATASET_TYPE=${DATASET_TYPE:-multicam}
RE10K_OUTPUT_PATH=${RE10K_OUTPUT_PATH:-}
RE10K_INDEX_PATH=${RE10K_INDEX_PATH:-}
PIPELINE_TYPE=${PIPELINE_TYPE:-v2v}
NO_RESUME=${NO_RESUME:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    -s=*|--dataset-path=*)
      DATASET_PATH="${1#*=}"; shift ;;
    -m|--metadata-path)
      METADATA_PATH="$2"; shift 2 ;;
    -m=*|--metadata-path=*)
      METADATA_PATH="${1#*=}"; shift ;;
    -t|--dataset-type)
      DATASET_TYPE="$2"; shift 2 ;;
    -t=*|--dataset-type=*)
      DATASET_TYPE="${1#*=}"; shift ;;
    --re10k-output)
      RE10K_OUTPUT_PATH="$2"; shift 2 ;;
    --re10k-output=*)
      RE10K_OUTPUT_PATH="${1#*=}"; shift ;;
    --re10k-index)
      RE10K_INDEX_PATH="$2"; shift 2 ;;
    --re10k-index=*)
      RE10K_INDEX_PATH="${1#*=}"; shift ;;
    -y|--pipeline-type)
      PIPELINE_TYPE="$2"; shift 2 ;;
    -y=*|--pipeline-type=*)
      PIPELINE_TYPE="${1#*=}"; shift ;;
    --no-resume)
      NO_RESUME=1; shift ;;
    -h|--help)
      usage ;;
    -*)
      echo "未知参数: $1" >&2
      usage ;;
    *)
      # 兼容旧的“位置参数=dataset_path”写法
      DATASET_PATH="$1"; shift ;;
  esac
done


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

echo "Running VAE feature extraction on ${DATASET_PATH} (dataset_type=${DATASET_TYPE}, pipeline=${PIPELINE_TYPE})"

CMD=(
  python -u -m src.vae_feature
  --dataset_path "${DATASET_PATH}"
  --output_path ./models
  --text_encoder_path "${TEXT_ENCODER_PATH}"
  --vae_path "${VAE_PATH}"
  --tiled
  --num_frames 81
  --height 480
  --width 832
  --dataloader_num_workers 24
  --pipeline_type "${PIPELINE_TYPE}"
  --tensor_suffix "${TENSOR_SUFFIX}"
  --dataset_type "${DATASET_TYPE}"
)

if [[ "$DATASET_TYPE" == "re10k" ]]; then
  if [[ -z "$RE10K_OUTPUT_PATH" ]]; then
    echo "RE10K_OUTPUT_PATH 不能为空 (dataset_type=re10k)" >&2
    exit 1
  fi
  CMD+=(--re10k_output_path "${RE10K_OUTPUT_PATH}")
  if [[ -n "$RE10K_INDEX_PATH" ]]; then
    CMD+=(--re10k_index_path "${RE10K_INDEX_PATH}")
  fi
else
  CMD+=(--metadata_path "${METADATA_PATH}")
fi

if [[ "${NO_RESUME}" == "1" ]]; then
  CMD+=(--no_resume)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
