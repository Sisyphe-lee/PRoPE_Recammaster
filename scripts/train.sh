#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法: scripts/train.sh [选项]
常用参数：
  -y, --pipeline-type        训练模式 v2v / i2v
  -s, --dataset-path         逗号分隔的数据集路径
  -m, --metadata-path        逗号分隔的 metadata CSV（无则填 none）
  -S, --dataset-type         逗号分隔的数据集类型（multicam, rel10k ...）
      --dataset-weights      采样权重，与 dataset-type 数量一致
  其余参数参见脚本内默认值。
EOF
  exit 1
}

# -------- 默认值 --------
export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +'%m-%d-%H%M%S')}
export TOKENIZERS_PARALLELISM=false
DEFAULT_MULTICAM="/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train"
DEFAULT_REL10K="/nas/datasets/relestate10k"
DEFAULT_METADATA="$(pwd)/metadata/metadata_all.csv"
I2V_CKPT_TYPE="wan21"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,3,4,5,6,7"}
DEBUG_MODE=false
OUTPUT_DIR="$(pwd)/models/train"
RESUME_CHECKPOINT_PATH=""
WANDB_NAME="Exp07c"
DATASET_PATH="$DEFAULT_MULTICAM"
METADATA_PATH="$DEFAULT_METADATA"
DATASET_TYPE="multicam"
DATASET_WEIGHTS=""
GLOBAL_SEED=42
T_HIGHFREQ_RATIO=0.5
BATCH_SIZE=1
FRAME_DOWNSAMPLE_TO=0
USE_REAL_TEMPORAL_INDICES=false
USE_PHYSICAL_INDEX=false
PIPELINE_TYPE="v2v"
MODEL_BASE_PATH=""
VAL_SIZE=12
VAL_CHECK_INTERVAL_BATCHES=100
VAL_STEPS=20
VAL_GUIDANCE_SCALE=""
TENSOR_SUFFIX=""

DATASET_PATH_SET=false
METADATA_PATH_SET=false
DATASET_TYPE_SET=false
DATASET_WEIGHTS_SET=false

# -------- 解析参数 --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cuda-devices) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    -d|--debug) DEBUG_MODE=true; shift ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -R|--resume-checkpoint) RESUME_CHECKPOINT_PATH="$2"; shift 2 ;;
    -w|--wandb-name) WANDB_NAME="$2"; shift 2 ;;
    -s|--dataset-path) DATASET_PATH="$2"; DATASET_PATH_SET=true; shift 2 ;;
    -m|--metadata-path) METADATA_PATH="$2"; METADATA_PATH_SET=true; shift 2 ;;
    -S|--dataset-type) DATASET_TYPE="$2"; DATASET_TYPE_SET=true; shift 2 ;;
    --dataset-weights) DATASET_WEIGHTS="$2"; DATASET_WEIGHTS_SET=true; shift 2 ;;
    -g|--global-seed) GLOBAL_SEED="$2"; shift 2 ;;
    -t|--t-highfreq-ratio) T_HIGHFREQ_RATIO="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -F|--frame-downsample-to) FRAME_DOWNSAMPLE_TO="$2"; shift 2 ;;
    -T|--use-real-temporal-indices) USE_REAL_TEMPORAL_INDICES=true; shift ;;
    -P|--use-physical-index) USE_PHYSICAL_INDEX=true; shift ;;
    -y|--pipeline-type) PIPELINE_TYPE="$2"; shift 2 ;;
    -M|--model-base-path) MODEL_BASE_PATH="$2"; shift 2 ;;
    -v|--val-size) VAL_SIZE="$2"; shift 2 ;;
    -i|--val-check-interval-batches) VAL_CHECK_INTERVAL_BATCHES="$2"; shift 2 ;;
    -u|--ckpt_type) I2V_CKPT_TYPE="$2"; shift 2 ;;
    -h|--help) usage ;;
    
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

# -------- 启动命令 --------
if [[ -z "$MODEL_BASE_PATH" ]]; then
  if [[ "$PIPELINE_TYPE" == "i2v" ]]; then
    if [[ "$I2V_CKPT_TYPE" == "wan21" ]]; then
      MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
    else
      MODEL_BASE_PATH="models/Wan-AI/Wan2.2-TI2V-5B"
    fi
    VAL_GUIDANCE_SCALE="${VAL_GUIDANCE_SCALE:-5.0}"
    VAL_STEPS=10
  else
    MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
    VAL_GUIDANCE_SCALE="${VAL_GUIDANCE_SCALE:-1.0}"
  fi
fi

if [[ "$PIPELINE_TYPE" == "i2v" && "$DATASET_TYPE_SET" == false && "$DATASET_PATH_SET" == false ]]; then
  DATASET_TYPE="multicam,rel10k"
  DATASET_PATH="$DEFAULT_MULTICAM,$DEFAULT_REL10K"
  [[ "$METADATA_PATH_SET" == false ]] && METADATA_PATH="$DEFAULT_METADATA,none"
  if [[ "$DATASET_WEIGHTS_SET" == false || -z "$DATASET_WEIGHTS" ]]; then
    DATASET_WEIGHTS="0.7,0.3"
  fi
fi

shopt -s nullglob
diffusion_files=("$MODEL_BASE_PATH"/diffusion_pytorch_model*.safetensors)
shopt -u nullglob
if [[ ${#diffusion_files[@]} -eq 0 ]]; then
  echo "未找到扩散模型权重: $MODEL_BASE_PATH" >&2
  exit 1
fi
if [[ -f "$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors" ]]; then
  DIT_PATH="$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors"
else
  DIT_PATH=$(IFS=,; echo "${diffusion_files[*]}")
fi

if [[ -f "$MODEL_BASE_PATH/Wan2.1_VAE.pth" ]]; then
  VAE_PATH="$MODEL_BASE_PATH/Wan2.1_VAE.pth"
elif [[ -f "$MODEL_BASE_PATH/Wan2.2_VAE.pth" ]]; then
  VAE_PATH="$MODEL_BASE_PATH/Wan2.2_VAE.pth"
else
  echo "未找到 VAE 权重: $MODEL_BASE_PATH" >&2
  exit 1
fi

TEXT_ENCODER_PATH=""
TOKENIZER_PATH=""
if [[ -f "$MODEL_BASE_PATH/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
  TEXT_ENCODER_PATH="$MODEL_BASE_PATH/models_t5_umt5-xxl-enc-bf16.pth"
fi
if [[ -d "$MODEL_BASE_PATH/google/umt5-xxl" ]]; then
  TOKENIZER_PATH="$MODEL_BASE_PATH/google/umt5-xxl"
fi
if [[ "$PIPELINE_TYPE" == "i2v" ]]; then
  if [[ -z "$TEXT_ENCODER_PATH" ]]; then
    echo "i2v 模式需要 text encoder 权重 (models_t5_umt5-xxl-enc-bf16.pth)" >&2
    exit 1
  fi
  if [[ -z "$TOKENIZER_PATH" ]]; then
    echo "i2v 模式需要 tokenizer 目录 (google/umt5-xxl)" >&2
    exit 1
  fi
fi


mkdir -p "$OUTPUT_DIR"
if [[ "$DEBUG_MODE" == true ]]; then
  set -x
  EFFECTIVE_BATCH_SIZE=2
  EFFECTIVE_DATALOADER_WORKERS=0
else
  EFFECTIVE_BATCH_SIZE="$BATCH_SIZE"
  EFFECTIVE_DATALOADER_WORKERS=16
fi

CMD=(
  python -u -m src.train_recammaster
  --task train
  --dataset_path "$DATASET_PATH"
  --output_path "$OUTPUT_DIR"
  --dit_path "$DIT_PATH"
  --vae_path "$VAE_PATH"
  --steps_per_epoch 40000
  --max_epochs 100
  --learning_rate 1e-4
  --accumulate_grad_batches 1
  --use_gradient_checkpointing
  --dataloader_num_workers "$EFFECTIVE_DATALOADER_WORKERS"
  --batch_size "$EFFECTIVE_BATCH_SIZE"
  --global_seed "$GLOBAL_SEED"
  --val_steps "$VAL_STEPS"
  --val_size "$VAL_SIZE"
  --wandb_name "$WANDB_NAME"
  --val_check_interval_batches "$VAL_CHECK_INTERVAL_BATCHES"
  --training_strategy deepspeed_stage_2
  --distributed_timeout_seconds 1800
  --t_highfreq_ratio "$T_HIGHFREQ_RATIO"
  --frame_downsample_to "$FRAME_DOWNSAMPLE_TO"
  --pipeline_type "$PIPELINE_TYPE"
  --val_guidance_scale "$VAL_GUIDANCE_SCALE"
  --dataset_type "$DATASET_TYPE"
)
if [[ "$PIPELINE_TYPE" == "i2v" ]]; then
  CMD+=(--i2v_ckpt_type "$I2V_CKPT_TYPE")
  if [[ "$I2V_CKPT_TYPE" == "wan21" ]]; then
    TENSOR_SUFFIX=".tensors.pth"
  fi
fi
[[ -n "$TENSOR_SUFFIX" ]] && CMD+=(--tensor_suffix "$TENSOR_SUFFIX")
[[ -n "$TEXT_ENCODER_PATH" ]] && CMD+=(--text_encoder_path "$TEXT_ENCODER_PATH")
[[ -n "$TOKENIZER_PATH" ]] && CMD+=(--tokenizer_path "$TOKENIZER_PATH")
[[ -n "$METADATA_PATH" ]] && CMD+=(--metadata_path "$METADATA_PATH")
[[ -n "$DATASET_WEIGHTS" ]] && CMD+=(--dataset_weights "$DATASET_WEIGHTS")
[[ -n "$RESUME_CHECKPOINT_PATH" ]] && CMD+=(--resume_ckpt_path "$RESUME_CHECKPOINT_PATH")
[[ "$USE_REAL_TEMPORAL_INDICES" == true ]] && CMD+=(--use_real_temporal_indices)
[[ "$USE_PHYSICAL_INDEX" == true ]] && CMD+=(--use_physical_index)
[[ "$DEBUG_MODE" == true ]] && CMD+=(--debug)

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_BLOCKING_WAIT=1 \
NCCL_DEBUG=${NCCL_DEBUG:-ERROR} \
PYTHONUNBUFFERED=1 \
"${CMD[@]}"
