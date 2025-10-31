#!/usr/bin/env bash
set -euo pipefail

echo "Train ReCamMaster With PRoPE Attention $(which python), 当前目录: $(pwd)"

usage() {
  cat <<'EOF'
用法: scripts/train.sh [选项]
  -c, --cuda-devices LIST        设置 CUDA_VISIBLE_DEVICES (默认: 2,3,4,5,6,7)
  -d, --debug                    启用调试模式 (batch=1, workers=0)
  -o, --output-dir PATH          训练输出目录 (默认: ./models/train)
  -R, --resume-checkpoint PATH   Lightning checkpoint 路径 (可选)
  -w, --wandb-name NAME          WandB 实验名称 (默认: Exp07c)
  -s, --dataset-path PATH        数据集路径 (默认: /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train)
  -m, --metadata-path PATH       元数据 CSV (默认: ./metadata/metadata_all.csv)
  -g, --global-seed SEED         全局随机种子 (默认: 42)
  -t, --t-highfreq-ratio VALUE   自注意高频屏蔽比例 (默认: 0.5)
  -b, --batch-size VALUE         每卡 batch size (默认: 1)
  -F, --frame-downsample-to N    每半段采样帧数 (默认: 0 = 不降采样)
  -T, --use-real-temporal-indices 使用真实帧索引计算 RoPE
  -P, --use-physical-index       物理索引模式
  -y, --pipeline-type TYPE       训练管线类型 v2v / i2v (默认: v2v)
  -M, --model-base-path PATH     基础权重根目录 (默认按 pipeline 推断)
  -v, --val-size N               验证集 batch 数 (默认: 12)
  -i, --val-check-interval-batches N  每多少个 batch 运行一次验证 (默认: 50)
  -h, --help                     显示帮助
EOF
  exit 1
}

export RUN_TIMESTAMP=$(date +'%m-%d-%H%M%S')

# 默认参数
CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
DEBUG_MODE=false
OUTPUT_DIR="$(pwd)/models/train"
RESUME_CHECKPOINT_PATH=""
WANDB_NAME="Exp07c"
DATASET_PATH="/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train"
METADATA_PATH="$(pwd)/metadata/metadata_all.csv"
GLOBAL_SEED=42
T_HIGHFREQ_RATIO=0.5
BATCH_SIZE=1
FRAME_DOWNSAMPLE_TO=0
USE_REAL_TEMPORAL_INDICES=false
USE_PHYSICAL_INDEX=false
PIPELINE_TYPE="v2v"
DATALOADER_WORKERS_DEFAULT=36
MODEL_BASE_PATH=""
VAL_SIZE=12
VAL_CHECK_INTERVAL_BATCHES=100
VAL_STEPS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cuda-devices)
      CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --cuda-devices=*)
      CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
    -d|--debug)
      DEBUG_MODE=true; shift ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"; shift ;;
    -R|--resume-checkpoint)
      RESUME_CHECKPOINT_PATH="$2"; shift 2 ;;
    --resume-checkpoint=*)
      RESUME_CHECKPOINT_PATH="${1#*=}"; shift ;;
    -w|--wandb-name)
      WANDB_NAME="$2"; shift 2 ;;
    --wandb-name=*)
      WANDB_NAME="${1#*=}"; shift ;;
    -s|--dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --dataset-path=*)
      DATASET_PATH="${1#*=}"; shift ;;
    -m|--metadata-path)
      METADATA_PATH="$2"; shift 2 ;;
    --metadata-path=*)
      METADATA_PATH="${1#*=}"; shift ;;
    -g|--global-seed)
      GLOBAL_SEED="$2"; shift 2 ;;
    --global-seed=*)
      GLOBAL_SEED="${1#*=}"; shift ;;
    -t|--t-highfreq-ratio)
      T_HIGHFREQ_RATIO="$2"; shift 2 ;;
    --t-highfreq-ratio=*)
      T_HIGHFREQ_RATIO="${1#*=}"; shift ;;
    -b|--batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"; shift ;;
    -F|--frame-downsample-to)
      FRAME_DOWNSAMPLE_TO="$2"; shift 2 ;;
    --frame-downsample-to=*)
      FRAME_DOWNSAMPLE_TO="${1#*=}"; shift ;;
    -T|--use-real-temporal-indices)
      USE_REAL_TEMPORAL_INDICES=true; shift ;;
    --use-real-temporal-indices=*)
      USE_REAL_TEMPORAL_INDICES="${1#*=}"; shift ;;
    -P|--use-physical-index)
      USE_PHYSICAL_INDEX=true; shift ;;
    --use-physical-index=*)
      USE_PHYSICAL_INDEX="${1#*=}"; shift ;;
    -y|--pipeline-type)
      PIPELINE_TYPE="$2"; shift 2 ;;
    --pipeline-type=*)
      PIPELINE_TYPE="${1#*=}"; shift ;;
    -M|--model-base-path)
      MODEL_BASE_PATH="$2"; shift 2 ;;
    --model-base-path=*)
      MODEL_BASE_PATH="${1#*=}"; shift ;;
    -v|--val-size)
      VAL_SIZE="$2"; shift 2 ;;
    -v=*)
      VAL_SIZE="${1#*=}"; shift ;;
    --val-size=*)
      VAL_SIZE="${1#*=}"; shift ;;
    -i|--val-check-interval-batches)
      VAL_CHECK_INTERVAL_BATCHES="$2"; shift 2 ;;
    -i=*)
      VAL_CHECK_INTERVAL_BATCHES="${1#*=}"; shift ;;
    --val-check-interval-batches=*)
      VAL_CHECK_INTERVAL_BATCHES="${1#*=}"; shift ;;

    -h|--help)
      usage ;;
    *)
      echo "未知参数: $1" >&2
      usage ;;
  esac
done

if [[ -z "$MODEL_BASE_PATH" ]]; then
  if [[ "$PIPELINE_TYPE" == "i2v" ]]; then
    MODEL_BASE_PATH="models/Wan-AI/Wan2.2-TI2V-5B"
    VAL_GUIDANCE_SCALE="${VAL_GUIDANCE_SCALE:-5.0}"
    VAL_STEPS=40
  else
    MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
    VAL_GUIDANCE_SCALE="${VAL_GUIDANCE_SCALE:-1.0}"
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

mkdir -p "$OUTPUT_DIR"

if [[ "$DEBUG_MODE" == true ]]; then
  set -x
  EFFECTIVE_BATCH_SIZE=1
  EFFECTIVE_DATALOADER_WORKERS=0
else
  EFFECTIVE_BATCH_SIZE="$BATCH_SIZE"
  EFFECTIVE_DATALOADER_WORKERS="$DATALOADER_WORKERS_DEFAULT"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}

CMD=(
  python -u -m src.train_recammaster
  --task train
  --dataset_path "$DATASET_PATH"
  --output_path "$OUTPUT_DIR"
  --dit_path "$DIT_PATH"
  --vae_path "$VAE_PATH"
  --steps_per_epoch 10000
  --max_epochs 100
  --learning_rate 1e-4
  --accumulate_grad_batches 1
  --use_gradient_checkpointing
  --dataloader_num_workers "$EFFECTIVE_DATALOADER_WORKERS"
  --batch_size "$EFFECTIVE_BATCH_SIZE"
  --global_seed "$GLOBAL_SEED"
  --val_steps "$VAL_STEPS"
  --val_size "$VAL_SIZE"
  --metadata_path "$METADATA_PATH"
  --wandb_name "$WANDB_NAME"
  --val_check_interval_batches "$VAL_CHECK_INTERVAL_BATCHES"
  --training_strategy deepspeed_stage_2
  --distributed_timeout_seconds 1800
  --t_highfreq_ratio "$T_HIGHFREQ_RATIO"
  --frame_downsample_to "$FRAME_DOWNSAMPLE_TO"
  --pipeline_type "$PIPELINE_TYPE"
  --height 704
  --width 1280
  --val_guidance_scale "$VAL_GUIDANCE_SCALE"
)

if [[ -n "$RESUME_CHECKPOINT_PATH" ]]; then
  CMD+=(--resume_ckpt_path "$RESUME_CHECKPOINT_PATH")
fi
if [[ "$USE_REAL_TEMPORAL_INDICES" == true ]]; then
  CMD+=(--use_real_temporal_indices)
fi
if [[ "$USE_PHYSICAL_INDEX" == true ]]; then
  CMD+=(--use_physical_index)
fi
if [[ "$DEBUG_MODE" == true ]]; then
  CMD+=(--debug)
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
"${CMD[@]}"
