#!/usr/bin/env bash
# Optional debug trace will be enabled later based on flag
echo "Train ReCamMaster With PRoPE Attention $(which python), Current directory: $(pwd)"
# Color definitions
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --cuda-devices CUDA_DEVICES    CUDA visible devices (default: 0,1,2,3,4,5,6,7)"
    echo "  -d, --debug                        Enable debug mode (default: disabled)"
    echo "  -o, --output-dir OUTPUT_DIR        Output directory (default: ./models/train)"
    echo "  -r, --recammaster-checkpoint PATH  ReCamMaster checkpoint path (default: /data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt)"
    echo "  -R, --wan21-resume-checkpoint PATH Wan2.1 resume checkpoint path (optional; used only when provided and ckpt_type=wan21)"
    echo "  -w, --wandb-name WANDB_NAME        Wandb experiment name (default: Exp07c)"
    echo "  -s, --dataset-path DATASET_PATH    Dataset path (default: /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train)"
    echo "  -u, --select-random-latents        Randomly select frames per half (keep first/last, randomize the rest with global seed). Default: disabled"
    echo "  -m, --metadata-path METADATA_PATH  Metadata file path (default: ./metadata/metadata_all.csv)"
    echo "  -g, --global-seed SEED             Global seed (default: 42)"
    echo "  -t, --t-highfreq-ratio RATIO      Temporal low-frequency masking ratio for self-attn (default: 0.0)"
    echo "  -b, --batch-size BATCH_SIZE        Training batch size (default: 1)"
    echo "  -F, --frame-downsample-to N       Per-half frames to sample (two-halves). Default: 0 (disabled); e.g., 5 means each half picks 5 frames"
    echo "  -T, --use-real-temporal-indices   Use real temporal indices for RoPE instead of continuous indices (default: false)"
    echo "                                    When enabled, RoPE uses actual frame positions instead of [0,1,2,3...]"
    echo "  -P, --use-physical-index          Duplicate first-half temporal indices to second-half so tgt/cond do not share timestamps (default: false)"
    echo "  -y, --pipeline-type TYPE          Pipeline type for training (recammaster or wan; default: recammaster)"
    echo "  -h, --help                         Show this help message"
    exit 1
}

export RUN_TIMESTAMP=$(date +'%m-%d-%H%M%S')

# Default values
CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
DEBUG_FLAG=""
OUTPUT_DIR="$(pwd)/models/train"
METADATA_PATH="$(pwd)/metadata/metadata_all.csv"
RECAMMASTER_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt"
WANDB_NAME="Exp07c"
DATASET_PATH="/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train"
WAN21_RESUME_CHECKPOINT_PATH=""
GLOBAL_SEED="42"
T_HIGHFREQ_RATIO="0.5"
FRAME_DOWNSAMPLE_TO="0"
BATCH_SIZE="1"
DATALOADER_DEFAULT=36
USE_REAL_TEMPORAL_INDICES="false"
USE_PHYSICAL_INDEX="false"
SELECT_RANDOM_LATENTS="false"
PIPELINE_TYPE="recammaster"
MODEL_BASE_PATH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cuda-devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -c=*)
            CUDA_VISIBLE_DEVICES="${1#*=}"
            shift
            ;;
        -c?*)
            CUDA_VISIBLE_DEVICES="${1#-c}"
            shift
            ;;
        --cuda-devices=*)
            CUDA_VISIBLE_DEVICES="${1#*=}"
            shift
            ;;
        -d|--debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --no-debug)
            DEBUG_FLAG=""
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--recammaster-checkpoint)
            RECAMMASTER_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -R|--wan21-resume-checkpoint)
            WAN21_RESUME_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --wan21-resume-checkpoint=*)
            WAN21_RESUME_CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        -w|--wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        -s|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--metadata-path)
            METADATA_PATH="$2"
            shift 2
            ;;
        -g|--global-seed)
            GLOBAL_SEED="$2"
            shift 2
            ;;
        -t|--t-highfreq-ratio)
            T_HIGHFREQ_RATIO="$2"
            shift 2
            ;;
        -t=*)
            T_HIGHFREQ_RATIO="${1#*=}"
            shift
            ;;
        --t-highfreq-ratio=*)
            T_HIGHFREQ_RATIO="${1#*=}"
            shift
            ;;
        -y|--pipeline-type)
            PIPELINE_TYPE="$2"
            shift 2
            ;;
        --pipeline-type=*)
            PIPELINE_TYPE="${1#*=}"
            shift
            ;;
        -M|--model-base-path)
            MODEL_BASE_PATH="$2"
            shift 2
            ;;
        --model-base-path=*)
            MODEL_BASE_PATH="${1#*=}"
            shift
            ;;
        -F|--frame-downsample-to)
            FRAME_DOWNSAMPLE_TO="$2"
            shift 2
            ;;
        -F=*)
            FRAME_DOWNSAMPLE_TO="${1#*=}"
            shift
            ;;
        --frame-downsample-to=*)
            FRAME_DOWNSAMPLE_TO="${1#*=}"
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -b=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        -T|--use-real-temporal-indices)
            USE_REAL_TEMPORAL_INDICES="true"
            shift
            ;;
        --use-real-temporal-indices=*)
            USE_REAL_TEMPORAL_INDICES="${1#*=}"
            shift
            ;;
        -P|--use-physical-index)
            USE_PHYSICAL_INDEX="true"
            shift
            ;;
        --use-physical-index=*)
            USE_PHYSICAL_INDEX="${1#*=}"
            shift
            ;;
        -u|--select-random-latents)
            SELECT_RANDOM_LATENTS="true"
            shift
            ;;
        --select-random-latents=*)
            SELECT_RANDOM_LATENTS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# For clarity, define paths as variables
if [[ -z "$MODEL_BASE_PATH" ]]; then
    if [[ "$PIPELINE_TYPE" == "wan" ]]; then
        MODEL_BASE_PATH="models/Wan-AI/Wan2.2-TI2V-5B"
    else
        MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
    fi
fi

# Resolve diffusion and VAE weights under the chosen base path
DIT_PATH="$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors"
if [[ ! -f "$DIT_PATH" ]]; then
    mapfile -t __diff_shards < <(ls "$MODEL_BASE_PATH"/diffusion_pytorch_model-*.safetensors 2>/dev/null | sort)
    if [[ ${#__diff_shards[@]} -gt 0 ]]; then
        DIT_PATH="$(IFS=,; echo "${__diff_shards[*]}")"
    else
        DIT_PATH=""
    fi
fi

if [[ -f "$MODEL_BASE_PATH/Wan2.1_VAE.pth" ]]; then
    VAE_PATH="$MODEL_BASE_PATH/Wan2.1_VAE.pth"
elif [[ -f "$MODEL_BASE_PATH/Wan2.2_VAE.pth" ]]; then
    VAE_PATH="$MODEL_BASE_PATH/Wan2.2_VAE.pth"
else
    VAE_PATH=""
fi

if [[ -z "$DIT_PATH" ]]; then
    echo -e "${RED}Error:${NC} diffusion weights not found under $MODEL_BASE_PATH" >&2
    exit 1
fi
if [[ -z "$VAE_PATH" ]]; then
    echo -e "${RED}Error:${NC} VAE weights not found under $MODEL_BASE_PATH" >&2
    exit 1
fi
# Set the path to the checkpoint you want to resume from.
# If you want to train from scratch, you can remove the --resume_ckpt_path line.
# RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/train/wandb/ReCamMaster/08-21-151648_exp02b/checkpoints/step1079.ckpt"

# Choose checkpoint type: defaults depend on pipeline
RESUME_CHECKPOINT_PATH=""
CHECKPOINT_TYPE="wan21"

if [[ "$PIPELINE_TYPE" == "wan" ]]; then
    ENABLE_CAM_LAYERS=""
else
    # For Wan2.1 original model:
    WAN21_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    if [[ "$CHECKPOINT_TYPE" == "wan21" ]]; then
        if [[ -n "$WAN21_RESUME_CHECKPOINT_PATH" ]]; then
            RESUME_CHECKPOINT_PATH="$WAN21_RESUME_CHECKPOINT_PATH"
        else
            RESUME_CHECKPOINT_PATH="$WAN21_CHECKPOINT_PATH"
        fi
        ENABLE_CAM_LAYERS=""
    else
        RESUME_CHECKPOINT_PATH="$RECAMMASTER_CHECKPOINT_PATH"
        ENABLE_CAM_LAYERS="--enable_cam_layers"
    fi
fi

if [ -n "$DEBUG_FLAG" ]; then
    DEBUG_BOOL=true
else
    DEBUG_BOOL=false
fi

# Wan pipeline does not support camera layer injection
if [[ "${PIPELINE_TYPE}" != "recammaster" ]]; then
    ENABLE_CAM_LAYERS=""
fi

# Build log file and redirect all outputs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
BASENAME_DATASET=$(basename "$DATASET_PATH")
SANITIZED_WANDB_NAME=$(echo "$WANDB_NAME" | tr -cs 'A-Za-z0-9._-' '-')
SANITIZED_DATASET=$(echo "$BASENAME_DATASET" | tr -cs 'A-Za-z0-9._-' '-')
SANITIZED_CKPT_TYPE=$(echo "$CHECKPOINT_TYPE" | tr -cs 'A-Za-z0-9._-' '-')
LOG_FILE="$OUTPUT_DIR/logs/${RUN_TIMESTAMP}_${SANITIZED_WANDB_NAME}_${SANITIZED_CKPT_TYPE}_${SANITIZED_DATASET}_seed${GLOBAL_SEED}.log"
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
echo "Logging to $LOG_FILE"

# Enable xtrace only in debug mode (after tee so traces also go to log)
if [ "$DEBUG_BOOL" = true ]; then
    set -x
fi

if [ "$DEBUG_BOOL" = true ]; then
    EFFECTIVE_BATCH_SIZE=1
    EFFECTIVE_DATALOADER_WORKERS=0
else
    EFFECTIVE_BATCH_SIZE="$BATCH_SIZE"
    EFFECTIVE_DATALOADER_WORKERS="$DATALOADER_DEFAULT"
fi

echo "Configuration:"
cat <<CONFIG_EOF
{
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "debug": $DEBUG_BOOL,
  "output_dir": "$OUTPUT_DIR",
  "resume_ckpt_path": "$RESUME_CHECKPOINT_PATH",
  "wandb_name": "$WANDB_NAME",
  "dataset_path": "$DATASET_PATH",
  "metadata_path": "$METADATA_PATH",
  "model_base_path": "$MODEL_BASE_PATH",
  "dit_path": "$DIT_PATH",
  "vae_path": "$VAE_PATH",
  "global_seed": $GLOBAL_SEED,
  "pipeline_type": "$PIPELINE_TYPE",
  "t_highfreq_ratio": $T_HIGHFREQ_RATIO,
  "batch_size": $BATCH_SIZE,
  "frame_downsample_to": $FRAME_DOWNSAMPLE_TO,
  "use_real_temporal_indices": $USE_REAL_TEMPORAL_INDICES,
  "use_physical_index": $USE_PHYSICAL_INDEX,
  "select_random_latents": $SELECT_RANDOM_LATENTS,
  "effective_dataloader_workers": $EFFECTIVE_DATALOADER_WORKERS
}
CONFIG_EOF

# Distributed/NCCL failure fast settings
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
# Reduce chatty logs unless debugging
export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}

# # set to num_workers 0 and batch_size 1 for debug
CMD=(
  python -u -m src.train_recammaster
  --task train
  --dataset_path "$DATASET_PATH"
  --output_path "$OUTPUT_DIR"
  --dit_path "$DIT_PATH"
  --vae_path "$VAE_PATH"
  --steps_per_epoch 10000
  --max_epochs 100
  --learning_rate 1e-5
  --accumulate_grad_batches 1
  --use_gradient_checkpointing
  --dataloader_num_workers "$EFFECTIVE_DATALOADER_WORKERS"
  --batch_size "$EFFECTIVE_BATCH_SIZE"
  --num_val_scenes 2
  --global_seed "$GLOBAL_SEED"
  --enable_test_step
  --test_samples 10
  --test_inference_steps 10
  --val_size 12
  --metadata_path "$METADATA_PATH"
  --wandb_name "$WANDB_NAME"
  --val_check_interval_batches 50
  --training_strategy deepspeed_stage_2
  --distributed_timeout_seconds 1800
  --t_highfreq_ratio "$T_HIGHFREQ_RATIO"
  --frame_downsample_to "$FRAME_DOWNSAMPLE_TO"
  --pipeline_type "$PIPELINE_TYPE"
)
if [[ -n "$RESUME_CHECKPOINT_PATH" ]]; then
  CMD+=(--resume_ckpt_path "$RESUME_CHECKPOINT_PATH")
fi
if [[ "$CHECKPOINT_TYPE" == "wan21" || "$CHECKPOINT_TYPE" == "recammaster" ]]; then
  CMD+=(--ckpt_type "$CHECKPOINT_TYPE")
fi
if [[ "$USE_REAL_TEMPORAL_INDICES" == "true" ]]; then
  CMD+=(--use_real_temporal_indices)
fi
if [[ "$USE_PHYSICAL_INDEX" == "true" ]]; then
  CMD+=(--use_physical_index)
fi
if [[ "$SELECT_RANDOM_LATENTS" == "true" ]]; then
  CMD+=(--select_random_latents)
fi
if [[ -n "$ENABLE_CAM_LAYERS" ]]; then
  CMD+=(--enable_cam_layers)
fi
if [[ "$DEBUG_BOOL" == true ]]; then
  CMD+=(--debug)
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
"${CMD[@]}"
