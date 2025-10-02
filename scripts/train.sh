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
    echo "  -d, --debug                        Enable debug mode (default: enabled)"
    echo "  -o, --output-dir OUTPUT_DIR        Output directory (default: ./models/train)"
    echo "  -r, --recammaster-checkpoint PATH  ReCamMaster checkpoint path (default: /data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt)"
    echo "  -w, --wandb-name WANDB_NAME        Wandb experiment name (default: Exp07c)"
    echo "  -s, --dataset-path DATASET_PATH    Dataset path (default: /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10)"
    echo "  -m, --metadata-path METADATA_PATH  Metadata file path (default: ./metadata_subset.csv)"
    echo "  -g, --global-seed SEED             Global seed (default: 42)"
    echo "  -h, --help                         Show this help message"
    exit 1
}

export RUN_TIMESTAMP=$(date +'%m-%d-%H%M%S')

# Default values
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DEBUG_FLAG="--debug"
OUTPUT_DIR="$(pwd)/models/train"
METADATA_PATH="$(pwd)/metadata/metadata_subset.csv"
RECAMMASTER_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt"
WANDB_NAME="Exp07c"
DATASET_PATH="/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10"

GLOBAL_SEED="42"

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
            # support forms like -c7 or -c0,7
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
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
# Set the path to the checkpoint you want to resume from.
# If you want to train from scratch, you can remove the --resume_ckpt_path line.
# For Wan2.1 original model:
WAN21_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
WAN21_RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/09-19-191944_Exp04d/checkpoints/step1344.ckpt"
# RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/train/wandb/ReCamMaster/08-21-151648_exp02b/checkpoints/step1079.ckpt"

# Choose checkpoint type: "wan21" for original Wan2.1 model, "recammaster" for ReCamMaster fine-tuned model
CHECKPOINT_TYPE="wan21"  # Change to "recammaster" if you want to use ReCamMaster checkpoint

# Set checkpoint path based on type
if [ "$CHECKPOINT_TYPE" = "wan21" ]; then
    RESUME_CHECKPOINT_PATH="$WAN21_CHECKPOINT_PATH"
    ENABLE_CAM_LAYERS=""
else
    RESUME_CHECKPOINT_PATH="$RECAMMASTER_CHECKPOINT_PATH"
    ENABLE_CAM_LAYERS="--enable_cam_layers"
fi

if [ -n "$DEBUG_FLAG" ]; then
    DEBUG_BOOL=true
else
    DEBUG_BOOL=false
fi

# Build log file and redirect all outputs
mkdir -p "$OUTPUT_DIR"
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
  "global_seed": $GLOBAL_SEED
}
CONFIG_EOF

# # set to num_workers 0 and batch_size 1 for debug
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 python -u -m src.train_recammaster  \
 --task train  \
 --dataset_path "$DATASET_PATH"  \
 --output_path "$OUTPUT_DIR"   \
 --dit_path "$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors"   \
 --vae_path "$MODEL_BASE_PATH/Wan2.1_VAE.pth"   \
 --steps_per_epoch 8000   \
 --max_epochs 100   \
 --learning_rate 1e-5   \
 --accumulate_grad_batches  4  \
 --use_gradient_checkpointing  \
 --dataloader_num_workers $(if [ "$DEBUG_BOOL" = true ]; then echo 0; else echo 4; fi) \
 --batch_size $(if [ "$DEBUG_BOOL" = true ]; then echo 1; else echo 2; fi) \
 --num_val_scenes 2 \
 --global_seed "$GLOBAL_SEED" \
 --enable_test_step \
 --test_samples 10 \
 --test_inference_steps 20 \
 --val_size 10 \
 --resume_ckpt_path "$RESUME_CHECKPOINT_PATH" \
 --ckpt_type "$CHECKPOINT_TYPE" \
 $ENABLE_CAM_LAYERS \
 --metadata_path "$METADATA_PATH" \
 --wandb_name "$WANDB_NAME" \
 --val_check_interval_batches 100 \
 --training_strategy deepspeed_stage_2 \
 $DEBUG_FLAG \

