#!/bin/bash

# ReCamMaster Inference Script
# Similar to train.sh but for inference

# For clarity, define paths as variables
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="./results"
# Set the path to the checkpoint you want to use for inference
# For Wan2.1 original model:
WAN21_CHECKPOINT_PATH="./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
# For ReCamMaster fine-tuned model:
RECAMMASTER_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/09-16-231550_Exp04d/checkpoints/validation_step150.ckpt"
WAN21_RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/09-27-130130_Exp07b/checkpoints/step966.ckpt"
# Choose checkpoint type: "wan21" for original Wan2.1 model, "recammaster" for ReCamMaster fine-tuned model
CHECKPOINT_TYPE="wan21"  # Change to "wan21" if you want to use Wan2.1 model

# Set checkpoint path based on type
if [ "$CHECKPOINT_TYPE" = "wan21" ]; then
    CHECKPOINT_PATH="$WAN21_RESUME_CHECKPOINT_PATH"
    ENABLE_CAM_LAYERS="--enable_cam_layers"
else
    CHECKPOINT_PATH="$RECAMMASTER_CHECKPOINT_PATH"
    ENABLE_CAM_LAYERS=""
fi

python inference_recammaster.py \
    --dataset_path "./example_test_data" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --ckpt_type "$CHECKPOINT_TYPE" \
    $ENABLE_CAM_LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --cfg_scale 5.0 \
    --dataloader_num_workers 1 \
    --camera_extrinsics_filename "camera_extrinsics_ori.json"