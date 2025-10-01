export RUN_TIMESTAMP=$(date +'%m-%d-%H%M%S')

# For clarity, define paths as variables
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="./models/train"
# Set the path to the checkpoint you want to resume from.
# If you want to train from scratch, you can remove the --resume_ckpt_path line.
# For Wan2.1 original model:
WAN21_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
WAN21_RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/09-19-191944_Exp04d/checkpoints/step1344.ckpt"
# For ReCamMaster fine-tuned model:
RECAMMASTER_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt"
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

CUDA_VISIBLE_DEVICES="7" python train_recammaster.py  \
 --task train  \
 --dataset_path /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10  \
 --output_path "$OUTPUT_DIR"   \
 --dit_path "$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors"   \
 --vae_path "$MODEL_BASE_PATH/Wan2.1_VAE.pth"   \
 --steps_per_epoch 8000   \
 --max_epochs 100   \
 --learning_rate 1e-5   \
 --accumulate_grad_batches  4  \
 --use_gradient_checkpointing  \
 --dataloader_num_workers 2 \
 --batch_size 1 \
 --num_val_scenes 2 \
 --global_seed 42 \
 --enable_test_step \
 --test_samples 10 \
 --test_inference_steps 20 \
 --val_size 10 \
 --resume_ckpt_path "$RESUME_CHECKPOINT_PATH" \
 --ckpt_type "$CHECKPOINT_TYPE" \
 $ENABLE_CAM_LAYERS \
 --metadata_path "./metadata_subset.csv" \
 --wandb_name "Exp07b" \
 --val_check_interval_batches 1000 \
 --training_strategy deepspeed_stage_2 \
 --debug \

