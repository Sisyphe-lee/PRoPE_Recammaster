export RUN_TIMESTAMP=$(date +'%m-%d-%H%M%S')

# For clarity, define paths as variables
MODEL_BASE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="./models/train"
# Set the path to the checkpoint you want to resume from.
# If you want to train from scratch, you can remove the --resume_ckpt_path line.
RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/models/ReCamMaster/checkpoints/step20000.ckpt"

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" python train_recammaster.py  \
 --task train  \
 --dataset_path /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10  \
 --output_path "$OUTPUT_DIR"   \
 --dit_path "$MODEL_BASE_PATH/diffusion_pytorch_model.safetensors"   \
 --vae_path "$MODEL_BASE_PATH/Wan2.1_VAE.pth"   \
 --steps_per_epoch 4000   \
 --max_epochs 100   \
 --learning_rate 5e-5   \
 --accumulate_grad_batches 4   \
 --use_gradient_checkpointing  \
 --dataloader_num_workers 4 \
 --resume_ckpt_path "$RESUME_CHECKPOINT_PATH" \
#  --debug
