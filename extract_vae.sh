CUDA_VISIBLE_DEVICES="2,3" python vae_feature.py   --task data_process   \
 --dataset_path /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f50_aperture2.4   \
 --output_path ./models   --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"   --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"   --tiled   --num_frames 81   --height 480   --width 832 --dataloader_num_workers 2
