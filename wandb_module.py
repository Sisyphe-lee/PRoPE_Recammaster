"""
WandB Video Logging Module for ReCamMaster Training

This module contains all WandB-related functionality for video logging,
compression, and upload management, separated from the core training logic.
"""

import os
import numpy as np
import torch
import imageio
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class WandBVideoLogger:
    """Handles WandB video logging with compression and selective upload strategies"""
    
    def __init__(
        self,
        strategy="selective",
        max_videos_per_epoch=5,
        quality_threshold=25.0,
        compress_videos=True,
        video_fps=4,
        video_scale=0.5,
        output_dir="./wandb_videos"
    ):
        """
        Initialize WandB video logger
        
        Args:
            strategy: Upload strategy ("none", "selective", "quality_based", "all")
            max_videos_per_epoch: Maximum videos to upload per validation epoch
            quality_threshold: PSNR threshold for quality-based selection
            compress_videos: Whether to compress videos before upload
            video_fps: FPS for WandB video uploads
            video_scale: Scale factor for video resolution (0.5 = half size)
            output_dir: Directory to save temporary video files
        """
        self.strategy = strategy
        self.max_videos_per_epoch = max_videos_per_epoch
        self.quality_threshold = quality_threshold
        self.compress_videos = compress_videos
        self.video_fps = video_fps
        self.video_scale = video_scale
        self.output_dir = output_dir
        
        # Track videos for batch upload
        self.queued_videos = []
        self.video_count = 0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def reset_epoch(self):
        """Reset video tracking for new validation epoch"""
        self.queued_videos = []
        self.video_count = 0
    
    def should_upload_video(self, batch_idx, psnr_value):
        """
        Determine if video should be uploaded based on strategy
        
        Args:
            batch_idx: Current batch index
            psnr_value: PSNR value for the video
            
        Returns:
            bool: Whether to upload this video
        """
        if self.strategy == "none":
            return False
        
        if self.video_count >= self.max_videos_per_epoch:
            return False
        
        if self.strategy == "all":
            return True
        elif self.strategy == "selective":
            # Upload every Nth sample
            if batch_idx % 5 == 0:  # Every 5th sample
                return True
            return False
        elif self.strategy == "quality_based":
            # Upload based on PSNR quality (very good or very bad results)
            if psnr_value < self.quality_threshold * 0.8:  # Poor quality
                return True
            elif psnr_value > self.quality_threshold * 1.2:  # Excellent quality
                return True
            return False
        
        return False
    
    def compress_video_frames(self, frames):
        """
        Compress video frames for WandB upload
        
        Args:
            frames: List of numpy arrays representing video frames
            
        Returns:
            List of compressed frames
        """
        if not self.compress_videos:
            return frames
        
        compressed_frames = []
        # Adjust frame sampling based on target FPS
        step = max(1, len(frames) // (len(frames) * self.video_fps // 8))
        
        for i in range(0, len(frames), step):
            frame = frames[i]
            if self.video_scale != 1.0:
                # Resize frame
                h, w = frame.shape[:2]
                new_h, new_w = int(h * self.video_scale), int(w * self.video_scale)
                frame_pil = Image.fromarray(frame)
                frame_pil = frame_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                frame = np.array(frame_pil)
            compressed_frames.append(frame)
        
        return compressed_frames
    
    def queue_video(self, frames, metadata, psnr_value, batch_idx):
        """
        Queue a video for upload at epoch end
        
        Args:
            frames: Video frames as numpy arrays
            metadata: Dictionary containing video metadata
            psnr_value: PSNR value for the video
            batch_idx: Batch index
        """
        if not self.should_upload_video(batch_idx, psnr_value):
            return False
        
        self.queued_videos.append({
            'frames': frames,
            'metadata': metadata,
            'psnr': psnr_value,
            'batch_idx': batch_idx
        })
        self.video_count += 1
        
        print(f"[W&B] Queued video for upload: batch {batch_idx}, PSNR {psnr_value:.2f}, "
              f"scene {metadata.get('scene_id', 'unknown')}")
        return True
    
    def upload_videos(self, logger, global_step):
        """
        Upload all queued videos to WandB at epoch end
        
        Args:
            logger: WandB logger instance
            global_step: Current training step (fallback if metadata doesn't contain step)
        """
        if not self.queued_videos:
            print("[W&B] No videos queued for upload this epoch")
            return
        
        if not hasattr(logger, "experiment") or logger is None:
            print("[W&B] Logger not available, skipping video upload")
            return
        
        try:
            # Group videos by step (from metadata) for better organization
            videos_by_step = {}
            for video_info in self.queued_videos:
                metadata = video_info['metadata']
                # Use step from metadata if available, otherwise fallback to global_step
                step_int = metadata.get('current_step', global_step)
                step_int = int(step_int)
                
                if step_int not in videos_by_step:
                    videos_by_step[step_int] = []
                videos_by_step[step_int].append(video_info)
            
            # Upload videos grouped by step
            for step_int, videos in videos_by_step.items():
                # Sort videos by PSNR for better organization
                sorted_videos = sorted(videos, key=lambda x: x['psnr'], reverse=True)
                
                upload_data = {}
                table_data = []
                
                for i, video_info in enumerate(sorted_videos):
                    frames = video_info['frames']
                    metadata = video_info['metadata']
                    psnr = video_info['psnr']
                    batch_idx = video_info['batch_idx']
                    
                    # Create compressed video for WandB
                    compressed_frames = self.compress_video_frames(frames)
                    
                    # Create temporary file for WandB upload
                    temp_path = os.path.join(
                        self.output_dir, 
                        f"wandb_temp_step{step_int}_batch{batch_idx}.mp4"
                    )
                    imageio.mimsave(temp_path, compressed_frames, fps=self.video_fps, quality=6)
                    
                    # Create descriptive key for WandB - separate from PSNR metrics
                    scene_id = metadata.get('scene_id', 'unknown')
                    cond_cam = metadata.get('condition_cam_type', 'unknown')
                    tgt_cam = metadata.get('target_cam_type', 'unknown')
                    
                    # Create separate video section for each step
                    video_key = f"videos/step_{step_int}/scene_{scene_id}_{cond_cam}_to_{tgt_cam}_psnr_{psnr:.1f}"
                    upload_data[video_key] = wandb.Video(temp_path, fps=self.video_fps, format="mp4")
                    
                    # Add to metadata table
                    table_data.append([
                        step_int, batch_idx, scene_id, cond_cam, tgt_cam, f"{psnr:.2f}"
                    ])
                
                # Upload all videos for this step at once
                logger.experiment.log(upload_data, step=step_int)
                
                # Upload metadata table in a separate section
                table = wandb.Table(
                    columns=["Step", "Batch", "Scene", "Source_Cam", "Target_Cam", "PSNR"],
                    data=table_data
                )
                logger.experiment.log({f"videos/step_{step_int}/metadata": table}, step=step_int)
                
                print(f"[W&B] Successfully uploaded {len(sorted_videos)} validation videos at step {step_int}")
                
                # Clean up temporary files for this step
                for video_info in sorted_videos:
                    batch_idx = video_info['batch_idx']
                    temp_path = os.path.join(
                        self.output_dir, 
                        f"wandb_temp_step{step_int}_batch{batch_idx}.mp4"
                    )
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
        except Exception as e:
            print(f"Failed to upload videos to WandB: {e}")


class VideoDecoder:
    """Handles video decoding and processing for validation"""
    
    def __init__(self, pipe):
        """
        Initialize video decoder
        
        Args:
            pipe: The diffusion pipeline for video processing
        """
        self.pipe = pipe
    
    def decode_and_create_combined_video(
        self, 
        noise_pred, 
        noisy_latents, 
        tgt_latent_len, 
        origin_latents, 
        timestep, 
        batch,
        output_path=None
    ):
        """
        Decode video and create combined visualization
        
        Args:
            noise_pred: Predicted noise from the model
            noisy_latents: Noisy latent representations
            tgt_latent_len: Length of target latent
            origin_latents: Original latent representations
            timestep: Current timestep
            batch: Batch data containing metadata
            output_path: Path to save the combined video (optional)
            
        Returns:
            Tuple of (psnr_value, combined_frames, metadata)
        """
        # 1. PREPARE PREDICTED AND GT LATENTS
        noise_pred_sample = noise_pred[0:1, :, :tgt_latent_len, ...]
        noisy_latents_sample = noisy_latents[0:1, :, :tgt_latent_len, ...]

        # For FlowMatch, project to the final (sigma=0) in one step using scheduler
        # This aligns validation restoration with the pipeline's inference update rule
        pred_original_sample = self.pipe.scheduler.step(
            noise_pred_sample, timestep, noisy_latents_sample, to_final=True
        )

        # Prepare ground truth and condition latents
        gt_original_sample = origin_latents[0:1, :, :tgt_latent_len, ...]
        cond_original_sample = origin_latents[0:1, :, tgt_latent_len:, ...]

        # 2. DECODE ALL THREE VIDEOS
        self.pipe.load_models_to_device(['vae'])
        
        pred_frames_tensor = self.pipe.decode_video(pred_original_sample.to(dtype=self.pipe.torch_dtype))[0]
        gt_frames_tensor = self.pipe.decode_video(gt_original_sample.to(dtype=self.pipe.torch_dtype))[0]
        cond_frames_tensor = self.pipe.decode_video(cond_original_sample.to(dtype=self.pipe.torch_dtype))[0]
        
        self.pipe.load_models_to_device([])

        # 3. CALCULATE PSNR
        pred_frames_norm = (pred_frames_tensor.clamp(-1, 1) + 1) / 2
        gt_frames_norm = (gt_frames_tensor.clamp(-1, 1) + 1) / 2
        
        mse = torch.nn.functional.mse_loss(pred_frames_norm, gt_frames_norm)
        psnr_value = 100.0 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

        # 4. CALCULATE ERROR MAP WITH HEATMAP VISUALIZATION
        error_frames_tensor = torch.abs(gt_frames_tensor - pred_frames_tensor)
        
        # Create heatmap error visualization
        error_video_frames = self._create_heatmap_error_frames(error_frames_tensor)

        # 5. CREATE COMBINED VIDEO (2x2 layout)
        pred_video_frames = self.pipe.tensor2video(pred_frames_tensor)
        gt_video_frames = self.pipe.tensor2video(gt_frames_tensor)
        cond_video_frames = self.pipe.tensor2video(cond_frames_tensor)
        
        combined_frames = []
        for i in range(len(pred_video_frames)):
            # Create 2x2 grid: [condition, gt]
            #                  [pred, error_map]
            top_row = np.concatenate([cond_video_frames[i], gt_video_frames[i]], axis=1)
            bottom_row = np.concatenate([pred_video_frames[i], error_video_frames[i]], axis=1)
            combined_frame = np.concatenate([top_row, bottom_row], axis=0)
            combined_frames.append(combined_frame)
        
        # 6. EXTRACT METADATA
        try:
            scene_id = batch.get('scene_id', ['unknown'])[0] if isinstance(batch.get('scene_id'), list) else str(batch.get('scene_id', 'unknown'))
            condition_cam_type = batch.get('condition_cam_type', ['unknown'])[0] if isinstance(batch.get('condition_cam_type'), list) else str(batch.get('condition_cam_type', 'unknown'))
            target_cam_type = batch.get('target_cam_type', ['unknown'])[0] if isinstance(batch.get('target_cam_type'), list) else str(batch.get('target_cam_type', 'unknown'))
        except Exception as e:
            print(f"Error extracting metadata from batch: {e}")
            scene_id = "unknown"
            condition_cam_type = "unknown"
            target_cam_type = "unknown"
        
        metadata = {
            'scene_id': scene_id,
            'condition_cam_type': condition_cam_type,
            'target_cam_type': target_cam_type
        }
        
        return psnr_value.detach().float().cpu().item(), combined_frames, metadata

    def _create_heatmap_error_frames(self, error_frames_tensor):
        """Convert error tensor to heatmap visualization using colormap"""
        error_frames_list = []
        
        # error_frames_tensor shape: (C, T, H, W)
        for t in range(error_frames_tensor.shape[1]):  # 遍历时间维度
            # 计算每个像素的总误差（RGB三通道平均）
            frame_error = error_frames_tensor[:, t, :, :].mean(dim=0)  # (H, W)
            
            # 归一化到 [0, 1]
            frame_min, frame_max = frame_error.min(), frame_error.max()
            if frame_max > frame_min:
                frame_norm = (frame_error - frame_min) / (frame_max - frame_min)
            else:
                frame_norm = torch.zeros_like(frame_error)
            
            # 应用热力图colormap (使用jet colormap)
            # 转换为float32以避免BFloat16不支持numpy转换的问题
            frame_np = frame_norm.float().cpu().numpy()
            colored_frame = cm.jet(frame_np)[:, :, :3]  # 去掉alpha通道
            colored_frame = (colored_frame * 255).astype(np.uint8)
            
            # 转换为PIL Image
            error_frames_list.append(Image.fromarray(colored_frame))
        
        return error_frames_list
