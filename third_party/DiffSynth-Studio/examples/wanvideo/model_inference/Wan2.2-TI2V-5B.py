import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import imageio

LOCAL_MODELS_ROOT = "./models"
WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            local_model_path=LOCAL_MODELS_ROOT,
            skip_download=True,
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=WAN22_MODEL_ID,
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            local_model_path=LOCAL_MODELS_ROOT,
            skip_download=True,
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=WAN22_MODEL_ID,
            origin_file_pattern="Wan2.2_VAE.pth",
            local_model_path=LOCAL_MODELS_ROOT,
            skip_download=True,
            offload_device="cpu",
        ),
    ],
    redirect_common_files=False,
)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
# input_image = Image.open("/data1/lcy/projects/ReCamMaster/i2v_input.JPG").resize((1248, 704))
TARGET_HEIGHT = 480
TARGET_WIDTH = 832

### input_image is the first frame of the video:/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10/scene999/videos/cam10.mp4
#### load video 
video = imageio.get_reader("/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10/scene999/videos/cam10.mp4")
input_image = Image.fromarray(video.get_data(0))


def resize_and_center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize while keeping aspect ratio, then center-crop to the target canvas."""
    src_w, src_h = image.size
    scale = max(target_width / src_w, target_height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.BICUBIC)

    left = max(0, (resized_w - target_width) // 2)
    top = max(0, (resized_h - target_height) // 2)
    right = left + target_width
    bottom = top + target_height
    return resized.crop((left, top, right, bottom))


input_image = resize_and_center_crop(input_image, TARGET_WIDTH, TARGET_HEIGHT)

video = pipe(
    prompt="A man in a yellow shirt and dark pants stands confidently, shifting weight slightly while facing forward. His posture relaxes as he subtly adjusts his stance, hands resting at his sides. The luxurious room remains static—only his minor movements animate the scene.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=TARGET_HEIGHT, width=TARGET_WIDTH,
    input_image=input_image,
    num_frames=121,
    num_inference_steps=10,
    
)
save_video(video, "video2.mp4", fps=15, quality=5)
