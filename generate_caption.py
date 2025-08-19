import os
# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before import all hf related pkgs

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import subprocess
import tyro
from tqdm.auto import tqdm
import random
from PIL import Image
import numpy as np
import csv
import decord
from sklearn.metrics.pairwise import cosine_similarity

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl import load_image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

def set_random_seed(seed=666):
    """Set random seeds for reproducibility while maintaining performance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set random seed
set_random_seed(666)

querys = [
    "<|User|> <ImageHere>; Combine the image to describe the scene in detail. The first sentence should be active voice with a starting of 'A man' or 'A person' or 'A couple'.",
    "<|User|> Please compress the description into two or three sentences . ",
    "<|User|> Here is a example:A man in a white shirt and suspenders sits against a stone wall in a rural setting, reading a piece of paper. He appears to be deeply focused on the content of the paper, occasionally glancing up and then back down to read. The scene is calm and contemplative, with the man's expressions reflecting his engagement with the document. The main subject is a man with gray hair, wearing a white shirt and suspenders. He is holding a piece of paper with handwritten text. The man is seated against a stone wall, with a hat and a jacket placed on the wall beside him. His posture is relaxed, and he is engaged in reading the paper, occasionally looking up and then back down to continue reading. The background consists of a rural setting with a stone wall and a grassy field. There is a hat and a jacket placed on the wall behind the man, suggesting a casual, outdoor environment. The scene is well-lit, indicating it is daytime, with natural light illuminating the area.",
]

@torch.no_grad()
def generate_internlm_captions(
    path: str = "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10", 
    model_type: str = 'internlmx',
    category: str = 'all',
    gpu_id: int = 4
):    
    # init model and tokenizer
    if model_type == 'internlmx':
        backend_config = TurbomindEngineConfig(dtype='float16', session_len=1048576)  # 1M context length
        pipe = pipeline('internlm/internlm-xcomposer2d5-7b', log_level='ERROR', backend_config=backend_config)
    else:
        raise ValueError('Not supported model type!')
    
    # exit(0)
    # Initialize CLIP model with FP16
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    basedirs = []
    if "dl3dv" in path.lower(): #prasing dl3dv
        img_folder_name = 'images_4'
        for scene_id in tqdm(sorted(os.listdir(os.path.join(path))), desc="Searching basedirs", leave=False):
            if not os.path.isdir(os.path.join(path, scene_id)) or not os.path.exists(os.path.join(path, scene_id, img_folder_name)) or \
                os.path.exists(os.path.join(path, scene_id, 'captions.txt')):
                continue
            basedirs.append(os.path.join(path, scene_id))
    elif "multicamvideo" in path.lower():
        img_folder_name = 'videos'
        for scene_id in tqdm(sorted(os.listdir(os.path.join(path))), desc="Searching basedirs", leave=False):
            if not os.path.isdir(os.path.join(path, scene_id)) or not os.path.exists(os.path.join(path, scene_id, img_folder_name)) or \
                os.path.exists(os.path.join(path, scene_id, 'captions.txt')):
                continue
            basedirs.append(os.path.join(path, scene_id))
    else:
        raise ValueError
    
    result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, text=True)
    gpu_count = result.stdout.count('GPU ')

    begin, end = int(gpu_id * len(basedirs)//gpu_count), int((gpu_id + 1) * len(basedirs)//gpu_count)
    if gpu_id == gpu_count - 1:
        basedirs = basedirs[begin:]
    else:
        basedirs = basedirs[begin:end]

    # --- CSV INITIALIZATION & PROCESSED VIDEO CHECK ---
    output_csv_path = "metadata.csv"
    processed_videos = set()
    if os.path.exists(output_csv_path):
        with open(output_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
                for row in reader:
                    if row:
                        processed_videos.add(row[0])
            except StopIteration:
                pass  # File is empty
    else:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["video_absolute_path", "caption"])



    # loop over data
    for idx, basedir in enumerate(tqdm(basedirs, desc="Generate Captions")):
        # get sequence, category from batch
        filenames = [f for f in sorted(os.listdir(os.path.join(basedir, img_folder_name))) if (f.endswith('mp4'))]
        for filename in tqdm(filenames, desc="Generate Captions in One Scene"):
            video_path = os.path.join(basedir, img_folder_name, filename)
            video_abs_path = os.path.abspath(video_path)
            if video_abs_path in processed_videos:
                tqdm.write(f"Skipping already processed video: {video_abs_path}")
                continue
            ## TODO: 修改下面的脚本成处理.mp4而不是图片的list, filename是一个video的path
            # Load and process all images with CLIP
            ## TODO: image_features
            '''
            video = load(video_path)
            ...
            for ... 
            batch_images = select from video
            '''

            # -- Start of modified code --
            # This block processes a video file to extract frames and their CLIP features.
            # video_path is already defined above.
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except decord.DECORDError as e:
                print(f"Skipping corrupted or unreadable video: {video_path}, error: {e}")
                continue

            # Sample frames at ~1 FPS (given FPS=15)
            fps = 15
            sample_rate = int(fps)
            frame_indices = list(range(0, len(vr), sample_rate))

            if not frame_indices:
                print(f"Skipping video with no frames to sample: {video_path}")
                continue

            pil_frames = vr.get_batch(frame_indices).asnumpy()
            pil_frames = [Image.fromarray(frame) for frame in pil_frames]

            # Extract CLIP features from the frames
            image_features = []
            batch_size = 32
            for i in tqdm(range(0, len(pil_frames), batch_size), desc="Processing frames with CLIP", leave=False):
                batch_frames = pil_frames[i:i + batch_size]
                inputs = clip_processor(images=batch_frames, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    batch_features = clip_model.get_image_features(**inputs).cpu().numpy()
                    image_features.append(batch_features)
            
            if not image_features:
                image_features = np.array([]) # Ensure image_features is a numpy array
            else:
                image_features = np.concatenate(image_features, axis=0)
            # -- End of modified code --
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(image_features)
            
            # Select 10 most diverse images
            selected_indices = []
            remaining_indices = list(range(len(pil_frames)))
            
            # Start with a random image
            start_idx = random.choice(remaining_indices)
            selected_indices.append(start_idx)
            remaining_indices.remove(start_idx)
            
            # Select remaining images based on minimum similarity to already selected images
            while len(selected_indices) < min(10, len(pil_frames)):
                min_similarities = []
                for idx in remaining_indices:
                    similarities = similarity_matrix[idx, selected_indices]
                    min_similarities.append((idx, np.min(similarities)))
                
                # Select the image with minimum similarity to already selected images
                next_idx = max(min_similarities, key=lambda x: x[1])[0]
                selected_indices.append(next_idx)
                remaining_indices.remove(next_idx)
            
            # Get selected filenames
            images = [pil_frames[i] for i in selected_indices]

            # run captioning
            history = ''
            results = []

            for i, query in enumerate(querys):
                if i == 0:
                    image_tags = [f"Image{i+1} <ImageHere>" for i in range(len(images))]
                    query = query.replace("<ImageHere>", " ".join(image_tags))
                    if len(images) > 1:
                        query = query.replace("image", "images")
                history += query
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    response = pipe(
                        (history, images),
                        gen_config=GenerationConfig(
                            top_k=0, 
                            top_p=0.8, 
                            temperature=0.3,
                        )
                    ).text
                history += " <|Bot|>" + response
                if i > 0:
                    results += [response]

                # save captions
            # --- FIXED SAVING LOGIC ---
            video_abs_path = os.path.abspath(video_path)
            caption_text = " ".join(results)
            with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_abs_path, caption_text])



if __name__ == "__main__":
    tyro.cli(generate_internlm_captions)
