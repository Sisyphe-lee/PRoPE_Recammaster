"""
Borrow from https://github.com/nv-tlabs/vipe/blob/main/scripts/download_dataset.py
# Replace YOUR_PREFIX with the prefix of the dataset to be downloaded (see prefix column in the table above)
# You can also use more specific prefixes, e.g. wsdg-003e2c86 to download a specific shard of the dataset.
python scripts/download_dataset.py --prefix YOUR_PREFIX --output_base YOUR_OUTPUT_DIR --rgb --depth



HF_ENDPOINT="https://hf-mirror.com" && python -m tools.download_datasets --prefix wsdg-003e2c86 --output_base /nas/datasets/vipe_wild_sdg_1m --rgb
HF_ENDPOINT="https://hf-mirror.com" && python -m tools.download_datasets --prefix dpsp-034b0a25 --output_base /nas/datasets/vipe_dynpose_100kpp --rgb

dpsp-034b0a25
"""
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HF_ENDPOINT = "https://hf-mirror.com"
import argparse
import os
import shutil
import tarfile
import tempfile
import zipfile

from pathlib import Path

import pandas as pd

from huggingface_hub import HfApi


def download_clips_from_url(
    url: str, clips_timestamps: list[tuple[str, str]], output_paths: list[Path], cookies_from_browser: bool = False
):
    import datetime

    import ffmpeg
    import yt_dlp

    def _get_seconds(t: str) -> float:
        time_format = "%H:%M:%S.%f"
        t_obj = datetime.datetime.strptime(t, time_format).time()
        return t_obj.second + t_obj.microsecond / 1e6 + t_obj.minute * 60 + t_obj.hour * 3600

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "video.mp4"

        ydl_opts = {
            "outtmpl": str(video_path),
            "format": "wv*[height>=720][ext=mp4]/w[height>=720][ext=mp4]/bv[ext=mp4]/b[ext=mp4]",
            "quiet": True,
            "no_warnings": True,
        }
        if cookies_from_browser:
            # 使用本地 cookies 文件
            cookies_file = Path(__file__).parent / "www.youtube.com_cookies.txt"
            if cookies_file.exists():
                ydl_opts["cookiefile"] = str(cookies_file)
                print(f"使用 cookies 文件: {cookies_file}")
            else:
                print("警告: 未找到 cookies 文件，尝试从浏览器读取...")
                try:
                    ydl_opts["cookiesfrombrowser"] = ("chrome",)
                except Exception as e:
                    print(f"无法从 Chrome 读取 cookies: {e}")
                    print("继续下载，但可能无法访问需要认证的视频")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)

        for idx, (s, e) in enumerate(clips_timestamps):
            s_time = _get_seconds(s)
            e_time = _get_seconds(e)
            duration = e_time - s_time
            _ = ffmpeg.input(video_path, ss=s_time, t=duration).output(str(output_paths[idx])).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="wsdg-003e2c86", type=str, help="Prefix of the dataset to be downloaded")
    parser.add_argument("--output_base", default="/nas/datasets/vipe_wild_sdg_1m", type=str, help="Base directory to save the dataset")
    parser.add_argument("--rgb", action="store_true", help="Download RGB components of the videos")
    parser.add_argument("--depth", action="store_true", help="Download depth components of the videos")
    parser.add_argument("--no-cookies", action="store_true", help="Disable browser cookies for YouTube downloads")

    args = parser.parse_args()
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    attributes_to_download = ["intrinsics", "pose"]
    if args.rgb:
        attributes_to_download.append("rgb")
    if args.depth:
        attributes_to_download.append("depth")

    if args.prefix.startswith("dpsp"):
        repo_id = "nvidia/vipe-dynpose-100kpp"
    elif args.prefix.startswith("wsdg"):
        repo_id = "nvidia/vipe-wild-sdg-1m"
    elif args.prefix.startswith("w360"):
        repo_id = "nvidia/vipe-web360"
    else:
        raise ValueError(f"Invalid prefix: {args.prefix}")

    # Grab the metadata of the dataset to download
    api = HfApi(token=os.getenv("HF_TOKEN"), endpoint=HF_ENDPOINT)

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = api.hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="meta.parquet", local_dir=tmp_dir
        )
        df = pd.read_parquet(meta_file)

    # Select rows where tar_name starts with the given prefix
    related_videos = df.loc[df["tar_name"].str.startswith(args.prefix)]
    related_tar_names = list(set(related_videos["tar_name"].tolist()))

    print(f"Found {len(related_videos)} videos to download within {len(related_tar_names)} tar files")

    for attribute in attributes_to_download:
        for tar_name in related_tar_names:
            remote_path = f"payload/{tar_name}/{attribute}.tar"
            (output_base / attribute).mkdir(parents=True, exist_ok=True)

            if api.file_exists(repo_id=repo_id, repo_type="dataset", filename=remote_path):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tar_file = api.hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=remote_path,
                        local_dir=tmp_dir,
                    )
                    # Extract the tar file
                    with tarfile.open(tar_file, "r") as tar:
                        tar.extractall(path=output_base / attribute)

            else:
                if attribute == "rgb" and args.prefix.startswith("dpsp"):
                    # Merge videos with the same youtube_link
                    merged_videos = {}
                    for video_info in related_videos.iterrows():
                        link = video_info[1]["youtube_link"]
                        if link not in merged_videos:
                            merged_videos[link] = []
                        merged_videos[link].append(video_info)

                    for link, merged_video in merged_videos.items():
                        time_slices = [t[1]["youtube_timestamp"].split("-") for t in merged_video]
                        output_links = [t[1]["sequence"] for t in merged_video]
                        download_links = [output_base / attribute / f"{ft}.mp4" for ft in output_links]
                        download_clips_from_url(
                            link,
                            time_slices,
                            download_links,
                            cookies_from_browser=not args.no_cookies,
                        )

                elif attribute == "rgb" and args.prefix.startswith("w360"):
                    # Download the depth frames from Official website
                    import gdown
                    gdown.download(
                        "https://drive.google.com/file/d/1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs/view",
                        output=str(output_base / "web360_raw.zip"),
                        fuzzy=True,
                    )
                    with zipfile.ZipFile(output_base / "web360_raw.zip", "r") as zip_ref:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            zip_ref.extractall(tmp_dir)
                            for video_file in Path(tmp_dir).glob("**/*.mp4"):
                                shutil.copy(video_file, output_base / attribute / video_file.name)

                else:
                    raise ValueError(f"Attribute {attribute} is not supported for {args.prefix}")


if __name__ == "__main__":
    main()