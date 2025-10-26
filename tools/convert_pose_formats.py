#!/usr/bin/env python3
"""
将 ReCamMaster 相机位姿 JSON 转换为每相机一个 NPZ 文件。
修正：按要求，对输出 data 的最后两个维度 (4x4) 进行转置。

NPZ 内容：
- data: (N, 4, 4) float32，每帧 4x4 矩阵（已对每个 4x4 做转置）
- inds: (N,) int64，帧索引 [0..N-1]
"""

import json
import numpy as np
import os
import argparse
import re


def parse_matrix_string(matrix_str: str) -> np.ndarray:
    """将诸如
    "[1 0 0 0] [-0 1 0 0] [0 -0 1 0] [3390 1380 240 1]"
    的一行转换为 4x4 numpy.float32 矩阵。
    """
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', matrix_str)
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers, got {len(numbers)}: {matrix_str}")
    mat = np.array([float(x) for x in numbers], dtype=np.float32).reshape(4, 4)
    return mat


def convert_json_to_npz(json_file_path: str, output_dir: str) -> None:
    with open(json_file_path, 'r') as f:
        content = json.load(f)

    frames = sorted([k for k in content.keys() if k.startswith('frame')],
                    key=lambda x: int(x.replace('frame', '')))
    if not frames:
        raise ValueError('JSON 中未找到任何 frame* 键')

    cameras = sorted([k for k in content[frames[0]].keys() if k.startswith('cam')],
                     key=lambda x: int(x.replace('cam', '')))
    if not cameras:
        raise ValueError('JSON 中未找到任何 cam* 键')

    os.makedirs(output_dir, exist_ok=True)
    print(f"发现 {len(frames)} 帧，{len(cameras)} 个相机")

    for cam_idx, cam_key in enumerate(cameras, 1):
        cam_mats = []
        inds = []
        for fi, frame_key in enumerate(frames):
            try:
                mat_str = content[frame_key][cam_key]
                mat = parse_matrix_string(mat_str)
                cam_mats.append(mat)
                inds.append(fi)
            except Exception as e:
                print(f"警告: 无法解析 {frame_key}/{cam_key}: {e}")
        if not cam_mats:
            print(f"警告: 相机 {cam_key} 无有效帧，跳过")
            continue

        data_arr = np.stack(cam_mats, axis=0).astype(np.float32)   # (N,4,4)
        # 关键修改：对每个 4x4 矩阵进行转置
        data_arr = np.transpose(data_arr, (0, 2, 1))               # (N,4,4)
        inds_arr = np.asarray(inds, dtype=np.int64)

        out_path = os.path.join(output_dir, f"{cam_idx}.npz")
        np.savez(out_path, data=data_arr, inds=inds_arr)
        print(f"保存 {cam_key} -> {out_path} | data={data_arr.shape}, inds={inds_arr.shape}")

    print('转换完成。')


def main():
    ap = argparse.ArgumentParser(description='将相机位姿 JSON 转换为多个 NPZ 文件（对4x4做转置）。')
    ap.add_argument('input_json', help='输入 JSON 路径，例如 camera_extrinsics_ori.json')
    ap.add_argument('-o', '--output-dir', default='./output_npz', help='输出目录，默认 ./output_npz')
    args = ap.parse_args()

    if not os.path.exists(args.input_json):
        print(f"错误：找不到输入文件 {args.input_json}")
        return 1
    try:
        convert_json_to_npz(args.input_json, args.output_dir)
        return 0
    except Exception as e:
        print(f"转换失败：{e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

