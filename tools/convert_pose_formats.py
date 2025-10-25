#!/usr/bin/env python3
"""
脚本用于转换位姿文件格式
从JSON格式（包含10个轨迹，每个轨迹81帧）转换为NPZ格式（每个轨迹一个文件）
"""

import json
import numpy as np
import os
import argparse
import re


def parse_matrix_string(matrix_str):
    """
    解析矩阵字符串，将其转换为4x4 numpy数组
    输入格式: "[1 0 0 0] [-0 1 0 0] [0 -0 1 0] [3390 1380 240 1]"
    """
    # 使用正则表达式提取所有数字（包括负数和小数）
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', matrix_str)
    
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers, got {len(numbers)}: {matrix_str}")
    
    # 转换为float并重塑为4x4矩阵
    matrix = np.array([float(x) for x in numbers], dtype=np.float32).reshape(4, 4)
    return matrix


def convert_json_to_npz(json_file_path, output_dir):
    """
    将JSON文件转换为10个NPZ文件
    
    Args:
        json_file_path: 输入JSON文件路径
        output_dir: 输出目录路径
    """
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 获取帧数和相机数
    frames = sorted([k for k in data.keys() if k.startswith('frame')], 
                   key=lambda x: int(x.replace('frame', '')))
    cameras = sorted([k for k in data[frames[0]].keys() if k.startswith('cam')],
                    key=lambda x: int(x.replace('cam', '')))
    
    print(f"发现 {len(frames)} 帧，{len(cameras)} 个相机")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个相机创建一个NPZ文件
    for cam_idx, camera in enumerate(cameras, 1):
        print(f"处理相机 {camera}...")
        
        # 收集该相机的所有帧数据
        camera_matrices = []
        frame_indices = []
        
        for frame_idx, frame in enumerate(frames):
            try:
                matrix_str = data[frame][camera]
                matrix = parse_matrix_string(matrix_str)
                camera_matrices.append(matrix)
                frame_indices.append(frame_idx)
            except Exception as e:
                print(f"警告: 无法解析 {frame}/{camera}: {e}")
                continue
        
        if not camera_matrices:
            print(f"警告: 相机 {camera} 没有有效数据")
            continue
        
        # 转换为numpy数组
        data_array = np.stack(camera_matrices, axis=0)  # shape: (num_frames, 4, 4)
        inds_array = np.array(frame_indices, dtype=np.int64)
        
        # 保存为NPZ文件
        output_file = os.path.join(output_dir, f"{cam_idx}.npz")
        np.savez(output_file, data=data_array, inds=inds_array)
        
        print(f"保存 {camera} 到 {output_file}, 形状: data={data_array.shape}, inds={inds_array.shape}")
    
    print("转换完成！")


def main():
    parser = argparse.ArgumentParser(description='转换位姿文件格式从JSON到NPZ')
    parser.add_argument('input_json', help='输入JSON文件路径')
    parser.add_argument('-o', '--output-dir', default='./output_npz', 
                       help='输出目录路径 (默认: ./output_npz)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"错误: 输入文件不存在: {args.input_json}")
        return 1
    
    try:
        convert_json_to_npz(args.input_json, args.output_dir)
        return 0
    except Exception as e:
        print(f"转换失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())