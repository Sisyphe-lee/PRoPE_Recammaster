#!/usr/bin/env python3
"""
使用 HfApi 列出 Hugging Face 数据集中的所有文件
支持命令行参数配置
HF_ENDPOINT="https://hf-mirror.com" &&python -m tools.list_hf_dataset_files --repo-id=nvidia/vipe-wild-sdg-1m
HF_ENDPOINT="https://hf-mirror.com" &&python -m tools.list_hf_dataset_files --repo-id=nvidia/vipe-web360
HF_ENDPOINT="https://hf-mirror.com" &&python -m tools.list_hf_dataset_files --repo-id=nvidia/vipe-dynpose-100kpp
"""

import os
import click
from huggingface_hub import HfApi

def list_dataset_files(repo_id, token, save_to_file, output_file, filter_pattern, show_details):
    """列出数据集中的所有文件"""
    # print(f"正在连接到 {endpoint}")
    print(f"正在列出数据集: {repo_id}")
    print("=" * 60)
    
    # 创建 HfApi 实例
    api = HfApi(token=token)
    
    try:
        # 获取数据集的所有文件列表
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # 应用过滤模式
        if filter_pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, filter_pattern)]
            print(f"应用过滤模式 '{filter_pattern}'，找到 {len(files)} 个匹配文件")
        
        print(f"找到 {len(files)} 个文件:")
        print("-" * 60)
        
        # 按文件类型分类
        file_types = {
            "metadata": [],
            "payload": [],
            "other": []
        }
        
        for file_path in sorted(files):
            if file_path.startswith("meta"):
                file_types["metadata"].append(file_path)
            elif file_path.startswith("payload"):
                file_types["payload"].append(file_path)
            else:
                file_types["other"].append(file_path)
        
        # 显示元数据文件
        if file_types["metadata"]:
            print("📊 元数据文件:")
            for file_path in file_types["metadata"]:
                print(f"  - {file_path}")
            print()
        
        # 显示载荷文件（按目录分组）
        if file_types["payload"]:
            print("📦 载荷文件:")
            payload_dirs = {}
            for file_path in file_types["payload"]:
                # 提取目录结构
                parts = file_path.split("/")
                if len(parts) >= 2:
                    dir_name = parts[1]  # payload/目录名/...
                    if dir_name not in payload_dirs:
                        payload_dirs[dir_name] = []
                    payload_dirs[dir_name].append(file_path)
            
            for dir_name in sorted(payload_dirs.keys()):
                print(f"  📁 {dir_name}/")
                for file_path in sorted(payload_dirs[dir_name]):
                    # 只显示文件名，不显示完整路径
                    filename = file_path.split("/")[-1]
                    if show_details:
                        # 获取文件详细信息
                        file_info = get_file_info(repo_id, file_path, api)
                        if file_info:
                            size_mb = file_info.size / (1024 * 1024) if file_info.size else 0
                            print(f"    - {filename} ({size_mb:.2f} MB)")
                        else:
                            print(f"    - {filename}")
                    else:
                        print(f"    - {filename}")
                print()
        
        # 显示其他文件
        if file_types["other"]:
            print("📄 其他文件:")
            for file_path in file_types["other"]:
                print(f"  - {file_path}")
            print()
        
        # 统计信息
        print("=" * 60)
        print("📈 统计信息:")
        print(f"  总文件数: {len(files)}")
        print(f"  元数据文件: {len(file_types['metadata'])}")
        print(f"  载荷文件: {len(file_types['payload'])}")
        print(f"  其他文件: {len(file_types['other'])}")
        
        # 分析载荷文件结构
        if file_types["payload"]:
            print("\n🔍 载荷文件分析:")
            tar_files = [f for f in file_types["payload"] if f.endswith('.tar')]
            print(f"  .tar 文件数量: {len(tar_files)}")
            
            # 按属性分类
            attributes = set()
            for file_path in file_types["payload"]:
                if "/" in file_path:
                    parts = file_path.split("/")
                    if len(parts) >= 3:
                        attr = parts[2].replace(".tar", "")
                        attributes.add(attr)
            
            if attributes:
                print(f"  包含的属性: {', '.join(sorted(attributes))}")
        
        # 保存到文件
        if save_to_file:
            save_files_to_file(files, output_file, file_types)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return True

def get_file_info(repo_id, file_path, api):
    """获取特定文件的详细信息"""
    try:
        file_info = api.repo_file_info(repo_id=repo_id, repo_type="dataset", filename=file_path)
        return file_info
    except Exception as e:
        print(f"无法获取文件 {file_path} 的信息: {e}")
        return None

def save_files_to_file(files, output_file, file_types):
    """保存文件列表到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("数据集文件列表\n")
            f.write(f"总文件数: {len(files)}\n")
            f.write(f"生成时间: {os.popen('date').read().strip()}\n")
            f.write("=" * 60 + "\n\n")
            
            # 元数据文件
            if file_types["metadata"]:
                f.write("📊 元数据文件:\n")
                for file_path in file_types["metadata"]:
                    f.write(f"  - {file_path}\n")
                f.write("\n")
            
            # 载荷文件
            if file_types["payload"]:
                f.write("📦 载荷文件:\n")
                for file_path in sorted(file_types["payload"]):
                    f.write(f"  - {file_path}\n")
                f.write("\n")
            
            # 其他文件
            if file_types["other"]:
                f.write("📄 其他文件:\n")
                for file_path in file_types["other"]:
                    f.write(f"  - {file_path}\n")
                f.write("\n")
        
        print(f"✅ 文件列表已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

@click.command()
@click.option('--repo-id', '-r', 
              default='nvidia/vipe-wild-sdg-1m',
              help='数据集仓库ID (默认: nvidia/vipe-wild-sdg-1m)')
# @click.option('--endpoint', '-e',
#               default='https://hf-mirror.com',
#               help='Hugging Face 端点 (默认: https://hf-mirror.com)')
@click.option('--token', '-t',
              default=None,
              help='Hugging Face 访问令牌 (默认: 从环境变量 HF_TOKEN 获取)')
@click.option('--save', '-s',
              is_flag=True,
              help='保存文件列表到文件')
@click.option('--output', '-o',
              default='dataset_files_list.txt',
              help='输出文件名 (默认: dataset_files_list.txt)')
@click.option('--filter', '-f',
              default=None,
              help='文件过滤模式 (支持通配符，如: *.tar, payload/*)')
@click.option('--details', '-d',
              is_flag=True,
              help='显示文件详细信息 (大小等)')
@click.option('--count-only', '-c',
              is_flag=True,
              help='只显示文件数量统计')
def main(repo_id, token, save, output, filter, details, count_only):
    """列出 Hugging Face 数据集中的所有文件"""
    
    # 获取 token
    if not token:
        token = os.getenv("HF_TOKEN")
    
    print("🚀 开始列出数据集文件...")
    print(f"数据集: {repo_id}")
    # print(f"端点: {endpoint}")
    if filter:
        print(f"过滤模式: {filter}")
    print()
    
    success = list_dataset_files(repo_id, token, save, output, filter, details)
    
    if success:
        print("\n✅ 文件列表获取完成!")
    else:
        print("\n❌ 文件列表获取失败!")

if __name__ == "__main__":
    main()
