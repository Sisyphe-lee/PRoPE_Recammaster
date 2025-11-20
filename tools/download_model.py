#!/usr/bin/env python3
"""
模型下载工具
支持从 Hugging Face Hub 和 ModelScope 下载模型到指定路径
python tools/download_model.py -m Wan-AI/Wan2.1-T2V-1.3B -d /nas/datasets/huggingface/ -s hf
python tools/download_model.py -m black-forest-labs/FLUX.1-Kontext-dev -d /nas/datasets/modelscope/black-forest-labs/FLUX.1-Kontext-dev -s ms
python tools/download_model.py -m Wan-AI/Wan2.2-TI2V-5B -d /nas/datasets/modelscope/Wan-AI/Wan2.2-TI2V-5B -s ms
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import click


def download_from_huggingface(
    model_id: str, 
    local_dir: str, 
    token: Optional[str] = None,
    endpoint: str = "https://hf-mirror.com",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> bool:
    """从 Hugging Face Hub 下载模型"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"🚀 从 Hugging Face Hub 下载模型: {model_id}")
        print(f"目标路径: {local_dir}")
        print(f"端点: {endpoint}")
        
        # 创建本地目录
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # 下载参数
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": local_dir,
            "endpoint": endpoint,
            "resume_download": True,
        }
        
        if token:
            download_kwargs["token"] = token
        if include_patterns:
            download_kwargs["include_patterns"] = include_patterns
        if exclude_patterns:
            download_kwargs["ignore_patterns"] = exclude_patterns
            
        # 执行下载
        local_path = snapshot_download(**download_kwargs)
        
        print(f"✅ 下载完成: {local_path}")
        return True
        
    except ImportError:
        print("❌ 错误: 需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def download_from_modelscope(
    model_id: str, 
    local_dir: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> bool:
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        
        print(f"🚀 从 ModelScope 下载模型: {model_id}")
        print(f"目标路径: {local_dir}")
        
        # 创建本地目录
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # 下载参数
        download_kwargs = {
            "model_id": model_id,
            "local_dir": local_dir,
            "cache_dir": None,  # 使用默认缓存
        }
        
        if include_patterns:
            download_kwargs["include_patterns"] = include_patterns
        if exclude_patterns:
            download_kwargs["ignore_patterns"] = exclude_patterns
            
        # 执行下载
        local_path = snapshot_download(**download_kwargs)
        
        print(f"✅ 下载完成: {local_path}")
        return True
        
    except ImportError:
        print("❌ 错误: 需要安装 modelscope")
        print("请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


@click.command()
@click.option('--model-id', '-m', 
              required=True,
              help='模型ID (如: Wan-AI/Wan2.1-T2V-1.3B)')
@click.option('--local-dir', '-d', 
              required=True,
              help='本地保存目录')
@click.option('--source', '-s',
              type=click.Choice(['hf', 'ms', 'auto']),
              default='auto',
              help='下载源: hf(HuggingFace), ms(ModelScope), auto(自动选择)')
@click.option('--token', '-t',
              default=None,
              help='访问令牌 (HuggingFace 需要)')
@click.option('--endpoint', '-e',
              default='https://hf-mirror.com',
              help='HuggingFace 端点')
@click.option('--include', '-i',
              multiple=True,
              help='包含文件模式 (可多次使用)')
@click.option('--exclude', '-x',
              multiple=True,
              help='排除文件模式 (可多次使用)')
@click.option('--force', '-f',
              is_flag=True,
              help='强制重新下载')
def main(model_id, local_dir, source, token, endpoint, include, exclude, force):
    """下载模型到指定路径
    
    示例:
    python download_model.py -m "Wan-AI/Wan2.1-T2V-1.3B" -d "./models/wan2.1"
    python download_model.py -m "stabilityai/stable-diffusion-2-1" -d "./models/sd21" -s hf
    python download_model.py -m "damo/nlp_structbert_sentence-embedding_chinese-base" -d "./models/structbert" -s ms
    """
    
    # 检查本地目录
    local_path = Path(local_dir)
    if local_path.exists() and not force:
        print(f"⚠️  目录已存在: {local_dir}")
        print("使用 --force 强制重新下载")
        return
    
    # 获取环境变量中的 token
    if not token:
        token = os.getenv("HF_TOKEN")
    
    # 处理文件模式
    include_patterns = list(include) if include else None
    exclude_patterns = list(exclude) if exclude else None
    
    print("=" * 60)
    print("📦 模型下载工具")
    print("=" * 60)
    print(f"模型ID: {model_id}")
    print(f"本地目录: {local_dir}")
    print(f"下载源: {source}")
    if include_patterns:
        print(f"包含模式: {include_patterns}")
    if exclude_patterns:
        print(f"排除模式: {exclude_patterns}")
    print()
    
    success = False
    
    if source == 'hf':
        success = download_from_huggingface(
            model_id, local_dir, token, endpoint, 
            include_patterns, exclude_patterns
        )
    elif source == 'ms':
        success = download_from_modelscope(
            model_id, local_dir, 
            include_patterns, exclude_patterns
        )
    elif source == 'auto':
        # 自动选择下载源
        print("🔍 自动选择下载源...")
        
        # 先尝试 HuggingFace
        print("尝试从 HuggingFace 下载...")
        success = download_from_huggingface(
            model_id, local_dir, token, endpoint, 
            include_patterns, exclude_patterns
        )
        
        if not success:
            print("\n尝试从 ModelScope 下载...")
            success = download_from_modelscope(
                model_id, local_dir, 
                include_patterns, exclude_patterns
            )
    
    if success:
        print("\n🎉 模型下载完成!")
        print(f"模型位置: {Path(local_dir).absolute()}")
    else:
        print("\n❌ 模型下载失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()