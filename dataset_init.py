#!/usr/bin/env python3
"""
从 Hugging Face 下载并解压 VR-Bench 数据集
"""

import argparse
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()


def download_and_extract(
    repo_id: str = "amagipeng/VR-Bench",
    output_dir: str = "./dataset_VR",
    token: str = None
):
    """
    下载并解压数据集
    
    Args:
        repo_id: Hugging Face 仓库 ID
        output_dir: 输出目录
        token: Hugging Face token (可选)
    """
    output_path = Path(output_dir)
    
    if output_path.exists() and any(output_path.iterdir()):
        logging.warning(f"目录 {output_dir} 已存在且非空")
        response = input("是否继续并覆盖? (y/n): ")
        if response.lower() != 'y':
            logging.info("取消下载")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    logging.info(f"开始下载数据集: {repo_id}")
    
    # 下载 train.tar.gz
    logging.info("\n下载 train.tar.gz...")
    train_file = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="train.tar.gz",
        token=token
    )
    
    logging.info("解压 train.tar.gz...")
    with tarfile.open(train_file, 'r:gz') as tar:
        tar.extractall(output_path)
    logging.info("✓ train 解压完成")
    
    # 下载 eval.tar.gz
    logging.info("\n下载 eval.tar.gz...")
    eval_file = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="eval.tar.gz",
        token=token
    )
    
    logging.info("解压 eval.tar.gz...")
    with tarfile.open(eval_file, 'r:gz') as tar:
        tar.extractall(output_path)
    logging.info("✓ eval 解压完成")
    
    # 下载 README
    try:
        logging.info("\n下载 README.md...")
        readme_file = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="README.md",
            token=token
        )
        import shutil
        shutil.copy(readme_file, output_path / "README.md")
        logging.info("✓ README.md 下载完成")
    except Exception as e:
        logging.warning(f"README.md 下载失败: {e}")
    
    logging.info(f"\n✓ 数据集下载并解压完成!")
    logging.info(f"数据集位置: {output_path.absolute()}")
    
    # 显示数据集结构
    logging.info("\n数据集结构:")
    for split in ['train', 'eval']:
        split_dir = output_path / split
        if split_dir.exists():
            games = [d.name for d in split_dir.iterdir() if d.is_dir()]
            logging.info(f"  {split}/: {', '.join(games)}")


def main():
    parser = argparse.ArgumentParser(
        description='从 Hugging Face 下载并解压 VR-Bench 数据集'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='amagipeng/VR-Bench',
        help='Hugging Face 仓库 ID (默认: amagipeng/VR-Bench)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset_VR',
        help='输出目录 (默认: ./dataset_VR)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face token (默认: 从 .env 文件读取 HF_TOKEN)'
    )
    
    args = parser.parse_args()
    
    download_and_extract(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token
    )


if __name__ == '__main__':
    main()

