#!/bin/bash
# 为数据集生成所有最优路径的视频
# 用法: ./scripts/generate_videos.sh <dataset_dir> [workers]
# 示例: ./scripts/generate_videos.sh dataset/maze/1

DATASET_DIR=${1:-dataset/maze/1/easy}
WORKERS=${2:-8}
SKIN=/data/pengyiran/cvpr_v1/skins/maze/1
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

python generation/generate_videos.py \
    "$DATASET_DIR" \
    --workers "$WORKERS" \
    --skin "$SKIN" \
    2>&1 | grep -v "^Processing" | grep -v "^Frame"

