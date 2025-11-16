#!/bin/bash

# SGLang 服务启动脚本

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
HOST="0.0.0.0"
PORT=8123
CACHE_DIR="/data/pengyiran/cvpr_v1/huggingface_model"

# GPU 配置
# 方式 1: 使用单个 GPU
# GPU_IDS="2"
# TP_SIZE=1

# 方式 2: 使用多个 GPU (Tensor Parallelism)
GPU_IDS="2,3"
TP_SIZE=2

echo "启动 SGLang 服务..."
echo "模型: $MODEL_NAME"
echo "地址: http://$HOST:$PORT"
echo "缓存目录: $CACHE_DIR"
echo "GPU: $GPU_IDS"

# 单 GPU 模式
# CUDA_VISIBLE_DEVICES=$GPU_IDS python -m sglang.launch_server \
#     --model-path $MODEL_NAME \
#     --host $HOST \
#     --port $PORT \
#     --cache-dir $CACHE_DIR \
#     --trust-remote-code

# 设置 HuggingFace 缓存目录
export HF_HOME=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

# 多 GPU 模式
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tp $TP_SIZE \
    --download-dir $CACHE_DIR \
    --trust-remote-code \
    --skip-server-warmup

# 其他可选参数：
# --mem-fraction-static 0.9 # GPU 显存使用比例
# --chat-template qwen      # 指定 chat template
# --context-length 8192     # 最大上下文长度

