#!/bin/bash
# 批量评估生成的视频（自动匹配所有难度）
# 用法: ./scripts/evaluate.sh

# 设置 CUDA 环境变量（修复 CuPy 编译问题）
export CUDA_PATH=${CUDA_HOME:-/usr/local/cuda}
export CPATH=$CUDA_PATH/include:${CPATH}
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}

NAME=sokoban
DATASET_DIR=dataset_VR/train/$NAME/4
OUTPUT_DIR=dataset_VR/train/$NAME/4
RESULT_DIR=eval_results/$NAME
WORKERS=4
NUM_SAMPLES=1000
THRESHOLD=0.05
FIDELITY_PIXEL_THRESHOLD=5
FRAME_STEP=1
TRACKER_TYPE=template  # 追踪器类型: csrt, template, optical_flow
SEARCH_MARGIN=50       # 模板匹配搜索边距（0=全图搜索，>0=局部搜索范围）
USE_GPU=gpu

# 构建Python命令
CMD="python evaluation/dataset_eval/batch_evaluate.py \
    \"$DATASET_DIR\" \
    \"$OUTPUT_DIR\" \
    \"$RESULT_DIR\" \
    --threshold \"$THRESHOLD\" \
    --num-samples \"$NUM_SAMPLES\" \
    --workers \"$WORKERS\" \
    --fidelity-pixel-threshold \"$FIDELITY_PIXEL_THRESHOLD\" \
    --frame-step \"$FRAME_STEP\" \
    --tracker-type \"$TRACKER_TYPE\" \
    --search-margin \"$SEARCH_MARGIN\""

# 如果指定了gpu参数，添加--gpu标志
if [ "$USE_GPU" = "gpu" ] || [ "$USE_GPU" = "GPU" ]; then
    CMD="$CMD --gpu"
    echo "启用GPU加速模式"
fi

# 执行命令
eval $CMD

