#!/bin/bash
# 根据皮肤批量生成关卡
# 用法: ./scripts/generate_by_skins.sh [config_file]

CONFIG_FILE=${1:-config/config.yaml}

python generation/batch_generate.py "$CONFIG_FILE"

