#!/bin/bash

CONFIG_FILE=config/vlm/maze3d_eval.yaml

python -m evaluation.vlm_eval.run_vlm_eval "$CONFIG_FILE"

