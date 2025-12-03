"""评估系统 - 包含视频模型评估和VLM评估"""
from . import vlm_eval
from . import videomodel_eval

# 从videomodel_eval导出
from evaluation.videomodel_eval.extractor import CSRTTracker
from evaluation.videomodel_eval.evaluator import TrajectoryEvaluator
from evaluation.videomodel_eval.metrics import (
    PrecisionRateMetric,
    StepMetric,
    ExactMatchMetric,
    normalize_trajectory,
    resample_by_length,
    compute_path_length
)

__all__ = [
    'vlm_eval',
    'videomodel_eval',
    'CSRTTracker',
    'TrajectoryEvaluator',
    'PrecisionRateMetric',
    'StepMetric',
    'ExactMatchMetric',
    'normalize_trajectory',
    'resample_by_length',
    'compute_path_length'
]