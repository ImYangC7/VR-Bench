"""评估系统 - 包含数据集评估和VLM评估"""

from . import vlm_eval
from . import dataset_eval

# 为了向后兼容，保留旧的导入路径
from evaluation.dataset_eval.extractor import TrajectoryExtractor
from evaluation.dataset_eval.evaluator import TrajectoryEvaluator
from evaluation.dataset_eval.metrics import (
    PrecisionRateMetric,
    StepMetric,
    ExactMatchMetric,
    normalize_trajectory,
    resample_by_length,
    compute_path_length
)

__all__ = [
    'vlm_eval',
    'dataset_eval',
    # 向后兼容
    'TrajectoryExtractor',
    'TrajectoryEvaluator',
    'PrecisionRateMetric',
    'StepMetric',
    'ExactMatchMetric',
    'normalize_trajectory',
    'resample_by_length',
    'compute_path_length'
]
