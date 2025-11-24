#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, Any, Optional, Tuple
from core.schema import UnifiedState

# 尝试导入GPU版本，如果失败则使用CPU版本
try:
    from evaluation.videomodel_eval.metrics_gpu import (
        PrecisionRateMetricGPU as PrecisionRateMetric,
        StepMetric,
        ExactMatchMetric,
        SuccessRateMetric,
        FidelityMetric,
        normalize_trajectory,
        GPU_AVAILABLE
    )
    if GPU_AVAILABLE:
        print("✓ GPU加速已启用")
except ImportError:
    from evaluation.videomodel_eval.metrics import (
        PrecisionRateMetric,
        StepMetric,
        ExactMatchMetric,
        normalize_trajectory
    )
    GPU_AVAILABLE = False
    print("✓ 使用CPU计算")
    # CPU版本没有SR和Fidelity，需要从GPU版本导入
    from evaluation.videomodel_eval.metrics_gpu import SuccessRateMetric, FidelityMetric


class TrajectoryEvaluator:
    """
    轨迹评估器 - 基于 GUI 路径跟踪一致性评测指标定义文档 v1.1
    """

    def __init__(self, eps_ratio: float = 0.01, num_samples: int = 1000,
                 pr_threshold: float = 0.98, step_threshold: float = 0.1,
                 fidelity_frame_step: int = 5, fidelity_pixel_threshold: int = 5):
        """
        Args:
            eps_ratio: 匹配阈值（相对对角线比例），默认 0.01 (1%)
            num_samples: 采样点数量，默认 1000
            pr_threshold: Exact Match 的 PR 阈值，默认 0.98
            step_threshold: Exact Match 的 Step 阈值，默认 0.1
            fidelity_frame_step: 保真度计算的帧采样步长，默认 5
            fidelity_pixel_threshold: 保真度计算的像素差异阈值，默认 5（±5灰度值）
        """
        self.eps_ratio = eps_ratio
        self.num_samples = num_samples
        self.pr_threshold = pr_threshold
        self.step_threshold = step_threshold

        self.metrics = [
            PrecisionRateMetric(eps_ratio, num_samples),
            SuccessRateMetric(),
            StepMetric(),
            ExactMatchMetric(pr_threshold, step_threshold),
            FidelityMetric(frame_step=fidelity_frame_step, pixel_threshold=fidelity_pixel_threshold),
        ]
    
    def evaluate(self,
                 gt_traj: np.ndarray,
                 gen_traj: np.ndarray,
                 video_width: int,
                 video_height: int,
                 state: Optional[UnifiedState] = None,
                 gen_box_traj: Optional[np.ndarray] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        评估两条轨迹

        Args:
            gt_traj: Ground truth 轨迹 (N, 2)，像素坐标（玩家轨迹）
            gen_traj: Generated 轨迹 (M, 2)，像素坐标（玩家轨迹）
            video_width: 视频宽度
            video_height: 视频高度
            state: UnifiedState 对象（可选）
            gen_box_traj: Generated 箱子轨迹（仅推箱子游戏，用于SR计算）
            **kwargs: 额外参数

        Returns:
            result: 评估结果字典
        """
        if len(gt_traj) < 2 or len(gen_traj) < 2:
            return self._empty_result()

        # 归一化轨迹到 [0,1]×[0,1]
        gt_traj_norm = normalize_trajectory(gt_traj, video_width, video_height)
        gen_traj_norm = normalize_trajectory(gen_traj, video_width, video_height)

        result = {}
        shared_data = {
            'state': state,
            'video_width': video_width,
            'video_height': video_height,
        }

        # 如果有state，提取goal bbox并归一化
        if state is not None:
            goal_bbox_pixel = state.goal.bbox
            # 使用 state 中记录的原始图片尺寸来归一化 bbox
            # 因为 bbox 坐标是基于原始图片的，而不是生成视频的尺寸
            state_width = state.render.image_width
            state_height = state.render.image_height
            goal_bbox_norm = (
                goal_bbox_pixel.x / state_width,
                goal_bbox_pixel.y / state_height,
                goal_bbox_pixel.width / state_width,
                goal_bbox_pixel.height / state_height
            )
            shared_data['goal_bbox'] = goal_bbox_norm

        # 如果是推箱子游戏，归一化箱子轨迹并用于 SR 计算
        if gen_box_traj is not None:
            gen_box_traj_norm = normalize_trajectory(gen_box_traj, video_width, video_height)
            shared_data['gen_box_traj'] = gen_box_traj_norm

        # 依次计算所有 metrics
        for metric in self.metrics:
            value, extra = metric.compute(gt_traj_norm, gen_traj_norm, **kwargs, **shared_data)
            result[metric.name] = value
            shared_data.update(extra)
            # 将 metric 的值也加入 shared_data，供后续 metric 使用
            shared_data[metric.name] = value

        # 添加额外数据（用于可视化和调试）
        result.update({
            'gt_length': shared_data.get('gt_length', 0.0),
            'gen_length': shared_data.get('gen_length', 0.0),
            'is_perfect': shared_data.get('is_perfect', False),
            'gt_resampled': shared_data.get('gt_resampled'),
            'gen_resampled': shared_data.get('gen_resampled'),
            'distances': shared_data.get('distances')
        })

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'pr': 0.0,
            'sd': 0.0,
            'em': 0.0,
            'sr': 0.0,
            'mf': 0.0,
            'gt_length': 0.0,
            'gen_length': 0.0,
            'is_perfect': False
        }


