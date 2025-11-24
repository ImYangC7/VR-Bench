#!/usr/bin/env python3
import numpy as np
from typing import Dict, Any, Tuple

def normalize_trajectory(traj, video_width, video_height):
    normalized = traj.copy().astype(np.float64)
    normalized[:, 0] /= video_width
    normalized[:, 1] /= video_height
    return normalized

def resample_by_length(points, N):
    if len(points) < 2:
        return np.tile(points[0] if len(points) > 0 else [0, 0], (N, 1))
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg_lengths)])
    if s[-1] < 1e-10:
        return np.tile(points[0], (N, 1))
    target_s = np.linspace(0, s[-1], N)
    new_x = np.interp(target_s, s, points[:, 0])
    new_y = np.interp(target_s, s, points[:, 1])
    return np.column_stack([new_x, new_y])

def compute_path_length(traj):
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

class BaseMetric:
    @property
    def name(self):
        raise NotImplementedError
    def compute(self, gt_traj, gen_traj, **kwargs):
        raise NotImplementedError

class PrecisionRateMetric(BaseMetric):
    def __init__(self, eps_ratio=0.01, num_samples=100):
        self.eps_ratio = eps_ratio
        self.num_samples = num_samples
    @property
    def name(self):
        return "pr"
    def compute(self, gt_traj, gen_traj, **kwargs):
        if len(gt_traj) < 2 or len(gen_traj) < 2:
            return 0.0, {}
        L_gt = compute_path_length(gt_traj)
        L_gen = compute_path_length(gen_traj)

        # 按物理距离比对：以GT的长度为基准
        # GT重采样：在0到L_gt之间均匀取num_samples个点
        gt_resampled = resample_by_length(gt_traj, self.num_samples)

        # Gen重采样：在相同的物理距离点上采样
        # 需要在Gen轨迹上找到对应物理距离的位置
        gen_resampled = self._resample_gen_by_gt_length(gen_traj, L_gt, self.num_samples)

        # 计算每对点的距离
        distances = np.linalg.norm(gt_resampled - gen_resampled, axis=1)

        # 优化：向量化查找第一个不匹配的点
        # 创建布尔数组：True表示匹配，False表示不匹配
        matches = distances <= self.eps_ratio

        # 查找第一个False的位置
        # 如果全部匹配，则matched_count = num_samples
        # 否则matched_count = 第一个False的索引
        first_mismatch = np.argmax(~matches)  # argmax找到第一个True（即第一个不匹配）

        if matches[first_mismatch]:
            # 如果第一个"不匹配"的位置实际上是匹配的，说明全部匹配
            matched_count = self.num_samples
        else:
            matched_count = first_mismatch

        pr = float(matched_count / self.num_samples)

        return pr, {
            "gt_length": L_gt,
            "gen_length": L_gen,
            "gt_resampled": gt_resampled,
            "gen_resampled": gen_resampled,
            "distances": distances,
            "matched_count": matched_count
        }

    def _resample_gen_by_gt_length(self, gen_traj, gt_length, num_samples):
        """
        按GT的物理长度重采样Gen轨迹（优化版：向量化操作）

        Args:
            gen_traj: Gen轨迹点
            gt_length: GT的总长度
            num_samples: 采样点数

        Returns:
            重采样后的Gen轨迹点（num_samples个点）
        """
        # 计算Gen轨迹的累积距离
        seg_lengths = np.linalg.norm(np.diff(gen_traj, axis=0), axis=1)
        cumsum_distances = np.concatenate([[0], np.cumsum(seg_lengths)])

        # 在0到gt_length之间均匀取num_samples个点
        target_distances = np.linspace(0, gt_length, num_samples)

        # 优化：向量化插值，一次性处理所有点
        gen_max_dist = cumsum_distances[-1]

        # 对于超出Gen长度的目标距离，截断到Gen的最大距离
        # 这样插值会自动返回Gen的终点坐标
        target_distances_clipped = np.minimum(target_distances, gen_max_dist)

        # 向量化插值：一次性插值所有x和y坐标
        x_resampled = np.interp(target_distances_clipped, cumsum_distances, gen_traj[:, 0])
        y_resampled = np.interp(target_distances_clipped, cumsum_distances, gen_traj[:, 1])

        # 组合成二维数组
        return np.column_stack([x_resampled, y_resampled])

class StepMetric(BaseMetric):
    @property
    def name(self):
        return "step"
    def compute(self, gt_traj, gen_traj, **kwargs):
        # Step 只在 SR=1 时计算
        sr = kwargs.get("sr", 0.0)
        if sr < 1.0:
            return None, {"gt_length": 0.0, "gen_length": 0.0}

        gt_length = kwargs.get("gt_length") or compute_path_length(gt_traj)
        gen_length = kwargs.get("gen_length") or compute_path_length(gen_traj)
        if gt_length < 1e-10:
            return 0.0, {"gt_length": gt_length, "gen_length": gen_length}
        step = gen_length / gt_length - 1.0

        # 当 Step 为负数时，返回 None（记录为 N/A）
        if step < 0:
            return None, {"gt_length": gt_length, "gen_length": gen_length}

        return float(step), {"gt_length": gt_length, "gen_length": gen_length}

class ExactMatchMetric(BaseMetric):
    def __init__(self, pr_threshold=0.98, step_threshold=0.1):
        self.pr_threshold = pr_threshold
        self.step_threshold = step_threshold
    @property
    def name(self):
        return "em"
    def compute(self, gt_traj, gen_traj, **kwargs):
        # EM 只在 SR=1 时计算
        sr = kwargs.get("sr", 0.0)
        if sr < 1.0:
            return 0.0, {"is_perfect": False}

        pr = kwargs.get("pr", 0.0)
        step = kwargs.get("step", 0.0)
        # 处理 step=None 的情况（当 SR<1 时）
        if step is None:
            step = 0.0
        is_perfect = (pr >= self.pr_threshold) and (abs(step) <= self.step_threshold)
        return float(is_perfect), {"is_perfect": is_perfect}


class SuccessRateMetric(BaseMetric):
    """SR指标：只要采样点中任意一个点到达goal的bbox，就认为成功"""

    @property
    def name(self):
        return "sr"

    def compute(self, gt_traj, gen_traj, **kwargs):
        goal_bbox = kwargs.get("goal_bbox")
        gen_resampled = kwargs.get("gen_resampled")

        if goal_bbox is None or gen_resampled is None:
            return 0.0, {"reached_goal": False}

        # goal_bbox: (x, y, width, height) 归一化坐标
        x, y, w, h = goal_bbox

        # CPU版本：检查是否有点在goal bbox内
        in_x = (gen_resampled[:, 0] >= x) & (gen_resampled[:, 0] <= x + w)
        in_y = (gen_resampled[:, 1] >= y) & (gen_resampled[:, 1] <= y + h)
        in_bbox = in_x & in_y
        reached = bool(np.any(in_bbox))

        return float(reached), {"reached_goal": reached}
