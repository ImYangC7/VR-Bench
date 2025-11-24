#!/usr/bin/env python3
"""
GPU加速版本的metrics计算
使用CuPy替代NumPy进行GPU加速

安装: pip install cupy-cuda12x  (根据你的CUDA版本选择)
或者: pip install cupy-cuda11x

如果没有GPU或CuPy，会自动回退到CPU版本
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

# 尝试导入CuPy，如果失败则使用NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU加速已启用 (CuPy)")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("GPU不可用，使用CPU计算 (NumPy)")


def normalize_trajectory(traj, video_width, video_height):
    """归一化轨迹到 [0,1]×[0,1]"""
    normalized = traj.copy().astype(np.float64)
    normalized[:, 0] /= video_width
    normalized[:, 1] /= video_height
    return normalized


def resample_by_length(points, N):
    """按物理距离重采样轨迹"""
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
    """计算路径总长度"""
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


class BaseMetric:
    @property
    def name(self):
        raise NotImplementedError
    def compute(self, gt_traj, gen_traj, **kwargs):
        raise NotImplementedError


class PrecisionRateMetricGPU(BaseMetric):
    """GPU加速版本的PR计算"""
    
    def __init__(self, eps_ratio=0.01, num_samples=100):
        self.eps_ratio = eps_ratio
        self.num_samples = num_samples
        self.use_gpu = GPU_AVAILABLE
    
    @property
    def name(self):
        return "pr"
    
    def compute(self, gt_traj, gen_traj, **kwargs):
        if len(gt_traj) < 2 or len(gen_traj) < 2:
            return 0.0, {}
        
        # CPU上计算路径长度（数据量小，不值得传输到GPU）
        L_gt = compute_path_length(gt_traj)
        L_gen = compute_path_length(gen_traj)
        
        # CPU上重采样（涉及插值，NumPy的interp很高效）
        gt_resampled = resample_by_length(gt_traj, self.num_samples)
        gen_resampled = self._resample_gen_by_gt_length(gen_traj, L_gt, self.num_samples)
        
        if self.use_gpu and self.num_samples >= 100:
            # 数据量足够大时使用GPU加速
            matched_count = self._compute_pr_gpu(gt_resampled, gen_resampled)
        else:
            # 数据量小或无GPU时使用CPU
            matched_count = self._compute_pr_cpu(gt_resampled, gen_resampled)
        
        pr = float(matched_count / self.num_samples)
        
        # 计算distances用于返回（在CPU上）
        distances = np.linalg.norm(gt_resampled - gen_resampled, axis=1)
        
        return pr, {
            "gt_length": L_gt,
            "gen_length": L_gen,
            "gt_resampled": gt_resampled,
            "gen_resampled": gen_resampled,
            "distances": distances,
            "matched_count": matched_count
        }
    
    def _compute_pr_gpu(self, gt_resampled, gen_resampled):
        """GPU加速的PR计算"""
        # 将数据传输到GPU
        gt_gpu = cp.asarray(gt_resampled)
        gen_gpu = cp.asarray(gen_resampled)
        
        # GPU上计算距离
        distances_gpu = cp.linalg.norm(gt_gpu - gen_gpu, axis=1)
        
        # GPU上查找第一个不匹配的点
        matches_gpu = distances_gpu <= self.eps_ratio
        first_mismatch = int(cp.argmax(~matches_gpu))
        
        # 检查是否全部匹配
        if bool(matches_gpu[first_mismatch]):
            matched_count = self.num_samples
        else:
            matched_count = first_mismatch
        
        return matched_count
    
    def _compute_pr_cpu(self, gt_resampled, gen_resampled):
        """CPU版本的PR计算（向量化）"""
        distances = np.linalg.norm(gt_resampled - gen_resampled, axis=1)
        matches = distances <= self.eps_ratio
        first_mismatch = np.argmax(~matches)
        
        if matches[first_mismatch]:
            matched_count = self.num_samples
        else:
            matched_count = first_mismatch
        
        return matched_count
    
    def _resample_gen_by_gt_length(self, gen_traj, gt_length, num_samples):
        """按GT的物理长度重采样Gen轨迹（向量化版本）"""
        seg_lengths = np.linalg.norm(np.diff(gen_traj, axis=0), axis=1)
        cumsum_distances = np.concatenate([[0], np.cumsum(seg_lengths)])
        target_distances = np.linspace(0, gt_length, num_samples)
        
        gen_max_dist = cumsum_distances[-1]
        target_distances_clipped = np.minimum(target_distances, gen_max_dist)
        
        x_resampled = np.interp(target_distances_clipped, cumsum_distances, gen_traj[:, 0])
        y_resampled = np.interp(target_distances_clipped, cumsum_distances, gen_traj[:, 1])
        
        return np.column_stack([x_resampled, y_resampled])


class StepMetric(BaseMetric):
    @property
    def name(self):
        return "sd"
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
        sd = kwargs.get("sd", 0.0)
        # ���� sd=None ��������� SR<1 ʱ�� 的情况（当 SR<1 时）
        if sd is None:
            sd = 0.0
        is_perfect = (pr >= self.pr_threshold) and (abs(sd) <= self.step_threshold)
        return float(is_perfect), {"is_perfect": is_perfect}


class SuccessRateMetric(BaseMetric):
    """SR指标：只要采样点中任意一个点到达goal的bbox，就认为成功

    对于推箱子游戏，使用箱子轨迹而不是玩家轨迹
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE

    @property
    def name(self):
        return "sr"

    def compute(self, gt_traj, gen_traj, **kwargs):
        goal_bbox = kwargs.get("goal_bbox")

        if goal_bbox is None:
            return 0.0, {"reached_goal": False}

        # goal_bbox: (x, y, width, height) 归一化坐标
        x, y, w, h = goal_bbox

        # 如果是推箱子游戏，使用箱子轨迹
        gen_box_traj = kwargs.get("gen_box_traj")
        if gen_box_traj is not None:
            traj_to_check = gen_box_traj
        else:
            traj_to_check = gen_traj

        # 使用原始的 gen_traj 而不是 gen_resampled
        # 因为 gen_resampled 是按 GT 长度重采样的，如果 gen 轨迹较短会被截断
        if self.use_gpu and len(traj_to_check) >= 100:
            reached = self._check_goal_gpu(traj_to_check, x, y, w, h)
        else:
            reached = self._check_goal_cpu(traj_to_check, x, y, w, h)

        return float(reached), {"reached_goal": reached}

    def _check_goal_gpu(self, points, x, y, w, h):
        """GPU版本：检查是否有点在goal bbox内"""
        points_gpu = cp.asarray(points)
        in_x = (points_gpu[:, 0] >= x) & (points_gpu[:, 0] <= x + w)
        in_y = (points_gpu[:, 1] >= y) & (points_gpu[:, 1] <= y + h)
        in_bbox = in_x & in_y
        return bool(cp.any(in_bbox))

    def _check_goal_cpu(self, points, x, y, w, h):
        """CPU版本：检查是否有点在goal bbox内"""
        in_x = (points[:, 0] >= x) & (points[:, 0] <= x + w)
        in_y = (points[:, 1] >= y) & (points[:, 1] <= y + h)
        in_bbox = in_x & in_y
        return bool(np.any(in_bbox))


class FidelityMetric(BaseMetric):
    """
    保真度指标：衡量视频背景的像素级稳定性

    计算方式：
    1. 从第一帧提取背景图A：移除 start_bbox 和 goal_bbox 区域
    2. 从每N帧提取背景图B：移除 goal_bbox、start_bbox 和 player_bbox 区域
    3. 计算每个图B与图A的像素级差异（MAE - Mean Absolute Error）
    4. 应用阈值容忍视频编码误差
    5. 取平均值得到保真度

    保真度 = 1.0 - (超过阈值的像素比例)
    范围：[0, 1]，1.0 表示背景完全不变，0.0 表示背景完全改变
    """

    def __init__(self, frame_step: int = 5, pixel_threshold: int = 5):
        """
        Args:
            frame_step: 帧采样步长，每隔多少帧采样一次（默认5）
            pixel_threshold: 像素差异阈值，容忍视频编码误差（默认5，即±5灰度值）
        """
        self.frame_step = frame_step
        self.pixel_threshold = pixel_threshold

    @property
    def name(self):
        return "mf"

    def compute(self, gt_traj, gen_traj, **kwargs):
        """
        计算保真度

        Args:
            gt_traj: GT轨迹（归一化坐标）
            gen_traj: 生成轨迹（归一化坐标）
            **kwargs: 必须包含：
                - gen_video_path: 生成视频路径
                - state: UnifiedState 对象
                - video_width: 视频宽度
                - video_height: 视频高度

        Returns:
            fidelity: 保真度分数 [0, 1]，越高越好
            extra: 额外信息
        """
        gen_video_path = kwargs.get("gen_video_path")
        state = kwargs.get("state")
        video_width = kwargs.get("video_width")
        video_height = kwargs.get("video_height")

        if gen_video_path is None or state is None:
            return 0.0, {"fidelity_scores": []}

        # 打开视频
        cap = cv2.VideoCapture(gen_video_path)
        if not cap.isOpened():
            return 0.0, {"fidelity_scores": []}

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return 0.0, {"fidelity_scores": []}

        # 获取 bbox（像素坐标）
        start_bbox = state.player.bbox
        goal_bbox = state.goal.bbox

        # 缩放 bbox 到视频尺寸
        state_width = state.render.image_width
        state_height = state.render.image_height
        scale_x = video_width / state_width
        scale_y = video_height / state_height

        start_bbox_scaled = self._scale_bbox(start_bbox, scale_x, scale_y)
        goal_bbox_scaled = self._scale_bbox(goal_bbox, scale_x, scale_y)

        # 将归一化轨迹转换为像素坐标
        # gen_traj 是归一化坐标 [0, 1]，需要转换为视频像素坐标
        gen_traj_pixels = gen_traj.copy()
        gen_traj_pixels[:, 0] *= video_width
        gen_traj_pixels[:, 1] *= video_height

        # 读取第一帧并提取背景A
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return 0.0, {"fidelity_scores": []}

        # 转换为灰度图
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # 不修改图像，只创建有效区域 mask
        # mask 标记哪些区域是有效的背景（未被抠除）
        valid_mask_A = np.ones((video_height, video_width), dtype=bool)
        valid_mask_A = self._mask_region(valid_mask_A, start_bbox_scaled, False)
        valid_mask_A = self._mask_region(valid_mask_A, goal_bbox_scaled, False)

        # 逐帧采样并计算相似度
        frame_idx = 0
        fidelity_scores = []

        # 重置视频到开头
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每 frame_step 帧采样一次
            if frame_idx % self.frame_step == 0 and frame_idx > 0:
                # 转换为灰度图
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 创建当前帧的有效区域 mask
                valid_mask_B = np.ones((video_height, video_width), dtype=bool)
                valid_mask_B = self._mask_region(valid_mask_B, start_bbox_scaled, False)
                valid_mask_B = self._mask_region(valid_mask_B, goal_bbox_scaled, False)

                # 根据轨迹动态计算当前帧的 player 位置并从 mask 中移除
                if len(gen_traj_pixels) > 0:
                    # 计算当前帧在轨迹中的位置（线性插值）
                    traj_idx = int((frame_idx / total_frames) * (len(gen_traj_pixels) - 1))
                    traj_idx = min(traj_idx, len(gen_traj_pixels) - 1)

                    # 获取当前帧的 player 位置
                    player_x, player_y = gen_traj_pixels[traj_idx]

                    # 创建 player bbox（以轨迹点为中心）
                    player_bbox_dynamic = {
                        'x': int(player_x - start_bbox_scaled['width'] / 2),
                        'y': int(player_y - start_bbox_scaled['height'] / 2),
                        'width': start_bbox_scaled['width'],
                        'height': start_bbox_scaled['height']
                    }

                    # 从 mask 中移除当前帧的 player 位置
                    valid_mask_B = self._mask_region(valid_mask_B, player_bbox_dynamic, False)

                # 计算像素级差异（MAE - Mean Absolute Error）
                # 取两个 mask 的交集，确保只比较两帧都有效的区域
                valid_mask = valid_mask_A & valid_mask_B

                if np.sum(valid_mask) > 0:
                    # 提取有效区域的像素值
                    pixels_A = first_frame_gray[valid_mask]
                    pixels_B = frame_gray[valid_mask]

                    # 计算绝对差异
                    pixel_diff = np.abs(pixels_A.astype(np.int16) - pixels_B.astype(np.int16))

                    # 应用阈值：超过阈值的像素认为是"背景变化"
                    changed_pixels = pixel_diff > self.pixel_threshold

                    # 计算保真度：1.0 - (变化像素的比例)
                    change_ratio = np.sum(changed_pixels) / len(changed_pixels)
                    fidelity_score = 1.0 - change_ratio
                else:
                    # 如果没有有效区域，保真度为 1.0（认为没有背景可比较）
                    fidelity_score = 1.0

                fidelity_scores.append(float(fidelity_score))

            frame_idx += 1

        cap.release()

        # 计算平均保真度
        if len(fidelity_scores) > 0:
            avg_fidelity = float(np.mean(fidelity_scores))
        else:
            avg_fidelity = 0.0

        return avg_fidelity, {
            "fidelity_scores": fidelity_scores,
            "num_frames_sampled": len(fidelity_scores)
        }

    def _scale_bbox(self, bbox, scale_x, scale_y):
        """缩放 bbox 到目标尺寸"""
        return {
            'x': int(bbox.x * scale_x),
            'y': int(bbox.y * scale_y),
            'width': int(bbox.width * scale_x),
            'height': int(bbox.height * scale_y)
        }

    def _mask_region(self, mask, bbox, value):
        """
        在 mask 上标记 bbox 区域

        Args:
            mask: bool 类型的 mask 数组
            bbox: bbox 字典 {'x', 'y', 'width', 'height'}
            value: 要设置的值（True 或 False）

        Returns:
            修改后的 mask
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        # 确保坐标在图像范围内
        x = max(0, min(x, mask.shape[1] - 1))
        y = max(0, min(y, mask.shape[0] - 1))
        w = min(w, mask.shape[1] - x)
        h = min(h, mask.shape[0] - y)

        # 设置区域值
        mask[y:y+h, x:x+w] = value
        return mask



