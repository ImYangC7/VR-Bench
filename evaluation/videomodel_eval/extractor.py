#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from core.schema import UnifiedState
from .advanced_trackers import TemplateMatchingTracker, OpticalFlowTracker


class TrajectoryExtractor:
    """从视频中提取轨迹"""

    def __init__(self, state_path: str, tracker_type: str = 'template', search_margin: int = 50):
        """
        Args:
            state_path: UnifiedState JSON 文件路径
            tracker_type: 追踪器类型 ('csrt', 'template', 'optical_flow')
            search_margin: 模板匹配搜索边距（0=全图搜索，>0=局部搜索）
        """
        self.state = UnifiedState.load(state_path)
        self.player_bbox = self.state.get_player_bbox()
        self.has_box = bool(self.state.boxes)
        if self.has_box:
            self.box_bbox = self.state.boxes[0].bbox
        self.tracker_type = tracker_type
        self.search_margin = search_margin
    
    def extract(self, video_path: str, target_size=None, frame_step=1):
        """
        从视频中提取轨迹（玩家轨迹）

        流程：
        1. 读取视频并统一规格（尺寸）
        2. 将 state 中的 bbox 归一化到视频尺寸
        3. 从第一帧提取起始位置的 bbox 区域图像
        4. 用 bbox 区域初始化 tracker
        5. 逐帧跟踪并采样

        Args:
            video_path: 视频文件路径
            target_size: 目标尺寸 (width, height)，None 表示使用原始尺寸
            frame_step: 采样步长，1 表示每帧采样，2 表示每 2 帧采样一次

        Returns:
            trajectory: numpy array of shape (N, 2)，像素坐标（玩家轨迹）
            first_frame: 第一帧图像（用于可视化，已调整尺寸）
            video_width: 视频宽度
            video_height: 视频高度
        """
        return self._extract_trajectory(video_path, self.player_bbox, target_size, frame_step)

    def extract_box(self, video_path: str, target_size=None, frame_step=1):
        """
        从视频中提取箱子轨迹（仅用于推箱子游戏）

        Args:
            video_path: 视频文件路径
            target_size: 目标尺寸 (width, height)，None 表示使用原始尺寸
            frame_step: 采样步长，1 表示每帧采样，2 表示每 2 帧采样一次

        Returns:
            trajectory: numpy array of shape (N, 2)，像素坐标（箱子轨迹）
            first_frame: 第一帧图像（用于可视化，已调整尺寸）
            video_width: 视频宽度
            video_height: 视频高度
        """
        if not self.has_box:
            raise ValueError("This state has no box, cannot extract box trajectory")
        return self._extract_trajectory(video_path, self.box_bbox, target_size, frame_step)

    def _extract_trajectory(self, video_path: str, tracking_bbox, target_size=None, frame_step=1):
        """
        内部方法：从视频中提取指定 bbox 的轨迹

        Args:
            video_path: 视频文件路径
            tracking_bbox: 要跟踪的 bbox
            target_size: 目标尺寸 (width, height)，None 表示使用原始尺寸
            frame_step: 采样步长，1 表示每帧采样，2 表示每 2 帧采样一次

        Returns:
            trajectory: numpy array of shape (N, 2)，像素坐标
            first_frame: 第一帧图像（用于可视化，已调整尺寸）
            video_width: 视频宽度
            video_height: 视频高度
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # 读取第一帧
        ret, first_frame_raw = cap.read()
        if not ret:
            raise ValueError(f"Cannot read first frame: {video_path}")

        original_h, original_w = first_frame_raw.shape[:2]
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # 确定目标尺寸
        if target_size:
            video_w, video_h = target_size
        else:
            video_w, video_h = original_w, original_h

        # 采样间隔：frame_step 表示每隔多少帧采样一次
        # frame_step = 1: 每帧采样
        # frame_step = 2: 每 2 帧采样一次
        frame_interval = float(frame_step)

        # 调整第一帧尺寸
        if target_size and (original_w != video_w or original_h != video_h):
            first_frame = cv2.resize(first_frame_raw, (video_w, video_h), interpolation=cv2.INTER_LINEAR)
        else:
            first_frame = first_frame_raw.copy()

        # 归一化 state 中的 bbox 到视频尺寸
        render_w = self.state.render.image_width
        render_h = self.state.render.image_height
        scale_x = video_w / render_w
        scale_y = video_h / render_h

        # 计算归一化后的 bbox
        bbox_x = int(tracking_bbox.x * scale_x)
        bbox_y = int(tracking_bbox.y * scale_y)
        bbox_w = int(tracking_bbox.width * scale_x)
        bbox_h = int(tracking_bbox.height * scale_y)

        # 确保 bbox 在图像范围内
        bbox_x = max(0, min(bbox_x, video_w - 1))
        bbox_y = max(0, min(bbox_y, video_h - 1))
        bbox_w = max(1, min(bbox_w, video_w - bbox_x))
        bbox_h = max(1, min(bbox_h, video_h - bbox_y))

        # 创建 tracker 并用第一帧的 bbox 区域初始化
        if self.tracker_type == 'template':
            tracker = TemplateMatchingTracker(first_frame, (bbox_x, bbox_y, bbox_w, bbox_h),
                                             search_margin=self.search_margin)
        elif self.tracker_type == 'optical_flow':
            tracker = OpticalFlowTracker(first_frame, (bbox_x, bbox_y, bbox_w, bbox_h))
        else:  # 'csrt' or default
            tracker = self._create_tracker()
            if tracker is None:
                raise RuntimeError("无法创建跟踪器，请安装 opencv-contrib-python")
            tracker.init(first_frame, (bbox_x, bbox_y, bbox_w, bbox_h))

        debug = False  # 设置为 True 启用调试输出

        if debug:
            print(f"Video: {original_w}x{original_h} -> {video_w}x{video_h}")
            print(f"First frame shape: {first_frame.shape}")
            print(f"BBox: ({bbox_x}, {bbox_y}, {bbox_w}, {bbox_h})")
            print(f"Tracker type: {self.tracker_type}")

        # 初始中心点（第一帧，frame_idx=0）
        cx = bbox_x + bbox_w / 2
        cy = bbox_y + bbox_h / 2
        trajectory = [(cx, cy)]

        # 从第二帧开始逐帧读取和跟踪
        frame_idx = 1
        next_sample = frame_interval

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调整帧尺寸
            if target_size and (original_w != video_w or original_h != video_h):
                frame = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_LINEAR)

            # 每帧都更新 tracker
            success, bbox = tracker.update(frame)

            if debug and frame_idx <= 5:
                print(f'Frame {frame_idx}: success={success}, bbox={bbox}')

            # 判断是否需要采样这一帧
            if frame_idx >= next_sample:
                if success:
                    x, y, w, h = bbox
                    cx = x + w / 2
                    cy = y + h / 2
                    trajectory.append((cx, cy))

                    if debug and len(trajectory) <= 5:
                        print(f'  Sampled: ({cx:.1f}, {cy:.1f})')
                else:
                    # 跟踪失败，使用上一个位置
                    trajectory.append(trajectory[-1])

                next_sample += frame_interval

            frame_idx += 1

        cap.release()

        trajectory = np.array(trajectory, dtype=np.float32)
        return trajectory, first_frame, video_w, video_h
    
    def _create_tracker(self):
        """创建跟踪器（使用 CSRT，最准确）"""
        # 优先使用 CSRT（最准确，适合游戏视频）
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()

        # 备选 KCF（较快）
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()

        # 最后尝试 MOSSE（最快但可能不准确）
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()

        return None

