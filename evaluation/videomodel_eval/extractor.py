#!/usr/bin/env python3
"""轨迹提取器模块

从视频中提取目标轨迹，支持多种追踪算法：
- ncc: 归一化互相关追踪（默认，适合固定外观目标）
- optical_flow: 光流追踪
- csrt: OpenCV CSRT 追踪器（需要 opencv-contrib-python）
"""
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from core.schema import UnifiedState


class NCCTracker:
    """
    基于归一化互相关 (NCC) 的追踪器

    使用 cv2.TM_CCOEFF_NORMED（归一化相关系数匹配）在搜索区域内定位目标。
    适合目标外观固定、只有平移运动的场景（如 puzzle 游戏中的玩家图标）。
    """

    # 匹配质量阈值，低于此值认为追踪失败
    MATCH_THRESHOLD = 0.6

    def __init__(self, first_frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 search_margin: int = 50):
        """
        初始化模板追踪器

        Args:
            first_frame: 第一帧图像 (BGR 或灰度)
            bbox: 初始边界框 (x, y, width, height)
            search_margin: 搜索区域边距（像素）
                          - 0: 全图搜索（慢但不会丢失）
                          - >0: 以上次位置为中心的局部搜索（快）
        """
        x, y, w, h = bbox

        # 从第一帧截取模板
        self.template = first_frame[y:y+h, x:x+w].copy()
        self.template_h, self.template_w = self.template.shape[:2]

        # 追踪状态
        self.last_pos = (x, y)
        self.search_margin = search_margin
        self.frame_h, self.frame_w = first_frame.shape[:2]
        self.full_search = (search_margin == 0)

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        """
        在新帧中追踪目标

        Args:
            frame: 当前帧图像

        Returns:
            (success, (x, y, width, height))
            - success: 是否追踪成功（匹配分数 > MATCH_THRESHOLD）
            - bbox: 目标位置边界框
        """
        last_x, last_y = self.last_pos

        # 确定搜索区域
        if self.full_search:
            search_x1, search_y1 = 0, 0
            search_x2, search_y2 = self.frame_w, self.frame_h
        else:
            search_x1 = max(0, last_x - self.search_margin)
            search_y1 = max(0, last_y - self.search_margin)
            search_x2 = min(self.frame_w, last_x + self.template_w + self.search_margin)
            search_y2 = min(self.frame_h, last_y + self.template_h + self.search_margin)

        search_region = frame[search_y1:search_y2, search_x1:search_x2]

        # 检查搜索区域是否足够大
        if search_region.shape[0] < self.template_h or search_region.shape[1] < self.template_w:
            return False, (last_x, last_y, self.template_w, self.template_h)

        # 执行模板匹配
        result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 转换到全局坐标
        match_x = search_x1 + max_loc[0]
        match_y = search_y1 + max_loc[1]

        # 更新位置
        self.last_pos = (match_x, match_y)

        # 判断匹配质量
        success = max_val > self.MATCH_THRESHOLD

        return success, (match_x, match_y, self.template_w, self.template_h)


class OpticalFlowTracker:
    """
    基于光流的追踪器

    使用 Lucas-Kanade 稀疏光流算法追踪目标区域内的特征点，
    通过计算特征点的中值位移来估计目标运动。
    适合连续平滑运动的场景。
    """

    # Lucas-Kanade 光流参数
    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # 特征点检测参数
    FEATURE_PARAMS = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # 最少特征点数量
    MIN_POINTS = 4

    def __init__(self, first_frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        初始化光流追踪器

        Args:
            first_frame: 第一帧图像（BGR 或灰度）
            bbox: 初始边界框 (x, y, width, height)
        """
        # 转换为灰度图
        if len(first_frame.shape) == 3:
            self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_gray = first_frame.copy()

        x, y, w, h = bbox

        # 在 bbox 内检测特征点
        mask = np.zeros_like(self.prev_gray)
        mask[y:y+h, x:x+w] = 255

        self.prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=mask, **self.FEATURE_PARAMS
        )

        # 如果特征点太少，使用网格点作为备选
        if self.prev_points is None or len(self.prev_points) < self.MIN_POINTS:
            grid_x = np.linspace(x + 5, x + w - 5, 5)
            grid_y = np.linspace(y + 5, y + h - 5, 5)
            xx, yy = np.meshgrid(grid_x, grid_y)
            self.prev_points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
            self.prev_points = self.prev_points.reshape(-1, 1, 2)

        self.bbox = bbox

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        """
        在新帧中追踪目标

        Args:
            frame: 当前帧图像

        Returns:
            (success, (x, y, width, height))
            - success: 是否追踪成功
            - bbox: 目标位置边界框
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()

        # 检查特征点数量
        if self.prev_points is None or len(self.prev_points) < self.MIN_POINTS:
            return False, self.bbox

        # 计算光流
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None, **self.LK_PARAMS
        )

        # 筛选成功追踪的点
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_curr) < self.MIN_POINTS:
            return False, self.bbox

        # 计算中值位移
        dx = np.median(good_curr[:, 0] - good_prev[:, 0])
        dy = np.median(good_curr[:, 1] - good_prev[:, 1])

        # 更新 bbox
        x, y, w, h = self.bbox
        self.bbox = (x + dx, y + dy, w, h)

        # 更新状态
        self.prev_gray = curr_gray
        self.prev_points = curr_points[status == 1].reshape(-1, 1, 2)

        return True, self.bbox


class CSRTTracker:
    """
    视频轨迹提取器 (CSRT Tracker)

    从视频中提取目标（玩家、箱子等）的运动轨迹。
    支持多种追踪算法，可根据场景选择最适合的方法。
    """

    def __init__(self, state_path: str, tracker_type: str = 'ncc', search_margin: int = 50):
        """
        初始化轨迹提取器

        Args:
            state_path: UnifiedState JSON 文件路径，包含初始 bbox 信息
            tracker_type: 追踪器类型
                - 'ncc': 归一化互相关追踪（默认，推荐用于 puzzle 游戏）
                - 'optical_flow': 光流追踪
                - 'csrt': OpenCV CSRT 追踪器（高精度，需要 opencv-contrib）
            search_margin: NCC 追踪器搜索边距（仅对 ncc 类型有效）
                - 0: 全图搜索
                - >0: 局部搜索范围（像素）
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
        if self.tracker_type == 'ncc':
            tracker = NCCTracker(first_frame, (bbox_x, bbox_y, bbox_w, bbox_h),
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
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2, 'TrackerMOSSE_create'):
            return cv2.TrackerMOSSE_create()

        return None
