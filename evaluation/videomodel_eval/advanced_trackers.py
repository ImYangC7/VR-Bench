#!/usr/bin/env python3
"""高级追踪算法"""

import cv2
import numpy as np
from typing import Tuple, Optional


class TemplateMatchingTracker:
    """基于模板匹配的追踪器"""

    def __init__(self, first_frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 search_margin: int = 50):
        """
        Args:
            first_frame: 第一帧图像
            bbox: 初始 bbox (x, y, w, h)
            search_margin: 搜索区域边距（像素），0 表示全图搜索，>0 表示局部搜索
        """
        x, y, w, h = bbox
        self.template = first_frame[y:y+h, x:x+w].copy()
        self.template_h, self.template_w = self.template.shape[:2]
        self.last_pos = (x, y)
        self.search_margin = search_margin
        self.frame_h, self.frame_w = first_frame.shape[:2]
        self.full_search = (search_margin == 0)
        
    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        """
        更新追踪

        Returns:
            (success, (x, y, w, h))
        """
        last_x, last_y = self.last_pos

        # 定义搜索区域
        if self.full_search:
            # 全图搜索
            search_x1 = 0
            search_y1 = 0
            search_x2 = self.frame_w
            search_y2 = self.frame_h
        else:
            # 局部搜索
            search_x1 = max(0, last_x - self.search_margin)
            search_y1 = max(0, last_y - self.search_margin)
            search_x2 = min(self.frame_w, last_x + self.template_w + self.search_margin)
            search_y2 = min(self.frame_h, last_y + self.template_h + self.search_margin)

        search_region = frame[search_y1:search_y2, search_x1:search_x2]

        # 模板匹配
        if search_region.shape[0] < self.template_h or search_region.shape[1] < self.template_w:
            return False, (last_x, last_y, self.template_w, self.template_h)

        result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 转换到全局坐标
        match_x = search_x1 + max_loc[0]
        match_y = search_y1 + max_loc[1]

        # 更新位置
        self.last_pos = (match_x, match_y)

        # 判断匹配质量
        success = max_val > 0.6  # 匹配阈值

        return success, (match_x, match_y, self.template_w, self.template_h)


class OpticalFlowTracker:
    """基于光流的追踪器"""
    
    def __init__(self, first_frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Args:
            first_frame: 第一帧图像（灰度或彩色）
            bbox: 初始 bbox (x, y, w, h)
        """
        if len(first_frame.shape) == 3:
            self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_gray = first_frame.copy()
        
        x, y, w, h = bbox
        
        # 在 bbox 内检测特征点
        mask = np.zeros_like(self.prev_gray)
        mask[y:y+h, x:x+w] = 255
        
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, 
                                                    **self.feature_params)
        
        if self.prev_points is None or len(self.prev_points) < 4:
            # 如果特征点太少，使用网格点
            grid_x = np.linspace(x + 5, x + w - 5, 5)
            grid_y = np.linspace(y + 5, y + h - 5, 5)
            xx, yy = np.meshgrid(grid_x, grid_y)
            self.prev_points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
            self.prev_points = self.prev_points.reshape(-1, 1, 2)
        
        self.bbox = bbox
        
    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        """
        更新追踪
        
        Returns:
            (success, (x, y, w, h))
        """
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()
        
        if self.prev_points is None or len(self.prev_points) < 4:
            return False, self.bbox
        
        # 计算光流
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None, **self.lk_params
        )
        
        # 选择成功追踪的点
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]
        
        if len(good_curr) < 4:
            return False, self.bbox
        
        # 计算平移
        dx = np.median(good_curr[:, 0] - good_prev[:, 0])
        dy = np.median(good_curr[:, 1] - good_prev[:, 1])
        
        # 更新 bbox
        x, y, w, h = self.bbox
        new_x = x + dx
        new_y = y + dy
        
        self.bbox = (new_x, new_y, w, h)
        
        # 更新特征点
        self.prev_gray = curr_gray
        self.prev_points = curr_points[status == 1].reshape(-1, 1, 2)
        
        return True, self.bbox

