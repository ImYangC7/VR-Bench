#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont


def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    获取视频信息
    
    Args:
        video_path: 视频路径
        
    Returns:
        width, height, fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, fps


def draw_trajectory_comparison(
    background: np.ndarray,
    gt_traj: np.ndarray,
    gen_traj: np.ndarray,
    output_path: str,
    title: str,
    metrics: Dict[str, Any],
    goal_center: Optional[Tuple[float, float]] = None,
    player_bbox: Optional[Any] = None,
    goal_bbox: Optional[Any] = None,
    render_config: Optional[Any] = None,
    box_bbox: Optional[Any] = None,
    gt_box_traj: Optional[np.ndarray] = None,
    gen_box_traj: Optional[np.ndarray] = None
):
    """
    绘制轨迹对比图

    Args:
        background: 背景图像（BGR 格式）
        gt_traj: Ground truth 轨迹 (N, 2) - 玩家轨迹
        gen_traj: Generated 轨迹 (M, 2) - 玩家轨迹
        output_path: 输出路径
        title: 标题
        metrics: 评估指标字典
        goal_center: 终点中心坐标（可选，已废弃）
        player_bbox: 起始位置 bbox（玩家）
        goal_bbox: 终点位置 bbox
        render_config: 渲染配置（用于计算缩放比例）
        box_bbox: 箱子初始位置 bbox（仅推箱子游戏）
        gt_box_traj: Ground truth 箱子轨迹（仅推箱子游戏）
        gen_box_traj: Generated 箱子轨迹（仅推箱子游戏）
    """
    # 转换为 RGB
    img = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # 计算缩放比例
    img_h, img_w = background.shape[:2]
    if render_config is not None:
        scale_x = img_w / render_config.image_width
        scale_y = img_h / render_config.image_height
    else:
        scale_x = scale_y = 1.0

    # 绘制起始位置 bbox（黄色矩形）
    if player_bbox is not None:
        x = player_bbox.x * scale_x
        y = player_bbox.y * scale_y
        w = player_bbox.width * scale_x
        h = player_bbox.height * scale_y
        cx = player_bbox.center_x * scale_x
        cy = player_bbox.center_y * scale_y

        draw.rectangle(
            [x, y, x + w, y + h],
            outline='yellow',
            width=3
        )
        # 绘制中心点
        draw.ellipse(
            [cx - 5, cy - 5, cx + 5, cy + 5],
            fill='yellow'
        )

    # 绘制终点 bbox（红色矩形）
    if goal_bbox is not None:
        x = goal_bbox.x * scale_x
        y = goal_bbox.y * scale_y
        w = goal_bbox.width * scale_x
        h = goal_bbox.height * scale_y
        cx = goal_bbox.center_x * scale_x
        cy = goal_bbox.center_y * scale_y

        draw.rectangle(
            [x, y, x + w, y + h],
            outline='red',
            width=3
        )
        # 绘制中心点
        draw.ellipse(
            [cx - 5, cy - 5, cx + 5, cy + 5],
            fill='red'
        )

    # 绘制箱子初始位置 bbox（橙色矩形，仅推箱子游戏）
    if box_bbox is not None:
        x = box_bbox.x * scale_x
        y = box_bbox.y * scale_y
        w = box_bbox.width * scale_x
        h = box_bbox.height * scale_y
        cx = box_bbox.center_x * scale_x
        cy = box_bbox.center_y * scale_y

        draw.rectangle(
            [x, y, x + w, y + h],
            outline='orange',
            width=3
        )
        # 绘制中心点
        draw.ellipse(
            [cx - 5, cy - 5, cx + 5, cy + 5],
            fill='orange'
        )

    # 绘制 GT 轨迹（绿色）
    if len(gt_traj) > 1:
        points = [(float(x), float(y)) for x, y in gt_traj]
        draw.line(points, fill='lime', width=4)
        # 起点（绿色圆圈）
        draw.ellipse(
            [points[0][0] - 8, points[0][1] - 8, points[0][0] + 8, points[0][1] + 8],
            fill='lime',
            outline='darkgreen',
            width=2
        )

    # 绘制 Generated 轨迹（蓝色）
    if len(gen_traj) > 1:
        points = [(float(x), float(y)) for x, y in gen_traj]
        draw.line(points, fill='cyan', width=4)
        # 起点（蓝色圆圈）
        draw.ellipse(
            [points[0][0] - 8, points[0][1] - 8, points[0][0] + 8, points[0][1] + 8],
            fill='cyan',
            outline='darkblue',
            width=2
        )

    # 绘制 GT 箱子轨迹（深绿色虚线，仅推箱子游戏）
    if gt_box_traj is not None and len(gt_box_traj) > 1:
        points = [(float(x), float(y)) for x, y in gt_box_traj]
        # 绘制虚线效果（每隔一段绘制一段）
        for i in range(0, len(points) - 1, 2):
            if i + 1 < len(points):
                draw.line([points[i], points[i + 1]], fill='darkgreen', width=3)
        # 起点（深绿色方块）
        draw.rectangle(
            [points[0][0] - 6, points[0][1] - 6, points[0][0] + 6, points[0][1] + 6],
            fill='darkgreen',
            outline='lime',
            width=2
        )

    # 绘制 Generated 箱子轨迹（深蓝色虚线，仅推箱子游戏）
    if gen_box_traj is not None and len(gen_box_traj) > 1:
        points = [(float(x), float(y)) for x, y in gen_box_traj]
        # 绘制虚线效果（每隔一段绘制一段）
        for i in range(0, len(points) - 1, 2):
            if i + 1 < len(points):
                draw.line([points[i], points[i + 1]], fill='darkblue', width=3)
        # 起点（深蓝色方块）
        draw.rectangle(
            [points[0][0] - 6, points[0][1] - 6, points[0][0] + 6, points[0][1] + 6],
            fill='darkblue',
            outline='cyan',
            width=2
        )

    # 绘制文本信息
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 处理 sd 可能为 None 的情况
    sd_value = metrics.get('sd')
    sd_str = f"{sd_value:.3f}" if sd_value is not None else "N/A"

    text_lines = [
        f"Video: {title}",
        f"PR: {metrics.get('pr', 0):.3f}",
        f"SD: {sd_str}",
        f"SR: {metrics.get('sr', 0):.3f}",
        f"EM: {metrics.get('em', 0):.3f}",
        f"MF: {metrics.get('mf', 0):.3f}"
    ]
    
    y_offset = 10
    for line in text_lines:
        draw.text((10, y_offset), line, fill='white', font=font)
        y_offset += 25
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path)


def find_matching_gt_videos(dataset_dir: Path, gen_video_name: str) -> list:
    """
    查找匹配的 GT 视频

    命名规则：
    - Generated: {difficulty}_{id}.mp4
    - GT 可能的格式：
      1. {difficulty}_{id}.mp4 (旧格式，单条路径)
      2. {difficulty}_{id}_{path_id}.mp4 (新格式，多条路径)

    Args:
        dataset_dir: 数据集目录
        gen_video_name: Generated 视频文件名

    Returns:
        匹配的 GT 视频路径列表
    """
    # 解析文件名
    stem = Path(gen_video_name).stem
    parts = stem.split('_')

    if len(parts) < 2:
        return []

    # 先查找新格式（带 path_id）
    pattern_new = f"{stem}_*.mp4"
    gt_videos = list(dataset_dir.rglob(pattern_new))

    # 如果没找到，查找旧格式（不带 path_id）
    if not gt_videos:
        pattern_old = f"{stem}.mp4"
        gt_videos = list(dataset_dir.rglob(pattern_old))

    return [str(v) for v in gt_videos]


def find_state_file(dataset_dir: Path, gen_video_name: str) -> Optional[str]:
    """
    查找对应的 state 文件

    Args:
        dataset_dir: 数据集目录
        gen_video_name: Generated 视频文件名

    Returns:
        state 文件路径，如果未找到返回 None
    """
    stem = Path(gen_video_name).stem
    parts = stem.split('_')

    if len(parts) < 2:
        return None

    # state 文件命名：{difficulty}_{id}.json
    state_name = f"{parts[0]}_{parts[1]}.json"

    # 在 states 目录中查找
    state_files = list(dataset_dir.rglob(f"states/{state_name}"))

    if state_files:
        return str(state_files[0])

    return None
