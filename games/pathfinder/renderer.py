"""
PathFinder 渲染器 - 使用 OpenCV 渲染平滑曲线
支持纹理贴图：起点/终点图标，道路纹理沿曲线平铺
"""

import math
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple
import imageio

from .board import PathFinderBoard
from .constants import (
    START_COLOR, END_COLOR,
    ROAD_COLOR, BG_COLOR, CURVE_SEGMENTS, FRAMES_PER_SECOND
)
from .texture_handler import get_texture_handler


def bezier_curve_opencv(control_points: List[Tuple[float, float]], num_points: int = CURVE_SEGMENTS) -> np.ndarray:
    """生成贝塞尔曲线点"""
    n = len(control_points) - 1
    points = []
    
    for i in range(num_points + 1):
        t = i / num_points
        x, y = 0.0, 0.0
        
        for j, (px, py) in enumerate(control_points):
            # 贝塞尔基函数
            from math import comb
            coef = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
            x += coef * px
            y += coef * py
        
        points.append([x, y])
    
    return np.array(points, dtype=np.int32)


def render_pathfinder_board(
    board: PathFinderBoard,
    output_path: Optional[str] = None,
    show_solution: bool = False,
    assets_folder: Optional[str] = None
) -> Image.Image:
    """
    渲染游戏板

    Args:
        board: 游戏板对象
        output_path: 输出路径（可选）
        show_solution: 是否显示解决方案（暂未实现）
        assets_folder: 纹理资源文件夹路径，如果为 None 则使用纯色渲染

    Returns:
        渲染后的图片
    """
    size = board.image_size
    road_width = board.road_width  # 使用board中保存的道路宽度

    # 尝试加载纹理
    texture_handler = None
    use_textures = False

    if assets_folder is not None:
        try:
            texture_handler = get_texture_handler(assets_folder)
            is_valid, missing = texture_handler.validate_textures()
            if is_valid:
                use_textures = True
            else:
                print(f"Warning: Missing textures {missing}, falling back to solid colors")
        except Exception as e:
            print(f"Warning: Failed to load textures: {e}, using solid colors")

    if use_textures:
        # 使用纹理渲染
        pil_img = _render_with_textures(board, size, road_width, texture_handler)
    else:
        # 使用纯色渲染（原有逻辑）
        pil_img = _render_with_colors(board, size, road_width)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)

    return pil_img


def _render_with_colors(board: PathFinderBoard, size: int, road_width: int) -> Image.Image:
    """使用纯色渲染（原有逻辑）"""
    # 使用 OpenCV 渲染（更好的抗锯齿）
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = BG_COLOR

    # 绘制所有路径段
    for segment in board.segments:
        curve_points = bezier_curve_opencv(segment.control_points)
        cv2.polylines(img, [curve_points], False, ROAD_COLOR, road_width, lineType=cv2.LINE_AA)

    # 计算圆圈半径：道路宽度的40%，确保不超出轨道
    circle_radius = int(road_width * 0.4)

    # 绘制终点（绿色圆圈）
    end_x, end_y = int(board.end_point[0]), int(board.end_point[1])
    cv2.circle(img, (end_x, end_y), circle_radius, END_COLOR, -1, lineType=cv2.LINE_AA)

    # 绘制起点（红色圆圈）
    start_x, start_y = int(board.start_point[0]), int(board.start_point[1])
    cv2.circle(img, (start_x, start_y), circle_radius, START_COLOR, -1, lineType=cv2.LINE_AA)

    # 转换为 PIL Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    return pil_img


def _render_with_textures(board: PathFinderBoard, size: int, road_width: int, texture_handler) -> Image.Image:
    """使用纹理渲染"""
    # 创建 PIL 图片（RGBA 模式以支持透明度）
    img = Image.new('RGBA', (size, size), (0, 0, 0, 255))

    # 获取道路纹理
    road_texture = texture_handler.get_road_texture()

    # 绘制所有路径段（使用纹理）
    for segment in board.segments:
        curve_points = bezier_curve_opencv(segment.control_points)
        _draw_textured_path(img, curve_points, road_width, road_texture)

    # 计算图标尺寸：道路宽度的80%
    icon_size = int(road_width * 0.8)

    # 绘制终点图标
    end_icon = texture_handler.get_end_icon(icon_size)
    if end_icon:
        end_x, end_y = int(board.end_point[0]), int(board.end_point[1])
        _paste_icon(img, end_icon, end_x, end_y)

    # 绘制起点图标
    start_icon = texture_handler.get_start_icon(icon_size)
    if start_icon:
        start_x, start_y = int(board.start_point[0]), int(board.start_point[1])
        _paste_icon(img, start_icon, start_x, start_y)

    return img.convert('RGB')


def _draw_textured_path(img: Image.Image, curve_points: np.ndarray, road_width: int, texture: Image.Image):
    """
    沿着曲线绘制纹理道路（优化版，减少锯齿）

    方法：
    1. 使用 OpenCV 绘制平滑的道路蒙版（抗锯齿）
    2. 创建平铺纹理填充整个画布
    3. 使用蒙版合成到目标图片

    Args:
        img: 目标图片
        curve_points: 曲线点数组
        road_width: 道路宽度
        texture: 道路纹理
    """
    if texture is None:
        # 降级到纯色
        img_array = np.array(img)
        cv2.polylines(img_array, [curve_points], False, (255, 255, 255, 255), road_width, lineType=cv2.LINE_AA)
        img.paste(Image.fromarray(img_array))
        return

    # 获取图片尺寸
    width, height = img.size

    # 步骤 1: 创建道路蒙版（使用 OpenCV 的抗锯齿线条）
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, [curve_points], False, 255, road_width, lineType=cv2.LINE_AA)

    # 步骤 2: 创建平铺纹理画布
    # 将纹理平铺到整个画布
    texture_np = np.array(texture.convert('RGB'))
    tex_h, tex_w = texture_np.shape[:2]

    # 创建平铺纹理画布
    textured_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 平铺纹理
    for y in range(0, height, tex_h):
        for x in range(0, width, tex_w):
            # 计算当前块的尺寸
            h_end = min(y + tex_h, height)
            w_end = min(x + tex_w, width)
            h_size = h_end - y
            w_size = w_end - x

            # 粘贴纹理块
            textured_canvas[y:h_end, x:w_end] = texture_np[:h_size, :w_size]

    # 步骤 3: 使用蒙版合成
    # 将蒙版转换为 3 通道
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # 转换 PIL 图片为 numpy 数组
    img_array = np.array(img.convert('RGB'))

    # 使用蒙版混合：只在道路区域应用纹理
    # alpha = mask / 255.0
    alpha = mask_3ch.astype(float) / 255.0
    blended = (textured_canvas * alpha + img_array * (1 - alpha)).astype(np.uint8)

    # 转换回 PIL 图片并更新
    result = Image.fromarray(blended)
    img.paste(result)


def _paste_icon(img: Image.Image, icon: Image.Image, x: int, y: int):
    """
    在指定位置粘贴图标（居中）

    Args:
        img: 目标图片
        icon: 图标
        x: 中心 x 坐标
        y: 中心 y 坐标
    """
    paste_x = x - icon.width // 2
    paste_y = y - icon.height // 2

    try:
        img.paste(icon, (paste_x, paste_y), icon)
    except:
        pass  # 忽略超出边界的情况


def _render_video_background_with_colors(board: PathFinderBoard, size: int, road_width: int) -> np.ndarray:
    """渲染视频背景（纯色模式）"""
    circle_radius = int(road_width * 0.4)

    background = np.zeros((size, size, 3), dtype=np.uint8)
    background[:] = BG_COLOR

    # 绘制所有路径段
    for segment in board.segments:
        curve_points = bezier_curve_opencv(segment.control_points)
        cv2.polylines(background, [curve_points], False, ROAD_COLOR, road_width, lineType=cv2.LINE_AA)

    # 绘制终点（绿色圆圈）
    end_x, end_y = int(board.end_point[0]), int(board.end_point[1])
    cv2.circle(background, (end_x, end_y), circle_radius, END_COLOR, -1, lineType=cv2.LINE_AA)

    # 转换为 RGB（只转换一次）
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    return background_rgb


def _render_video_background_with_textures(board: PathFinderBoard, size: int, road_width: int, texture_handler) -> Image.Image:
    """渲染视频背景（纹理模式）"""
    # 创建 PIL 图片
    img = Image.new('RGB', (size, size), (0, 0, 0))

    # 获取道路纹理
    road_texture = texture_handler.get_road_texture()

    # 绘制所有路径段（使用纹理）
    for segment in board.segments:
        curve_points = bezier_curve_opencv(segment.control_points)
        _draw_textured_path(img, curve_points, road_width, road_texture)

    # 计算图标尺寸
    icon_size = int(road_width * 0.8)

    # 绘制终点图标
    end_icon = texture_handler.get_end_icon(icon_size)
    if end_icon:
        end_x, end_y = int(board.end_point[0]), int(board.end_point[1])
        _paste_icon(img, end_icon, end_x, end_y)

    return img


def render_solution_video(
    board: PathFinderBoard,
    output_path: str,
    fps: int = FRAMES_PER_SECOND,
    use_gpu: bool = True,
    assets_folder: Optional[str] = None
) -> Optional[str]:
    """
    渲染解决方案视频（连续移动）

    GPU 加速优化：
    1. 预渲染背景图（只渲染一次道路）
    2. 使用 NVENC 硬件编码（如果可用）
    3. 批量处理帧

    Args:
        board: 游戏板对象
        output_path: 输出视频路径
        fps: 帧率
        use_gpu: 是否使用 GPU 加速
        assets_folder: 纹理资源文件夹路径，如果为 None 则使用纯色渲染

    Returns:
        输出路径或 None（如果失败）
    """
    if not board.is_solvable():
        return None

    size = board.image_size
    road_width = board.road_width

    # 尝试加载纹理
    texture_handler = None
    use_textures = False

    if assets_folder is not None:
        try:
            texture_handler = get_texture_handler(assets_folder)
            is_valid, missing = texture_handler.validate_textures()
            if is_valid:
                use_textures = True
        except:
            pass

    if use_textures:
        # 使用纹理渲染背景
        background_rgb = _render_video_background_with_textures(board, size, road_width, texture_handler)
        icon_size = int(road_width * 0.8)
        start_icon = texture_handler.get_start_icon(icon_size)
    else:
        # 使用纯色渲染背景
        background_rgb = _render_video_background_with_colors(board, size, road_width)
        circle_radius = int(road_width * 0.4)
        start_icon = None

    # 使用节点路径生成连续的曲线轨迹
    all_solution_points = []

    if board.solution_path and len(board.solution_path) >= 2:
        # 使用保存的节点路径
        for i in range(len(board.solution_path) - 1):
            start_pt = board.solution_path[i]
            end_pt = board.solution_path[i + 1]

            # 在两个节点之间生成平滑曲线
            # 找到对应的segment
            found_segment = None
            for seg_idx in board.solution_segments:
                segment = board.segments[seg_idx]
                seg_start = segment.get_start()
                seg_end = segment.get_end()

                # 检查这个segment是否连接这两个节点
                dist1 = math.sqrt((seg_start[0] - start_pt[0])**2 + (seg_start[1] - start_pt[1])**2)
                dist2 = math.sqrt((seg_end[0] - end_pt[0])**2 + (seg_end[1] - end_pt[1])**2)
                dist3 = math.sqrt((seg_start[0] - end_pt[0])**2 + (seg_start[1] - end_pt[1])**2)
                dist4 = math.sqrt((seg_end[0] - start_pt[0])**2 + (seg_end[1] - start_pt[1])**2)

                if (dist1 < 10 and dist2 < 10) or (dist3 < 10 and dist4 < 10):
                    found_segment = segment
                    # 判断方向
                    if dist1 < 10 and dist2 < 10:
                        # 正向
                        curve_points = bezier_curve_opencv(segment.control_points)
                    else:
                        # 反向
                        reversed_points = segment.control_points[::-1]
                        curve_points = bezier_curve_opencv(reversed_points)
                    break

            if found_segment:
                # 添加曲线点（跳过第一个点以避免重复，除了第一条曲线）
                if i == 0:
                    all_solution_points.extend(curve_points)
                else:
                    all_solution_points.extend(curve_points[1:])

        all_solution_points = np.array(all_solution_points)
    else:
        # 降级方案：使用旧方法
        for seg_idx in board.solution_segments:
            segment = board.segments[seg_idx]
            curve_points = bezier_curve_opencv(segment.control_points)
            all_solution_points.extend(curve_points)
        all_solution_points = np.array(all_solution_points)
    
    # 计算总路程（沿着曲线的实际距离）
    total_distance = 0.0
    segment_distances = []  # 记录每段的距离

    for i in range(len(board.solution_path) - 1):
        start_node = board.solution_path[i]
        end_node = board.solution_path[i + 1]

        # 找到对应的segment
        for seg_idx in board.solution_segments:
            segment = board.segments[seg_idx]
            seg_start = segment.get_start()
            seg_end = segment.get_end()

            # 检查这个segment是否连接这两个节点
            dist1 = math.sqrt((seg_start[0] - start_node[0])**2 + (seg_start[1] - start_node[1])**2)
            dist2 = math.sqrt((seg_end[0] - end_node[0])**2 + (seg_end[1] - end_node[1])**2)
            dist3 = math.sqrt((seg_start[0] - end_node[0])**2 + (seg_start[1] - end_node[1])**2)
            dist4 = math.sqrt((seg_end[0] - start_node[0])**2 + (seg_end[1] - start_node[1])**2)

            if (dist1 < 10 and dist2 < 10) or (dist3 < 10 and dist4 < 10):
                # 判断方向
                if dist1 < 10 and dist2 < 10:
                    # 正向
                    curve_points = bezier_curve_opencv(segment.control_points)
                else:
                    # 反向
                    reversed_points = segment.control_points[::-1]
                    curve_points = bezier_curve_opencv(reversed_points)

                # 计算这段曲线的长度
                segment_dist = 0.0
                for j in range(len(curve_points) - 1):
                    p1 = curve_points[j]
                    p2 = curve_points[j + 1]
                    segment_dist += np.linalg.norm(p2 - p1)

                segment_distances.append(segment_dist)
                total_distance += segment_dist
                break

    if total_distance == 0:
        return None

    # 使用恒定速度：每秒移动的像素数
    # 假设我们希望以合理的速度移动，比如每秒移动 100 像素
    pixels_per_second = 100.0
    duration = total_distance / pixels_per_second  # 总时长（秒）
    num_frames = int(duration * fps)  # 总帧数

    # 确保至少有一些帧
    num_frames = max(num_frames, fps * 2)  # 至少2秒

    frames = []

    # 生成所有曲线点（按顺序连接）
    all_curve_points = []
    for i in range(len(board.solution_path) - 1):
        start_node = board.solution_path[i]
        end_node = board.solution_path[i + 1]

        # 找到对应的segment
        for seg_idx in board.solution_segments:
            segment = board.segments[seg_idx]
            seg_start = segment.get_start()
            seg_end = segment.get_end()

            # 检查这个segment是否连接这两个节点
            dist1 = math.sqrt((seg_start[0] - start_node[0])**2 + (seg_start[1] - start_node[1])**2)
            dist2 = math.sqrt((seg_end[0] - end_node[0])**2 + (seg_end[1] - end_node[1])**2)
            dist3 = math.sqrt((seg_start[0] - end_node[0])**2 + (seg_start[1] - end_node[1])**2)
            dist4 = math.sqrt((seg_end[0] - start_node[0])**2 + (seg_end[1] - start_node[1])**2)

            if (dist1 < 10 and dist2 < 10) or (dist3 < 10 and dist4 < 10):
                # 判断方向
                if dist1 < 10 and dist2 < 10:
                    # 正向
                    curve_points = bezier_curve_opencv(segment.control_points)
                else:
                    # 反向
                    reversed_points = segment.control_points[::-1]
                    curve_points = bezier_curve_opencv(reversed_points)

                # 添加曲线点（跳过第一个点以避免重复，除了第一段）
                if i == 0:
                    all_curve_points.extend(curve_points)
                else:
                    all_curve_points.extend(curve_points[1:])
                break

    all_curve_points = np.array(all_curve_points)

    # ========== 关键：计算每个点的累积距离，实现真正的匀速移动 ==========
    # 计算每段的距离
    cumulative_distances = [0.0]  # 累积距离
    for i in range(len(all_curve_points) - 1):
        p1 = all_curve_points[i]
        p2 = all_curve_points[i + 1]
        dist = np.linalg.norm(p2 - p1)
        cumulative_distances.append(cumulative_distances[-1] + dist)

    total_path_distance = cumulative_distances[-1]

    if total_path_distance == 0:
        return None

    # ========== 优化 2: 批量生成帧 ==========
    # 预先计算所有位置，减少循环开销
    frames = []

    for frame_idx in range(num_frames):
        # 计算当前应该走过的距离（基于实际距离，而不是点索引）
        progress = frame_idx / (num_frames - 1) if num_frames > 1 else 0
        target_distance = progress * total_path_distance

        # 二分查找：找到对应距离的点
        # 找到 cumulative_distances[i] <= target_distance < cumulative_distances[i+1]
        left, right = 0, len(cumulative_distances) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if cumulative_distances[mid] <= target_distance:
                left = mid
            else:
                right = mid

        # 在 left 和 left+1 之间插值
        if left < len(all_curve_points) - 1:
            d1 = cumulative_distances[left]
            d2 = cumulative_distances[left + 1]
            if d2 - d1 > 0:
                t = (target_distance - d1) / (d2 - d1)
                p1 = all_curve_points[left]
                p2 = all_curve_points[left + 1]
                current_pos = p1 + t * (p2 - p1)
            else:
                current_pos = all_curve_points[left]
        else:
            current_pos = all_curve_points[-1]

        # 复制背景图（比重新绘制快得多）
        if use_textures:
            # 纹理模式：背景是 PIL Image
            img_pil = background_rgb.copy()

            # 绘制移动的起点图标
            curr_x, curr_y = int(current_pos[0]), int(current_pos[1])
            if start_icon:
                _paste_icon(img_pil, start_icon, curr_x, curr_y)

            # 转换为 numpy 数组
            img_rgb = np.array(img_pil)
        else:
            # 纯色模式：背景是 numpy 数组
            img_rgb = background_rgb.copy()

            # 只绘制移动的红色圆圈
            curr_x, curr_y = int(current_pos[0]), int(current_pos[1])
            # 注意：background_rgb 是 RGB 格式，所以颜色也要用 RGB
            cv2.circle(img_rgb, (curr_x, curr_y), circle_radius,
                      (START_COLOR[2], START_COLOR[1], START_COLOR[0]),  # BGR -> RGB
                      -1, lineType=cv2.LINE_AA)

        frames.append(img_rgb)

    # ========== 优化 3: 使用 GPU 硬件编码 ==========
    # 尝试使用 NVENC (NVIDIA GPU 编码)，失败则降级到 CPU
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 首先尝试 GPU 编码
        codec = "h264_nvenc" if use_gpu else "libx264"

        try:
            with imageio.get_writer(str(output_path), format="FFMPEG", mode="I", fps=fps,
                                   codec=codec, pixelformat="yuv420p",
                                   macro_block_size=1,
                                   output_params=["-preset", "fast"]) as writer:
                for frame in frames:
                    writer.append_data(frame)
            return output_path
        except Exception as gpu_error:
            # GPU 编码失败，降级到 CPU
            if use_gpu:
                print(f"GPU encoding failed ({gpu_error}), falling back to CPU encoding...")
                with imageio.get_writer(str(output_path), format="FFMPEG", mode="I", fps=fps,
                                       codec="libx264", pixelformat="yuv420p",
                                       macro_block_size=1,
                                       output_params=["-preset", "fast"]) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                return output_path
            else:
                raise
    except Exception as e:
        print(f"Failed to save video: {e}")
        return None
