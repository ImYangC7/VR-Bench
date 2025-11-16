"""
3D Maze 游戏适配器
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.game_adapter import GameAdapter
from core.schema import UnifiedState, Grid, Entity, RenderConfig, BBox, Position as SchemaPosition
from games.maze3d.main import QAGenerator, draw_puzzle, generate_solution_video
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def render_and_extract_bbox(puzzle, output_image_path, dpi=100):
    """
    渲染 3D 场景并提取 player 和 goal 的 bbox

    通过在 3D 空间中绘制标记，然后渲染得到 2D bbox

    Returns:
        (player_bbox, goal_bbox, image_size)
        player_bbox: (center_x, center_y, x, y, w, h)
        goal_bbox: (center_x, center_y, x, y, w, h)
        image_size: (width, height)
    """
    import tempfile
    from PIL import Image
    import numpy as np

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_player:
        player_mask_path = tmp_player.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_goal:
        goal_mask_path = tmp_goal.name

    try:
        # 1. 渲染 player 标记（蓝色球体在起点）
        _render_marker(puzzle, puzzle.start_pos, player_mask_path, color='blue', dpi=dpi, is_ball=True)

        # 2. 渲染 goal 标记（红色球体在终点，表示小球到达终点时的位置）
        _render_marker(puzzle, puzzle.goal_pos, goal_mask_path, color='red', dpi=dpi, is_ball=True)

        # 3. 从渲染图像中提取 bbox
        player_img = Image.open(player_mask_path).convert('RGB')
        goal_img = Image.open(goal_mask_path).convert('RGB')

        player_bbox = _extract_bbox_from_image(player_img, target_color=(0, 0, 255))
        goal_bbox = _extract_bbox_from_image(goal_img, target_color=(255, 0, 0))

        image_size = player_img.size

        return player_bbox, goal_bbox, image_size

    finally:
        import os
        try:
            os.unlink(player_mask_path)
            os.unlink(goal_mask_path)
        except:
            pass


def _render_marker(puzzle, pos, output_path, color='blue', dpi=100, is_ball=True):
    """渲染单个标记（球体）"""
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20, azim=45)

    max_range = max(puzzle.grid_size)
    ax.set_box_aspect([puzzle.grid_size[0]/max_range,
                       puzzle.grid_size[1]/max_range,
                       puzzle.grid_size[2]/max_range])

    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    ax.set_xlim(0, puzzle.grid_size[0])
    ax.set_ylim(0, puzzle.grid_size[1])
    ax.set_zlim(0, puzzle.grid_size[2])

    # 绘制球体（在方块顶部）
    ball_x = pos.x + 0.5
    ball_y = pos.y + 0.5
    ball_z = pos.z + 1.35  # 球体在方块顶部的高度

    avg_grid_size = (puzzle.grid_size[0] + puzzle.grid_size[1]) / 2
    ball_size = 1200 * (8 / avg_grid_size) ** 2

    ax.scatter([ball_x], [ball_y], [ball_z],
              c=color, s=ball_size, alpha=1.0,
              edgecolors='none', depthshade=False, marker='o')

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=-0.3,
                facecolor='white', dpi=dpi)
    plt.close()


def _extract_bbox_from_image(img, target_color):
    """从图像中提取指定颜色的 bbox"""
    import numpy as np

    img_array = np.array(img)

    # 找到目标颜色的像素（允许一定误差）
    mask = np.all(np.abs(img_array - target_color) < 10, axis=2)

    # 找到非零像素的坐标
    coords = np.argwhere(mask)

    if len(coords) == 0:
        # 没有找到目标颜色，返回默认值
        return (0, 0, 0, 0, 1, 1)

    # 计算 bbox
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox_x = int(x_min)
    bbox_y = int(y_min)
    bbox_w = int(x_max - x_min + 1)
    bbox_h = int(y_max - y_min + 1)

    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)

    return (center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h)


class Maze3DAdapter(GameAdapter):
    """3D Maze 游戏适配器"""
    
    def __init__(self):
        self.qa_generator = QAGenerator()
    
    def get_game_name(self) -> str:
        return "3d_maze"
    
    def get_required_texture_files(self) -> list:
        """3D Maze 不需要纹理文件"""
        return []
    
    def generate_level(
        self,
        difficulty_config: Dict[str, Any],
        assets_folder: str,
        **kwargs
    ) -> Optional[Any]:
        """生成 3D Maze 关卡（带内部重试）"""
        qa_type = difficulty_config.get('qa_type', 'path_finding')
        index = kwargs.get('level_id', 0)
        max_attempts = difficulty_config.get('max_attempts', 100)

        # 获取网格大小配置
        grid_size = difficulty_config.get('grid_size', None)
        if grid_size:
            grid_size = tuple(grid_size)  # 转换为元组

        # 内部重试生成
        for attempt in range(max_attempts):
            try:
                # 生成问答对和谜题（传入 grid_size）
                data, puzzle = self.qa_generator.generate_qa_pair(index, qa_type, grid_size=grid_size)

                # 返回包含数据和谜题的对象
                return {
                    'data': data,
                    'puzzle': puzzle,
                    'qa_type': qa_type
                }
            except Exception as e:
                if attempt == max_attempts - 1:
                    # 最后一次尝试失败，记录错误
                    logging.error(f"Failed to generate 3D Maze level after {max_attempts} attempts: {e}")
                # 否则静默重试
                continue

        return None
    
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty_name: str,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """保存关卡（图片、视频和状态）"""
        try:
            images_dir = output_dir / "images"
            states_dir = output_dir / "states"
            videos_dir = output_dir / "videos"
            images_dir.mkdir(parents=True, exist_ok=True)
            states_dir.mkdir(parents=True, exist_ok=True)
            videos_dir.mkdir(parents=True, exist_ok=True)

            # 文件名
            image_filename = f"3d_maze_{level_id:05d}.png"
            state_filename = f"3d_maze_{level_id:05d}.json"
            video_filename = f"3d_maze_{level_id:05d}.mp4"

            image_path = images_dir / image_filename
            state_path = states_dir / state_filename
            video_path = videos_dir / video_filename

            # 获取颜色配置
            colors = kwargs.get('colors', None)

            # 绘制谜题图片（起点位置的小球）
            draw_puzzle(level['puzzle'], str(image_path), player_pos=level['puzzle'].start_pos, colors=colors)

            # 生成解决方案视频（自动根据路径长度计算时长）
            generate_video = kwargs.get('generate_video', True)
            if generate_video:
                fps = kwargs.get('fps', 24)
                speed = kwargs.get('speed', 2.0)  # 格子/秒
                generate_solution_video(level['puzzle'], str(video_path), fps=fps, speed=speed, colors=colors)

            # 保存状态（使用 UnifiedState 格式）
            puzzle = level['puzzle']
            grid_size = puzzle.grid_size

            # 通过渲染提取 bbox
            player_bbox_data, goal_bbox_data, (image_w, image_h) = render_and_extract_bbox(
                puzzle, str(image_path)
            )

            # 解包 bbox 数据
            start_x, start_y, start_bx, start_by, start_bw, start_bh = player_bbox_data
            goal_cx, goal_cy, goal_bx, goal_by, goal_bw, goal_bh = goal_bbox_data

            # 创建 Entity（使用渲染后提取的 2D 坐标，无需 grid_pos）
            from core.schema import BBox

            player_entity = Entity(
                pixel_pos=(start_x, start_y),
                bbox=BBox(
                    x=start_bx,
                    y=start_by,
                    width=start_bw,
                    height=start_bh
                ),
                grid_pos=None
            )

            goal_entity = Entity(
                pixel_pos=(goal_cx, goal_cy),
                bbox=BBox(
                    x=goal_bx,
                    y=goal_by,
                    width=goal_bw,
                    height=goal_bh
                ),
                grid_pos=None
            )

            # Maze3D 是 3D 游戏，没有 2D grid
            # 3D 数据存储在 metadata 中
            state = UnifiedState(
                version="1.0",
                game_type="maze3d",
                player=player_entity,
                goal=goal_entity,
                render=RenderConfig(
                    cell_size=max(start_bw, start_bh),
                    image_width=image_w,
                    image_height=image_h
                ),
                grid=None,
                boxes=[],
                metadata={
                    'qa_data': level['data'],
                    'grid_size': list(grid_size),
                    'cubes': [pos.to_tuple() for pos in puzzle.cubes],
                    'start_pos': puzzle.start_pos.to_tuple(),
                    'goal_pos': puzzle.goal_pos.to_tuple(),
                    'ladders': [
                        {
                            'base_pos': ladder.base_pos.to_tuple(),
                            'direction': ladder.direction,
                            'height': ladder.height
                        }
                        for ladder in puzzle.ladders
                    ],
                    'path': [
                        {
                            'start': seg.start.to_tuple(),
                            'end': seg.end.to_tuple(),
                            'type': seg.type
                        }
                        for seg in puzzle.path
                    ]
                }
            )
            state.save(str(state_path))

            return {
                'image': f"images/{image_filename}",
                'state': f"states/{state_filename}",
                'video': f"videos/{video_filename}" if generate_video else None
            }
        except Exception as e:
            logging.error(f"Failed to save 3D Maze level {level_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'image': None,
                'state': None,
                'video': None
            }
    
    def get_level_hash(self, level: Any) -> str:
        """获取关卡的哈希值"""
        try:
            # 使用谜题的方块位置和路径生成哈希
            puzzle = level['puzzle']
            cubes_str = str(sorted([pos.to_tuple() for pos in puzzle.cubes]))
            path_str = str([(seg.start.to_tuple(), seg.end.to_tuple(), seg.type) 
                           for seg in puzzle.path])
            hash_input = f"{cubes_str}_{path_str}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            logging.error(f"Failed to get level hash: {e}")
            return ""
    
    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """检查关卡是否重复"""
        level_hash = self.get_level_hash(level)
        return level_hash in existing_hashes
    
    def validate_difficulty_config(self, difficulty_config: Dict[str, Any]) -> bool:
        """验证难度配置"""
        # 检查必需字段
        if 'count' not in difficulty_config:
            return False
        if 'qa_type' not in difficulty_config:
            return False
        
        # 检查 qa_type 是否有效
        valid_qa_types = ['path_finding', 'sequence_finding', 'height_comparison', 'main_path']
        if difficulty_config['qa_type'] not in valid_qa_types:
            logging.error(f"Invalid qa_type: {difficulty_config['qa_type']}")
            return False
        
        return True
    
    def cleanup(self):
        """清理资源"""
        pass

