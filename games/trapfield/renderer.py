"""
TrapField 渲染器
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Iterable, Tuple
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.texture_handler import get_texture_handler
from core.renderer import BaseRenderer
from .config import REQUIRED_TEXTURES
from . import constants

Coordinate = Tuple[int, int]


class TrapFieldRenderer:
    """TrapField 渲染器"""
    
    def __init__(self, assets_folder: Optional[str] = None, cell_size: int = constants.CELL_SIZE):
        """
        初始化渲染器
        
        Args:
            assets_folder: 资源文件夹路径
            cell_size: 单元格大小（像素）
        """
        if assets_folder is None:
            assets_folder = str(Path(__file__).parent / "assets")
        
        # 获取纹理处理器
        self.handler = get_texture_handler(
            assets_folder=assets_folder,
            cell_size=cell_size,
            texture_names=REQUIRED_TEXTURES
        )
        
        # 创建渲染器
        self.renderer = BaseRenderer(self.handler)
        self.cell_size = cell_size
    
    def render_grid(self, grid: Sequence[Sequence[int]], save_path: str):
        """
        渲染静态图片
        
        Args:
            grid: 2D 网格
            save_path: 保存路径
        """
        from PIL import Image
        
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        
        width = cols * self.cell_size
        height = rows * self.cell_size
        
        img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        
        # Layer 1: 地板
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for row in range(rows):
                for col in range(cols):
                    img.paste(floor_tile, (col * self.cell_size, row * self.cell_size), floor_tile)
        
        # Layer 2: 陷阱
        if self.handler.has_texture('trap'):
            trap_tile = self.handler.get_texture('trap')
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col] == constants.TRAP_CELL:
                        img.paste(trap_tile, (col * self.cell_size, row * self.cell_size), trap_tile)
        
        # Layer 3: 目标
        if self.handler.has_texture('goal'):
            goal_tile = self.handler.get_texture('goal')
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col] == constants.GOAL_CELL:
                        img.paste(goal_tile, (col * self.cell_size, row * self.cell_size), goal_tile)
        
        # Layer 4: 玩家
        if self.handler.has_texture('player'):
            player_tile = self.handler.get_texture('player')
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col] == constants.PLAYER_CELL:
                        img.paste(player_tile, (col * self.cell_size, row * self.cell_size), player_tile)
        
        img.save(save_path)
    
    def render_video(
        self, 
        grid: Sequence[Sequence[int]], 
        path: Iterable[Coordinate],
        save_path: str, 
        frame_duration_ms: int = 300
    ):
        """
        渲染解决方案视频（连续移动，24fps）
        
        Args:
            grid: 2D 网格
            path: 解决方案路径（坐标列表）
            save_path: 保存路径
            frame_duration_ms: 每步移动的持续时间（毫秒，已废弃）
        """
        from PIL import Image
        
        path_list = list(path)
        if len(path_list) < 2:
            return
        
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        width = cols * self.cell_size
        height = rows * self.cell_size
        
        # 固定24fps
        fps = 24
        frames_per_step = 12  # 每步移动12帧（0.5秒）
        frames = []
        
        # 生成连续移动的帧
        for i in range(len(path_list) - 1):
            start_row, start_col = path_list[i]
            end_row, end_col = path_list[i + 1]
            
            # 生成中间帧（连续移动）
            for frame_idx in range(frames_per_step):
                progress = frame_idx / frames_per_step
                
                # 计算当前位置（像素级插值）
                current_col = start_col + (end_col - start_col) * progress
                current_row = start_row + (end_row - start_row) * progress
                
                # 创建帧
                img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
                
                # Layer 1: 地板
                if self.handler.has_texture('floor'):
                    floor_tile = self.handler.get_texture('floor')
                    for row in range(rows):
                        for col in range(cols):
                            img.paste(floor_tile, (col * self.cell_size, row * self.cell_size), floor_tile)
                
                # Layer 2: 陷阱
                if self.handler.has_texture('trap'):
                    trap_tile = self.handler.get_texture('trap')
                    for row in range(rows):
                        for col in range(cols):
                            if grid[row][col] == constants.TRAP_CELL:
                                img.paste(trap_tile, (col * self.cell_size, row * self.cell_size), trap_tile)
                
                # Layer 3: 目标
                if self.handler.has_texture('goal'):
                    goal_tile = self.handler.get_texture('goal')
                    for row in range(rows):
                        for col in range(cols):
                            if grid[row][col] == constants.GOAL_CELL:
                                img.paste(goal_tile, (col * self.cell_size, row * self.cell_size), goal_tile)
                
                # Layer 4: 玩家（在插值位置）
                if self.handler.has_texture('player'):
                    player_tile = self.handler.get_texture('player')
                    player_x = int(current_col * self.cell_size)
                    player_y = int(current_row * self.cell_size)
                    img.paste(player_tile, (player_x, player_y), player_tile)
                
                frames.append(img)
        
        # 添加最后一帧（停留在目标位置）
        final_row, final_col = path_list[-1]
        img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        
        # Layer 1: 地板
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for row in range(rows):
                for col in range(cols):
                    img.paste(floor_tile, (col * self.cell_size, row * self.cell_size), floor_tile)
        
        # Layer 2: 陷阱
        if self.handler.has_texture('trap'):
            trap_tile = self.handler.get_texture('trap')
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col] == constants.TRAP_CELL:
                        img.paste(trap_tile, (col * self.cell_size, row * self.cell_size), trap_tile)
        
        # Layer 3: 目标
        if self.handler.has_texture('goal'):
            goal_tile = self.handler.get_texture('goal')
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col] == constants.GOAL_CELL:
                        img.paste(goal_tile, (col * self.cell_size, row * self.cell_size), goal_tile)
        
        # Layer 4: 玩家（在最终位置）
        if self.handler.has_texture('player'):
            player_tile = self.handler.get_texture('player')
            player_x = final_col * self.cell_size
            player_y = final_row * self.cell_size
            img.paste(player_tile, (player_x, player_y), player_tile)
        
        frames.append(img)
        
        # 保存视频
        try:
            # 转换为 numpy 数组
            import numpy as np
            frame_arrays = [np.array(frame.convert('RGB')) for frame in frames]
            
            # 保存为 MP4
            imageio.mimsave(save_path, frame_arrays, fps=fps)
        except Exception as e:
            # 如果 MP4 失败，尝试保存为 GIF
            import logging
            logging.warning(f"Failed to save MP4, falling back to GIF: {e}")
            gif_path = save_path.replace('.mp4', '.gif')
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0
            )


def create_solution_video(
    grid: Sequence[Sequence[int]],
    path: Iterable[Coordinate],
    cell_size: int,
    save_path: str,
    frame_duration_ms: int = 300,
    assets_folder: Optional[str] = None
) -> None:
    """
    创建解决方案视频的便捷函数
    
    Args:
        grid: 2D 网格
        path: 解决方案路径
        cell_size: 单元格大小
        save_path: 保存路径
        frame_duration_ms: 帧持续时间（已废弃，固定24fps）
        assets_folder: 资源文件夹路径
    """
    renderer = TrapFieldRenderer(assets_folder=assets_folder, cell_size=cell_size)
    renderer.render_video(grid, path, save_path, frame_duration_ms)

