"""
Maze renderer using unified core.
Replaces pymaze/texture_handler.py with minimal adapter code.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Iterable, Tuple

# Add parent directory to path to import core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.texture_handler import get_texture_handler
from core.renderer import BaseRenderer, LayeredRenderer
from .config import REQUIRED_TEXTURES, CELL_LAYER_MAP

try:
    from . import constants
except ImportError:
    import constants


Coordinate = Tuple[int, int]


class MazeRenderer:
    """Maze-specific renderer adapter"""
    
    def __init__(self, assets_folder: Optional[str] = None, cell_size: int = constants.CELL_SIZE):
        """
        Initialize Maze renderer.
        
        Args:
            assets_folder: Path to assets folder
            cell_size: Size of each cell in pixels
        """
        if assets_folder is None:
            assets_folder = str(Path(__file__).parent / "assets")
        
        # Get texture handler
        self.handler = get_texture_handler(
            assets_folder=assets_folder,
            cell_size=cell_size,
            texture_names=REQUIRED_TEXTURES
        )
        
        # Create renderer
        self.renderer = BaseRenderer(self.handler)
        self.cell_size = cell_size
    
    def render_maze(self, maze: Sequence[Sequence[int]], save_path: str):
        """
        Render maze to static image.
        
        Args:
            maze: 2D maze array
            save_path: Path to save image
        """
        from PIL import Image, ImageDraw
        
        rows = len(maze)
        cols = len(maze[0]) if rows else 0
        
        width = cols * self.cell_size
        height = rows * self.cell_size
        
        img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Layer 1: Floor
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for row in range(rows):
                for col in range(cols):
                    if maze[row][col] != constants.WALL_CELL:
                        img.paste(floor_tile, (col * self.cell_size, row * self.cell_size), floor_tile)
        
        # Layer 2: Walls and targets
        for row in range(rows):
            for col in range(cols):
                cell_value = maze[row][col]
                texture = None
                
                if cell_value == constants.WALL_CELL:
                    texture = self.handler.get_texture('wall')
                elif cell_value == constants.GOAL_CELL:
                    texture = self.handler.get_texture('target')
                
                if texture:
                    img.paste(texture, (col * self.cell_size, row * self.cell_size), texture)
                elif cell_value == constants.EMPTY_CELL and not self.handler.has_texture('floor'):
                    draw.rectangle([col * self.cell_size, row * self.cell_size,
                                  (col + 1) * self.cell_size, (row + 1) * self.cell_size],
                                 fill=(255, 255, 255), outline=(211, 211, 211))
        
        # Layer 3: Player
        for row in range(rows):
            for col in range(cols):
                if maze[row][col] == constants.PLAYER_CELL:
                    texture = self.handler.get_texture('player')
                    if texture:
                        img.paste(texture, (col * self.cell_size, row * self.cell_size), texture)
                    else:
                        # Fallback: red circle
                        if not self.handler.has_texture('floor'):
                            draw.rectangle([col * self.cell_size, row * self.cell_size,
                                          (col + 1) * self.cell_size, (row + 1) * self.cell_size],
                                         fill=(255, 255, 255), outline=(211, 211, 211))
                        margin = self.cell_size // 4
                        draw.ellipse([col * self.cell_size + margin, row * self.cell_size + margin,
                                    (col + 1) * self.cell_size - margin, (row + 1) * self.cell_size - margin],
                                   fill=(255, 0, 0))
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, format="PNG")
    
    def render_video(self, maze: Sequence[Sequence[int]], path: Iterable[Coordinate],
                    save_path: str, frame_duration_ms: int = 300):
        """
        Render maze solution as MP4 video with smooth continuous movement.

        Args:
            maze: 2D maze array
            path: Sequence of (row, col) positions
            save_path: Path to save MP4 video
            frame_duration_ms: Frame duration in milliseconds (deprecated, now uses 24fps)
        """
        from PIL import Image, ImageDraw
        import numpy as np
        import imageio

        try:
            from .utils import maze_utils
        except ImportError:
            from utils import maze_utils

        path = list(path)
        if not path:
            path = [maze_utils.find_position(maze, constants.PLAYER_CELL)]

        rows = len(maze)
        cols = len(maze[0]) if rows else 0

        width = cols * self.cell_size
        height = rows * self.cell_size

        base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(base)

        # Layer 1: Floor
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for row in range(rows):
                for col in range(cols):
                    if maze[row][col] != constants.WALL_CELL:
                        base.paste(floor_tile, (col * self.cell_size, row * self.cell_size), floor_tile)

        # Layer 2: Walls and targets
        for row in range(rows):
            for col in range(cols):
                cell_value = maze[row][col]
                texture = None

                if cell_value == constants.WALL_CELL:
                    texture = self.handler.get_texture('wall')
                elif cell_value == constants.GOAL_CELL:
                    texture = self.handler.get_texture('target')

                if texture:
                    base.paste(texture, (col * self.cell_size, row * self.cell_size), texture)
                elif cell_value == constants.EMPTY_CELL and not self.handler.has_texture('floor'):
                    draw.rectangle([col * self.cell_size, row * self.cell_size,
                                  (col + 1) * self.cell_size, (row + 1) * self.cell_size],
                                 fill=(255, 255, 255, 255), outline=(211, 211, 211, 255))

        # Generate frames with smooth continuous movement
        fps = 24  # 固定24fps
        frames_per_step = 12  # 每步移动12帧（0.5秒）
        frames = []

        for i in range(len(path) - 1):
            start_row, start_col = path[i]
            end_row, end_col = path[i + 1]

            # 生成中间帧（连续移动）
            for frame_idx in range(frames_per_step):
                progress = frame_idx / frames_per_step

                # 计算当前位置（像素级插值）
                current_col = start_col + (end_col - start_col) * progress
                current_row = start_row + (end_row - start_row) * progress

                frame = base.copy()

                # 如果玩家在目标上，先绘制目标
                cell_row = int(round(current_row))
                cell_col = int(round(current_col))
                if 0 <= cell_row < rows and 0 <= cell_col < cols:
                    if maze[cell_row][cell_col] == constants.GOAL_CELL:
                        target_texture = self.handler.get_texture('target')
                        if target_texture:
                            frame.paste(target_texture, (cell_col * self.cell_size, cell_row * self.cell_size), target_texture)

                # 绘制玩家（使用像素级位置）
                player_x = int(current_col * self.cell_size)
                player_y = int(current_row * self.cell_size)

                player_texture = self.handler.get_texture('player')
                if player_texture:
                    frame.paste(player_texture, (player_x, player_y), player_texture)
                else:
                    fdraw = ImageDraw.Draw(frame)
                    margin = self.cell_size // 4
                    fdraw.ellipse([player_x + margin, player_y + margin,
                                 player_x + self.cell_size - margin, player_y + self.cell_size - margin],
                                fill=(255, 0, 0, 255))

                frames.append(frame)

        # 添加最后一帧（停留在终点）
        final_row, final_col = path[-1]
        frame = base.copy()
        if maze[final_row][final_col] == constants.GOAL_CELL:
            target_texture = self.handler.get_texture('target')
            if target_texture:
                frame.paste(target_texture, (final_col * self.cell_size, final_row * self.cell_size), target_texture)

        player_texture = self.handler.get_texture('player')
        if player_texture:
            frame.paste(player_texture, (final_col * self.cell_size, final_row * self.cell_size), player_texture)
        else:
            fdraw = ImageDraw.Draw(frame)
            margin = self.cell_size // 4
            fdraw.ellipse([final_col * self.cell_size + margin, final_row * self.cell_size + margin,
                         (final_col + 1) * self.cell_size - margin, (final_row + 1) * self.cell_size - margin],
                        fill=(255, 0, 0, 255))
        frames.append(frame)

        # Save as MP4 using imageio
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            # Convert PIL images to numpy arrays
            frame_arrays = [np.array(frame.convert('RGB')) for frame in frames]

            try:
                # Save as MP4 with 24fps
                with imageio.get_writer(str(output_path), format="FFMPEG", mode="I", fps=fps,
                                       codec="libx264", pixelformat="yuv420p",
                                       macro_block_size=1) as writer:
                    for frame_array in frame_arrays:
                        writer.append_data(frame_array)
            except Exception as e:
                # Fallback to GIF if MP4 fails
                import logging
                logging.warning(f"Failed to save MP4, falling back to GIF: {e}")
                fallback_path = output_path.with_suffix('.gif')
                frames[0].save(
                    fallback_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=frame_duration_ms,
                    loop=0,
                    disposal=2
                )


# Global renderer cache
_shared_renderer: Optional[MazeRenderer] = None
_current_assets_folder: Optional[str] = None


def get_shared_renderer(assets_folder: Optional[str] = None) -> MazeRenderer:
    """
    Get or create shared Maze renderer.
    
    Args:
        assets_folder: Path to assets folder
        
    Returns:
        Cached or new renderer
    """
    global _shared_renderer, _current_assets_folder
    
    if assets_folder is None:
        assets_folder = str(Path(__file__).parent / "assets")
    
    if _shared_renderer is None or _current_assets_folder != assets_folder:
        _shared_renderer = MazeRenderer(assets_folder=assets_folder)
        _current_assets_folder = assets_folder
    
    return _shared_renderer

