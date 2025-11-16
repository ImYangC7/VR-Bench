"""
Unified renderer for all games.
Handles static image and video rendering with three-layer system.
"""

from pathlib import Path
from typing import Optional, Sequence, Callable
from PIL import Image, ImageDraw
import numpy as np

from .texture_handler import BaseTextureHandler
from .constants import WALL


class BaseRenderer:
    """Base renderer for game boards"""
    
    def __init__(self, texture_handler: BaseTextureHandler):
        """
        Initialize renderer.
        
        Args:
            texture_handler: Texture handler instance
        """
        self.handler = texture_handler
        self.cell_size = texture_handler.cell_size
    
    def render_static(self, 
                     board: np.ndarray,
                     output_path: str,
                     cell_classifier: Callable,
                     add_grid: bool = False) -> Image.Image:
        """
        Render a static image of the board.
        
        Args:
            board: 2D numpy array representing the board
            output_path: Path to save the image
            cell_classifier: Function that returns (layer, texture_name) for each cell value
            add_grid: Whether to add grid lines
            
        Returns:
            Rendered image
        """
        height, width = board.shape
        
        # Create canvas
        total_width = width * self.cell_size
        total_height = height * self.cell_size
        
        img = Image.new('RGB', (total_width, total_height), "#E0C9A6")
        draw = ImageDraw.Draw(img)
        
        # Three-layer rendering: floor -> walls/targets -> players/boxes
        
        # Layer 1: Floor
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for y in range(height):
                for x in range(width):
                    if board[y, x] != WALL:
                        img.paste(floor_tile, (x * self.cell_size, y * self.cell_size), floor_tile)
        
        # Layer 2: Walls and targets
        for y in range(height):
            for x in range(width):
                layer, texture_name = cell_classifier(board[y, x])
                
                if layer == 2 and texture_name:
                    texture = self.handler.get_texture(texture_name)
                    if texture:
                        img.paste(texture, 
                                (x * self.cell_size, y * self.cell_size),
                                texture if texture.mode == 'RGBA' else None)
        
        # Layer 3: Players and boxes
        for y in range(height):
            for x in range(width):
                layer, texture_name = cell_classifier(board[y, x])
                
                if layer == 3 and texture_name:
                    texture = self.handler.get_texture(texture_name)
                    if texture:
                        img.paste(texture,
                                (x * self.cell_size, y * self.cell_size),
                                texture if texture.mode == 'RGBA' else None)
        
        # Add grid if requested
        if add_grid:
            for i in range(height + 1):
                y_pos = i * self.cell_size
                draw.line([(0, y_pos), (total_width, y_pos)], fill="#000000", width=2)
            
            for i in range(width + 1):
                x_pos = i * self.cell_size
                draw.line([(x_pos, 0), (x_pos, total_height)], fill="#000000", width=2)
        
        # Save image
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
        return img
    
    def render_video_gif(self,
                        base_board: np.ndarray,
                        path: Sequence[tuple],
                        output_path: str,
                        cell_classifier: Callable,
                        player_renderer: Callable,
                        frame_duration_ms: int = 300) -> None:
        """
        Render an animated GIF video.
        
        Args:
            base_board: Base board state (without player)
            path: Sequence of (row, col) positions for animation
            output_path: Path to save the GIF
            cell_classifier: Function that returns (layer, texture_name) for each cell value
            player_renderer: Function that renders player at given position on a frame
            frame_duration_ms: Duration of each frame in milliseconds
        """
        height, width = base_board.shape
        
        # Create base image (without player)
        total_width = width * self.cell_size
        total_height = height * self.cell_size
        
        base = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(base)
        
        # Layer 1: Floor
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for y in range(height):
                for x in range(width):
                    if base_board[y, x] != WALL:
                        base.paste(floor_tile, (x * self.cell_size, y * self.cell_size), floor_tile)
        
        # Layer 2: Walls and targets
        for y in range(height):
            for x in range(width):
                layer, texture_name = cell_classifier(base_board[y, x])
                
                if layer == 2 and texture_name:
                    texture = self.handler.get_texture(texture_name)
                    if texture:
                        base.paste(texture,
                                 (x * self.cell_size, y * self.cell_size),
                                 texture if texture.mode == 'RGBA' else None)
        
        # Generate frames with player at each position
        frames = []
        for row, col in path:
            frame = base.copy()
            player_renderer(frame, row, col, base_board)
            frames.append(frame)
        
        # Save as GIF
        if frames:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                output_path_obj,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                disposal=2
            )


class LayeredRenderer:
    """Helper class for layer-based rendering logic"""
    
    @staticmethod
    def create_cell_classifier(layer_map: dict) -> Callable:
        """
        Create a cell classifier function from a layer map.
        
        Args:
            layer_map: Dict mapping cell values to (layer, texture_name)
            
        Returns:
            Classifier function
        """
        def classifier(cell_value):
            return layer_map.get(cell_value, (0, None))
        return classifier
    
    @staticmethod
    def create_player_renderer(texture_handler: BaseTextureHandler,
                              cell_size: int,
                              target_value: int) -> Callable:
        """
        Create a player renderer function.
        
        Args:
            texture_handler: Texture handler instance
            cell_size: Size of each cell
            target_value: Cell value representing target/goal
            
        Returns:
            Player renderer function
        """
        def render_player(frame: Image.Image, row: int, col: int, board: np.ndarray):
            x = col * cell_size
            y = row * cell_size
            
            # If player is on target, draw target first
            if board[row, col] == target_value:
                target_texture = texture_handler.get_texture('target')
                if target_texture:
                    frame.paste(target_texture, (x, y), target_texture)
            
            # Draw player
            player_texture = texture_handler.get_texture('player')
            if player_texture:
                frame.paste(player_texture, (x, y), player_texture)
            else:
                # Fallback: red circle
                draw = ImageDraw.Draw(frame)
                margin = cell_size // 4
                draw.ellipse([x + margin, y + margin,
                            x + cell_size - margin, y + cell_size - margin],
                           fill=(255, 0, 0, 255))
        
        return render_player

