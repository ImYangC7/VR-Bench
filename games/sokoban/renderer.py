"""
Sokoban renderer using unified core.
Replaces sokoban_texture_handler.py with minimal adapter code.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from core.texture_handler import get_texture_handler
from core.renderer import BaseRenderer, LayeredRenderer
from .config import REQUIRED_TEXTURES, CELL_LAYER_MAP, COMBINED_CELLS
from core.constants import TARGET, BOX_ON_TARGET, PLAYER_ON_TARGET


class SokobanRenderer:
    """Sokoban-specific renderer adapter"""
    
    def __init__(self, assets_folder: Optional[str] = None, texture_size: int = 64):
        """
        Initialize Sokoban renderer.
        
        Args:
            assets_folder: Path to assets folder
            texture_size: Size of each texture in pixels
        """
        # Get texture handler
        self.handler = get_texture_handler(
            assets_folder=assets_folder,
            cell_size=texture_size,
            texture_names=REQUIRED_TEXTURES
        )
        
        # Create renderer
        self.renderer = BaseRenderer(self.handler)
        self.texture_size = texture_size
    
    def render_board(self, sokoban_board, output_path: Optional[str] = None, 
                    add_grid: bool = False) -> Image.Image:
        """
        Render Sokoban board to image.
        
        Args:
            sokoban_board: Sokoban board object with .grid attribute
            output_path: Path to save image (optional)
            add_grid: Whether to add grid lines
            
        Returns:
            Rendered image
        """
        grid = sokoban_board.grid
        
        # Create cell classifier with special handling for combined cells
        def cell_classifier(cell_value):
            # Handle combined cells (box/player on target)
            if cell_value in COMBINED_CELLS:
                # Return target for layer 2
                return (2, 'target')
            return CELL_LAYER_MAP.get(cell_value, (0, None))
        
        # Render with layer 2 (walls and targets)
        height, width = grid.shape
        total_width = width * self.texture_size
        total_height = height * self.texture_size
        
        img = Image.new('RGB', (total_width, total_height), "#E0C9A6")
        
        # Layer 1: Floor
        if self.handler.has_texture('floor'):
            floor_tile = self.handler.get_texture('floor')
            for y in range(height):
                for x in range(width):
                    if grid[y, x] != 1:  # Not wall
                        img.paste(floor_tile, (x * self.texture_size, y * self.texture_size), floor_tile)
        
        # Layer 2: Walls and targets
        for y in range(height):
            for x in range(width):
                cell = grid[y, x]
                texture = None
                
                if cell == 1:  # Wall
                    texture = self.handler.get_texture('wall')
                elif cell in [3, 4, 6]:  # Target, box on target, player on target
                    texture = self.handler.get_texture('target')
                
                if texture:
                    img.paste(texture,
                            (x * self.texture_size, y * self.texture_size),
                            texture if texture.mode == 'RGBA' else None)
        
        # Layer 3: Boxes and players
        for y in range(height):
            for x in range(width):
                cell = grid[y, x]
                texture = None
                
                if cell in [2, 4]:  # Box or box on target
                    texture = self.handler.get_texture('box')
                elif cell in [5, 6]:  # Player or player on target
                    texture = self.handler.get_texture('player')
                
                if texture:
                    img.paste(texture,
                            (x * self.texture_size, y * self.texture_size),
                            texture if texture.mode == 'RGBA' else None)
        
        # Add grid if requested
        if add_grid:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for i in range(height + 1):
                y_pos = i * self.texture_size
                draw.line([(0, y_pos), (total_width, y_pos)], fill="#000000", width=2)
            
            for i in range(width + 1):
                x_pos = i * self.texture_size
                draw.line([(x_pos, 0), (x_pos, total_height)], fill="#000000", width=2)
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
        
        return img


# Global renderer cache
_shared_renderer: Optional[SokobanRenderer] = None
_current_assets_folder: Optional[str] = None


def get_shared_renderer(assets_folder: Optional[str] = None) -> SokobanRenderer:
    """
    Get or create shared Sokoban renderer.
    
    Args:
        assets_folder: Path to assets folder
        
    Returns:
        Cached or new renderer
    """
    global _shared_renderer, _current_assets_folder
    
    if assets_folder is None:
        assets_folder = str(Path(__file__).parent / "assets")
    
    if _shared_renderer is None or _current_assets_folder != assets_folder:
        _shared_renderer = SokobanRenderer(assets_folder=assets_folder)
        _current_assets_folder = assets_folder
    
    return _shared_renderer

