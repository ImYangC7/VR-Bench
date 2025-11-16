"""
Unified texture handler for all games.
Handles loading, caching, and resizing of game textures.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from PIL import Image

from .constants import SUPPORTED_IMAGE_FORMATS, DEFAULT_CELL_SIZE


class BaseTextureHandler:
    """Base texture handler shared by all games"""
    
    def __init__(self, assets_folder: Optional[str] = None, cell_size: int = DEFAULT_CELL_SIZE):
        """
        Initialize texture handler.
        
        Args:
            assets_folder: Path to assets folder. If None, uses default.
            cell_size: Size of each cell in pixels.
        """
        if assets_folder is None:
            assets_folder = Path(__file__).parent.parent / "assets"
        
        self.assets_path = Path(assets_folder)
        self.cell_size = cell_size
        self.textures: Dict[str, Image.Image] = {}
        
        # Ensure assets directory exists
        self.assets_path.mkdir(parents=True, exist_ok=True)
    
    def load_textures(self, texture_names: list):
        """
        Load specified textures.
        
        Args:
            texture_names: List of texture names to load (e.g., ['floor', 'wall', 'player'])
        """
        for name in texture_names:
            texture = self._load_texture(name)
            if texture:
                self.textures[name] = texture
    
    def _load_texture(self, name: str) -> Optional[Image.Image]:
        """
        Load a single texture from file.
        
        Args:
            name: Texture name (without extension)
            
        Returns:
            Loaded and resized texture, or None if not found
        """
        file_path = None
        for ext in SUPPORTED_IMAGE_FORMATS:
            candidate = self.assets_path / f"{name}{ext}"
            if candidate.exists():
                file_path = candidate
                break
        
        if file_path is None:
            return None
        
        try:
            img = Image.open(file_path).convert("RGBA")
            return self._resize_keep_aspect_ratio(img, self.cell_size)
        except Exception as e:
            print(f"Failed to load texture {name}: {e}")
            return None
    
    def _resize_keep_aspect_ratio(self, img: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image while maintaining aspect ratio, centered on transparent canvas.
        
        Args:
            img: Source image
            target_size: Target size (width and height)
            
        Returns:
            Resized image on transparent canvas
        """
        original_width, original_height = img.size
        
        # Calculate scaling ratio
        ratio = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize with high quality
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create transparent canvas
        result = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        
        # Center the resized image
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        result.paste(resized_img, (paste_x, paste_y), resized_img if resized_img.mode == 'RGBA' else None)
        
        return result
    
    def get_texture(self, name: str) -> Optional[Image.Image]:
        """
        Get a loaded texture by name.
        
        Args:
            name: Texture name
            
        Returns:
            Texture image or None if not loaded
        """
        return self.textures.get(name)
    
    def has_texture(self, name: str) -> bool:
        """
        Check if a texture is loaded.
        
        Args:
            name: Texture name
            
        Returns:
            True if texture is loaded
        """
        return name in self.textures


# Global texture handler cache
_texture_handlers: Dict[str, BaseTextureHandler] = {}


def get_texture_handler(assets_folder: Optional[str] = None, 
                       cell_size: int = DEFAULT_CELL_SIZE,
                       texture_names: Optional[list] = None) -> BaseTextureHandler:
    """
    Get or create a cached texture handler.
    
    Args:
        assets_folder: Path to assets folder
        cell_size: Size of each cell
        texture_names: List of textures to load
        
    Returns:
        Cached or new texture handler
    """
    if assets_folder is None:
        assets_folder = str(Path(__file__).parent.parent / "assets")
    
    cache_key = f"{assets_folder}:{cell_size}"
    
    if cache_key not in _texture_handlers:
        handler = BaseTextureHandler(assets_folder, cell_size)
        if texture_names:
            handler.load_textures(texture_names)
        _texture_handlers[cache_key] = handler
    
    return _texture_handlers[cache_key]


def clear_texture_cache():
    """Clear the texture handler cache."""
    global _texture_handlers
    _texture_handlers.clear()

