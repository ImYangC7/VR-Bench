"""
Core module for game rendering and texture handling.
Shared by all games (Sokoban, Maze, etc.)
"""

from .constants import *
from .texture_handler import BaseTextureHandler
from .renderer import BaseRenderer

__all__ = [
    'BaseTextureHandler',
    'BaseRenderer',
    'EMPTY',
    'WALL',
    'PLAYER',
    'TARGET',
    'BOX',
    'BOX_ON_TARGET',
    'PLAYER_ON_TARGET',
]

