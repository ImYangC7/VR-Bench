"""
Unified constants for all games.
"""

# Cell types - common across all games
EMPTY = 0
WALL = 1
PLAYER = 2
TARGET = 3

# Sokoban-specific cell types (Sokoban uses different values)
SOKOBAN_EMPTY = 0
SOKOBAN_WALL = 1
SOKOBAN_BOX = 2
SOKOBAN_TARGET = 3
SOKOBAN_BOX_ON_TARGET = 4
SOKOBAN_PLAYER = 5
SOKOBAN_PLAYER_ON_TARGET = 6

# Legacy Sokoban constants (for backward compatibility with core constants)
BOX = 4
BOX_ON_TARGET = 5
PLAYER_ON_TARGET = 6

# Maze-specific aliases (for compatibility)
EMPTY_CELL = EMPTY
WALL_CELL = WALL
PLAYER_CELL = PLAYER
GOAL_CELL = TARGET

# Rendering configuration
DEFAULT_CELL_SIZE = 64
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg')

# Texture names - unified across all games
TEXTURE_NAMES = {
    'floor': 'floor',
    'wall': 'wall',
    'player': 'player',
    'target': 'target',
    'box': 'box'
}

