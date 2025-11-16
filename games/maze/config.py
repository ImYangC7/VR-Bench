"""
Maze game configuration.
"""

from core.constants import *

# Required textures for Maze
REQUIRED_TEXTURES = ['floor', 'wall', 'player', 'target']

# Cell type to layer and texture mapping
# Format: cell_value -> (layer, texture_name)
# Layer 1: floor (handled separately)
# Layer 2: walls and targets (goals)
# Layer 3: player
CELL_LAYER_MAP = {
    EMPTY_CELL: (0, None),
    WALL_CELL: (2, 'wall'),
    PLAYER_CELL: (3, 'player'),
    GOAL_CELL: (2, 'target')
}

