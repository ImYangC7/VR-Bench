"""
Sokoban game configuration.
"""

from core.constants import *

# Required textures for Sokoban
REQUIRED_TEXTURES = ['floor', 'wall', 'player', 'target', 'box']

# Cell type to layer and texture mapping
# Format: cell_value -> (layer, texture_name)
# Layer 1: floor (handled separately)
# Layer 2: walls and targets
# Layer 3: boxes and players
CELL_LAYER_MAP = {
    EMPTY: (0, None),
    WALL: (2, 'wall'),
    PLAYER: (3, 'player'),
    TARGET: (2, 'target'),
    BOX: (3, 'box'),
    BOX_ON_TARGET: (3, 'box'),      # Box on target: target in layer 2, box in layer 3
    PLAYER_ON_TARGET: (3, 'player')  # Player on target: target in layer 2, player in layer 3
}

# Special handling for combined cells
COMBINED_CELLS = {
    BOX_ON_TARGET: ('target', 'box'),      # (layer 2, layer 3)
    PLAYER_ON_TARGET: ('target', 'player')  # (layer 2, layer 3)
}

