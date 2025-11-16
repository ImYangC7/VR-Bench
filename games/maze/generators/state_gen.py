from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.schema import UnifiedState, Grid, Entity, RenderConfig
from .. import constants
from ..utils import maze_utils


def save_state(maze: Sequence[Sequence[int]], save_path: str) -> None:
    player_pos = maze_utils.find_position(maze, constants.PLAYER_CELL)
    goal_pos = maze_utils.find_position(maze, constants.GOAL_CELL)

    height = len(maze)
    width = len(maze[0]) if maze else 0
    cell_size = constants.CELL_SIZE

    state = UnifiedState(
        version="1.0",
        game_type="maze",
        grid=Grid.from_2d_list([list(row) for row in maze]),
        player=Entity.from_grid_pos(player_pos[0], player_pos[1], cell_size),
        goal=Entity.from_grid_pos(goal_pos[0], goal_pos[1], cell_size),
        boxes=[],
        render=RenderConfig.from_grid_size(height, width, cell_size),
        metadata={}
    )

    state.save(save_path)
