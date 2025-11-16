from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import random

from .. import constants
from ..templates import (
    AvailableDirections,
    FindPathToGoal,
    GoalPosition,
    PlayerPosition,
    PositionAfterMoving,
    TurnCount,
)
from ..templates.base_template import BaseTemplate
from ..utils import maze_utils
from . import image_gen, maze_gen, state_gen, video_gen


def generate_json_data(maze: List[List[int]], data_id: int) -> List[BaseTemplate]:
    return [
        PlayerPosition(maze, data_id),
        GoalPosition(maze, data_id),
        PositionAfterMoving(maze, data_id),
        AvailableDirections(maze, data_id),
        FindPathToGoal(maze, data_id),
        TurnCount(maze, data_id),
    ]


def generate_data(
    id_begin: int,
    amount: int,
    maze_size: int,
    images_dir: str,
    states_dir: str,
    video_dir: str,
    assets_folder: Optional[str] = None,
) -> List[BaseTemplate]:
    dataset: List[BaseTemplate] = []
    for internal_id in range(id_begin, id_begin + amount):
        maze = maze_gen.generate_maze(maze_size, maze_size)

        image_path = Path(images_dir) / f"image_{internal_id:05d}.png"
        state_path = Path(states_dir) / f"state_{internal_id:05d}.json"
        video_path = Path(video_dir) / f"video_{internal_id:05d}.gif"

        image_gen.draw_maze(maze, constants.CELL_SIZE, str(image_path), assets_folder=assets_folder)
        state_gen.save_state(maze, str(state_path))

        solver_rng = random.Random(internal_id)
        path = maze_utils.dfs_solve_maze(maze, [], rng=solver_rng)
        video_gen.create_solution_video(
            maze,
            path,
            constants.CELL_SIZE,
            str(video_path),
            assets_folder=assets_folder,
        )

        dataset.extend(generate_json_data(maze, internal_id))
    return dataset


def save_data_to_json(records: List[BaseTemplate], file_path: str) -> None:
    payload: List[Dict[str, object]] = [item.to_dict() for item in records]
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
