from __future__ import annotations

import random
from typing import List, Tuple

from .. import constants

Coordinate = Tuple[int, int]

_RANDOM = random.Random()


def generate_maze(rows: int, cols: int) -> List[List[int]]:
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("The number of rows and columns in the maze must be odd!")

    maze = [[constants.WALL_CELL for _ in range(cols)] for _ in range(rows)]
    maze[1][1] = constants.EMPTY_CELL
    _dfs(maze, 1, 1)
    _place_player_and_goal(maze)
    return maze


def _dfs(maze: List[List[int]], row: int, col: int) -> None:
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    _RANDOM.shuffle(directions)

    for d_row, d_col in directions:
        next_row = row + d_row
        next_col = col + d_col
        if _is_in_bounds(maze, next_row, next_col) and maze[next_row][next_col] == constants.WALL_CELL:
            maze[row + d_row // 2][col + d_col // 2] = constants.EMPTY_CELL
            maze[next_row][next_col] = constants.EMPTY_CELL
            _dfs(maze, next_row, next_col)


def _place_player_and_goal(maze: List[List[int]]) -> None:
    rows = len(maze)
    cols = len(maze[0]) if rows else 0

    empty_cells: List[Coordinate] = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if maze[r][c] == constants.EMPTY_CELL
    ]
    if not empty_cells:
        raise ValueError("There are no empty cells in the maze to place the player and goal.")

    player_row, player_col = _RANDOM.choice(empty_cells)

    distances = []
    for cell in empty_cells:
        if cell == (player_row, player_col):
            continue
        distance = abs(cell[0] - player_row) + abs(cell[1] - player_col)
        distances.append((cell, distance))

    if not distances:
        raise ValueError("There are not enough empty cells to place the goal.")

    unique_distances = sorted({distance for _, distance in distances}, reverse=True)
    if len(unique_distances) >= 3:
        target_distance = unique_distances[2]
    elif len(unique_distances) == 2:
        target_distance = unique_distances[1]
    else:
        target_distance = unique_distances[0]

    candidates = [cell for cell, distance in distances if distance == target_distance]
    if not candidates:
        raise ValueError("No cells found with the target distance to place the goal.")

    goal_row, goal_col = _RANDOM.choice(candidates)

    maze[player_row][player_col] = constants.PLAYER_CELL
    maze[goal_row][goal_col] = constants.GOAL_CELL


def _is_in_bounds(maze: List[List[int]], row: int, col: int) -> bool:
    return 0 < row < len(maze) and 0 < col < len(maze[0])
