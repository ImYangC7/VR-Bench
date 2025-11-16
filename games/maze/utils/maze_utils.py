from __future__ import annotations

import random
from random import Random
from typing import Iterable, List, Sequence, Tuple

from .. import constants

Coordinate = Tuple[int, int]


def find_position(maze: Sequence[Sequence[int]], target_value: int) -> Coordinate:
    for row_index, row in enumerate(maze):
        for col_index, cell in enumerate(row):
            if cell == target_value:
                return row_index, col_index
    raise ValueError(f"Target value not found in the maze: {target_value}")


def dfs_solve_maze(
    maze: Sequence[Sequence[int]],
    info: List[str],
    rng: Random | None = None,
) -> List[Coordinate]:
    random_source = rng if rng is not None else random

    maze_state = [list(row) for row in maze]
    path: List[Coordinate] = []
    backtrack: List[Tuple[int, int, int]] = []

    start_row, start_col = find_position(maze, constants.PLAYER_CELL)
    goal_row, goal_col = find_position(maze, constants.GOAL_CELL)

    stack: List[Coordinate] = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    maze_state[start_row][start_col] = -1

    shuffled = directions[:]
    random_source.shuffle(shuffled)
    branch_count = 0
    for d_row, d_col in shuffled:
        next_row = start_row + d_row
        next_col = start_col + d_col
        if _is_valid(maze_state, next_row, next_col):
            stack.append((next_row, next_col))
            branch_count += 1

    if branch_count > 1:
        backtrack.append((start_row, start_col, branch_count - 1))

    path.append((start_row, start_col))
    current_row, current_col = start_row, start_col
    step = 1

    info.append("Let's figure out the path to the goal step by step.\n")

    while stack:
        info.append(f"Step {step}. ")
        step += 1

        row, col = stack.pop()
        maze_state[row][col] = -1
        path.append((row, col))

        direction = get_direction((current_row, current_col), (row, col))
        info.append(f"Go {direction}, from ({current_row}, {current_col}) to ({row}, {col}). ")

        current_row, current_col = row, col

        if row == goal_row and col == goal_col:
            info.append("Achieved the goal!")
            return path

        branch_count = 0
        shuffled = directions[:]
        random_source.shuffle(shuffled)
        for d_row, d_col in shuffled:
            next_row = row + d_row
            next_col = col + d_col
            if _is_valid(maze_state, next_row, next_col):
                stack.append((next_row, next_col))
                branch_count += 1

        if branch_count > 1:
            backtrack.append((row, col, branch_count - 1))

        if branch_count == 0 and backtrack:
            back_row, back_col, remaining = backtrack.pop()
            info.append(
                f"Oops! We hit a dead end. Going back to the last unexplored branch at ({back_row}, {back_col})."
            )
            if remaining > 1:
                backtrack.append((back_row, back_col, remaining - 1))

            while path and path[-1] != (back_row, back_col):
                path.pop()
            current_row, current_col = back_row, back_col

        info.append("\n")

    return path


def path_to_string(path: Iterable[Coordinate]) -> str:
    return ", ".join(f"({row}, {col})" for row, col in path)


def count_turns(path: Sequence[Coordinate], info: List[str]) -> int:
    if len(path) < 2:
        return 0

    info.append("Let's count the number of turns step by step.\n")

    turns = 0
    prev_direction = get_direction(path[0], path[1])

    for index in range(2, len(path)):
        current_direction = get_direction(path[index - 1], path[index])
        info.append(f"Step {index}. ")
        if current_direction != prev_direction:
            info.append(f"Turn detected: from {prev_direction} to {current_direction}.\n")
            turns += 1
            prev_direction = current_direction
        else:
            info.append("No turn detected.\n")

    return turns


def get_direction(origin: Coordinate, target: Coordinate) -> str:
    row_delta = target[0] - origin[0]
    col_delta = target[1] - origin[1]

    if row_delta == -1 and col_delta == 0:
        return "up"
    if row_delta == 1 and col_delta == 0:
        return "down"
    if row_delta == 0 and col_delta == -1:
        return "left"
    if row_delta == 0 and col_delta == 1:
        return "right"
    return "invalid"


def get_available_directions(maze: Sequence[Sequence[int]]) -> List[str]:
    player_row, player_col = find_position(maze, constants.PLAYER_CELL)
    directions = [
        ("up", (-1, 0)),
        ("down", (1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
    ]

    options: List[str] = []
    rows = len(maze)
    cols = len(maze[0]) if maze else 0

    for label, (d_row, d_col) in directions:
        next_row = player_row + d_row
        next_col = player_col + d_col
        if 0 <= next_row < rows and 0 <= next_col < cols:
            if maze[next_row][next_col] in {constants.EMPTY_CELL, constants.GOAL_CELL}:
                options.append(label)

    return options


def _is_valid(maze_state: Sequence[Sequence[int]], row: int, col: int) -> bool:
    return (
        0 <= row < len(maze_state)
        and 0 <= col < len(maze_state[0])
        and maze_state[row][col] in {constants.EMPTY_CELL, constants.GOAL_CELL}
    )
