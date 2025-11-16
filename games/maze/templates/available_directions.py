from __future__ import annotations

import random
from typing import List, Set

from .. import constants
from ..utils import maze_utils
from .base_template import BaseTemplate


class AvailableDirections(BaseTemplate):
    def __init__(self, maze: List[List[int]], image_id: int) -> None:
        super().__init__(maze, image_id)

        self.question_id = 5
        self.data_id = f"maze_{image_id:05d}_{self.question_id:02d}"
        self.qa_type = "StateInfo"
        self.question_description = "Ask for the available directions to move are currently."
        self.qa_level = "Easy"
        self.question += "Which directions are available to move now?\n\n**Options:**"

        answers = maze_utils.get_available_directions(maze)
        answer_str = ", ".join(answers)

        rng = random.Random(image_id)
        option_sets = [
            ["up", "down", "left", "right"],
            ["up, down", "up, left", "up, right", "down, left", "down, right", "left, right"],
            ["up, down, left", "up, down, right", "up, left, right", "down, left, right"],
            ["up, down, left, right"],
        ]
        counts = [2, 2, 2, 1]

        pool: Set[str] = set()
        for choices, count in zip(option_sets, counts):
            _add_random_options(pool, choices, count, rng)
        pool.add(answer_str)

        option_list = sorted(pool, key=lambda item: (len(item), item))
        self.options = []
        label_code = ord("A")
        for entry in option_list:
            label = chr(label_code)
            self.options.append(f"{label}. {entry}")
            if entry == answer_str:
                self.answer = label
            label_code += 1

        for option in self.options:
            self.question += f"\n{option}"

        player_row, player_col = maze_utils.find_position(maze, constants.PLAYER_CELL)
        segments = [f"The player is on ({player_row}, {player_col})"]
        if "up" in answer_str:
            segments.append(f"({player_row - 1}, {player_col}) is empty")
        if "down" in answer_str:
            segments.append(f"({player_row + 1}, {player_col}) is empty")
        if "left" in answer_str:
            segments.append(f"({player_row}, {player_col - 1}) is empty")
        if "right" in answer_str:
            segments.append(f"({player_row}, {player_col + 1}) is empty")

        detail = ", and ".join(segments)
        self.analysis = f"{detail}. The player can move {answer_str}. Therefore, The option is {self.answer}"


def _add_random_options(
    bucket: Set[str],
    choices: List[str],
    count: int,
    rng: random.Random,
) -> None:
    target = len(bucket) + count
    if not choices:
        return
    while len(bucket) < target:
        bucket.add(rng.choice(choices))
