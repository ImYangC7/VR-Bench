from __future__ import annotations

import random
from typing import List, Set

from .. import constants
from ..utils import maze_utils
from .base_template import BaseTemplate


class PositionAfterMoving(BaseTemplate):
    def __init__(self, maze: List[List[int]], image_id: int) -> None:
        super().__init__(maze, image_id)

        self.qa_type = "ActionOutcome"
        self.question_id = 6
        self.data_id = f"maze_{image_id:05d}_{self.question_id:02d}"
        self.question_description = "The position after moving."
        self.qa_level = "Medium"

        rng = random.Random(image_id)

        directions = maze_utils.get_available_directions(maze)
        if not directions:
            raise ValueError("Player has no available moves to build question")
        direction = rng.choice(directions)

        self.question += f"What are the coordinates of player after moving {direction}?\n\n**Options:**"

        row, col = maze_utils.find_position(maze, constants.PLAYER_CELL)
        if direction == "up":
            answer_str = f"({row - 1}, {col})"
        elif direction == "down":
            answer_str = f"({row + 1}, {col})"
        elif direction == "left":
            answer_str = f"({row}, {col - 1})"
        else:
            answer_str = f"({row}, {col + 1})"

        choices: Set[str] = {
            answer_str,
            f"({row + 1}, {col})",
            f"({row - 1}, {col})",
            f"({row}, {col + 1})",
            f"({row}, {col - 1})",
            f"({row}, {col})",
        }

        option_list = sorted(choices)
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

        self.analysis = (
            f"Observe the screen, the position of player is ({row}, {col}). "
            f"After moving {direction}, the player is in {answer_str}. "
            f"Therefore, the right option is {self.answer}"
        )
