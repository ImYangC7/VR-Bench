from __future__ import annotations

from typing import List, Set

from .. import constants
from ..utils import maze_utils
from .base_template import BaseTemplate


class PlayerPosition(BaseTemplate):
    def __init__(self, maze: List[List[int]], image_id: int) -> None:
        super().__init__(maze, image_id)

        self.qa_type = "StateInfo"
        self.question_id = 1
        self.data_id = f"maze_{image_id:05d}_{self.question_id:02d}"
        self.question_description = "Ask for the position of player."
        self.qa_level = "Easy"
        self.question += "Which of the following are the coordinates of the player?\n\n**Options:**"

        row, col = maze_utils.find_position(maze, constants.PLAYER_CELL)
        answer_str = f"({row}, {col})"

        choices: Set[str] = {
            answer_str,
            f"({row + 1}, {col})",
            f"({row - 1}, {col})",
            f"({row}, {col + 1})",
            f"({row}, {col - 1})",
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
            "Take a look at the game screen, the red circle represents the player.\n"
            f"The coordinates of player are {answer_str}, so the right option is {self.answer}"
        )
