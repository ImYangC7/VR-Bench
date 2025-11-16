from __future__ import annotations

import random
from typing import List, Set

from ..utils import maze_utils
from .base_template import BaseTemplate


class FindPathToGoal(BaseTemplate):
    def __init__(self, maze: List[List[int]], image_id: int) -> None:
        super().__init__(maze, image_id)

        self.question_id = 3
        self.data_id = f"maze_{image_id:05d}_{self.question_id:02d}"
        self.qa_type = "TransitionPath"
        self.question_description = "Find the path to the goal"
        self.qa_level = "Medium"
        self.question += "Which sequence of movements will allow the player to reach the destination?\n\n**Options:**"

        solver_rng = random.Random(image_id)
        info: List[str] = []
        path = maze_utils.dfs_solve_maze(maze, info, rng=solver_rng)
        actions = _path_to_actions(path)
        answer_str = ", ".join(actions)

        variant_rng = random.Random(image_id + 1)
        variants: Set[str] = {answer_str}
        for _ in range(4):
            variants.add(_random_path(len(actions), variant_rng))

        option_list = sorted(variants)
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

        self.analysis = "".join(info)
        self.analysis += (
            f"\n\nTherefore, the right sequence of movements are: {answer_str}\n"
            f"The right option is {self.answer}"
        )


def _path_to_actions(path: List[maze_utils.Coordinate]) -> List[str]:
    actions: List[str] = []
    for index in range(1, len(path)):
        actions.append(maze_utils.get_direction(path[index - 1], path[index]))
    return actions


def _random_path(length: int, rng: random.Random) -> str:
    directions = ["up", "down", "left", "right"]
    sequence = [rng.choice(directions) for _ in range(max(1, length))]
    return ", ".join(sequence)
