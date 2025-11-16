from __future__ import annotations

import random
from typing import List

from ..utils import maze_utils
from .base_template import BaseTemplate


class TurnCount(BaseTemplate):
    def __init__(self, maze: List[List[int]], image_id: int) -> None:
        super().__init__(maze, image_id)

        self.question_id = 4
        self.data_id = f"maze_{image_id:05d}_{self.question_id:02d}"
        self.qa_type = "TransitionPath"
        self.question_description = "Count how many turns it takes to reach the finish."
        self.qa_level = "Hard"
        self.question += (
            "Find the path to the finish and count the number of turns it takes to get there. "
            "You only need to provide one number."
        )

        solver_rng = random.Random(image_id)
        path_info: List[str] = []
        path = maze_utils.dfs_solve_maze(maze, path_info, rng=solver_rng)
        turn_info: List[str] = []
        turns = maze_utils.count_turns(path, turn_info)

        self.answer = str(turns)
        self.options = None

        self.analysis = "First," + "".join(path_info)
        self.analysis += f"Therefore, the path is: {maze_utils.path_to_string(path)}\n\nThen,"
        self.analysis += "".join(turn_info)
        self.analysis += f"\nIn summary, the total number of turns is {turns}"
