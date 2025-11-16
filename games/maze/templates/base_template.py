from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .. import constants

_BASE_RULES = (
    "**Rules:**\n"
    "1. This is a maze mini-game.The player needs to navigate around obstacles to reach the destination and achieve victory.\n"
    "2. The red circle represents the player, the green block is the goal and the blue blocks are obstacles.\n"
    "3. The player can only move within the white blocks.\n"
    "4. The coordinates are given in the format (row, col), where row represents the vertical position and col represents the horizontal position.\n\n"
    "**Question:** "
)


class BaseTemplate:
    data_id: str
    qa_type: str
    question_id: int
    question_description: str
    image: str
    state: str
    plot_level: str
    qa_level: str
    question: str
    answer: str
    options: Optional[List[str]]
    analysis: str

    def __init__(self, maze: Sequence[Sequence[int]], image_id: int) -> None:
        self.image = f"{constants.IMAGES_DIR}/image_{image_id:05d}.png"
        self.state = f"{constants.STATES_DIR}/state_{image_id:05d}.json"
        self.plot_level = constants.PLOT_LEVELS.get(len(maze), "Unknown")
        self.question = _BASE_RULES
        self.answer = ""
        self.analysis = ""
        self.options: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "data_id": getattr(self, "data_id", ""),
            "qa_type": getattr(self, "qa_type", ""),
            "question_id": getattr(self, "question_id", None),
            "question_description": getattr(self, "question_description", ""),
            "image": self.image,
            "state": self.state,
            "plot_level": getattr(self, "plot_level", ""),
            "qa_level": getattr(self, "qa_level", ""),
            "question": getattr(self, "question", ""),
            "answer": getattr(self, "answer", ""),
            "options": getattr(self, "options", None),
            "analysis": getattr(self, "analysis", ""),
        }
        return payload
