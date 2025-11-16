from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .position import Position, BBox
from .entity import Entity
from .grid import Grid
from .render import RenderConfig


@dataclass
class UnifiedState:
    version: str
    game_type: str
    player: Entity
    goal: Entity
    render: RenderConfig
    grid: Optional[Grid] = None
    boxes: List[Entity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "game_type": self.game_type,
            "entities": {
                "player": self.player.to_dict(),
                "goal": self.goal.to_dict(),
                "boxes": [box.to_dict() for box in self.boxes]
            },
            "render": self.render.to_dict(),
            "metadata": self.metadata
        }

        if self.grid is not None:
            result["grid"] = self.grid.to_dict()

        return result

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedState':
        entities = data["entities"]
        grid = Grid.from_dict(data["grid"]) if "grid" in data else None

        return cls(
            version=data["version"],
            game_type=data["game_type"],
            player=Entity.from_dict(entities["player"]),
            goal=Entity.from_dict(entities["goal"]),
            render=RenderConfig.from_dict(data["render"]),
            grid=grid,
            boxes=[Entity.from_dict(box) for box in entities.get("boxes", [])],
            metadata=data.get("metadata", {})
        )

    @classmethod
    def load(cls, path: str) -> 'UnifiedState':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_player_bbox(self) -> BBox:
        return self.player.bbox

    def get_goal_bbox(self) -> BBox:
        return self.goal.bbox

    def get_player_grid_pos(self) -> Optional[Position]:
        return self.player.grid_pos

    def get_goal_grid_pos(self) -> Optional[Position]:
        return self.goal.grid_pos

