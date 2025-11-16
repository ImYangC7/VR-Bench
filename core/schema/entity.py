from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from .position import Position, BBox


@dataclass
class Entity:
    pixel_pos: Tuple[int, int]
    bbox: BBox
    grid_pos: Optional[Position] = None

    def to_dict(self) -> Dict:
        result = {
            "pixel_pos": {"x": self.pixel_pos[0], "y": self.pixel_pos[1]},
            "bbox": self.bbox.to_dict()
        }

        if self.grid_pos is not None:
            result["grid_pos"] = self.grid_pos.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        pixel_data = data["pixel_pos"]
        grid_pos = Position.from_dict(data["grid_pos"]) if "grid_pos" in data else None

        return cls(
            pixel_pos=(pixel_data["x"], pixel_data["y"]),
            bbox=BBox.from_dict(data["bbox"]),
            grid_pos=grid_pos
        )

    @classmethod
    def from_grid_pos(cls, row: int, col: int, cell_size: int) -> 'Entity':
        pixel_x = col * cell_size + cell_size // 2
        pixel_y = row * cell_size + cell_size // 2

        return cls(
            pixel_pos=(pixel_x, pixel_y),
            bbox=BBox.from_grid_pos(row, col, cell_size),
            grid_pos=Position(row=row, col=col)
        )

    @classmethod
    def from_pixel_pos(cls, x: int, y: int, bbox_size: int) -> 'Entity':
        """从像素坐标创建 Entity（用于非网格游戏）"""
        return cls(
            pixel_pos=(x, y),
            bbox=BBox(
                x=x - bbox_size // 2,
                y=y - bbox_size // 2,
                width=bbox_size,
                height=bbox_size
            ),
            grid_pos=None
        )

