from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class Position:
    row: int
    col: int
    
    def to_dict(self) -> Dict[str, int]:
        return {"row": self.row, "col": self.col}
    
    def to_list(self) -> List[int]:
        return [self.row, self.col]
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'Position':
        if "row" in data:
            return cls(row=data["row"], col=data["col"])
        elif "y" in data:
            return cls(row=data["y"], col=data["x"])
        raise ValueError(f"Unknown position format: {data}")
    
    @classmethod
    def from_list(cls, data: List[int]) -> 'Position':
        return cls(row=data[0], col=data[1])


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y
        }
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BBox':
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"]
        )
    
    @classmethod
    def from_grid_pos(cls, row: int, col: int, cell_size: int) -> 'BBox':
        return cls(
            x=col * cell_size,
            y=row * cell_size,
            width=cell_size,
            height=cell_size
        )

