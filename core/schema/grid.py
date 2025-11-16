from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Grid:
    data: List[List[int]]
    height: int
    width: int
    
    def to_dict(self) -> Dict:
        return {
            "data": self.data,
            "height": self.height,
            "width": self.width
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Grid':
        return cls(
            data=data["data"],
            height=data["height"],
            width=data["width"]
        )
    
    @classmethod
    def from_2d_list(cls, grid: List[List[int]]) -> 'Grid':
        return cls(
            data=grid,
            height=len(grid),
            width=len(grid[0]) if grid else 0
        )

