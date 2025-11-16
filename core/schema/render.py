from dataclasses import dataclass
from typing import Dict


@dataclass
class RenderConfig:
    cell_size: int
    image_width: int
    image_height: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "cell_size": self.cell_size,
            "image_width": self.image_width,
            "image_height": self.image_height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'RenderConfig':
        return cls(
            cell_size=data["cell_size"],
            image_width=data["image_width"],
            image_height=data["image_height"]
        )
    
    @classmethod
    def from_grid_size(cls, height: int, width: int, cell_size: int) -> 'RenderConfig':
        return cls(
            cell_size=cell_size,
            image_width=width * cell_size,
            image_height=height * cell_size
        )

