"""
PathFinder 游戏板 - 基于曲线路径
"""

from typing import List, Tuple


class PathSegment:
    """路径段"""
    
    def __init__(self, control_points: List[Tuple[float, float]]):
        self.control_points = control_points
    
    def get_start(self) -> Tuple[float, float]:
        return self.control_points[0]
    
    def get_end(self) -> Tuple[float, float]:
        return self.control_points[-1]


class PathFinderBoard:
    """PathFinder 游戏板"""

    def __init__(
        self,
        segments: List[PathSegment],
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        solution_segments: List[int],
        solution_path: List[Tuple[float, float]] = None,  # 新增：解决方案的节点路径
        image_size: int = 800,
        road_width: int = 35  # 新增：道路宽度
    ):
        self.segments = segments
        self.start_point = start_point
        self.end_point = end_point
        self.solution_segments = solution_segments
        self.solution_path = solution_path or []  # 节点序列
        self.image_size = image_size
        self.road_width = road_width  # 保存道路宽度
    
    def is_solvable(self) -> bool:
        return len(self.solution_segments) > 0
    
    def to_dict(self) -> dict:
        return {
            'segments': [[pt for pt in seg.control_points] for seg in self.segments],
            'start_point': list(self.start_point),
            'end_point': list(self.end_point),
            'solution_segments': self.solution_segments,
            'solution_path': [list(pt) for pt in self.solution_path],
            'image_size': self.image_size,
            'road_width': self.road_width
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PathFinderBoard':
        segments = [PathSegment(points) for points in data['segments']]
        return cls(
            segments=segments,
            start_point=tuple(data['start_point']),
            end_point=tuple(data['end_point']),
            solution_segments=data['solution_segments'],
            solution_path=[tuple(pt) for pt in data.get('solution_path', [])],
            image_size=data.get('image_size', 800),
            road_width=data.get('road_width', 35)
        )
