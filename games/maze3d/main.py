import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

@dataclass(frozen=True)  # Make the class immutable
class Position:
    x: int
    y: int
    z: int
    
    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def to_tuple(self):
        return (self.x, self.y, self.z)

@dataclass
class PathSegment:
    start: Position
    end: Position
    type: str  # 'walk' or 'ladder'

@dataclass
class Ladder:
    base_pos: Position
    direction: str
    height: int

@dataclass
class Branch:
    pos: Position
    branch_id: int
    paths: Dict[str, List[PathSegment]] # 'A' for main path, 'B' for alternative

@dataclass
class BasePuzzleState:
    grid_size: Tuple[int, int, int]
    cubes: Set[Position]
    start_pos: Position
    goal_pos: Position
    ladders: List[Ladder]
    path: List[PathSegment]

@dataclass
class PathFindingState(BasePuzzleState):
    branches: List[Branch]
    all_paths: List[PathSegment]

@dataclass
class SequencePoint:
    pos: Position
    label: int
    
@dataclass
class SequenceState(BasePuzzleState):
    sequence_points: List[SequencePoint]
    
def create_cube_verts(pos: Position, cubes: Set[Position]):
    """Create vertices for a cube at given position, showing all visible faces"""
    x, y, z = pos.x, pos.y, pos.z
    verts = []

    # 顶面（总是显示）
    if Position(x, y, z+1) not in cubes:
        verts.append([(x, y, z+1), (x+1, y, z+1), (x+1, y+1, z+1), (x, y+1, z+1)])

    # 底面（如果下方没有方块）
    if Position(x, y, z-1) not in cubes:
        verts.append([(x, y, z), (x+1, y, z), (x+1, y+1, z), (x, y+1, z)])

    # 前面（-Y方向，朝向观察者）
    if Position(x, y-1, z) not in cubes:
        verts.append([(x, y, z), (x+1, y, z), (x+1, y, z+1), (x, y, z+1)])

    # 后面（+Y方向）
    if Position(x, y+1, z) not in cubes:
        verts.append([(x, y+1, z), (x+1, y+1, z), (x+1, y+1, z+1), (x, y+1, z+1)])

    # 左面（-X方向）
    if Position(x-1, y, z) not in cubes:
        verts.append([(x, y, z), (x, y+1, z), (x, y+1, z+1), (x, y, z+1)])

    # 右面（+X方向）
    if Position(x+1, y, z) not in cubes:
        verts.append([(x+1, y, z), (x+1, y+1, z), (x+1, y+1, z+1), (x+1, y, z+1)])

    return verts

def draw_ladder(ax, ladder: Ladder, zorder=3):
    base = ladder.base_pos
    height = ladder.height

    if ladder.direction == '+x':
        x, y, z = base.x + 0.5, base.y, base.z + 0.5
        ax.plot([x, x], [y, y], [z, z + height], 'k-', linewidth=3, zorder=zorder)
        ax.plot([x, x], [y + 0.2, y + 0.2], [z, z + height], 'k-', linewidth=3, zorder=zorder)
        for h in np.linspace(z, z + height, 6):
            ax.plot([x, x], [y, y + 0.2], [h, h], 'k-', linewidth=2, zorder=zorder)

def draw_puzzle(puzzle, filename: str, player_pos: Position = None, colors: dict = None):
    """
    使用图片合成方法：先绘制场景，再绘制球体，最后合成

    Args:
        puzzle: 谜题对象
        filename: 输出文件名
        player_pos: 玩家位置
        colors: 颜色配置字典，包含 'start_pos', 'goal_pos', 'default_cube'
    """
    import tempfile
    from PIL import Image

    if player_pos is None:
        # 没有球体，直接绘制场景
        _draw_scene(puzzle, filename, colors=colors)
    else:
        # 有球体，使用合成方法
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_scene:
            scene_path = tmp_scene.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_ball:
            ball_path = tmp_ball.name

        try:
            # 1. 绘制场景（立方体+梯子）
            _draw_scene(puzzle, scene_path, colors=colors)

            # 2. 绘制球体（透明背景）
            _draw_ball_only(puzzle, player_pos, ball_path, colors=colors)

            # 3. 合成图片
            scene_img = Image.open(scene_path).convert('RGBA')
            ball_img = Image.open(ball_path).convert('RGBA')

            # 确保两张图片尺寸一致
            if scene_img.size != ball_img.size:
                # 调整球体图片尺寸以匹配场景
                ball_img = ball_img.resize(scene_img.size, Image.Resampling.LANCZOS)

            # 合成：球体图层在上
            scene_img.paste(ball_img, (0, 0), ball_img)

            # 保存最终结果
            scene_img.convert('RGB').save(filename, dpi=(100, 100))

        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(scene_path)
                os.unlink(ball_path)
            except:
                pass


def _draw_scene(puzzle, filename: str, dpi: int = 100, colors: dict = None):
    """绘制场景（立方体+梯子，无球体）

    Args:
        puzzle: 谜题对象
        filename: 输出文件名
        dpi: 图片分辨率
        colors: 颜色配置字典，包含 'start_pos', 'goal_pos', 'default_cube'
    """
    # 默认颜色配置
    if colors is None:
        colors = {
            'start_pos': '#4444FF',
            'goal_pos': '#FF4444',
            'default_cube': '#888888'
        }

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 优化视角：更好的等轴测视角
    ax.view_init(elev=20, azim=45)

    # 设置等比例缩放（重要！）
    max_range = max(puzzle.grid_size)
    ax.set_box_aspect([puzzle.grid_size[0]/max_range,
                       puzzle.grid_size[1]/max_range,
                       puzzle.grid_size[2]/max_range])

    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    margin = 0.5
    ax.set_xlim(-margin, puzzle.grid_size[0] + margin)
    ax.set_ylim(-margin, puzzle.grid_size[1] + margin)
    ax.set_zlim(-margin, puzzle.grid_size[2] + margin)

    # Draw base grid with lowest z-order
    x = np.arange(-margin, puzzle.grid_size[0] + margin, 1)
    y = np.arange(-margin, puzzle.grid_size[1] + margin, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray', zorder=1)

    # Sort cubes by distance from camera view
    camera_pos = np.array([puzzle.grid_size[0] + 2, puzzle.grid_size[1] + 2, puzzle.grid_size[2] + 2])
    sorted_cubes = sorted(
        puzzle.cubes,
        key=lambda pos: -np.linalg.norm(
            np.array([pos.x, pos.y, pos.z]) - camera_pos
        )
    )

    # Draw cubes
    for cube_pos in sorted_cubes:
        verts = create_cube_verts(cube_pos, puzzle.cubes)

        # 恢复透明度
        alpha = 0.85

        # 根据配置设置颜色
        if cube_pos == puzzle.start_pos:
            color = colors.get('start_pos', '#4444FF')
        elif cube_pos == puzzle.goal_pos:
            color = colors.get('goal_pos', '#FF4444')
        else:
            color = colors.get('default_cube', '#888888')

        pc = Poly3DCollection(verts, alpha=alpha, zorder=2)
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)
        ax.add_collection3d(pc)

    # Draw ladders
    for ladder in puzzle.ladders:
        draw_ladder(ax, ladder, zorder=300)

    # Set axis limits
    ax.set_xlim(0, puzzle.grid_size[0])
    ax.set_ylim(0, puzzle.grid_size[1])
    ax.set_zlim(0, puzzle.grid_size[2])

    # Remove extra white space
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Save
    plt.savefig(filename, bbox_inches='tight', pad_inches=-0.3, facecolor='white', dpi=dpi)
    plt.close()


def _draw_ball_only(puzzle, player_pos: Position, filename: str, dpi: int = 100, colors: dict = None):
    """只绘制球体（透明背景）- 简单的圆形标记

    Args:
        puzzle: 谜题对象
        player_pos: 玩家位置
        filename: 输出文件名
        dpi: 图片分辨率
        colors: 颜色配置字典，包含 'ball', 'ball_edge'
    """
    # 默认颜色配置
    if colors is None:
        colors = {
            'ball': '#FFD700',
            'ball_edge': '#FF8C00'
        }

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 使用相同的视角
    ax.view_init(elev=20, azim=45)

    max_range = max(puzzle.grid_size)
    ax.set_box_aspect([puzzle.grid_size[0]/max_range,
                       puzzle.grid_size[1]/max_range,
                       puzzle.grid_size[2]/max_range])

    ax.grid(False)
    ax.set_facecolor('none')  # 透明背景
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    # 球的中心位置
    ball_x = player_pos.x + 0.5
    ball_y = player_pos.y + 0.5
    ball_z = player_pos.z + 1.35

    # 根据网格大小动态调整球的大小（相对大小）
    # 基准：8x8 网格使用 s=1200
    base_grid_size = 8
    avg_grid_size = (puzzle.grid_size[0] + puzzle.grid_size[1]) / 2
    ball_size = 1200 * (base_grid_size / avg_grid_size) ** 2

    # 使用简单的 scatter 绘制圆形标记，使用配置的颜色
    ax.scatter([ball_x], [ball_y], [ball_z],
              c=colors.get('ball', '#FFD700'), s=ball_size, alpha=1.0,
              edgecolors=colors.get('ball_edge', '#FF8C00'), linewidths=2,
              depthshade=False, marker='o')

    # Set axis limits (与场景相同)
    ax.set_xlim(0, puzzle.grid_size[0])
    ax.set_ylim(0, puzzle.grid_size[1])
    ax.set_zlim(0, puzzle.grid_size[2])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 保存为透明背景
    plt.savefig(filename, bbox_inches='tight', pad_inches=-0.3,
                facecolor='none', transparent=True, dpi=dpi)
    plt.close()
    
def draw_number(ax, pos: Position, number: int, zorder=4):
    """Helper function to draw numbers with white outline for better visibility"""
    # Draw white outline/background
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        ax.text(pos.x + 0.5 + dx*0.01,
                pos.y + 0.5 + dy*0.01,
                pos.z + 1.1,
                str(number),
                size=24,
                ha='center',
                va='center',
                weight='bold',
                color='white', zorder=zorder)

    # Draw main text in black
    ax.text(pos.x + 0.5,
            pos.y + 0.5,
            pos.z + 1.1,
            str(number),
            size=24,
            ha='center',
            va='center',
            weight='bold',
            color='black', zorder=zorder)


def generate_solution_video(puzzle, output_path: str, fps: int = 24, speed: float = 3.0, colors: dict = None):
    """
    生成小球从起点匀速移动到终点的视频

    Args:
        puzzle: 谜题对象
        output_path: 输出视频路径
        fps: 帧率（默认24）
        speed: 移动速度（格子/秒，默认2.0）
        colors: 颜色配置字典，包含 'start_pos', 'goal_pos', 'default_cube'
    """
    import tempfile
    import shutil
    import imageio
    import math
    from PIL import Image

    if len(puzzle.path) < 1:
        draw_puzzle(puzzle, output_path.replace('.mp4', '.png'), player_pos=puzzle.start_pos, colors=colors)
        return

    # 计算每段的距离和累积距离
    segment_distances = []
    total_distance = 0.0
    for segment in puzzle.path:
        dist = math.sqrt(
            (segment.end.x - segment.start.x)**2 +
            (segment.end.y - segment.start.y)**2 +
            (segment.end.z - segment.start.z)**2
        )
        segment_distances.append(dist)
        total_distance += dist

    cumulative_distances = [0.0]
    for dist in segment_distances:
        cumulative_distances.append(cumulative_distances[-1] + dist)

    # 计算帧数
    duration = total_distance / speed
    total_frames = int(fps * duration)
    if total_frames < 10:
        total_frames = 10

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 只绘制一次场景（不含球体）
        scene_path = os.path.join(temp_dir, "scene.png")
        _draw_scene(puzzle, scene_path, colors=colors)
        scene_img = Image.open(scene_path).convert('RGBA')

        frames = []

        for frame_idx in range(total_frames):
            # 计算当前应该走过的距离
            target_distance = (frame_idx / (total_frames - 1)) * total_distance if total_frames > 1 else 0

            # 找到当前在哪一段
            segment_idx = 0
            for i in range(len(cumulative_distances) - 1):
                if cumulative_distances[i] <= target_distance <= cumulative_distances[i + 1]:
                    segment_idx = i
                    break

            if segment_idx >= len(puzzle.path):
                segment_idx = len(puzzle.path) - 1

            # 计算在当前段内的插值比例
            segment_start_dist = cumulative_distances[segment_idx]
            segment_end_dist = cumulative_distances[segment_idx + 1]
            segment_length = segment_end_dist - segment_start_dist
            segment_t = (target_distance - segment_start_dist) / segment_length if segment_length > 0 else 0

            # 插值计算当前位置
            current_segment = puzzle.path[segment_idx]
            interp_x = current_segment.start.x + (current_segment.end.x - current_segment.start.x) * segment_t
            interp_y = current_segment.start.y + (current_segment.end.y - current_segment.start.y) * segment_t
            interp_z = current_segment.start.z + (current_segment.end.z - current_segment.start.z) * segment_t

            # 创建浮点位置对象
            class FloatPosition:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z

            # 只绘制球体
            ball_path = os.path.join(temp_dir, f"ball_{frame_idx:05d}.png")
            _draw_ball_only(puzzle, FloatPosition(interp_x, interp_y, interp_z), ball_path, colors=colors)

            # 合成场景和球体
            ball_img = Image.open(ball_path).convert('RGBA')
            if scene_img.size != ball_img.size:
                ball_img = ball_img.resize(scene_img.size, Image.Resampling.LANCZOS)

            # 复制场景并粘贴球体
            frame_img = scene_img.copy()
            frame_img.paste(ball_img, (0, 0), ball_img)

            # 转换为RGB并添加到帧列表
            frames.append(np.array(frame_img.convert('RGB')))

        # 保存视频
        imageio.mimsave(output_path, frames, fps=fps)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_adjacent_positions(pos: Position) -> Set[Position]:
    """Get all positions adjacent to the given position"""
    adjacent = set()
    for dx, dy, dz in [(x, y, z) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1]]:
        if dx == dy == dz == 0:
            continue
        adjacent.add(Position(pos.x + dx, pos.y + dy, pos.z + dz))
    return adjacent

def is_path_valid(start_pos: Position, delta: Position, existing_cubes: Set[Position], 
                 grid_size: Tuple[int, int, int], check_adjacency: bool = False) -> bool:
    """Check if a proposed path segment would collide with existing cubes or their adjacent spaces"""
    new_pos = start_pos + delta
    
    # Check grid boundaries
    if not all(0 <= coord < size for coord, size in zip((new_pos.x, new_pos.y, new_pos.z), grid_size)):
        return False

    # 2 格移动需要检查中间位置
    if delta.z == 0:  # Walking path
        intermediate_pos = Position(
            start_pos.x + delta.x // 2,
            start_pos.y + delta.y // 2,
            start_pos.z
        )
        if intermediate_pos in existing_cubes:
            return False
        # if check_adjacency: # this part is not needed because the begin position is always adjacent to the intermediate position. Enable this will forbid all walking paths
        #     if any(adj in existing_cubes for adj in get_adjacent_positions(intermediate_pos)):
        #         return False

    # Check end position
    if new_pos in existing_cubes:
        return False
        
    if check_adjacency:
        if any(adj in existing_cubes for adj in get_adjacent_positions(new_pos)):
            return False
    
    return True

def generate_valid_path(start: Position, grid_size: Tuple[int, int, int], 
                       existing_cubes: Set[Position] = None,
                       segRange: Tuple[int, int]=(5,7),
                       check_adjacency: bool = False) -> Tuple[List[PathSegment], Set[Position]]:
    """Generate a valid path from start position"""
    if existing_cubes is None:
        existing_cubes = set()
    
    current_pos = start
    path_segments = []
    path_cubes = {start}
    all_cubes = existing_cubes | path_cubes
    
    num_segments = random.randint(*segRange)
    
    for _ in range(num_segments):
        moves = []
        # Check possible moves (2 格水平移动，3 格垂直移动)
        possible_moves = []
        if current_pos.y - 2 > 0:
            possible_moves.append(('walk', Position(0, -2, 0)))
        if current_pos.x - 2 > 0:
            possible_moves.append(('walk', Position(-2, 0, 0)))
        if current_pos.z + 3 < grid_size[2]:
            possible_moves.append(('ladder', Position(0, 0, 3)))
            
        for move_type, delta in possible_moves:
            if delta and is_path_valid(current_pos, delta, all_cubes, grid_size, check_adjacency):
                moves.append((move_type, delta))
        
        if not moves:
            break
            
        move_type, delta = random.choice(moves)
        new_pos = current_pos + delta
        
        segment = PathSegment(current_pos, new_pos, move_type)
        path_segments.append(segment)

        # 2 格移动需要添加中间立方体
        if move_type == 'walk':
            intermediate_pos = Position(
                current_pos.x + delta.x // 2,
                current_pos.y + delta.y // 2,
                current_pos.z
            )
            path_cubes.add(intermediate_pos)
            all_cubes.add(intermediate_pos)

        path_cubes.add(new_pos)
        all_cubes.add(new_pos)
        current_pos = new_pos
    
    return path_segments, path_cubes

def get_ordered_path_cubes(path: List[PathSegment]) -> List[Position]:
    """Convert path segments into an ordered list of cubes (2 格移动，包含中间位置)"""
    ordered_cubes = []

    if path:
        ordered_cubes.append(path[0].start)

    for segment in path:
        # 2 格移动需要添加中间位置
        if segment.type == 'walk':
            delta = Position(
                segment.end.x - segment.start.x,
                segment.end.y - segment.start.y,
                segment.end.z - segment.start.z
            )
            intermediate_pos = Position(
                segment.start.x + delta.x // 2,
                segment.start.y + delta.y // 2,
                segment.start.z
            )
            ordered_cubes.append(intermediate_pos)
        ordered_cubes.append(segment.end)

    return ordered_cubes

def is_positions_too_close(pos1: Position, pos2: Position) -> bool:
    """Check if two positions are too close (less than 2 blocks apart)"""
    return all(abs(getattr(pos1, attr) - getattr(pos2, attr)) < 2 
              for attr in ['x', 'y', 'z'])

def get_segment_direction(segment: PathSegment) -> str:
    """Determine the direction of a path segment"""
    if segment.type == 'ladder':
        return 'up'
    
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    
    if delta_x == 0:
        return 'left-forward'
    elif delta_y == 0:
        return 'right-forward'
    return '??'

def get_plot_level(cubes: Set[Position]) -> str:
    """Determine plot level based on number of blocks"""
    num_blocks = len(cubes)
    if num_blocks < 15:
        return "Easy"
    elif num_blocks < 25:
        return "Medium"
    return "Hard"

def get_path_direction(segment: PathSegment) -> str:
    """Determine the direction of a path segment"""
    if segment.type == 'ladder':
        return 'up'
    
    # For walking paths, determine direction based on coordinate changes
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    
    if delta_x == 0:
        return 'left-forward'
    elif delta_y == 0:
        return 'right-forward'  # or could be called 'straight'
    else:
        return '??'
    
def get_branch_order(main_path: List[PathSegment], branches: List[Branch]) -> List[Branch]:
    """Get branches in the order they appear along the main path"""
    ordered_positions = get_ordered_path_cubes(main_path)
    ordered_branches = []
    
    for pos in ordered_positions:
        for branch in branches:
            if pos == branch.pos and branch not in ordered_branches:
                ordered_branches.append(branch)
    
    return ordered_branches

def find_path_between_points(start: Position, end: Position, cubes: Set[Position], path: List[PathSegment]) -> List[PathSegment]:
    """Find the path segments between two points on the main path"""
    ordered_cubes = get_ordered_path_cubes(path)
    start_idx = ordered_cubes.index(start) if start in ordered_cubes else -1
    end_idx = ordered_cubes.index(end) if end in ordered_cubes else -1
    reverse=start_idx>end_idx
    
    if start_idx == -1 or end_idx == -1:
        return []
    
    # Get the relevant segment of the path
    relevant_cubes = ordered_cubes[min(start_idx, end_idx):max(start_idx, end_idx) + 1]
    relevant_segments = []
    
    for i in range(len(path)):
        segment = path[i]
        segment_cubes = [segment.start]
        if segment.type == 'walk':
            delta = Position(
                segment.end.x - segment.start.x,
                segment.end.y - segment.start.y,
                segment.end.z - segment.start.z
            )
            intermediate_pos = Position(
                segment.start.x + delta.x // 2,
                segment.start.y + delta.y // 2,
                segment.start.z
            )
            segment_cubes.append(intermediate_pos)
        segment_cubes.append(segment.end)
        
        if sum(cube in relevant_cubes for cube in segment_cubes)>1:
            relevant_segments.append(segment)
    
    return relevant_segments,reverse

def normalize_height_relation(heights):
    """Convert raw heights into a valid relation string that matches possible_relations format"""
    # Sort points by height, then by label for equal heights
    sorted_points = sorted(heights, key=lambda x: (x[1], x[0]))
    
    # Build relation string ensuring valid format
    relations = []
    current_height = None
    current_group = []
    
    # Group points by height
    for label, height in sorted_points:
        if height != current_height:
            if current_group:
                relations.append(current_group)
            current_group = [label]
            current_height = height
        else:
            current_group.append(label)
    relations.append(current_group)
    
    # Convert groups into relation string
    result = []
    for i, group in enumerate(relations):
        # Sort labels within group
        group.sort()
        result.append(" = ".join(str(x) for x in group))
    
    return " < ".join(result)


class PuzzleGenerator:
    def __init__(self, grid_size=None):
        grid_size = grid_size or (8,8,7)
        self.grid_size = grid_size
        self.start_pos = Position(grid_size[0]-1, grid_size[1]-1, 0)
    
    def _extract_ladders(self, path: List[PathSegment]) -> List[Ladder]:
        return [Ladder(segment.start, '+x', segment.end.z - segment.start.z)
                for segment in path if segment.type == 'ladder']
    
    def generate_path_finding_puzzle(self, main_path_length: Tuple[int, int] = (5, 7), side_path_num: Tuple[int,int] = (3, 4), side_path_length: Tuple[int, int] = (1, 2)) -> PathFindingState:
        """Generate a puzzle with branching paths"""
        main_path, main_cubes = generate_valid_path(self.start_pos, self.grid_size, segRange=main_path_length)
        goal_pos = main_path[-1].end
        
        all_cubes = main_cubes.copy()
        branches = []
        
        # Select branch positions
        ordered_cubes = get_ordered_path_cubes(main_path)
        valid_branch_positions = ordered_cubes[0:-3]
        random.shuffle(valid_branch_positions)
        
        selected_positions = []
        for pos in valid_branch_positions:
            if not any(is_positions_too_close(pos, selected_pos) 
                      for selected_pos in selected_positions) and pos.x%2==1 and pos.y%2==1:
                selected_positions.append(pos)
        
        side_path_num = min(len(selected_positions), random.randint(*side_path_num))
        selected_positions, alters = selected_positions[:side_path_num], selected_positions[side_path_num:]
        
        # Generate branches
        for i, branch_pos in enumerate(selected_positions, 1):
            for _ in range(10):
                alt_path, alt_cubes = generate_valid_path(
                    branch_pos, self.grid_size,
                    existing_cubes=all_cubes,
                    segRange=side_path_length,
                    check_adjacency=True
                )
                if alt_path:
                    break
                if alters:
                    branch_pos = random.choice(alters)
            if not alt_path:
                raise Exception("Cannot generate valid path")
            
            branches.append(Branch(branch_pos, i, {'A': main_path, 'B': alt_path}))
            all_cubes.update(alt_cubes)
        
        return PathFindingState(
            self.grid_size, all_cubes, self.start_pos, goal_pos,
            self._extract_ladders(main_path + sum((b.paths['B'] for b in branches), [])),
            main_path, branches, main_path
        )
    
    def choose_labeled_cubes(self, valid_positions: List[Position], num_labels: int) -> List[SequencePoint]:
        """Choose labeled cubes from the valid positions, ensuring they are not too close"""
        num_labels = min(num_labels, len(valid_positions))
        sequence_points = []
        label_positions = []
        
        while len(sequence_points) < num_labels and valid_positions:
            pos = random.choice(valid_positions)
            if not any(is_positions_too_close(pos, prev_pos) for prev_pos in label_positions):
                label_positions.append(pos)
                sequence_points.append(SequencePoint(pos, len(sequence_points) + 1))
            valid_positions.remove(pos)
        return sequence_points
    
    def generate_sequence_puzzle(self, label_num_range: Tuple[int, int] = (3, 4)) -> SequenceState:
        """Generate a puzzle with labeled sequence points"""
        main_path, main_cubes = generate_valid_path(self.start_pos, self.grid_size)
        goal_pos = main_path[-1].end
        
        
        ordered_cubes = get_ordered_path_cubes(main_path)
        valid_positions = ordered_cubes[1:-1]
        
        num_labels = random.randint(*label_num_range)
        sequence_points = self.choose_labeled_cubes(valid_positions, num_labels)
        
        return SequenceState(
            self.grid_size, main_cubes, self.start_pos, goal_pos,
            self._extract_ladders(main_path), main_path, sequence_points
        )

class QAGenerator:
    def __init__(self):
        self.puzzle_generator = PuzzleGenerator()

    def generate_qa_pair(self, index: int, qa_type: str, grid_size=None):
        """Generate a Q&A pair based on the specified type

        Args:
            index: 随机种子索引
            qa_type: 问题类型
            grid_size: 网格大小 (width, depth, height)，如果为 None 则使用默认值
        """
        # 如果指定了 grid_size，创建新的生成器
        if grid_size:
            self.puzzle_generator = PuzzleGenerator(grid_size=grid_size)
        else:
            self.puzzle_generator = PuzzleGenerator()

        if qa_type == 'path_finding':
            return self._generate_path_finding_qa(index)
        elif qa_type == 'sequence_finding':
            return self._generate_sequence_qa(index)
        elif qa_type == 'height_comparison':
            return self._generate_height_comparison_qa(index)
        elif qa_type == 'main_path':
            return self._generate_main_path_qa(index)
        raise ValueError(f"Unknown qa_type: {qa_type}")
    
    def _generate_path_finding_qa(self, index: int):
        """Generate a path-finding question and answer pair"""
        puzzle = self.puzzle_generator.generate_path_finding_puzzle()
        
        # Get branches in order of appearance
        ordered_branches = get_branch_order(puzzle.all_paths, puzzle.branches)
        
        def get_main_path_direction_at_branch(branch_pos: Position, path: List[PathSegment]) -> str:
            """Get the direction of the main path at the branch position"""
            current_segment = None
            for segment in path:
                if segment.start == branch_pos:
                    current_segment = segment
                    break
                    
                # For walking paths, check if branch is at intermediate position
                if segment.type == 'walk':
                    delta = Position(
                        segment.end.x - segment.start.x,
                        segment.end.y - segment.start.y,
                        segment.end.z - segment.start.z
                    )
                    intermediate_pos = Position(
                        segment.start.x + delta.x // 2,
                        segment.start.y + delta.y // 2,
                        segment.start.z
                    )
                    if intermediate_pos == branch_pos:
                        # Find the next segment that continues from this position
                        for next_segment in path:
                            if next_segment.start == segment.end:
                                current_segment = next_segment
                                break
                        break
            
            if current_segment:
                return get_path_direction(current_segment)
            return "??"

        # Create the correct path sequence with directional descriptions
        correct_path = ""
        for b in puzzle.branches:
            main_direction = get_main_path_direction_at_branch(b.pos, b.paths['A'])
            alt_segment = b.paths['B'][0]  # Alternative path always starts at branch position
            alt_direction = get_path_direction(alt_segment)
            correct_path += f"{b.branch_id}-{main_direction}, "
        correct_path = correct_path[:-2]  # Remove trailing comma and space
        
        # Create the correct path sequence with directional descriptions for ordered branches
        correct_path_ordered = ""
        for b in ordered_branches:
            main_direction = get_main_path_direction_at_branch(b.pos, b.paths['A'])
            alt_segment = b.paths['B'][0]  # Alternative path always starts at branch position
            alt_direction = get_path_direction(alt_segment)
            correct_path_ordered += f"{b.branch_id}-{main_direction}, "
        correct_path_ordered = correct_path_ordered[:-2]  # Remove trailing comma and space
        
        # Generate all possible combinations
        options = []
        for i in range(8):
            path = ""
            for branch in puzzle.branches:
                main_direction = get_main_path_direction_at_branch(branch.pos, branch.paths['A'])
                alt_segment = branch.paths['B'][0]
                alt_direction = get_path_direction(alt_segment)
                chosen_direction = main_direction if (i & (1 << (branch.branch_id - 1))) else alt_direction
                path += f"{branch.branch_id}-{chosen_direction}, "
            options.append(path[:-2])
        
        # Ensure correct path is in options
        if correct_path not in options:
            options[0] = correct_path
            
        random.shuffle(options)
        
        correct_answer = options.index(correct_path) + 1
        
        question = """Rules:
1. Player can only walk on top of cubes
2. Player can climb ladders if they can reach the cube under the ladder
3. From a ladder, player can reach the top of the last cube with the ladder
4. Blue cube is start position, red cube is goal position
5. Numbered cubes are branch points where player must choose a path

Which combination of path choices leads to the goal?"""
        
        # Create enhanced analysis with path progression
        branch_order_desc = "From the start point, "
        if ordered_branches:
            branch_order_desc += f"you first meet branch {ordered_branches[0].branch_id}"
            for b in ordered_branches[1:]:
                branch_order_desc += f", then branch {b.branch_id}"
            branch_order_desc += ", before finally reaching the goal."
        else:
            branch_order_desc += "you proceed directly to the goal."
        
        analysis = f"{branch_order_desc}\n\nAnalyzing each branch point:\n"
        for b in ordered_branches:
            main_direction = get_main_path_direction_at_branch(b.pos, b.paths['A'])
            alt_direction = get_path_direction(b.paths['B'][0])
            next_branch = None
            for i, next_b in enumerate(ordered_branches):
                if next_b.branch_id == b.branch_id:
                    if i + 1 < len(ordered_branches):
                        next_branch = ordered_branches[i + 1]
                    break
            
            if next_branch:
                analysis += f"- At branch {b.branch_id}, going {main_direction} leads to branch {next_branch.branch_id}, "
            else:
                analysis += f"- At branch {b.branch_id}, going {main_direction} leads toward the goal, "
            analysis += f"while going {alt_direction} leads to a dead end\n"
        
        analysis += f"\nTherefore, the correct sequence is {correct_path_ordered}, that is {correct_path}, making the answer Option {correct_answer}."
        
        data = {
            "qa_type": "State Prediction",
            "question_description":"path_finding",
            "question_id": 0,
            "data_id": f"path-mcq-{index:05d}-path_finding",
            "image": f"images/path-mcq-{index:05d}.png",
            "state": f"states/path-mcq-{index:05d}.json",
            "plot_level": get_plot_level(puzzle.cubes),
            "qa_level": "Hard",  # path_finding questions are more complex
            "question": f"{question}\n\nOptions:\n" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options)),
            "answer": correct_answer,
            "options": options,
            "analysis": analysis
        }
        
        return data, puzzle

    def _generate_sequence_qa(self, index: int):
        """Generate a sequence question Q&A pair"""
        puzzle = self.puzzle_generator.generate_sequence_puzzle()
        
        # Get the correct sequence order
        ordered_cubes = get_ordered_path_cubes(puzzle.path)
        sequence = ["Start"]
        
        # Create ordered sequence by checking each cube in path order
        for cube in ordered_cubes:
            for sp in sorted(puzzle.sequence_points, key=lambda x: x.label):
                if sp.pos == cube:
                    sequence.append(str(sp.label))
        sequence.append("Goal")
        
        correct_sequence = " -> ".join(sequence)
        
        # Generate wrong options by shuffling middle numbers
        middle_numbers = sequence[1:-1]
        options = [correct_sequence]
        
        while len(options) < 6:  # Generate 7 wrong options
            random.shuffle(middle_numbers)
            wrong_sequence = " -> ".join(["Start"] + middle_numbers + ["Goal"])
            if wrong_sequence not in options:
                options.append(wrong_sequence)
        
        random.shuffle(options)
        correct_answer = options.index(correct_sequence) + 1
        
        question = """Rules:
1. Player can only walk on top of cubes
2. Player can climb ladders if they can reach the cube under the ladder
3. From a ladder, player can reach the top of the last cube with the ladder
4. Blue cube is start position, red cube is goal position
5. Green cubes are numbered checkpoints

What is the correct sequence of numbered checkpoints when following the path from start to goal?"""
        
        # Create analysis explaining the path
        analysis = "Following the path from start to goal:\n"
        current_pos = puzzle.start_pos
        step = 1
        
        for segment in puzzle.path:
            direction = get_segment_direction(segment)
            
            # Check if there's a sequence point at the current position
            for sp in puzzle.sequence_points:
                if sp.pos == current_pos:
                    analysis += f"\nStep {step}: At checkpoint {sp.label}"
                    step += 1
            
            # For walking paths, check intermediate position for checkpoints
            if segment.type == 'walk':
                delta = Position(
                    segment.end.x - segment.start.x,
                    segment.end.y - segment.start.y,
                    segment.end.z - segment.start.z
                )
                intermediate_pos = Position(
                    segment.start.x + delta.x // 2,
                    segment.start.y + delta.y // 2,
                    segment.start.z
                )
                # Check for checkpoint at intermediate position
                for sp in puzzle.sequence_points:
                    if sp.pos == intermediate_pos:
                        analysis += f"\nStep {step}: At checkpoint {sp.label}"
                        step += 1
        
            analysis += f"\nStep {step}: Move {direction}"
            current_pos = segment.end
            step += 1
        
        # Add final checkpoint if it exists
        for sp in puzzle.sequence_points:
            if sp.pos == current_pos:
                analysis += f"\nStep {step}: At checkpoint {sp.label}"
        
        analysis += f"\n\nTherefore, the correct sequence is {correct_sequence}, making the answer Option {correct_answer}."
        
        data = {
            "qa_type": "State Prediction",
            "question_description":"sequence_finding",
            "question_id": 1 ,
            "data_id": f"path-mcq-{index:05d}",
            "image": f"images/path-mcq-{index:05d}.png",
            "state": f"states/path-mcq-{index:05d}.json",
            "plot_level": get_plot_level(puzzle.cubes),
            "qa_level": "Medium",  # sequence_finding questions are medium complexity
            "question": f"{question}\n\nOptions:\n" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options)),
            "answer": correct_answer,
            "options": options,
            "analysis": analysis
        }
        
        return data, puzzle

    def _generate_height_comparison_qa(self, index: int):
        """Generate a height comparison question Q&A pair with corrected relation handling"""
        puzzle = self.puzzle_generator.generate_sequence_puzzle((3, 3))
        
        # Get heights of comparison points
        heights = [(p.label, p.pos.z) for p in puzzle.sequence_points]
        
        # Generate the correct height relation string using normalized format
        correct_relation = normalize_height_relation(heights)
        
        # List of all possible height relations
        possible_relations = [
            "1 < 2 < 3", "1 < 3 < 2", "2 < 1 < 3", "2 < 3 < 1", "3 < 1 < 2", "3 < 2 < 1",
            "1 < 2 = 3", "2 < 1 = 3", "3 < 1 = 2", "1 = 2 < 3", "1 = 3 < 2", "2 = 3 < 1",
            "1 = 2 = 3"
        ]
        
        # Generate options
        options = [correct_relation]
        while len(options) < 8:
            wrong_relation = random.choice(possible_relations)
            if wrong_relation not in options:
                options.append(wrong_relation)
        
        random.shuffle(options)
        correct_answer = options.index(correct_relation) + 1
        
        question = """Rules:
1. Player can only walk on top of cubes
2. Player can climb ladders if they can reach the cube under the ladder
3. From a ladder, player can reach the top of the last cube with the ladder
4. Blue cube is start position, red cube is goal position
5. Green cubes are numbered points (1, 2, and 3)

What is the correct height relationship between the three numbered points? Use '<' for 'lower than' and '=' for 'same height as'."""
        
        # Create analysis explaining the height relationships
        analysis = "Analyzing the heights of each point:\n"
        
        # Compare each pair of points
        for i, point1 in enumerate(puzzle.sequence_points):
            for j, point2 in enumerate(puzzle.sequence_points[i+1:], i+1):
                path_segments, reverse = find_path_between_points(point1.pos, point2.pos, puzzle.cubes, puzzle.path)
                current_pos = point1.pos
                
                height_diff = point2.pos.z - point1.pos.z
                relation = "same height as" if height_diff == 0 else ("higher than" if height_diff > 0 else "lower than")
                
                analysis += f"\nComparing points {point1.label} and {point2.label}:\n"
                analysis += f" Found a path from {(point2 if reverse else point1).label} to {(point1 if reverse else point2).label}:\n"
                for segment in path_segments:
                    direction = get_segment_direction(segment)
                    if segment.type == 'ladder':
                        analysis += f"  * Go up {segment.end.z - segment.start.z} blocks\n"
                    else:
                        analysis += f"  * Move {direction}\n"
                    current_pos = segment.end
                    
                analysis += f"- Point {point2.label} is {relation} point {point1.label}\n"
        
        analysis += f"\nTherefore, the correct height relationship is {correct_relation}, making the answer Option {correct_answer}."
        
        data = {
            "qa_type": "Target Perception",
            "question_description":"height_comparison",
            "question_id": 2,
            "data_id": f"path-mcq-{index:05d}",
            "image": f"images/path-mcq-{index:05d}.png",
            "state": f"states/path-mcq-{index:05d}.json",
            "plot_level": get_plot_level(puzzle.cubes),
            "qa_level": "Easy",  # height comparison questions are simpler
            "question": f"{question}\n\nOptions:\n" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options)),
            "answer": correct_answer,
            "options": options,
            "analysis": analysis
        }
        
        return data, puzzle

    def _generate_main_path_qa(self, index: int):
        """Generate a question about which numbered blocks are on the main path"""
        puzzle = self.puzzle_generator.generate_path_finding_puzzle()
        
        # Get main path cubes in order
        cubes = puzzle.cubes
        
        # Remove start and goal positions from cubes when choosing labeled cubes
        cubes_remove = cubes.copy()
        cubes_remove.remove(puzzle.start_pos)
        cubes_remove.remove(puzzle.goal_pos)
        
        main_path_cubes = get_ordered_path_cubes(puzzle.path)
        
        label_cubes = self.puzzle_generator.choose_labeled_cubes(list(cubes_remove), random.randint(3, 4))
        
        puzzle.branches = [Branch(cube.pos, cube.label, {'A': [], 'B': []}) for cube in label_cubes] # so that draw_puzzle can draw the labels
        
        # Create list of branch numbers that are on main path vs side paths
        main_path_branches = []
        side_path_branches = []
        
        for cube in label_cubes:
            if cube.pos in main_path_cubes:
                main_path_branches.append(str(cube.label))
            else:
                side_path_branches.append(str(cube.label))
        
        # Generate the correct answer
        correct_answer = ", ".join(sorted(main_path_branches, key=int))
        if correct_answer == "":
            correct_answer = "None"
        
        # Generate wrong options by mixing main path and side path branches
        all_branches = [str(b.label) for b in label_cubes]
        options = [correct_answer]
        
        # Generate wrong options by taking different combinations
        while len(options) < 8:
            num_choices = random.randint(0, len(all_branches))
            if num_choices == 0:
                wrong_answer = "None"
            else:
                wrong_branches = random.sample(all_branches, num_choices)
                wrong_answer = ", ".join(sorted(wrong_branches, key=int))
            if wrong_answer not in options:
                options.append(wrong_answer)
        
        random.shuffle(options)
        correct_option = options.index(correct_answer) + 1
        
        question = """Rules:
    1. Player can only walk on top of cubes
    2. Player can climb ladders if they can reach the cube under the ladder
    3. From a ladder, player can reach the top of the last cube with the ladder
    4. Blue cube is start position, red cube is goal position
    5. Numbered cubes are branch points

    Which numbered blocks are passed through when following the most direct path from start to goal?"""

        # Create detailed path analysis
        analysis = "Following the main path from start to goal:\n"
        current_pos = puzzle.start_pos
        step = 1
        
        for segment in puzzle.path:
            direction = get_segment_direction(segment)
            
            # Check if current position is a branch point
            branch_num = None
            for branch in puzzle.branches:
                if branch.pos == current_pos:
                    branch_num = branch.branch_id
                    break
            
            if branch_num is not None:
                analysis += f"\nStep {step}: At Block {branch_num}"
                step += 1
            
            # For walking paths, check intermediate position for branch points
            if segment.type == 'walk':
                delta = Position(
                    segment.end.x - segment.start.x,
                    segment.end.y - segment.start.y,
                    segment.end.z - segment.start.z
                )
                intermediate_pos = Position(
                    segment.start.x + delta.x // 2,
                    segment.start.y + delta.y // 2,
                    segment.start.z
                )
                # Check for branch at intermediate position
                for branch in puzzle.branches:
                    if branch.pos == intermediate_pos:
                        analysis += f"\nStep {step}: At Block {branch.branch_id}"
                        step += 1
            
            analysis += f"\nStep {step}: Move {direction}"
            current_pos = segment.end
            step += 1
        
        # Add information about side paths
        if side_path_branches:
            analysis += f"\n\nBlocks not on main path: {', '.join(sorted(side_path_branches, key=int))}"
        
        analysis += f"\n\nTherefore, the blocks passed through on the main path are: {correct_answer}, making the answer Option {correct_option}."
        
        data = {
            "qa_type": "State Prediction",
            "question_description":"main_path",
            "question_id": 3,
            "data_id": f"path-mcq-{index:05d}",
            "image": f"images/path-mcq-{index:05d}.png",
            "state": f"states/path-mcq-{index:05d}.json",
            "plot_level": get_plot_level(puzzle.cubes),
            "qa_level": "Medium",
            "question": f"{question}\n\nOptions:\n" + "\n".join(f"{i+1}: {opt}" for i, opt in enumerate(options)),
            "answer": correct_option,
            "options": options,
            "analysis": analysis
        }
        
        return data, puzzle
        
def generate_mixed_dataset(num_problems: int):
    """Generate a dataset with mixed problem types"""
    ensure_output_dirs()
    dataset = []
    qa_generator = QAGenerator()
    
    qa_types = ['path_finding', 'sequence_finding', 'height_comparison', 'main_path']
    i = 0
    count=0
    while i < num_problems:
        try:
            # print(f"Generating problem {i+1}")
            qa_type = qa_types[i % len(qa_types)]
            data, puzzle = qa_generator.generate_qa_pair(i + 1, qa_type)
            draw_puzzle(puzzle, f"{outputPath}/images/path-mcq-{i+1:05d}.png")
            dataset.append(data)
            i += 1
            count=0
        except Exception as e:
            if str(e) != "Cannot generate valid path":
                raise e
            count+=1
            if count>10:
                print("Too many failures, aborting")
                raise e
    
    with open(f"{outputPath}/data.json", 'w') as f:
        json.dump(dataset, f, indent=2)

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs(f"{outputPath}/images", exist_ok=True)
    os.makedirs(f"{outputPath}/states", exist_ok=True)

if __name__ == "__main__":
    outputPath = "3d_maze_dataset"
    generate_mixed_dataset(15)