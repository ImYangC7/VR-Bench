"""
TrapField 游戏生成器
生成没有外围墙的陷阱场地图
"""

from __future__ import annotations

import random
from typing import List, Tuple, Set
from collections import deque

from . import constants

Coordinate = Tuple[int, int]

_RANDOM = random.Random()


def generate_trapfield(size: int, trap_density: float = 0.3) -> List[List[int]]:
    """
    生成陷阱场地图
    
    Args:
        size: 网格大小（size x size）
        trap_density: 陷阱密度（0.0 - 1.0）
    
    Returns:
        2D 网格，包含空地、陷阱、玩家和目标
    """
    if size < 5:
        raise ValueError("Grid size must be at least 5x5")
    
    # 初始化为全空地
    grid = [[constants.EMPTY_CELL for _ in range(size)] for _ in range(size)]
    
    # 随机放置玩家和目标（确保距离足够远）
    player_pos, goal_pos = _place_player_and_goal(grid, size)
    
    # 使用 BFS 找到从玩家到目标的最短路径
    solution_path = _find_path_bfs(grid, player_pos, goal_pos)
    
    if not solution_path:
        raise ValueError("Failed to find path from player to goal")
    
    # 将路径转换为集合以便快速查找
    path_set = set(solution_path)
    
    # 随机放置陷阱（不能在路径上）
    _place_traps(grid, size, trap_density, path_set, player_pos, goal_pos)
    
    # 设置玩家和目标
    grid[player_pos[0]][player_pos[1]] = constants.PLAYER_CELL
    grid[goal_pos[0]][goal_pos[1]] = constants.GOAL_CELL
    
    return grid


def _place_player_and_goal(grid: List[List[int]], size: int) -> Tuple[Coordinate, Coordinate]:
    """随机放置玩家和目标，确保距离足够远"""
    # 玩家随机位置
    player_row = _RANDOM.randint(0, size - 1)
    player_col = _RANDOM.randint(0, size - 1)
    player_pos = (player_row, player_col)
    
    # 目标位置：尽量远离玩家
    max_attempts = 100
    best_goal = None
    best_distance = 0
    
    for _ in range(max_attempts):
        goal_row = _RANDOM.randint(0, size - 1)
        goal_col = _RANDOM.randint(0, size - 1)
        
        if (goal_row, goal_col) == player_pos:
            continue
        
        # 曼哈顿距离
        distance = abs(goal_row - player_row) + abs(goal_col - player_col)
        
        if distance > best_distance:
            best_distance = distance
            best_goal = (goal_row, goal_col)
    
    if best_goal is None:
        # 如果没找到，就放在对角
        best_goal = (size - 1 - player_row, size - 1 - player_col)
    
    return player_pos, best_goal


def _find_path_bfs(grid: List[List[int]], start: Coordinate, goal: Coordinate) -> List[Coordinate]:
    """使用 BFS 找到从起点到终点的最短路径"""
    size = len(grid)
    queue = deque([(start, [start])])
    visited = {start}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    
    while queue:
        (row, col), path = queue.popleft()
        
        if (row, col) == goal:
            return path
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if 0 <= new_row < size and 0 <= new_col < size:
                if (new_row, new_col) not in visited:
                    # 只考虑空地（暂时忽略陷阱，因为还没放置）
                    if grid[new_row][new_col] == constants.EMPTY_CELL:
                        visited.add((new_row, new_col))
                        queue.append(((new_row, new_col), path + [(new_row, new_col)]))
    
    return []


def _place_traps(
    grid: List[List[int]], 
    size: int, 
    trap_density: float, 
    path_set: Set[Coordinate],
    player_pos: Coordinate,
    goal_pos: Coordinate
) -> None:
    """随机放置陷阱（不能在解决方案路径上）"""
    # 计算可以放置陷阱的位置
    available_cells = []
    for row in range(size):
        for col in range(size):
            pos = (row, col)
            # 不能在路径上、玩家位置或目标位置
            if pos not in path_set and pos != player_pos and pos != goal_pos:
                available_cells.append(pos)
    
    # 计算要放置的陷阱数量
    num_traps = int(len(available_cells) * trap_density)
    
    # 随机选择位置放置陷阱
    trap_positions = _RANDOM.sample(available_cells, min(num_traps, len(available_cells)))
    
    for row, col in trap_positions:
        grid[row][col] = constants.TRAP_CELL


def find_position(grid: List[List[int]], target_value: int) -> Coordinate:
    """在网格中找到指定值的位置"""
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == target_value:
                return row_index, col_index
    raise ValueError(f"Target value not found in grid: {target_value}")


def solve_trapfield(grid: List[List[int]]) -> List[Coordinate]:
    """
    求解陷阱场（使用 BFS 找最短路径，避开陷阱）
    
    Returns:
        从玩家到目标的路径（坐标列表）
    """
    player_pos = find_position(grid, constants.PLAYER_CELL)
    goal_pos = find_position(grid, constants.GOAL_CELL)
    
    size = len(grid)
    queue = deque([(player_pos, [player_pos])])
    visited = {player_pos}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    
    while queue:
        (row, col), path = queue.popleft()
        
        if (row, col) == goal_pos:
            return path
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if 0 <= new_row < size and 0 <= new_col < size:
                if (new_row, new_col) not in visited:
                    cell_value = grid[new_row][new_col]
                    # 可以走空地、玩家位置、目标位置，但不能走陷阱
                    if cell_value != constants.TRAP_CELL:
                        visited.add((new_row, new_col))
                        queue.append(((new_row, new_col), path + [(new_row, new_col)]))
    
    return []  # 没有找到路径


def get_direction(from_pos: Coordinate, to_pos: Coordinate) -> str:
    """获取移动方向的文字描述"""
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    
    if dr == -1:
        return "up"
    elif dr == 1:
        return "down"
    elif dc == -1:
        return "left"
    elif dc == 1:
        return "right"
    else:
        return "unknown"

