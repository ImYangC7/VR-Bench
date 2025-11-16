 #!/usr/bin/env python3
"""统一的路径查找接口"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple, Set
from collections import deque
import numpy as np
from core.schema import UnifiedState
from games.maze import constants as maze_constants
from games.trapfield import constants as trapfield_constants
from games.pathfinder.generator import Node, find_all_shortest_paths_bfs as pathfinder_bfs
from games.pathfinder.renderer import bezier_curve_opencv

Coordinate = Tuple[int, int]


def find_grid_paths(grid: List[List[int]], start: Coordinate, goal: Coordinate, 
                    walkable: Set[int]) -> List[List[Coordinate]]:
    """BFS查找网格游戏的所有最短路径"""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    queue = deque([(start, [start])])
    visited = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shortest_distance = None
    all_paths = []
    
    while queue:
        (row, col), path = queue.popleft()
        current_distance = len(path) - 1
        
        if shortest_distance and current_distance > shortest_distance:
            break
        
        if (row, col) == goal:
            if shortest_distance is None:
                shortest_distance = current_distance
            if current_distance == shortest_distance:
                all_paths.append(path)
            continue
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in walkable:
                new_distance = current_distance + 1
                if (nr, nc) not in visited or visited[(nr, nc)] == new_distance:
                    visited[(nr, nc)] = new_distance
                    queue.append(((nr, nc), path + [(nr, nc)]))
    
    return all_paths


def find_maze_paths(state: UnifiedState) -> List[List[Coordinate]]:
    """查找Maze游戏的所有最短路径"""
    grid = state.grid.data
    start = (state.player.grid_pos.row, state.player.grid_pos.col)
    goal = (state.goal.grid_pos.row, state.goal.grid_pos.col)
    walkable = {maze_constants.EMPTY_CELL, maze_constants.PLAYER_CELL, maze_constants.GOAL_CELL}
    return find_grid_paths(grid, start, goal, walkable)


def find_trapfield_paths(state: UnifiedState) -> List[List[Coordinate]]:
    """查找TrapField游戏的所有最短路径"""
    grid = state.grid.data
    start = (state.player.grid_pos.row, state.player.grid_pos.col)
    goal = (state.goal.grid_pos.row, state.goal.grid_pos.col)
    walkable = {trapfield_constants.EMPTY_CELL, trapfield_constants.PLAYER_CELL, trapfield_constants.GOAL_CELL}
    return find_grid_paths(grid, start, goal, walkable)


def find_sokoban_paths(state: UnifiedState) -> List[List[Coordinate]]:
    """查找Sokoban游戏的所有最短路径（使用求解器）"""
    from games.sokoban.board import Solution
    solver = Solution()
    grid_chars = []
    for row in state.grid.data:
        row_str = ""
        for cell in row:
            if cell == 0:
                row_str += "."
            elif cell == 1:
                row_str += "#"
            elif cell == 2:
                row_str += "B"
            elif cell == 3:
                row_str += "T"
            elif cell == 4:
                row_str += "*"
            elif cell == 5:
                row_str += "S"
            elif cell == 6:
                row_str += "+"
        grid_chars.append(list(row_str))

    total_moves, _ = solver.minPushBox(grid_chars)
    if total_moves == -1:
        return []

    moves = solver.get_solution_path()
    if not moves:
        return []

    path = [(int(moves[0].start_pos[0]), int(moves[0].start_pos[1]))]
    for move in moves:
        path.append((int(move.end_pos[0]), int(move.end_pos[1])))

    return [path]


def reconstruct_pathfinder_graph(state: UnifiedState) -> Tuple[Node, Node]:
    """从state重建PathFinder节点图"""
    segments = state.metadata.get('segments', [])
    solution_path = state.metadata.get('solution_path', [])

    if not solution_path or len(solution_path) < 2:
        raise ValueError("solution_path为空")

    start_point = tuple(solution_path[0])
    end_point = tuple(solution_path[-1])

    node_map = {}

    for segment in segments:
        if len(segment) < 2:
            continue
        p1, p2 = tuple(segment[0]), tuple(segment[-1])
        if p1 not in node_map:
            node_map[p1] = Node(p1[0], p1[1])
        if p2 not in node_map:
            node_map[p2] = Node(p2[0], p2[1])
        node_map[p1].pathways.add(node_map[p2])
        node_map[p2].pathways.add(node_map[p1])

    start_node = node_map.get(start_point)
    end_node = node_map.get(end_point)

    if not start_node or not end_node:
        raise ValueError(f"找不到起点或终点节点")

    return start_node, end_node


def calculate_pathfinder_distance(segments_data: list, node_path: List[Node]) -> float:
    """计算PathFinder路径的物理距离"""
    segment_map = {}
    for idx, segment in enumerate(segments_data):
        if len(segment) < 2:
            continue
        p1, p2 = tuple(segment[0]), tuple(segment[-1])
        segment_map[(p1, p2)] = idx
        segment_map[(p2, p1)] = idx
    
    total_distance = 0.0
    for i in range(len(node_path) - 1):
        node1, node2 = node_path[i], node_path[i + 1]
        edge = ((node1.x, node1.y), (node2.x, node2.y))
        if edge in segment_map:
            segment = segments_data[segment_map[edge]]
            curve_points = bezier_curve_opencv(segment)
            for j in range(len(curve_points) - 1):
                p1, p2 = curve_points[j], curve_points[j + 1]
                total_distance += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    return total_distance


def find_pathfinder_paths(state: UnifiedState) -> List[Tuple[List[Node], float]]:
    """查找PathFinder的所有最优路径（按物理距离）"""
    start_node, end_node = reconstruct_pathfinder_graph(state)
    all_paths = pathfinder_bfs(start_node, end_node)

    if not all_paths:
        return []

    segments_data = state.metadata.get('segments', [])
    path_distances = [(path, calculate_pathfinder_distance(segments_data, path))
                      for path in all_paths]
    path_distances.sort(key=lambda x: x[1])

    min_distance = path_distances[0][1]
    optimal_paths = [(path, dist) for path, dist in path_distances
                     if dist <= min_distance * 1.01]

    return optimal_paths


def find_optimal_paths(state: UnifiedState, game_type: str = None) -> List:
    """统一接口：查找任意游戏的最优路径"""
    if game_type is None:
        game_type = state.metadata.get('game_type', 'maze')
    
    if game_type == 'maze':
        return find_maze_paths(state)
    elif game_type == 'trapfield':
        return find_trapfield_paths(state)
    elif game_type == 'sokoban':
        return find_sokoban_paths(state)
    elif game_type == 'pathfinder':
        return find_pathfinder_paths(state)
    else:
        raise ValueError(f"不支持的游戏类型: {game_type}")

