"""
PathFinder 生成器 - Voronoi 版本

使用 Voronoi 图替代 Delaunay 三角化，天然避免道路重叠。

核心思想：
- Voronoi 图的边天然就是"安全走廊"，距离所有节点都最远
- 不需要复杂的重叠检测，因为 Voronoi 边本身就保证了最大间距
"""

import random
import time
from typing import List, Tuple, Optional, Set
import numpy as np
from scipy.spatial import Voronoi
from collections import deque

from games.pathfinder.board import PathFinderBoard, PathSegment
from games.pathfinder.constants import DIFFICULTY_CONFIG


class VoronoiNode:
    """Voronoi 节点（Voronoi 图的顶点）"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.adjacents: Set['VoronoiNode'] = set()
        self.pathways: Set['VoronoiNode'] = set()


def poisson_disk_sampling_voronoi(
    width: int,
    height: int,
    radius: float,
    k: int = 30
) -> List[Tuple[float, float]]:
    """
    Poisson Disk Sampling - 生成均匀分布的种子点
    
    这些种子点用于生成 Voronoi 图
    """
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    points = []
    active_list = []
    
    # 初始点
    first_x = random.uniform(width * 0.2, width * 0.8)
    first_y = random.uniform(height * 0.2, height * 0.8)
    first_point = (first_x, first_y)
    points.append(first_point)
    active_list.append(first_point)
    
    gx = int(first_x / cell_size)
    gy = int(first_y / cell_size)
    grid[gx][gy] = first_point
    
    while active_list:
        idx = random.randint(0, len(active_list) - 1)
        point = active_list[idx]
        found = False
        
        for _ in range(k):
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(radius, 2 * radius)
            new_x = point[0] + r * np.cos(angle)
            new_y = point[1] + r * np.sin(angle)
            
            if not (0 <= new_x < width and 0 <= new_y < height):
                continue
            
            gx = int(new_x / cell_size)
            gy = int(new_y / cell_size)
            
            # 检查周围格子
            valid = True
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    if grid[i][j] is not None:
                        dx = new_x - grid[i][j][0]
                        dy = new_y - grid[i][j][1]
                        if dx * dx + dy * dy < radius * radius:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                new_point = (new_x, new_y)
                points.append(new_point)
                active_list.append(new_point)
                grid[gx][gy] = new_point
                found = True
                break
        
        if not found:
            active_list.pop(idx)
    
    return points


def generate_voronoi_graph(
    seed_points: List[Tuple[float, float]],
    image_size: int,
    road_width: float
) -> Tuple[List[VoronoiNode], List[Tuple[VoronoiNode, VoronoiNode]]]:
    """
    生成 Voronoi 图
    
    返回：
    - nodes: Voronoi 顶点列表
    - edges: Voronoi 边列表
    
    关键：Voronoi 边天然就是"安全走廊"，距离所有种子点都最远
    """
    if len(seed_points) < 4:
        return [], []
    
    # 添加边界点，确保 Voronoi 图覆盖整个区域
    margin = road_width * 2
    boundary_points = [
        (-margin, -margin),
        (image_size + margin, -margin),
        (image_size + margin, image_size + margin),
        (-margin, image_size + margin),
    ]
    all_points = seed_points + boundary_points
    
    # 生成 Voronoi 图
    vor = Voronoi(all_points)
    
    # 提取 Voronoi 顶点（这些是我们的节点）
    nodes = []
    node_map = {}  # vor.vertices 索引 -> VoronoiNode
    
    bounds = (margin, image_size - margin)
    
    for i, vertex in enumerate(vor.vertices):
        x, y = vertex
        # 只保留在边界内的顶点
        if bounds[0] <= x <= bounds[1] and bounds[0] <= y <= bounds[1]:
            node = VoronoiNode(x, y)
            nodes.append(node)
            node_map[i] = node
    
    # 提取 Voronoi 边
    edges = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:  # 跳过无限边
            continue
        
        v1_idx, v2_idx = ridge_vertices
        if v1_idx in node_map and v2_idx in node_map:
            node1 = node_map[v1_idx]
            node2 = node_map[v2_idx]
            
            # 计算边长，过滤过长的边
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            length = np.sqrt(dx * dx + dy * dy)
            
            # 最大边长：避免过长的边
            max_edge_length = road_width * 8.0
            if length <= max_edge_length:
                node1.adjacents.add(node2)
                node2.adjacents.add(node1)
                edges.append((node1, node2))
    
    return nodes, edges


def create_spanning_tree_dfs(
    nodes: List[VoronoiNode],
    start_node: VoronoiNode,
    connectivity_ratio: float = 0.3
) -> None:
    """
    使用 DFS 创建生成树（随机化）

    Args:
        connectivity_ratio: 连通率（0-1）
            - 0.0: 只保留最小生成树（最稀疏）
            - 0.3: 保留 30% 的额外边（推荐，看起来像道路）
            - 1.0: 保留所有边（最密集，像网格）
    """
    visited = set()
    stack = [start_node]
    tree_edges = []  # 记录生成树的边

    # 第一步：创建最小生成树（保证连通性）
    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)

        # 随机打乱邻居顺序
        neighbors = list(current.adjacents)
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited:
                # 添加到生成树
                current.pathways.add(neighbor)
                neighbor.pathways.add(current)
                tree_edges.append((current, neighbor))
                stack.append(neighbor)

    # 第二步：根据连通率添加额外的边
    if connectivity_ratio > 0:
        # 收集所有不在生成树中的边
        non_tree_edges = []
        for node in nodes:
            for neighbor in node.adjacents:
                if neighbor not in node.pathways:
                    edge = tuple(sorted([id(node), id(neighbor)]))
                    non_tree_edges.append((edge, node, neighbor))

        # 去重
        unique_edges = {}
        for edge_id, node, neighbor in non_tree_edges:
            if edge_id not in unique_edges:
                unique_edges[edge_id] = (node, neighbor)

        # 随机选择一部分边添加
        num_to_add = int(len(unique_edges) * connectivity_ratio)
        edges_to_add = random.sample(list(unique_edges.values()), min(num_to_add, len(unique_edges)))

        for node, neighbor in edges_to_add:
            node.pathways.add(neighbor)
            neighbor.pathways.add(node)


def add_extra_edges(
    nodes: List[VoronoiNode],
    num_extra: int
) -> None:
    """
    添加额外的边，增加路径复杂度
    """
    added = 0
    attempts = 0
    max_attempts = num_extra * 10
    
    while added < num_extra and attempts < max_attempts:
        attempts += 1
        node = random.choice(nodes)
        
        # 找到不在 pathways 中的邻居
        candidates = [n for n in node.adjacents if n not in node.pathways]
        if candidates:
            neighbor = random.choice(candidates)
            node.pathways.add(neighbor)
            neighbor.pathways.add(node)
            added += 1


def find_longest_path_bfs(
    nodes: List[VoronoiNode],
    start_node: VoronoiNode
) -> Tuple[VoronoiNode, List[VoronoiNode]]:
    """
    使用 BFS 找到从 start_node 出发的最长路径
    """
    queue = deque([(start_node, [start_node])])
    visited = {start_node}
    longest_path = [start_node]
    farthest_node = start_node
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > len(longest_path):
            longest_path = path
            farthest_node = current
        
        for neighbor in current.pathways:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return farthest_node, longest_path


def generate_simple_curve(
    start: Tuple[float, float],
    end: Tuple[float, float],
    curvature: float = 0.05
) -> List[Tuple[float, float]]:
    """
    生成简单的 3 点 Bezier 曲线
    
    curvature: 曲率（0-1），越小越接近直线
    """
    sx, sy = start
    ex, ey = end
    
    # 中点
    mx = (sx + ex) / 2
    my = (sy + ey) / 2
    
    # 垂直方向
    dx = ex - sx
    dy = ey - sy
    length = np.sqrt(dx * dx + dy * dy)
    
    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length
        
        # 随机偏移
        offset = random.uniform(-curvature, curvature) * length
        cx = mx + perp_x * offset
        cy = my + perp_y * offset
    else:
        cx, cy = mx, my
    
    return [(sx, sy), (cx, cy), (ex, ey)]


def generate_pathfinder_board_voronoi(
    difficulty: str = "medium",
    timeout_seconds: float = 30.0,
    debug: bool = False
) -> Optional[PathFinderBoard]:
    """
    使用 Voronoi 图生成 PathFinder 关卡
    
    优势：
    - Voronoi 边天然保证最大间距，不会重叠
    - 不需要复杂的重叠检测
    - 生成速度快
    """
    config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG['medium'])
    image_size = config['image_size']
    road_width = config['road_width']
    node_spacing_ratio = config['node_spacing_ratio']
    min_solution_nodes = config['min_solution_nodes']
    connectivity_ratio = config.get('connectivity_ratio', 0.2)  # 默认 0.2

    start_time = time.time()
    max_attempts = 50

    # Voronoi 需要更大的节点间距（因为我们使用的是 Voronoi 顶点，不是种子点）
    # 种子点间距应该是道路宽度的 3-4 倍
    seed_radius = max(image_size * node_spacing_ratio, road_width * 3.0)

    for attempt in range(max_attempts):
        if time.time() - start_time > timeout_seconds:
            if debug:
                print(f'[Voronoi] Timeout after {timeout_seconds}s')
            return None

        # 1. 生成种子点
        seed_points = poisson_disk_sampling_voronoi(image_size, image_size, seed_radius)

        if len(seed_points) < 5:
            continue

        # 2. 生成 Voronoi 图
        nodes, _ = generate_voronoi_graph(seed_points, image_size, road_width)

        if len(nodes) < min_solution_nodes + 2:
            continue

        # 3. 创建生成树（使用连通率控制密度）
        start_node = random.choice(nodes)
        create_spanning_tree_dfs(nodes, start_node, connectivity_ratio)

        # 4. 不再需要 add_extra_edges，因为连通率已经控制了额外边的数量
        # add_extra_edges(nodes, extra_paths)
        
        # 5. 找到最长路径作为解决方案
        end_node, solution_path = find_longest_path_bfs(nodes, start_node)
        
        if len(solution_path) < min_solution_nodes:
            continue
        
        # 6. 生成路径段
        segments = []
        segment_map = {}
        
        for node in nodes:
            for neighbor in node.pathways:
                edge = tuple(sorted([id(node), id(neighbor)]))
                if edge not in segment_map:
                    control_points = generate_simple_curve(
                        (node.x, node.y),
                        (neighbor.x, neighbor.y),
                        curvature=0.04  # 更小的曲率，因为 Voronoi 边已经很安全了
                    )
                    segments.append(PathSegment(control_points))
                    segment_map[edge] = len(segments) - 1
        
        # 7. 找到解决方案对应的 segment 索引
        solution_segments = []
        solution_path_coords = [(start_node.x, start_node.y)]
        
        for i in range(len(solution_path) - 1):
            edge = tuple(sorted([id(solution_path[i]), id(solution_path[i+1])]))
            if edge in segment_map:
                solution_segments.append(segment_map[edge])
            solution_path_coords.append((solution_path[i+1].x, solution_path[i+1].y))
        
        if not solution_segments:
            continue
        
        # 8. 创建游戏板
        board = PathFinderBoard(
            segments=segments,
            start_point=(start_node.x, start_node.y),
            end_point=(end_node.x, end_node.y),
            solution_segments=solution_segments,
            solution_path=solution_path_coords,
            image_size=image_size,
            road_width=road_width
        )
        
        if debug:
            elapsed = time.time() - start_time
            print(f'[Voronoi] Success in {elapsed:.3f}s, {len(segments)} segments, {len(solution_path)} nodes in solution')
        
        return board
    
    if debug:
        print(f'[Voronoi] Failed after {max_attempts} attempts')
    
    return None

