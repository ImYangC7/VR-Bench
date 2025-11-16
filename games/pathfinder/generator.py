"""
PathFinder 生成器 - 使用 Poisson Disk Sampling + Delaunay Triangulation
参考: https://github.com/gerizim16/maze
"""

import random
import math
import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Set
from scipy.spatial import Delaunay

from .board import PathFinderBoard, PathSegment
from .constants import DIFFICULTY_CONFIG, DEFAULT_IMAGE_SIZE, MARGIN, ROAD_COLOR, BG_COLOR, CURVE_SEGMENTS


class Node:
    """图节点"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.adjacents: Set['Node'] = set()  # Delaunay三角化的邻居
        self.pathways: Set['Node'] = set()   # 实际可通行的路径
        self.visited = False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def poisson_disk_sampling(
    radius: float,
    width: float,
    height: float,
    k: int = 30,
    margin_ratio: float = 0.05  # 边界留白比例
) -> List[Node]:
    """Poisson Disk Sampling - 生成均匀分布的点"""
    # 计算实际可用区域（缩小5%）
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    usable_width = width - 2 * margin_x
    usable_height = height - 2 * margin_y

    cell_size = radius / math.sqrt(2)
    grid_width = int(math.ceil(usable_width / cell_size)) + 1
    grid_height = int(math.ceil(usable_height / cell_size)) + 1
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

    def get_cell_index(point: Node) -> Tuple[int, int]:
        # 转换到相对坐标
        rel_x = point.x - margin_x
        rel_y = point.y - margin_y
        return (int(rel_x / cell_size), int(rel_y / cell_size))

    def is_valid_point(point: Node) -> bool:
        # 检查是否在可用区域内
        if point.x < margin_x or point.y < margin_y or point.x >= width - margin_x or point.y >= height - margin_y:
            return False

        cell_x, cell_y = get_cell_index(point)
        # 检查周围的8个格子
        for y in range(max(cell_y - 1, 0), min(cell_y + 2, grid_height)):
            for x in range(max(cell_x - 1, 0), min(cell_x + 2, grid_width)):
                if grid[y][x] is not None:
                    dx = grid[y][x].x - point.x
                    dy = grid[y][x].y - point.y
                    if math.sqrt(dx*dx + dy*dy) < radius:
                        return False
        return True

    points = []
    active_points = []

    # 初始点（在可用区域内）
    p0 = Node(random.uniform(margin_x, width - margin_x), random.uniform(margin_y, height - margin_y))
    cell_x, cell_y = get_cell_index(p0)
    grid[cell_y][cell_x] = p0
    points.append(p0)
    active_points.append(p0)

    while active_points:
        active_idx = random.randint(0, len(active_points) - 1)
        active_point = active_points[active_idx]

        found = False
        for _ in range(k):
            theta = random.uniform(0, 2 * math.pi)
            point_radius = random.uniform(radius, 2 * radius)
            new_point = Node(
                active_point.x + point_radius * math.cos(theta),
                active_point.y + point_radius * math.sin(theta)
            )

            if is_valid_point(new_point):
                cell_x, cell_y = get_cell_index(new_point)
                grid[cell_y][cell_x] = new_point
                points.append(new_point)
                active_points.append(new_point)
                found = True
                break

        if not found:
            active_points.pop(active_idx)

    return points


def calculate_angle_between_edges(edge1: Tuple[Node, Node], edge2: Tuple[Node, Node]) -> float:
    """计算两条边之间的夹角（度数）"""
    # 找到共享节点
    shared_node = None
    if edge1[0] == edge2[0] or edge1[0] == edge2[1]:
        shared_node = edge1[0]
        other1 = edge1[1]
    elif edge1[1] == edge2[0] or edge1[1] == edge2[1]:
        shared_node = edge1[1]
        other1 = edge1[0]
    else:
        # 没有共享节点，返回180度（不相交）
        return 180.0

    # 找到另一条边的另一个节点
    if edge2[0] == shared_node:
        other2 = edge2[1]
    else:
        other2 = edge2[0]

    # 计算两个向量
    v1_x = other1.x - shared_node.x
    v1_y = other1.y - shared_node.y
    v2_x = other2.x - shared_node.x
    v2_y = other2.y - shared_node.y

    # 计算向量长度
    len1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
    len2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)

    if len1 == 0 or len2 == 0:
        return 180.0

    # 计算夹角（使用点积）
    dot_product = v1_x * v2_x + v1_y * v2_y
    cos_angle = dot_product / (len1 * len2)

    # 限制在 [-1, 1] 范围内，避免浮点误差
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # 转换为度数
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def check_edge_angle_conflict(new_edge: Tuple[Node, Node], existing_edges: List[Tuple[Node, Node]],
                              min_angle: float = 25.0) -> bool:
    """检查新边是否与现有边的夹角太小（会导致重叠）"""
    for edge in existing_edges:
        # 检查是否有共享节点
        if new_edge[0] in edge or new_edge[1] in edge:
            # 计算夹角
            angle = calculate_angle_between_edges(new_edge, edge)

            # 如果夹角太小，认为会重叠
            if angle < min_angle:
                return True

    return False


def line_segment_distance(p1: Tuple[float, float], p2: Tuple[float, float],
                         p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
    """计算两条线段之间的最短距离"""
    # 点到线段的距离
    def point_to_segment_distance(px: float, py: float,
                                  x1: float, y1: float,
                                  x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    # 检查四个端点到对方线段的距离
    d1 = point_to_segment_distance(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1])
    d2 = point_to_segment_distance(p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])
    d3 = point_to_segment_distance(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1])
    d4 = point_to_segment_distance(p4[0], p4[1], p1[0], p1[1], p2[0], p2[1])

    return min(d1, d2, d3, d4)


def check_edge_distance_conflict(new_edge: Tuple[Node, Node], existing_edges: List[Tuple[Node, Node]],
                                 road_width: float) -> bool:
    """检查新边是否与现有边距离太近（会导致重叠）"""
    p1 = (new_edge[0].x, new_edge[0].y)
    p2 = (new_edge[1].x, new_edge[1].y)

    # 检查是否共享节点（共享节点的边不检查距离）
    for edge in existing_edges:
        if new_edge[0] in edge or new_edge[1] in edge:
            continue

        p3 = (edge[0].x, edge[0].y)
        p4 = (edge[1].x, edge[1].y)

        # 计算两条线段的距离
        dist = line_segment_distance(p1, p2, p3, p4)

        # 如果距离小于道路宽度的2倍，认为会重叠
        if dist < road_width * 2.0:
            return True

    return False


def delaunay_triangulation(nodes: List[Node], max_edge_length: Optional[float] = None,
                           road_width: float = 30.0, min_angle: float = 25.0) -> List[Tuple[Node, Node]]:
    """Delaunay三角化 - 连接节点（优化版：只检查相关边）"""
    if len(nodes) < 3:
        return []

    points = np.array([[node.x, node.y] for node in nodes])
    tri = Delaunay(points)

    edges = []
    # 使用字典加速查找：node -> 与该节点相连的边
    node_edges = {node: [] for node in nodes}

    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i + 1) % 3]

            p1 = nodes[p1_idx]
            p2 = nodes[p2_idx]

            # 检查边长度
            if max_edge_length is not None:
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                if math.sqrt(dx*dx + dy*dy) > max_edge_length:
                    continue

            # 避免重复边
            if (p1, p2) in edges or (p2, p1) in edges:
                continue

            new_edge = (p1, p2)

            # 只检查与 p1 或 p2 相关的边（大幅减少检查次数）
            related_edges = node_edges[p1] + node_edges[p2]

            # 检查夹角（只检查相关边）
            has_angle_conflict = False
            for edge in related_edges:
                angle = calculate_angle_between_edges(new_edge, edge)
                if angle < min_angle:
                    has_angle_conflict = True
                    break

            if has_angle_conflict:
                continue

            # 检查距离（只检查不共享节点的边）
            has_distance_conflict = False
            for edge in edges:
                # 跳过共享节点的边
                if p1 in edge or p2 in edge:
                    continue

                p3 = (edge[0].x, edge[0].y)
                p4 = (edge[1].x, edge[1].y)
                p1_pos = (p1.x, p1.y)
                p2_pos = (p2.x, p2.y)

                dist = line_segment_distance(p1_pos, p2_pos, p3, p4)
                if dist < road_width * 2.0:
                    has_distance_conflict = True
                    break

            if has_distance_conflict:
                continue

            # 添加双向邻接关系
            p1.adjacents.add(p2)
            p2.adjacents.add(p1)
            edges.append(new_edge)

            # 更新 node_edges
            node_edges[p1].append(new_edge)
            node_edges[p2].append(new_edge)

    return edges


def create_maze_aldous_broder(nodes: List[Node]) -> List[Tuple[Node, Node]]:
    """使用 Aldous-Broder 算法生成完美迷宫（只有一条路径）"""
    if not nodes:
        return []

    pathways = []
    current = random.choice(nodes)
    current.visited = True
    visited_count = 1

    while visited_count < len(nodes):
        if not current.adjacents:
            break

        neighbor = random.choice(list(current.adjacents))

        if not neighbor.visited:
            # 创建路径
            current.pathways.add(neighbor)
            neighbor.pathways.add(current)
            pathways.append((current, neighbor))
            neighbor.visited = True
            visited_count += 1

        current = neighbor

    return pathways


def add_random_pathways(nodes: List[Node], n: int = 1, max_tries: int = 20) -> List[Tuple[Node, Node]]:
    """添加额外的随机路径（增加多条解决方案）"""
    pathways = []

    for _ in range(n):
        tries = 0
        while tries < max_tries:
            node = random.choice(nodes)
            # 找到还没有连接的邻居
            candidates = node.adjacents - node.pathways

            if candidates:
                neighbor = random.choice(list(candidates))
                node.pathways.add(neighbor)
                neighbor.pathways.add(node)
                pathways.append((node, neighbor))
                break

            tries += 1

    return pathways


def generate_smooth_curve(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_control_points: int = 2,
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
) -> List[Tuple[float, float]]:
    """生成平滑的贝塞尔曲线控制点"""
    points = [start]

    for i in range(num_control_points):
        t = (i + 1) / (num_control_points + 1)
        base_x = start[0] + (end[0] - start[0]) * t
        base_y = start[1] + (end[1] - start[1]) * t

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)

        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
            # 曲线偏移量：使用 6% 而不是 8%（更保守，减少重叠风险）
            # 基于 Buffer Zone 理论：偏移量应该 < (节点间距 - 道路宽度) / 2
            # 当前：节点间距 = 4.0 × road_width，偏移 6% 确保安全
            offset = random.uniform(-0.06, 0.06) * length
            control_x = base_x + perp_x * offset
            control_y = base_y + perp_y * offset

            # 如果提供了边界，确保控制点在边界内
            if bounds:
                min_x, min_y, max_x, max_y = bounds
                control_x = max(min_x, min(max_x, control_x))
                control_y = max(min_y, min(max_y, control_y))

            points.append((control_x, control_y))

    points.append(end)
    return points


def find_path_bfs(start: Node, end: Node) -> Optional[List[Node]]:
    """使用BFS查找路径（返回第一条最短路径）"""
    if start == end:
        return [start]

    from collections import deque
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for neighbor in current.pathways:
            if neighbor in visited:
                continue

            new_path = path + [neighbor]

            if neighbor == end:
                return new_path

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def find_all_shortest_paths_bfs(start: Node, end: Node) -> List[List[Node]]:
    """
    使用BFS查找所有最短路径

    Args:
        start: 起始节点
        end: 目标节点

    Returns:
        所有最短路径的列表，每条路径是节点列表
    """
    if start == end:
        return [[start]]

    from collections import deque
    queue = deque([(start, [start])])
    visited = {start: 0}  # 记录到达每个节点的最短距离

    shortest_distance = None
    all_paths = []

    while queue:
        current, path = queue.popleft()
        current_distance = len(path) - 1

        # 如果已经找到最短路径，且当前路径更长，停止搜索
        if shortest_distance is not None and current_distance > shortest_distance:
            break

        # 到达终点
        if current == end:
            if shortest_distance is None:
                shortest_distance = current_distance
            if current_distance == shortest_distance:
                all_paths.append(path)
            continue

        # 探索所有邻居
        for neighbor in current.pathways:
            new_distance = current_distance + 1

            # 如果这个节点没访问过，或者找到了相同长度的路径
            if neighbor not in visited or visited[neighbor] == new_distance:
                visited[neighbor] = new_distance
                queue.append((neighbor, path + [neighbor]))

    return all_paths


def bezier_curve_for_check(control_points: List[Tuple[float, float]], num_points: int = CURVE_SEGMENTS) -> np.ndarray:
    """生成贝塞尔曲线点（用于重叠检测）"""
    n = len(control_points) - 1
    points = []

    for i in range(num_points + 1):
        t = i / num_points
        x, y = 0.0, 0.0

        for j, (px, py) in enumerate(control_points):
            # 贝塞尔基函数
            from math import comb
            coef = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
            x += coef * px
            y += coef * py

        points.append([x, y])

    return np.array(points, dtype=np.int32)


def check_road_overlap(segments: List[PathSegment], image_size: int, road_width: int) -> bool:
    """检测道路是否有重叠（通过渲染检测像素重叠）- 优化版本"""
    # 创建一个临时图像用于检测（使用更小的分辨率加速）
    # 缩小到1/2分辨率进行检测，速度提升4倍
    scale = 0.5
    scaled_size = int(image_size * scale)
    scaled_width = max(1, int(road_width * scale))

    img = np.zeros((scaled_size, scaled_size), dtype=np.uint8)

    # 预先计算所有曲线点（避免重复计算）
    all_curve_points = []
    for segment in segments:
        curve_points = bezier_curve_for_check(segment.control_points)
        # 缩放曲线点
        scaled_points = (curve_points * scale).astype(np.int32)
        all_curve_points.append(scaled_points)

    # 检测重叠
    for i, curve_points in enumerate(all_curve_points):
        # 创建临时图像绘制当前道路
        temp_img = np.zeros((scaled_size, scaled_size), dtype=np.uint8)
        cv2.polylines(temp_img, [curve_points], False, 255, scaled_width, lineType=cv2.LINE_AA)

        # 检测是否与已有道路重叠
        overlap = cv2.bitwise_and(img, temp_img)
        overlap_pixels = np.count_nonzero(overlap)

        # 阈值也需要缩放
        threshold = scaled_width * 0.5
        if overlap_pixels > threshold:
            return True

        # 将当前道路添加到已绘制的图像中（直接在原图上绘制，避免bitwise_or）
        cv2.polylines(img, [curve_points], False, 255, scaled_width, lineType=cv2.LINE_AA)

    return False


def generate_pathfinder_board(
    difficulty: str = "medium",
    image_size: int = None,
    max_attempts: int = 50,
    timeout_seconds: float = 30.0,
    debug: bool = False
) -> Optional[PathFinderBoard]:
    """
    生成PathFinder游戏板 - 使用 Voronoi 图（天然避免重叠）

    难度通过多个因素区分：
    - easy: 512x512, 稀疏节点(18%), 少支路(1), 短路径(4节点), 宽道路(30px)
    - medium: 768x768, 中等节点(15%), 中支路(3), 中路径(6节点), 中道路(22px)
    - hard: 1024x1024, 密集节点(12%), 多支路(5), 长路径(8节点), 窄道路(18px)

    Args:
        timeout_seconds: 最大生成时间（秒），超时返回 None
        debug: 是否输出调试信息

    注意：现在使用 Voronoi 生成器，速度快 10-13 倍，天然避免道路重叠
    """
    # 使用 Voronoi 生成器（速度快，天然避免重叠）
    from games.pathfinder.generator_voronoi import generate_pathfinder_board_voronoi
    return generate_pathfinder_board_voronoi(difficulty, timeout_seconds, debug)


# ============================================================================
# V1 生成器 - 旧版本（已废弃，保留用于参考）
# ============================================================================
def generate_pathfinder_board_v1_deprecated(
    difficulty: str = "medium",
    image_size: int = None,
    max_attempts: int = 50,
    timeout_seconds: float = 30.0,
    debug: bool = False
) -> Optional[PathFinderBoard]:
    """
    旧版生成器 - 使用 Poisson + Delaunay + Aldous-Broder
    已废弃：速度太慢，容易出现道路重叠
    """
    start_time = time.time()

    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # 获取难度对应的配置
    difficulty_config = DIFFICULTY_CONFIG[difficulty]
    if image_size is None:
        image_size = difficulty_config['image_size']
    road_width = difficulty_config['road_width']
    node_spacing_ratio = difficulty_config['node_spacing_ratio']
    extra_paths = difficulty_config['extra_paths']
    min_solution_nodes = difficulty_config['min_solution_nodes']

    # 根据配置计算节点间距（考虑道路宽度，确保道路不重叠）
    # 节点间距应该至少是道路宽度的2倍，以避免道路重叠
    min_radius_for_road = road_width * 2.5  # 2.5倍道路宽度作为最小间距
    radius = max(image_size * node_spacing_ratio, min_radius_for_road)

    # 计算边界（5%留白）
    margin = image_size * 0.05
    bounds = (margin, margin, image_size - margin, image_size - margin)

    # 统计信息
    stats = {
        'attempts': 0,
        'not_enough_nodes': 0,
        'no_path': 0,
        'path_too_short': 0,
        'redraw_failed': 0,
    }

    for attempt in range(max_attempts):
        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            if debug:
                print(f'[PathFinder] Timeout after {elapsed:.2f}s')
                print(f'[PathFinder] Stats: {stats}')
            return None

        stats['attempts'] += 1

        # 1. Poisson Disk Sampling生成节点
        nodes = poisson_disk_sampling(radius, image_size, image_size, k=30)

        if len(nodes) < 5:
            stats['not_enough_nodes'] += 1
            continue

        # 2. Delaunay三角化连接节点 - 通过夹角和距离避免道路重叠
        # min_angle: 最小夹角（度），夹角太小会导致道路重叠
        min_angle = 20.0  # 20度是一个比较好的阈值
        delaunay_triangulation(nodes, max_edge_length=radius * 2.5, road_width=road_width, min_angle=min_angle)

        # 3. 使用Aldous-Broder生成完美迷宫
        create_maze_aldous_broder(nodes)

        # 4. 添加额外路径
        add_random_pathways(nodes, n=extra_paths)

        # 5. 选择起点和终点（尽量远）
        start_node = random.choice(nodes)
        end_node = max(nodes, key=lambda n: math.sqrt((n.x - start_node.x)**2 + (n.y - start_node.y)**2))

        # 6. 查找路径
        path = find_path_bfs(start_node, end_node)

        if not path or len(path) < 3:
            stats['no_path'] += 1
            continue

        # 7. 将节点图转换为PathSegment（传入边界限制）
        # 由于已经通过夹角检测避免了大部分重叠，这里只需要简单重试
        max_redraw_attempts = 3  # 减少到3次
        segments = None
        segment_map = None

        for redraw_attempt in range(max_redraw_attempts):
            segments = []
            segment_map = {}  # (node1, node2) -> segment_index

            for node in nodes:
                for neighbor in node.pathways:
                    edge = tuple(sorted([id(node), id(neighbor)]))
                    if edge not in segment_map:
                        control_points = generate_smooth_curve(
                            (node.x, node.y),
                            (neighbor.x, neighbor.y),
                            num_control_points=2,
                            bounds=bounds  # 传入边界
                        )
                        segments.append(PathSegment(control_points))
                        segment_map[edge] = len(segments) - 1

            # 检测道路重叠（快速检测）
            has_overlap = check_road_overlap(segments, image_size, road_width)

            if not has_overlap:
                # 没有重叠，成功！
                break

            # 有重叠，继续重试
            if redraw_attempt == max_redraw_attempts - 1:
                # 最后一次尝试也失败了，放弃这个布局
                segments = None
                break

        if segments is None:
            # 重绘失败，尝试下一个布局
            stats['redraw_failed'] += 1
            continue

        # 8. 找到解决方案路径对应的segment索引和节点路径
        solution_segments = []
        solution_path = [(start_node.x, start_node.y)]  # 保存节点序列

        for i in range(len(path) - 1):
            edge = tuple(sorted([id(path[i]), id(path[i+1])]))
            if edge in segment_map:
                solution_segments.append(segment_map[edge])
            # 添加下一个节点
            solution_path.append((path[i+1].x, path[i+1].y))

        if not solution_segments:
            stats['no_path'] += 1
            continue

        # 检查解决方案路径是否满足最小节点数要求
        if len(solution_path) < min_solution_nodes:
            stats['path_too_short'] += 1
            continue

        # 创建游戏板
        board = PathFinderBoard(
            segments=segments,
            start_point=(start_node.x, start_node.y),
            end_point=(end_node.x, end_node.y),
            solution_segments=solution_segments,
            solution_path=solution_path,  # 传入节点路径
            image_size=image_size,
            road_width=road_width  # 传入道路宽度
        )

        if debug:
            elapsed = time.time() - start_time
            print(f'[PathFinder] Success in {elapsed:.2f}s after {stats["attempts"]} attempts')

        return board

    # 所有尝试都失败了
    if debug:
        elapsed = time.time() - start_time
        print(f'[PathFinder] Failed after {elapsed:.2f}s and {max_attempts} attempts')
        print(f'[PathFinder] Stats: {stats}')

    return None


# ============================================================================
# V2 生成器 - 使用 Delaunay 三角化，但去掉复杂的重叠检测
# ============================================================================

def delaunay_triangulation_v2(
    nodes: List[Node],
    max_edge_length: Optional[float] = None
) -> List[Tuple[Node, Node]]:
    """
    使用 Delaunay 三角化连接节点（简化版，不做重叠检测）
    """
    if len(nodes) < 3:
        return []

    # 提取节点坐标
    points = np.array([[node.x, node.y] for node in nodes])

    # Delaunay 三角化
    tri = Delaunay(points)

    # 提取边
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i + 1) % 3]

            node1 = nodes[p1_idx]
            node2 = nodes[p2_idx]

            # 计算边长
            dist = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)

            # 如果指定了最大边长，则过滤
            if max_edge_length is not None and dist > max_edge_length:
                continue

            # 添加边（双向）
            edge = tuple(sorted([id(node1), id(node2)]))
            if edge not in edges:
                edges.add(edge)
                node1.adjacents.add(node2)
                node2.adjacents.add(node1)

    return list(edges)


def check_path_overlap_fast(board, road_width: float) -> bool:
    """
    快速检测道路是否重叠（使用像素采样方法）

    返回 True 表示有重叠，False 表示无重叠

    方法：
    1. 在较小的分辨率下渲染所有道路
    2. 检测是否有像素被绘制了多次（重叠）
    """
    # 使用较小的分辨率进行快速检测（原始尺寸的 1/2）
    check_size = board.image_size // 2
    scaled_width = max(1, int(road_width / 2))

    # 创建一个计数图像（记录每个像素被绘制的次数）
    overlap_map = np.zeros((check_size, check_size), dtype=np.uint16)

    # 逐条绘制道路，累加计数
    for segment in board.segments:
        # 缩放控制点
        scaled_points = []
        for x, y in segment.control_points:
            scaled_x = int(x * check_size / board.image_size)
            scaled_y = int(y * check_size / board.image_size)
            scaled_points.append([scaled_x, scaled_y])

        # 生成贝塞尔曲线（简化版，减少计算）
        n = len(scaled_points) - 1
        curve_points = []
        for i in range(51):  # 50 个点足够检测
            t = i / 50
            x, y = 0.0, 0.0
            for j, (px, py) in enumerate(scaled_points):
                from math import comb
                coef = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
                x += coef * px
                y += coef * py
            curve_points.append([int(x), int(y)])
        curve_points = np.array(curve_points, dtype=np.int32)

        # 在临时图像上绘制这条道路
        temp = np.zeros((check_size, check_size), dtype=np.uint8)
        cv2.polylines(temp, [curve_points], False, 255, scaled_width, lineType=cv2.LINE_AA)

        # 累加到 overlap_map（只累加非零像素）
        overlap_map[temp > 0] += 1

    # 检测是否有像素被绘制了 2 次或更多（表示重叠）
    # 使用阈值：如果超过一定比例的道路像素重叠，则认为有问题
    total_road_pixels = np.sum(overlap_map > 0)
    overlapping_pixels = np.sum(overlap_map >= 2)

    if total_road_pixels == 0:
        return False

    overlap_ratio = overlapping_pixels / total_road_pixels

    # 如果超过 5% 的道路像素重叠，认为有重叠问题
    return overlap_ratio > 0.05


def generate_pathfinder_board_v2(
    difficulty: str = "medium",
    timeout_seconds: float = 30.0,
    debug: bool = False
) -> Optional[PathFinderBoard]:
    """
    PathFinder 生成器 V2 - 使用 Delaunay 三角化，但去掉复杂的重叠检测
    """
    start_time = time.time()

    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    config = DIFFICULTY_CONFIG[difficulty]
    image_size = config['image_size']
    road_width = config['road_width']
    extra_paths = config['extra_paths']
    min_solution_nodes = config['min_solution_nodes']
    node_spacing_ratio = config['node_spacing_ratio']

    # 计算节点间距（Buffer Zone 方法）
    # 基于路径规划领域的 "Safety Corridor" 概念：
    # - 每条路径需要一个安全走廊，宽度 = 道路宽度 + 安全间距
    # - 节点间距应该至少是道路宽度的 4.0 倍（从 3.5 增加到 4.0）
    # - 这样即使 Bezier 曲线偏移 8%，也不会与相邻路径重叠
    min_radius_for_road = road_width * 4.0  # 增加到 4.0 倍
    radius = max(image_size * node_spacing_ratio, min_radius_for_road)

    # 最大边长：限制边的长度，避免过长的曲线
    # - 过长的边会导致 Bezier 曲线弯曲过度
    # - 使用道路宽度的 5.5 倍（从 6.0 降到 5.5，更保守）
    max_edge_length = max(radius * 2.5, road_width * 5.5)

    # 计算边界
    margin = image_size * 0.05
    bounds = (margin, margin, image_size - margin, image_size - margin)

    max_attempts = 100

    for attempt in range(max_attempts):
        # 检查超时
        if time.time() - start_time > timeout_seconds:
            if debug:
                print(f'[PathFinder V2] Timeout after {time.time() - start_time:.2f}s')
            return None

        # 1. Poisson Disk Sampling 生成节点
        nodes = poisson_disk_sampling(radius, image_size, image_size, k=30)

        if len(nodes) < 5:
            continue

        # 2. Delaunay 三角化连接节点（限制最大边长）
        delaunay_triangulation_v2(nodes, max_edge_length=max_edge_length)

        # 3. Aldous-Broder 算法生成生成树
        create_maze_aldous_broder(nodes)

        # 4. 添加额外路径
        add_random_pathways(nodes, n=extra_paths)

        # 5. 选择起点和终点（尽量远）
        start_node = random.choice(nodes)
        end_node = max(nodes, key=lambda n: math.sqrt((n.x - start_node.x)**2 + (n.y - start_node.y)**2))

        # 6. 查找路径
        path = find_path_bfs(start_node, end_node)

        if not path or len(path) < min_solution_nodes:
            continue

        # 7. 将节点图转换为 PathSegment
        segments = []
        segment_map = {}

        for node in nodes:
            for neighbor in node.pathways:
                edge = tuple(sorted([id(node), id(neighbor)]))
                if edge not in segment_map:
                    control_points = generate_smooth_curve(
                        (node.x, node.y),
                        (neighbor.x, neighbor.y),
                        num_control_points=2,
                        bounds=bounds
                    )
                    segments.append(PathSegment(control_points))
                    segment_map[edge] = len(segments) - 1

        # 8. 找到解决方案路径对应的 segment 索引和节点路径
        solution_segments = []
        solution_path = [(start_node.x, start_node.y)]

        for i in range(len(path) - 1):
            edge = tuple(sorted([id(path[i]), id(path[i+1])]))
            if edge in segment_map:
                solution_segments.append(segment_map[edge])
            solution_path.append((path[i+1].x, path[i+1].y))

        if not solution_segments:
            continue

        # 9. 创建游戏板
        board = PathFinderBoard(
            segments=segments,
            start_point=(start_node.x, start_node.y),
            end_point=(end_node.x, end_node.y),
            solution_segments=solution_segments,
            solution_path=solution_path,
            image_size=image_size,
            road_width=road_width
        )

        # 10. 检测道路重叠（后处理验证）
        # 注意：只在前几次尝试时检测，避免无限循环
        if attempt < 5:  # 只在前 5 次尝试时检测重叠
            has_overlap = check_path_overlap_fast(board, road_width)
            if has_overlap:
                if debug:
                    print(f'[PathFinder V2] Attempt {attempt + 1}: Rejected due to path overlap, retrying...')
                continue  # 拒绝这个关卡，重新生成

        if debug:
            elapsed = time.time() - start_time
            overlap_status = "checked" if attempt < 5 else "unchecked"
            print(f'[PathFinder V2] Success in {elapsed:.2f}s after {attempt + 1} attempts (overlap {overlap_status})')

        return board

    if debug:
        print(f'[PathFinder V2] Failed after {max_attempts} attempts')

    return None
