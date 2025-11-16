import string
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from core.schema import UnifiedState
from evaluation.vlm_eval.game_executor import GameExecutor
from evaluation.vlm_eval.prompts.pathfinder_prompt import PATHFINDER_SYSTEM_PROMPT, PATHFINDER_USER_PROMPT
from games.pathfinder.board import PathFinderBoard
from games.pathfinder.renderer import render_pathfinder_board, bezier_curve_opencv
from games.pathfinder.constants import START_COLOR, END_COLOR, ROAD_COLOR, BG_COLOR
from games.pathfinder.texture_handler import get_texture_handler


class PathfinderExecutor(GameExecutor):
    """PathFinder 游戏执行器
    
    PathFinder 是一个路径查找游戏，玩家需要找到从起点到终点的最短路径。
    为了简化 VLM 的判定，我们在每个路径节点上标注字母（A, B, C, D...），
    VLM 输出一个字母序列表示路径。
    """
    
    def __init__(self, assets_folder: str = None):
        self.assets_folder = assets_folder
    
    def load_state(self, state_path: str) -> UnifiedState:
        """加载游戏状态"""
        return UnifiedState.load(state_path)
    
    def _extract_all_nodes(self, state: UnifiedState) -> List[Tuple[float, float]]:
        """从 segments 中提取所有唯一节点（排除起点和终点）

        Args:
            state: 游戏状态

        Returns:
            所有唯一节点的列表（去重后，按坐标排序）
        """
        metadata = state.metadata
        segments = metadata.get('segments', [])

        # 从 player 和 goal 获取起点和终点
        start_point = state.player.pixel_pos
        end_point = state.goal.pixel_pos

        # 收集所有节点（segment 的起点和终点）
        all_nodes = set()
        for seg_points in segments:
            # 每个 segment 是一个控制点列表，第一个是起点，最后一个是终点
            if seg_points:
                all_nodes.add(tuple(seg_points[0]))
                all_nodes.add(tuple(seg_points[-1]))

        # 移除起点和终点（使用近似匹配，因为可能有浮点误差）
        threshold = 5.0
        nodes_to_remove = set()
        for node in all_nodes:
            dist_to_start = ((node[0] - start_point[0]) ** 2 + (node[1] - start_point[1]) ** 2) ** 0.5
            dist_to_end = ((node[0] - end_point[0]) ** 2 + (node[1] - end_point[1]) ** 2) ** 0.5
            if dist_to_start < threshold or dist_to_end < threshold:
                nodes_to_remove.add(node)

        all_nodes -= nodes_to_remove

        # 转换为列表并排序（保证顺序一致性）
        nodes_list = sorted(list(all_nodes))

        return nodes_list

    def _build_node_to_letter_map(self, all_nodes: List[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
        """建立节点到字母的映射

        Args:
            all_nodes: 所有节点列表

        Returns:
            节点坐标到字母的映射字典
        """
        node_to_letter = {}
        for i, node in enumerate(all_nodes):
            if i < 26:
                letter = string.ascii_uppercase[i]
            else:
                # 超过26个节点，使用 AA, AB, AC...
                first = string.ascii_uppercase[(i // 26) - 1]
                second = string.ascii_uppercase[i % 26]
                letter = first + second
            node_to_letter[node] = letter

        return node_to_letter

    def _find_closest_node(self, point: Tuple[float, float], nodes: List[Tuple[float, float]],
                          threshold: float = 5.0) -> Optional[Tuple[float, float]]:
        """找到最接近给定点的节点

        Args:
            point: 目标点坐标
            nodes: 节点列表
            threshold: 距离阈值

        Returns:
            最接近的节点，如果没有足够接近的节点则返回 None
        """
        min_dist = float('inf')
        closest_node = None

        for node in nodes:
            dist = ((point[0] - node[0]) ** 2 + (point[1] - node[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        if min_dist <= threshold:
            return closest_node
        return None

    def get_optimal_solution(self, state: UnifiedState) -> List[List[Dict[str, Any]]]:
        """获取最优解

        PathFinder 的解决方案存储在 metadata 中的 solution_path 字段。
        我们将节点路径转换为字母数组。

        Returns:
            包含一个解决方案的列表，每个解决方案是一个字母数组
        """
        metadata = state.metadata
        solution_path = metadata.get('solution_path', [])

        if not solution_path or len(solution_path) < 2:
            return [[]]

        # 提取所有节点
        all_nodes = self._extract_all_nodes(state)

        # 建立节点到字母的映射
        node_to_letter = self._build_node_to_letter_map(all_nodes)

        # 将解决方案路径转换为字母数组
        # 跳过起点（第一个）和终点（最后一个）
        letters = []
        for i in range(1, len(solution_path) - 1):
            point = tuple(solution_path[i])
            # 找到最接近的节点
            closest_node = self._find_closest_node(point, all_nodes)
            if closest_node and closest_node in node_to_letter:
                letters.append(node_to_letter[closest_node])

        # 返回格式：[{"action": "path", "path": ["A", "C", "D"]}]
        return [[{"action": "path", "path": letters}]]
    
    def execute_action(self, state: UnifiedState, action: Dict[str, Any]) -> Tuple[UnifiedState, bool, str]:
        """执行动作

        PathFinder 游戏中，动作是一个字母数组，表示经过的路径节点。
        我们验证这个数组是否能到达终点（前缀匹配最优解）。

        Args:
            state: 当前状态
            action: 动作字典，格式为 {"action": "path", "path": ["A", "C", "D"]}

        Returns:
            (新状态, 是否成功, 消息)
        """
        if action.get('action') != 'path':
            return state, False, f"Invalid action type: {action.get('action')}"

        path = action.get('path', [])
        if not isinstance(path, list):
            return state, False, f"Invalid path type: {type(path)}, expected list"

        # 获取最优解
        optimal_solutions = self.get_optimal_solution(state)
        if not optimal_solutions or not optimal_solutions[0]:
            return state, False, "No solution available"

        optimal_path = optimal_solutions[0][0].get('path', [])

        # 检查是否是最优解的前缀（允许多走几步）
        # 只要前 len(optimal_path) 个节点匹配，就认为能到达终点
        if len(path) >= len(optimal_path):
            if path[:len(optimal_path)] == optimal_path:
                # 创建新状态并标记为成功
                new_state = state
                new_state._path_correct = True
                new_state._predicted_path = path
                new_state._optimal_path = optimal_path

                if path == optimal_path:
                    return new_state, True, "Path is optimal"
                else:
                    extra_steps = len(path) - len(optimal_path)
                    return new_state, True, f"Path is correct but with {extra_steps} extra steps"
            else:
                # 路径错误，不设置 _path_correct
                new_state = state
                new_state._path_correct = False
                return new_state, False, f"Path diverges from optimal. Expected prefix: {optimal_path}, Got: {path[:len(optimal_path)]}"
        else:
            # 路径太短，不设置 _path_correct
            new_state = state
            new_state._path_correct = False
            # 检查是否是正确的前缀
            if optimal_path[:len(path)] == path:
                return new_state, False, f"Path is incomplete. Got {len(path)} steps, need {len(optimal_path)}"
            else:
                return new_state, False, f"Path is incorrect. Expected: {optimal_path}, Got: {path}"
    
    def check_win(self, state: UnifiedState) -> bool:
        """检查是否获胜

        PathFinder 游戏中，我们通过验证路径序列来判断是否获胜。
        在 execute_action 中会设置 _path_correct 标记。
        """
        return getattr(state, '_path_correct', False)
    
    def render_state(self, state: UnifiedState, output_path: str = None) -> str:
        """渲染游戏状态，在路径节点上标注字母
        
        Args:
            state: 游戏状态
            output_path: 输出路径
        
        Returns:
            输出路径
        """
        # 从 state 重建 PathFinderBoard
        metadata = state.metadata

        from games.pathfinder.board import PathSegment

        segments = [PathSegment(points) for points in metadata['segments']]
        board = PathFinderBoard(
            segments=segments,
            start_point=state.player.pixel_pos,
            end_point=state.goal.pixel_pos,
            solution_segments=metadata['solution_segments'],
            solution_path=[tuple(pt) for pt in metadata['solution_path']],
            image_size=state.render.image_width,
            road_width=metadata['road_width']
        )
        
        # 渲染带标签的图像
        img = self._render_with_labels(state, board, self.assets_folder)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            return output_path

        return None

    def _render_with_labels(self, state: UnifiedState, board: PathFinderBoard,
                           assets_folder: str = None) -> Image.Image:
        """渲染游戏板并在所有路径节点上标注字母

        Args:
            state: 游戏状态
            board: 游戏板对象
            assets_folder: 资源文件夹路径

        Returns:
            渲染后的 PIL Image
        """
        # 先使用原有的渲染函数渲染基础图像
        img = render_pathfinder_board(board, assets_folder=assets_folder)

        # 提取所有节点
        all_nodes = self._extract_all_nodes(state)
        node_to_letter = self._build_node_to_letter_map(all_nodes)

        if not all_nodes:
            return img

        # 创建一个半透明层用于绘制标签
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 尝试加载字体 - 使用更小的字体
        try:
            font_size = max(14, int(board.road_width * 0.6))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            # 如果加载失败，使用默认字体
            font = ImageFont.load_default()

        # 在所有节点上标注字母（除了起点和终点）
        for node, letter in node_to_letter.items():
            x, y = node

            # 绘制半透明白色背景圆圈 - 使用更小的圆圈
            circle_radius = int(board.road_width * 0.35)
            # 使用半透明白色 (255, 255, 255, 200) - alpha=200 表示约78%不透明度
            draw.ellipse(
                [(x - circle_radius, y - circle_radius),
                 (x + circle_radius, y + circle_radius)],
                fill=(255, 255, 255, 200),
                outline=(0, 0, 0, 255),
                width=1
            )

            # 绘制字母 - 使用 anchor='mm' 实现完美居中
            # 'mm' 表示 middle-middle，即文本的中心点对齐到指定坐标
            draw.text((x, y), letter, fill=(0, 0, 0, 255), font=font, anchor='mm')

        # 将半透明层合并到原图像
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')

        return img
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return PATHFINDER_SYSTEM_PROMPT
    
    def get_user_prompt(self) -> str:
        """获取用户提示词"""
        return PATHFINDER_USER_PROMPT
    
    def get_game_type(self) -> str:
        """获取游戏类型"""
        return "pathfinder"

