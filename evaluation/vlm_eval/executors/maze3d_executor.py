"""
3D Maze 游戏执行器
"""

import copy
import threading
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.schema import UnifiedState
from evaluation.vlm_eval.game_executor import GameExecutor
from evaluation.vlm_eval.prompts import get_prompt
from games.maze3d.main import Position, Ladder, PathSegment, BasePuzzleState, draw_puzzle

# 全局渲染锁，防止多线程并发渲染时的资源竞争
# matplotlib 不是线程安全的，多线程同时渲染会导致图片混乱
_render_lock = threading.Lock()


def rebuild_puzzle_from_state(state: UnifiedState) -> BasePuzzleState:
    """从 UnifiedState 重建 BasePuzzleState 对象
    
    将 metadata 中序列化的数据转换回原版的 puzzle 对象
    """
    metadata = state.metadata
    
    cubes = set(Position(*c) for c in metadata['cubes'])
    start_pos = Position(*metadata['start_pos'])
    goal_pos = Position(*metadata['goal_pos'])
    
    ladders = [
        Ladder(Position(*ld['base_pos']), ld['direction'], ld['height'])
        for ld in metadata.get('ladders', [])
    ]
    
    path = [
        PathSegment(Position(*seg['start']), Position(*seg['end']), seg['type'])
        for seg in metadata.get('path', [])
    ]
    
    return BasePuzzleState(
        grid_size=tuple(metadata['grid_size']),
        cubes=cubes,
        start_pos=start_pos,
        goal_pos=goal_pos,
        ladders=ladders,
        path=path
    )


class Maze3DExecutor(GameExecutor):
    """3D Maze 游戏执行器
    
    移动规则：
    - 水平移动：跨越 2 个网格单位，需要中间位置和目标位置都有立方体
    - 垂直移动：通过梯子上下移动 3 个单位
    """
    
    DIRECTION_DELTAS = {
        'forward_right': (-2, 0, 0),
        'forward_left': (0, -2, 0),
        'backward_left': (2, 0, 0),
        'backward_right': (0, 2, 0),
        'up': (0, 0, 3),
        'down': (0, 0, -3),
    }
    
    DELTA_TO_DIRECTION = {
        (-2, 0, 0): 'forward_right',
        (0, -2, 0): 'forward_left',
        (2, 0, 0): 'backward_left',
        (0, 2, 0): 'backward_right',
        (0, 0, 3): 'up',
        (0, 0, -3): 'down',
    }
    
    def __init__(self, assets_folder: str = None):
        self.assets_folder = assets_folder
    
    def load_state(self, state_path: str) -> UnifiedState:
        """加载游戏状态"""
        state = UnifiedState.load(state_path)
        
        # 构建立方体集合用于快速查找
        if 'cubes' in state.metadata:
            state.metadata['_cube_set'] = set(tuple(c) for c in state.metadata['cubes'])
        
        # 构建梯子映射
        if 'ladders' in state.metadata:
            ladder_map = {}
            for ladder in state.metadata['ladders']:
                base = tuple(ladder['base_pos'])
                height = ladder['height']
                top = (base[0], base[1], base[2] + height)
                ladder_map[base] = top
                ladder_map[top] = base
            state.metadata['_ladder_map'] = ladder_map
        
        # 初始化当前位置
        if 'current_pos' not in state.metadata:
            state.metadata['current_pos'] = state.metadata['start_pos']
        
        return state
    
    def get_optimal_solution(self, state: UnifiedState) -> List[List[Dict[str, Any]]]:
        """从 metadata.path 提取最优解"""
        path_segments = state.metadata.get('path', [])
        if not path_segments:
            return [[]]
        
        actions = []
        for seg in path_segments:
            delta = (
                seg['end'][0] - seg['start'][0],
                seg['end'][1] - seg['start'][1],
                seg['end'][2] - seg['start'][2]
            )
            direction = self.DELTA_TO_DIRECTION.get(delta)
            if direction:
                actions.append({'action': 'move', 'direction': direction})
        
        return [actions]
    
    def execute_action(self, state: UnifiedState, action: Dict[str, Any]) -> Tuple[UnifiedState, bool, str]:
        """执行移动动作"""
        if action.get('action') != 'move':
            return state, False, f"Invalid action type: {action.get('action')}"
        
        direction = action.get('direction')
        if direction not in self.DIRECTION_DELTAS:
            return state, False, f"Invalid direction: {direction}"
        
        current_pos = tuple(state.metadata.get('current_pos', state.metadata['start_pos']))
        delta = self.DIRECTION_DELTAS[direction]
        new_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1], current_pos[2] + delta[2])
        
        cube_set = state.metadata.get('_cube_set', set(tuple(c) for c in state.metadata['cubes']))
        
        # 检查目标位置
        if new_pos not in cube_set:
            return state, False, f"No cube at position {new_pos}"
        
        # 水平移动：检查中间位置
        if direction in ['forward_left', 'forward_right', 'backward_left', 'backward_right']:
            mid_pos = (current_pos[0] + delta[0]//2, current_pos[1] + delta[1]//2, current_pos[2])
            if mid_pos not in cube_set:
                return state, False, f"No intermediate cube at {mid_pos}"
        
        # 垂直移动：检查梯子
        elif direction in ['up', 'down']:
            ladder_map = state.metadata.get('_ladder_map', {})
            if current_pos not in ladder_map:
                return state, False, f"No ladder at {current_pos}"
            expected_pos = ladder_map[current_pos]
            if new_pos != expected_pos:
                return state, False, f"Ladder leads to {expected_pos}, not {new_pos}"
        
        # 更新状态
        new_state = copy.deepcopy(state)
        new_state.metadata['current_pos'] = list(new_pos)
        
        return new_state, True, f"Moved {direction} to {new_pos}"
    
    def check_win(self, state: UnifiedState) -> bool:
        """检查是否到达终点"""
        current_pos = tuple(state.metadata.get('current_pos', state.metadata['start_pos']))
        goal_pos = tuple(state.metadata['goal_pos'])
        return current_pos == goal_pos
    
    def render_state(self, state: UnifiedState, output_path: str) -> None:
        """渲染游戏状态

        使用全局锁确保多线程环境下渲染的线程安全性。
        matplotlib 不是线程安全的，必须串行化渲染操作。
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 重建 puzzle 对象
        puzzle = rebuild_puzzle_from_state(state)

        # 获取当前玩家位置
        current_pos = state.metadata.get('current_pos', state.metadata['start_pos'])
        player_pos = Position(*current_pos)

        # 使用锁保护渲染操作，防止多线程并发渲染导致的图片混乱
        with _render_lock:
            draw_puzzle(puzzle, output_path, player_pos=player_pos)
    
    def get_system_prompt(self) -> str:
        return get_prompt('3dmaze', 'system')
    
    def get_user_prompt(self) -> str:
        return get_prompt('3dmaze', 'user')
    
    def get_game_type(self) -> str:
        return 'maze3d'
