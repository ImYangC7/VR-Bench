"""Sokoban游戏执行器"""
import copy
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.constants import (
    SOKOBAN_EMPTY, SOKOBAN_WALL, SOKOBAN_BOX, SOKOBAN_TARGET,
    SOKOBAN_BOX_ON_TARGET, SOKOBAN_PLAYER, SOKOBAN_PLAYER_ON_TARGET
)
from core.schema import UnifiedState
from evaluation.vlm_eval.game_executor import GameExecutor
from evaluation.vlm_eval.prompts import get_prompt


class SokobanExecutor(GameExecutor):
    """
    Sokoban游戏执行器

    处理Sokoban特有的逻辑：
    - 玩家移动
    - 推箱子（当玩家移动到箱子位置时，箱子会被推动）
    - 胜利条件：箱子到达目标位置
    """

    def __init__(self, assets_folder: str = None):
        self.assets_folder = assets_folder

    def load_state(self, state_path: str) -> UnifiedState:
        """加载UnifiedState"""
        return UnifiedState.load(state_path)

    def get_optimal_solution(self, state: UnifiedState) -> List[List[Dict[str, Any]]]:
        """
        获取Sokoban的最优解

        使用Solution类的minPushBox方法求解，返回Move序列
        然后转换为action序列
        """
        from games.sokoban.board import Solution

        # 将UnifiedState的grid转换为Solution需要的字符网格格式
        grid_chars = self._state_to_solver_grid(state)

        # 使用Solution求解
        solution = Solution()
        total_moves, _ = solution.minPushBox(grid_chars)

        if total_moves == -1:
            # 无解
            return []

        # 获取Move序列
        solution_moves = solution.get_solution_path()

        # 转换为action序列
        actions = self._moves_to_actions(solution_moves)

        # 返回单个解决方案的列表（Sokoban通常只有一个最优解）
        return [actions] if actions else []
    def execute_action(
        self, state: UnifiedState, action: Dict[str, Any]
    ) -> Tuple[UnifiedState, bool, str]:
        """
        执行一个动作

        处理：
        1. 玩家移动
        2. 推箱子（如果玩家移动到箱子位置）
        3. 更新grid状态
        """
        from core.schema.entity import Entity

        if action.get('action') != 'move':
            return state, False, f"Invalid action type: {action.get('action')}"

        direction = action.get('direction')
        if direction not in ['up', 'down', 'left', 'right']:
            return state, False, f"Invalid direction: {direction}"

        current_pos = state.player.grid_pos
        new_pos = self._calculate_new_position(current_pos, direction)

        grid = state.grid.data
        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        # 检查边界
        if not (0 <= new_pos.row < rows and 0 <= new_pos.col < cols):
            return state, False, "Out of bounds"

        cell_value = grid[new_pos.row][new_pos.col]

        # 检查墙
        if cell_value == SOKOBAN_WALL:
            return state, False, "Hit wall"

        # 检查是否推箱子
        is_pushing_box = cell_value in [SOKOBAN_BOX, SOKOBAN_BOX_ON_TARGET]

        if is_pushing_box:
            # 计算箱子的新位置
            box_new_pos = self._calculate_new_position(new_pos, direction)

            # 检查箱子新位置是否有效
            if not (0 <= box_new_pos.row < rows and 0 <= box_new_pos.col < cols):
                return state, False, "Cannot push box out of bounds"

            box_new_cell = grid[box_new_pos.row][box_new_pos.col]

            # 箱子不能推到墙或另一个箱子上
            if box_new_cell in [SOKOBAN_WALL, SOKOBAN_BOX, SOKOBAN_BOX_ON_TARGET]:
                return state, False, "Cannot push box"

        # 创建新状态
        new_state = copy.deepcopy(state)
        cell_size = state.render.cell_size

        # 更新玩家位置
        new_state.player = Entity.from_grid_pos(new_pos.row, new_pos.col, cell_size)

        # 更新grid
        new_grid = [list(row) for row in grid]

        # 清除旧的玩家位置
        if new_grid[current_pos.row][current_pos.col] == SOKOBAN_PLAYER_ON_TARGET:
            new_grid[current_pos.row][current_pos.col] = SOKOBAN_TARGET
        else:
            new_grid[current_pos.row][current_pos.col] = SOKOBAN_EMPTY

        if is_pushing_box:
            # 推箱子
            box_new_pos = self._calculate_new_position(new_pos, direction)

            # 更新箱子新位置
            if new_grid[box_new_pos.row][box_new_pos.col] == SOKOBAN_TARGET:
                new_grid[box_new_pos.row][box_new_pos.col] = SOKOBAN_BOX_ON_TARGET
            else:
                new_grid[box_new_pos.row][box_new_pos.col] = SOKOBAN_BOX

            # 更新箱子旧位置（玩家移动到这里）
            if cell_value == SOKOBAN_BOX_ON_TARGET:
                new_grid[new_pos.row][new_pos.col] = SOKOBAN_PLAYER_ON_TARGET
            else:
                new_grid[new_pos.row][new_pos.col] = SOKOBAN_PLAYER

            # 更新boxes列表
            new_state.boxes = [
                Entity.from_grid_pos(box_new_pos.row, box_new_pos.col, cell_size)
            ]
        else:
            # 不推箱子，只移动玩家
            if new_grid[new_pos.row][new_pos.col] == SOKOBAN_TARGET:
                new_grid[new_pos.row][new_pos.col] = SOKOBAN_PLAYER_ON_TARGET
            else:
                new_grid[new_pos.row][new_pos.col] = SOKOBAN_PLAYER

        new_state.grid.data = new_grid

        return new_state, True, "OK"
    def check_win(self, state: UnifiedState) -> bool:
        """
        检查胜利条件：箱子是否在目标位置上
        """
        if not state.boxes:
            return False

        # 检查第一个箱子是否在目标位置
        box = state.boxes[0]
        goal = state.goal

        return (box.grid_pos.row == goal.grid_pos.row and
                box.grid_pos.col == goal.grid_pos.col)

    def render_state(self, state: UnifiedState, output_path: str) -> None:
        """渲染状态到图片"""
        import numpy as np
        from games.sokoban.textured_board import TexturedSokobanBoard
        from games.sokoban.renderer import get_shared_renderer

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 转换grid为numpy数组
        grid_array = np.array(state.grid.data, dtype=int)

        # 创建临时board对象（不传assets_folder参数）
        board = TexturedSokobanBoard(
            grid=grid_array,
            player_x=state.player.grid_pos.col,
            player_y=state.player.grid_pos.row
        )

        # 使用共享renderer渲染
        renderer = get_shared_renderer(self.assets_folder)
        renderer.render_board(board, output_path=output_path)

    def get_system_prompt(self) -> str:
        """获取系统提示"""
        return get_prompt('sokoban', 'system')

    def get_user_prompt(self) -> str:
        """获取用户提示"""
        return get_prompt('sokoban', 'user')

    def get_game_type(self) -> str:
        """获取游戏类型"""
        return 'sokoban'
    def _state_to_solver_grid(self, state: UnifiedState) -> List[List[str]]:
        """
        将UnifiedState的grid转换为Solution需要的字符网格格式

        格式：
        - '#': 墙
        - '.': 空地
        - 'S': 玩家
        - 'B': 箱子
        - 'T': 目标
        """
        grid = state.grid.data
        grid_chars = []

        for row in grid:
            char_row = []
            for cell in row:
                if cell == SOKOBAN_WALL:
                    char_row.append('#')
                elif cell in [SOKOBAN_PLAYER, SOKOBAN_PLAYER_ON_TARGET]:
                    char_row.append('S')
                elif cell in [SOKOBAN_BOX, SOKOBAN_BOX_ON_TARGET]:
                    char_row.append('B')
                elif cell in [SOKOBAN_TARGET]:
                    char_row.append('T')
                else:
                    char_row.append('.')
            grid_chars.append(char_row)

        return grid_chars

    def _moves_to_actions(self, moves: List) -> List[Dict[str, Any]]:
        """
        将Solution的Move序列转换为action序列

        Move对象包含：
        - is_push: 是否是推箱子
        - start_pos: 起始位置 (row, col)
        - end_pos: 结束位置 (row, col)
        - box_start: 箱子起始位置（如果是推箱子）
        - box_end: 箱子结束位置（如果是推箱子）
        """
        actions = []

        for move in moves:
            # 计算方向
            start_row, start_col = move.start_pos
            end_row, end_col = move.end_pos

            if end_row < start_row:
                direction = 'up'
            elif end_row > start_row:
                direction = 'down'
            elif end_col < start_col:
                direction = 'left'
            else:
                direction = 'right'

            actions.append({'action': 'move', 'direction': direction})

        return actions

    def _calculate_new_position(self, pos, direction: str):
        """计算新位置"""
        from core.schema.position import Position

        if direction == 'up':
            return Position(row=pos.row - 1, col=pos.col)
        if direction == 'down':
            return Position(row=pos.row + 1, col=pos.col)
        if direction == 'left':
            return Position(row=pos.row, col=pos.col - 1)
        return Position(row=pos.row, col=pos.col + 1)
