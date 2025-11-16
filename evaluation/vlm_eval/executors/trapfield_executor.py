import copy
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.schema import UnifiedState
from evaluation.vlm_eval.game_executor import GameExecutor
from evaluation.vlm_eval.prompts import get_prompt
from games.trapfield import constants
from games.trapfield.renderer import TrapFieldRenderer
from generation.path_finder import find_trapfield_paths


class TrapFieldExecutor(GameExecutor):
    def __init__(self, assets_folder: str = None):
        self.assets_folder = assets_folder
    
    def load_state(self, state_path: str) -> UnifiedState:
        return UnifiedState.load(state_path)
    
    def get_optimal_solution(self, state: UnifiedState) -> List[List[Dict[str, Any]]]:
        all_paths = find_trapfield_paths(state)
        return [self._path_to_actions(path) for path in all_paths]
    
    def execute_action(self, state: UnifiedState, action: Dict[str, Any]) -> Tuple[UnifiedState, bool, str]:
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
        
        if not (0 <= new_pos.row < rows and 0 <= new_pos.col < cols):
            return state, False, "Out of bounds"
        
        cell_value = grid[new_pos.row][new_pos.col]
        
        if cell_value == constants.TRAP_CELL:
            return state, False, "Hit trap"
        
        new_state = copy.deepcopy(state)
        
        from core.schema.entity import Entity
        cell_size = state.render.cell_size
        new_state.player = Entity.from_grid_pos(new_pos.row, new_pos.col, cell_size)
        
        new_grid = [list(row) for row in grid]
        new_grid[current_pos.row][current_pos.col] = constants.EMPTY_CELL
        if new_grid[new_pos.row][new_pos.col] != constants.GOAL_CELL:
            new_grid[new_pos.row][new_pos.col] = constants.PLAYER_CELL
        else:
            new_grid[new_pos.row][new_pos.col] = constants.PLAYER_CELL
        
        new_state.grid.data = new_grid
        
        return new_state, True, "OK"
    
    def check_win(self, state: UnifiedState) -> bool:
        return (state.player.grid_pos.row == state.goal.grid_pos.row and 
                state.player.grid_pos.col == state.goal.grid_pos.col)
    
    def render_state(self, state: UnifiedState, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        renderer = TrapFieldRenderer(cell_size=state.render.cell_size, assets_folder=self.assets_folder)
        renderer.render_grid(state.grid.data, output_path)
    
    def get_system_prompt(self) -> str:
        return get_prompt('trapfield', 'system')
    
    def get_user_prompt(self) -> str:
        return get_prompt('trapfield', 'user')
    
    def get_game_type(self) -> str:
        return 'trapfield'
    
    def _path_to_actions(self, path: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        actions = []
        for i in range(len(path) - 1):
            curr_row, curr_col = path[i]
            next_row, next_col = path[i + 1]
            
            if next_row < curr_row:
                direction = 'up'
            elif next_row > curr_row:
                direction = 'down'
            elif next_col < curr_col:
                direction = 'left'
            else:
                direction = 'right'
            
            actions.append({'action': 'move', 'direction': direction})
        
        return actions
    
    def _calculate_new_position(self, pos, direction: str):
        from core.schema.position import Position
        
        if direction == 'up':
            return Position(row=pos.row - 1, col=pos.col)
        elif direction == 'down':
            return Position(row=pos.row + 1, col=pos.col)
        elif direction == 'left':
            return Position(row=pos.row, col=pos.col - 1)
        else:
            return Position(row=pos.row, col=pos.col + 1)

