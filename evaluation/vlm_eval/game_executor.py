from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from core.schema import UnifiedState


class GameExecutor(ABC):
    @abstractmethod
    def load_state(self, state_path: str) -> UnifiedState:
        pass
    
    @abstractmethod
    def get_optimal_solution(self, state: UnifiedState) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def execute_action(self, state: UnifiedState, action: Dict[str, Any]) -> Tuple[UnifiedState, bool, str]:
        pass
    
    @abstractmethod
    def check_win(self, state: UnifiedState) -> bool:
        pass
    
    @abstractmethod
    def render_state(self, state: UnifiedState, output_path: str) -> None:
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_user_prompt(self) -> str:
        pass

    def get_game_type(self) -> str:
        return 'default'

