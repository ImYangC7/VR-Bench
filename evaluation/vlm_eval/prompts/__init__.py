from .maze_prompt import MAZE_SYSTEM_PROMPT, MAZE_USER_PROMPT
from .sokoban_prompt import SOKOBAN_SYSTEM_PROMPT, SOKOBAN_USER_PROMPT
from .trapfield_prompt import TRAPFIELD_SYSTEM_PROMPT, TRAPFIELD_USER_PROMPT
from .pathfinder_prompt import PATHFINDER_SYSTEM_PROMPT, PATHFINDER_USER_PROMPT
from .maze3d_prompt import MAZE3D_SYSTEM_PROMPT, MAZE3D_USER_PROMPT

PROMPTS = {
    'maze': {
        'system': MAZE_SYSTEM_PROMPT,
        'user': MAZE_USER_PROMPT,
    },
    'sokoban': {
        'system': SOKOBAN_SYSTEM_PROMPT,
        'user': SOKOBAN_USER_PROMPT,
    },
    'trapfield': {
        'system': TRAPFIELD_SYSTEM_PROMPT,
        'user': TRAPFIELD_USER_PROMPT,
    },
    'pathfinder': {
        'system': PATHFINDER_SYSTEM_PROMPT,
        'user': PATHFINDER_USER_PROMPT,
    },
    '3dmaze': {
        'system': MAZE3D_SYSTEM_PROMPT,
        'user': MAZE3D_USER_PROMPT,
    },
    'maze3d': {
        'system': MAZE3D_SYSTEM_PROMPT,
        'user': MAZE3D_USER_PROMPT,
    }
}


def get_prompt(game_name: str, prompt_type: str = 'system') -> str:
    if game_name not in PROMPTS:
        raise ValueError(f"Unknown game: {game_name}")
    if prompt_type not in PROMPTS[game_name]:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return PROMPTS[game_name][prompt_type]

