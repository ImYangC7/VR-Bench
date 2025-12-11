# -*- coding: utf-8 -*-
"""
Video Model Prompts 模块。

提供基于皮肤配置的动态 prompt 生成功能。
"""
import json
from pathlib import Path
from typing import Optional

from .videomodel_maze_prompt import get_maze_prompt
from .videomodel_maze3d_prompt import get_maze3d_prompt
from .videomodel_sokoban_prompt import get_sokoban_prompt
from .videomodel_trapfield_prompt import get_trapfield_prompt
from .videomodel_pathfinder_prompt import get_pathfinder_prompt

# 游戏类型别名映射
GAME_ALIASES = {
    "irregular_maze": "pathfinder",
    "regular_maze": "maze",
    "3d_maze": "maze3d",
}

# 游戏类型到 prompt 生成函数的映射
PROMPT_GENERATORS = {
    "maze": get_maze_prompt,
    "maze3d": get_maze3d_prompt,
    "sokoban": get_sokoban_prompt,
    "trapfield": get_trapfield_prompt,
    "pathfinder": get_pathfinder_prompt,
}


def load_skin_description(skins_root: Path, game_type: str, skin_id: str) -> Optional[dict]:
    """
    加载皮肤的 description.json 文件。
    
    Args:
        skins_root: skins 目录的根路径
        game_type: 游戏类型 (maze, maze3d, sokoban, trapfield, pathfinder)
        skin_id: 皮肤 ID (1, 2, 3, ...)
    
    Returns:
        description.json 的内容，或 None（如果文件不存在）
    """
    # 处理游戏类型别名
    canonical_game_type = GAME_ALIASES.get(game_type, game_type)
    
    desc_path = skins_root / canonical_game_type / skin_id / "description.json"
    
    if not desc_path.exists():
        return None
    
    with open(desc_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dynamic_prompt(
    game_type: str,
    skin_id: str,
    skins_root: Optional[Path] = None,
) -> str:
    """
    根据游戏类型和皮肤 ID 生成动态 prompt。
    
    Args:
        game_type: 游戏类型 (maze, maze3d, sokoban, trapfield, pathfinder, irregular_maze, regular_maze)
        skin_id: 皮肤 ID
        skins_root: skins 目录的根路径，默认为 VR-Bench/skins
    
    Returns:
        生成的 prompt 字符串
    
    Raises:
        ValueError: 如果游戏类型不支持或找不到皮肤描述文件
    """
    # 处理游戏类型别名
    canonical_game_type = GAME_ALIASES.get(game_type, game_type)
    
    # 检查游戏类型是否支持
    if canonical_game_type not in PROMPT_GENERATORS:
        raise ValueError(f"Unsupported game type: {game_type}")
    
    # 确定 skins 目录路径
    if skins_root is None:
        # 默认路径: VR-Bench/skins (相对于此文件)
        skins_root = Path(__file__).parent.parent / "skins"
    
    # 加载皮肤描述
    description = load_skin_description(skins_root, canonical_game_type, skin_id)
    
    if description is None:
        raise ValueError(
            f"Skin description not found: skins/{canonical_game_type}/{skin_id}/description.json"
        )
    
    visual_description = description.get("visual_description", {})
    
    if not visual_description:
        raise ValueError(
            f"visual_description is empty in skins/{canonical_game_type}/{skin_id}/description.json"
        )
    
    # 生成 prompt
    generator = PROMPT_GENERATORS[canonical_game_type]
    return generator(visual_description)


__all__ = [
    "get_dynamic_prompt",
    "load_skin_description",
    "GAME_ALIASES",
    "PROMPT_GENERATORS",
]

