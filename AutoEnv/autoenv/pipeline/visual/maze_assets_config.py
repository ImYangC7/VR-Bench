"""
Maze Assets Configuration

"""

from typing import Any


# ==================== Asset Prompt Templates ====================

def get_asset_prompt_template(asset_type: str, theme: str) -> str:
    """
    根据资产类型和主题生成 prompt。
    
    Args:
        asset_type: 资产类型（player, wall, floor, target, box, trap, start, end, road, goal）
        theme: 视觉主题（如 "cyberpunk neon city"）
    
    Returns:
        完整的图像生成 prompt
    """
    # 基础描述映射
    base_descriptions = {
        "player": "character sprite, game avatar, movable entity",
        "wall": "solid barrier tile, impassable obstacle",
        "floor": "walkable ground tile, path surface",
        "target": "goal marker, destination indicator",
        "box": "pushable crate, movable box object",
        "trap": "danger zone tile, hazard area",
        "start": "starting point marker, origin indicator",
        "end": "endpoint marker, finish line",
        "road": "pathway tile, traversable route",
        "goal": "objective marker, target destination",
    }
    
    base_desc = base_descriptions.get(asset_type, asset_type)
    
    # 构建完整 prompt
    prompt = f"""A {base_desc} in {theme} style.
Art requirements:
- Top-down 2D game asset
- Square tile format (suitable for grid-based game)
- Clean, recognizable silhouette
- Centered in frame, filling 70-85% of canvas
- Solid white background
- NO glow effects, NO bloom, NO shadows extending beyond tile
- Sharp, clean edges suitable for pixel-perfect rendering
- Consistent art style with other game assets
- Game-ready quality"""
    
    return prompt


# ==================== Maze Type Configurations ====================

MAZE_ASSETS_CONFIG = {
    "maze": {
        "name": "Regular Maze",
        "description": "Classic maze with walls, player navigates to target",
        "assets": [
            {
                "id": "wall",
                "name": "Wall Tile",
                "type": "tile",
                "priority": 100,  # Style anchor
                "is_tileable": True,
                "description": "Impassable wall barrier",
            },
            {
                "id": "floor",
                "name": "Floor Tile",
                "type": "tile",
                "priority": 90,
                "is_tileable": True,
                "description": "Walkable ground path",
            },
            {
                "id": "player",
                "name": "Player Character",
                "type": "character",
                "priority": 80,
                "is_tileable": False,
                "description": "Player avatar sprite",
            },
            {
                "id": "target",
                "name": "Goal Target",
                "type": "object",
                "priority": 70,
                "is_tileable": False,
                "description": "Destination marker",
            },
        ],
        "style_anchor": "wall",
    },
    "pathfinder": {
        "name": "Pathfinder (Irregular Maze)",
        "description": "Irregular maze with start, road, and end points",
        "assets": [
            {
                "id": "road",
                "name": "Road Tile",
                "type": "tile",
                "priority": 100,  # Style anchor
                "is_tileable": True,
                "description": "Traversable pathway",
            },
            {
                "id": "start",
                "name": "Start Point",
                "type": "marker",
                "priority": 90,
                "is_tileable": False,
                "description": "Starting position marker",
            },
            {
                "id": "end",
                "name": "End Point",
                "type": "marker",
                "priority": 80,
                "is_tileable": False,
                "description": "Finish line marker",
            },
        ],
        "style_anchor": "road",
    },
    "sokoban": {
        "name": "Sokoban Puzzle",
        "description": "Box-pushing puzzle game",
        "assets": [
            {
                "id": "wall",
                "name": "Wall Tile",
                "type": "tile",
                "priority": 100,  # Style anchor
                "is_tileable": True,
                "description": "Impassable wall barrier",
            },
            {
                "id": "floor",
                "name": "Floor Tile",
                "type": "tile",
                "priority": 90,
                "is_tileable": True,
                "description": "Walkable ground path",
            },
            {
                "id": "player",
                "name": "Player Character",
                "type": "character",
                "priority": 80,
                "is_tileable": False,
                "description": "Player avatar sprite",
            },
            {
                "id": "box",
                "name": "Pushable Box",
                "type": "object",
                "priority": 70,
                "is_tileable": False,
                "description": "Movable crate object",
            },
            {
                "id": "target",
                "name": "Goal Target",
                "type": "object",
                "priority": 60,
                "is_tileable": False,
                "description": "Box destination marker",
            },
        ],
        "style_anchor": "wall",
    },
    "trapfield": {
        "name": "Trap Field",
        "description": "Navigate through trap-filled maze to goal",
        "assets": [
            {
                "id": "floor",
                "name": "Floor Tile",
                "type": "tile",
                "priority": 100,  # Style anchor
                "is_tileable": True,
                "description": "Safe walkable ground",
            },
            {
                "id": "trap",
                "name": "Trap Tile",
                "type": "tile",
                "priority": 90,
                "is_tileable": True,
                "description": "Dangerous hazard area",
            },
            {
                "id": "player",
                "name": "Player Character",
                "type": "character",
                "priority": 80,
                "is_tileable": False,
                "description": "Player avatar sprite",
            },
            {
                "id": "goal",
                "name": "Goal Marker",
                "type": "object",
                "priority": 70,
                "is_tileable": False,
                "description": "Objective destination",
            },
        ],
        "style_anchor": "floor",
    },
}


# ==================== Helper Functions ====================

def get_maze_config(maze_type: str) -> dict[str, Any]:
    """
    获取指定迷宫类型的配置。

    Args:
        maze_type: 迷宫类型（maze, pathfinder, sokoban, trapfield）

    Returns:
        迷宫配置字典

    Raises:
        ValueError: 如果迷宫类型不支持
    """
    # 处理别名
    aliases = {
        "irregular_maze": "pathfinder",
        "regular_maze": "maze",
    }
    canonical_type = aliases.get(maze_type, maze_type)

    if canonical_type not in MAZE_ASSETS_CONFIG:
        supported = ", ".join(MAZE_ASSETS_CONFIG.keys())
        raise ValueError(
            f"Unsupported maze type: {maze_type}. "
            f"Supported types: {supported}"
        )

    return MAZE_ASSETS_CONFIG[canonical_type]


def generate_strategy_for_maze(maze_type: str, theme: str) -> dict[str, Any]:
    """
    为指定迷宫类型和主题生成完整的资产生成策略。

    Args:
        maze_type: 迷宫类型
        theme: 视觉主题

    Returns:
        完整的 strategy.json 格式数据
    """
    config = get_maze_config(maze_type)

    assets = []
    for asset_def in config["assets"]:
        asset_id = asset_def["id"]
        prompt = get_asset_prompt_template(asset_id, theme)

        asset_entry = {
            "id": asset_id,
            "name": asset_def["name"],
            "type": asset_def["type"],
            "priority": asset_def["priority"],
            "is_tileable": asset_def["is_tileable"],
            "description": asset_def["description"],
            "prompt_strategy": {
                "base_prompt": prompt,
            },
        }

        # 第一个资产（style_anchor）用 text-to-image
        if asset_id == config["style_anchor"]:
            asset_entry["generation_method"] = "text-to-image"
            asset_entry["dependencies"] = []
        else:
            # 其他资产用 image-to-image
            asset_entry["generation_method"] = "image-to-image"
            asset_entry["dependencies"] = [config["style_anchor"]]
            asset_entry["reference_assets"] = [config["style_anchor"]]

        assets.append(asset_entry)

    return {
        "maze_type": maze_type,
        "theme": theme,
        "rendering_approach": {
            "type": "tilemap",
            "rationale": f"Grid-based {config['name']} game with tile-based rendering",
        },
        "style_anchor": config["style_anchor"],
        "assets": assets,
    }


def get_supported_maze_types() -> list[str]:
    """返回所有支持的迷宫类型列表。"""
    return list(MAZE_ASSETS_CONFIG.keys())

