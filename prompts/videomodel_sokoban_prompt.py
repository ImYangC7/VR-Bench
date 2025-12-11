# -*- coding: utf-8 -*-
"""
Sokoban 游戏的视频模型 prompt 模板。

占位符: {player}, {box}, {goal}, {wall}, {floor}
从 description.json 的 visual_description 中读取。
"""
from string import Template

SOKOBAN_PROMPT_TEMPLATE = Template("""Create a 2D animation based on the provided image of a grid puzzle.
The $player moves into position behind the $box and smoothly pushes it toward the $goal.
The $box only slides when pushed from behind by the $player and moves in a straight line along the $floor tiles.
When the direction of the $box's movement needs to change, the $player must reposition itself to a new side of the $box.
The $box never crosses or overlaps any $wall.

Gameplay Rules:
The floor area is $floor, and the walls are $wall.
The $box can only move when pushed by the $player from behind.
The $player cannot pull the $box or move through walls.
The $box slides smoothly in one direction until it reaches the $goal.
The animation stops perfectly when the $box aligns with the $goal.

Scene:
No change in grid layout or tile design.
The camera remains static, showing the entire play area.
The movement is smooth, with no speed variation, camera shake, or visual artifacts.""")


def get_sokoban_prompt(visual_description: dict) -> str:
    """
    生成 sokoban 游戏的动态 prompt。
    
    Args:
        visual_description: 来自 description.json 的 visual_description 字段
            - player: 玩家描述 (如 "blue circle")
            - box: 箱子描述 (如 "yellow square")
            - goal: 目标描述 (如 "pink square")
            - wall: 墙壁描述 (如 "gray square")
            - floor: 地板描述 (如 "white square")
    """
    return SOKOBAN_PROMPT_TEMPLATE.substitute(
        player=visual_description.get("player", "blue ball"),
        box=visual_description.get("box", "yellow square"),
        goal=visual_description.get("goal", "red square"),
        wall=visual_description.get("wall", "gray wall"),
        floor=visual_description.get("floor", "white floor"),
    )

