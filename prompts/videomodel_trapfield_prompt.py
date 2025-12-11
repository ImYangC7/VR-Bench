# -*- coding: utf-8 -*-
"""
Trapfield 游戏的视频模型 prompt 模板。

占位符: {player}, {goal}, {trap}, {floor}
从 description.json 的 visual_description 中读取。
"""
from string import Template

TRAPFIELD_PROMPT_TEMPLATE = Template("""Create a 2D animation based on the provided image of a maze. The $player slides smoothly along the $floor path, stopping perfectly on the $goal. The $player never slides into or crosses the $trap (trap areas). The camera is a static, top-down view showing the entire maze.

Maze:
 The maze paths are $floor, and the trap areas are $trap.
 The $player moves to the goal position, represented by the $goal.
 The $player slides smoothly along the $floor path.
 The $player never slides into or crosses the $trap of the maze.
 The $player stops perfectly on the $goal.

Scene:
 No change in scene composition.
 No change in the layout of the maze.
 The $player travels along the $floor path without speeding up or slowing down.

Camera:
 Static camera.
 No zoom.
 No pan.
 No glitches, noise, or artifacts.""")


def get_trapfield_prompt(visual_description: dict) -> str:
    """
    生成 trapfield 游戏的动态 prompt。
    
    Args:
        visual_description: 来自 description.json 的 visual_description 字段
            - player: 玩家描述 (如 "blue circle")
            - goal: 目标描述 (如 "green circle")
            - trap: 陷阱描述 (如 "red x")
            - floor: 地板描述 (如 "white square")
    """
    return TRAPFIELD_PROMPT_TEMPLATE.substitute(
        player=visual_description.get("player", "blue circle"),
        goal=visual_description.get("goal", "green circle"),
        trap=visual_description.get("trap", "red cross"),
        floor=visual_description.get("floor", "gray path"),
    )

