# -*- coding: utf-8 -*-
"""
Maze 游戏的视频模型 prompt 模板。

占位符: {player}, {goal}, {wall}, {floor}
从 description.json 的 visual_description 中读取。
"""
from string import Template

MAZE_PROMPT_TEMPLATE = Template("""Create a 2D animation based on the provided image of a maze. The $player slides smoothly along the $floor path, stopping perfectly on the $goal. The $player never slides or crosses into the $wall areas of the maze. The camera is a static, top-down view showing the entire maze.

Maze:
 The maze paths are $floor, the walls are $wall.
 The $player moves to the goal position, represented by $goal.
 The $player slides smoothly along the $floor path.
 The $player never slides or crosses into the $wall areas of the maze.
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


def get_maze_prompt(visual_description: dict) -> str:
    """
    生成 maze 游戏的动态 prompt。
    
    Args:
        visual_description: 来自 description.json 的 visual_description 字段
            - player: 玩家描述 (如 "red circle")
            - goal: 目标描述 (如 "green square")
            - wall: 墙壁描述 (如 "light blue square")
            - floor: 地板描述 (如 "white square")
    """
    return MAZE_PROMPT_TEMPLATE.substitute(
        player=visual_description.get("player", "red circle"),
        goal=visual_description.get("goal", "green square"),
        wall=visual_description.get("wall", "blue"),
        floor=visual_description.get("floor", "white"),
    )

