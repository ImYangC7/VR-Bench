# -*- coding: utf-8 -*-
"""
Pathfinder (irregular_maze) 游戏的视频模型 prompt 模板。

占位符: {start}, {end}, {road}
从 description.json 的 visual_description 中读取。
"""
from string import Template

PATHFINDER_PROMPT_TEMPLATE = Template("""Create a 2D animation based on the provided image of a maze. The $start slides smoothly along the $road path, stopping perfectly on the $end. The $start never slides or crosses into the black areas of the maze. The camera is a static, top-down view showing the entire maze.

Maze:
 The maze paths are $road, the walls are black.
 The $start moves to the goal position, represented by $end.
 The $start slides smoothly along the $road path.
 The $start never slides or crosses into the black areas of the maze.
 The $start stops perfectly on the $end.

Scene:
 No change in scene composition.
 No change in the layout of the maze.
 The $start travels along the $road path without speeding up or slowing down.

Camera:
 Static camera.
 No zoom.
 No pan.
 No glitches, noise, or artifacts.""")


def get_pathfinder_prompt(visual_description: dict) -> str:
    """
    生成 pathfinder 游戏的动态 prompt。
    
    Args:
        visual_description: 来自 description.json 的 visual_description 字段
            - start: 起点描述 (如 "green circle")
            - end: 终点描述 (如 "red circle")
            - road: 道路描述 (如 "white square")
    """
    return PATHFINDER_PROMPT_TEMPLATE.substitute(
        start=visual_description.get("start", "green circle"),
        end=visual_description.get("end", "red circle"),
        road=visual_description.get("road", "white path"),
    )

