# -*- coding: utf-8 -*-
"""
Maze3D 游戏的视频模型 prompt 模板。

占位符: {ball}, {start_cube}, {goal_cube}, {default_cube}
从 description.json 的 visual_description 中读取。
"""
from string import Template

MAZE3D_PROMPT_TEMPLATE = Template("""Create a 3D animation based on the provided image of a cube maze. A $ball slides smoothly along the $default_cube pathway, climbs up the vertical ladders step by step, and finally stops perfectly on the $goal_cube at the top. The $ball never touches or passes through the $start_cube or any non-$default_cube areas of the maze. The camera remains static in an isometric, top-down angle showing the entire structure.

Maze:
 The maze consists of stacked transparent $default_cube forming a 3D pathway.
 The $goal_cube represents the goal position.
 The $start_cube marks the starting platform where the $ball begins.
 The $ball moves upward along the $default_cube path, climbing vertically via the ladders.
 The ball slides smoothly without sudden changes in direction or speed.
 The ball stops exactly on top of the $goal_cube at the end.

Scene:
 No structural or color changes during animation.
 The maze layout and cube arrangement remain unchanged.
 The $ball moves continuously at a constant speed along the 3D path.

Camera:
 Static, isometric camera view.
 No zoom or pan.
 Smooth animation without flicker, noise, or artifacts.""")


def get_maze3d_prompt(visual_description: dict) -> str:
    """
    生成 maze3d 游戏的动态 prompt。
    
    Args:
        visual_description: 来自 description.json 的 visual_description 字段
            - ball: 球的描述 (如 "golden ball with orange edge")
            - start_cube: 起点方块描述 (如 "blue cube")
            - goal_cube: 目标方块描述 (如 "red cube")
            - default_cube: 默认路径方块描述 (如 "gray cube")
    """
    return MAZE3D_PROMPT_TEMPLATE.substitute(
        ball=visual_description.get("ball", "yellow ball"),
        start_cube=visual_description.get("start_cube", "blue cube"),
        goal_cube=visual_description.get("goal_cube", "red cube"),
        default_cube=visual_description.get("default_cube", "gray cube"),
    )

