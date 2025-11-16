MAZE_SYSTEM_PROMPT = """You are given an image of a grid-based maze.
Black tiles represent walls and cannot be crossed.
White tiles represent open paths that can be moved through.
The green circle represents the starting point of the path.
The red circle represents the goal or destination.


Task:
Infer the shortest valid path from the green starting point to the red goal circle.
Movement can only occur between adjacent open tiles — up, down, left, or right.
Diagonal movement is not allowed, and the path must not cross or touch any black walls.


Output Format:
Return the entire movement sequence of the green circle as a JSON array of directions, where each element is one of "up", "down", "left", or "right".
Do not include any explanations or additional text.


Example of expected output:
{
  "path": ["up", "up", "left", "down", "right", "right"]
}
"""


MAZE_USER_PROMPT = """Infer the shortest valid path from the green starting point to the red goal circle.
"""

