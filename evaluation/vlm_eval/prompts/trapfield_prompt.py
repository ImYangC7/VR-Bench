TRAPFIELD_SYSTEM_PROMPT = """You are given an image of a grid-based maze.
 Red tiles marked with an “X” represent trap zones that must be avoided.
 White tiles represent open paths that can be moved through.
 The blue circle represents the starting point of the path.
 The green circle represents the goal or destination.
Task:
 Infer the shortest valid path for the blue circle to reach the green circle.
 Movement can only occur between adjacent open tiles — up, down, left, or right.
 Diagonal movement is not allowed.
 The path must not cross or touch any red trap tiles.
Output Format:
 Return the full movement sequence of the blue circle as a JSON array of directions, where each element is one of "up", "down", "left", or "right".
 Do not include any explanations, reasoning, or extra text.
Example of expected output:
{
  "path": ["left", "left", "down", "down"]
}
"""


TRAPFIELD_USER_PROMPT = """Infer the shortest valid path for the blue circle to reach the green circle.
"""

