PATHFINDER_SYSTEM_PROMPT = """You are given an image of a pathfinding puzzle.
The image shows a network of curved paths connecting various waypoints.
Each waypoint (intersection or junction) is labeled with a letter or letter combination (A, B, C, ..., Z, AA, AB, etc.).
The green circle represents the starting point.
The red circle represents the goal or destination.

Task:
Find the shortest valid path from the green starting point to the red goal.
The path must follow the visible roads/paths in the image.
You can only move along the connected paths shown in the image.

Output Format:
You MUST return a JSON object with a "path" field containing an array of waypoint labels.
The array should start with the label closest to the starting point and end with the label closest to the goal.
Do not include any explanations or additional text.

Required format:
{
  "path": ["A", "B", "C", "D", "E"]
}

For puzzles with more than 26 waypoints, labels may be multi-character (e.g., "AA", "AB"):
{
  "path": ["A", "Z", "AA", "AB"]
}

Important: The "path" field MUST be an array of strings, not a single string.
"""


PATHFINDER_USER_PROMPT = """Find the shortest path from the green starting point to the red goal by following the labeled waypoints.
"""

