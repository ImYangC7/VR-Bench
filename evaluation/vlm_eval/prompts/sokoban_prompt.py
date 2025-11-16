SOKOBAN_SYSTEM_PROMPT = """You are given an image of a grid-based Sokoban puzzle.
 Gray tiles represent walls and cannot be crossed.
 White tiles represent open floor tiles that can be moved through.
 The blue ball represents the player or agent.
 The yellow square represents the box that needs to be pushed.
 The red square represents the goal destination for the box.
Task:
 Infer the complete movement sequence required for the blue ball to push the yellow square onto the red goal square.
 The blue ball moves in four directions: up, down, left, right.
 When the blue ball moves into a box, it automatically pushes the box if there is space behind it.
 The box and the blue ball cannot cross or overlap any gray walls.
 Diagonal movement is not allowed, and the camera remains fixed from a top-down view.
Output Format:
 Return the entire movement sequence as a JSON array of directional actions, where each element is one of "up", "down", "left", or "right".
 Do not include any explanations or additional text.
Example of expected output:
{
  "actions": ["right", "right", "down", "left", "down"]
}
"""


SOKOBAN_USER_PROMPT = """Infer the complete movement sequence required for the blue ball to push the yellow square onto the red goal square.
"""
