MAZE3D_SYSTEM_PROMPT = """You are given an image of a 3D maze composed of \
gray cubes that represent walkable platforms suspended in space.
Each cube represents a solid tile that the ball can stand on or move across.
The yellow sphere represents the starting point.
The red cube represents the goal or destination.
The blue cubes represent the initial platform where the ball begins.

Task:
Infer the shortest valid 3D path for the yellow sphere to move from its \
starting position to the red goal cube.

Movement Rules:
- Horizontal movements (forward_left, forward_right, backward_left, backward_right): \
Each move spans 2 grid units horizontally. 
- Vertical movements (up, down): Each move spans 3 grid units vertically via a ladder. \
The ladder must be present at the starting position.
- The sphere cannot move through empty space or overlap any cube structure.
- All movements must follow valid cube surfaces and ladder connections.

The six valid directions of movement are:
"forward_left" – move diagonally forward and to the left (2 units) within the same layer
"forward_right" – move diagonally forward and to the right (2 units) within the same layer
"backward_left" – move diagonally backward and to the left (2 units) within the same layer
"backward_right" – move diagonally backward and to the right (2 units) within the same layer
"up" – move vertically upward (3 units) via a ladder
"down" – move vertically downward (3 units) via a ladder

Output Format:
Return the full sequence of movement directions as a JSON array, where each \
step is one of the six valid directions.
Do not include any explanations, reasoning, or extra text.

Example of expected output:
{
  "path": ["up", "forward_right", "forward_left", "up", "forward_right"]
}
"""


MAZE3D_USER_PROMPT = """Infer the shortest valid 3D path for the yellow \
sphere to move from its starting position to the red goal cube.
"""

