# Python Maze Q&A Generator

Python port of the maze dataset generator originally implemented in `src/main/java/com/zjt`.

## Requirements

```bash
pip install pillow
```

## Usage (no CLI arguments)

1. Adjust configuration in `pymaze/main.py`:
   - `DEFAULT_OUTPUT_DIR` for the destination folder.
   - `DEFAULT_COUNTS` for the number of 9℅9, 11℅11, and 13℅13 mazes.
2. Run the module:

```bash
python -m pymaze
```

The script reads the configuration above and produces the dataset automatically.

## Output Layout

- `<OUTPUT>/images/` 每 rendered maze PNG files (with row/column labels).
- `<OUTPUT>/states/` 每 maze state JSON files containing the grid and player/goal coordinates.
- `<OUTPUT>/video/` 每 animated GIFs showing the solution path step by step.
- `<OUTPUT>/data.json` 每 list of QA entries matching the Java schema.

## Registered Question Templates

- `PlayerPosition`
- `GoalPosition`
- `PositionAfterMoving`
- `AvailableDirections`
- `FindPathToGoal`
- `TurnCount`

Each entry contains the question text, answer, options when applicable, and an analysis section.
