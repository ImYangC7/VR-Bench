# Metadata Generation Tool

## Overview

Generate metadata.csv files for VR-Bench dataset with dynamic prompts based on skin configurations.

**Features:**
- Dynamic prompt generation from skin descriptions
- Flexible filtering by game type, skin, and difficulty
- Separate or merged output modes
- Support for both train and eval splits

## Quick Start

```bash
# Generate all metadata files
python test_dynamic_metadata.py

# Generate for specific game type
python test_dynamic_metadata.py --games maze

# Generate for specific skins and difficulties
python test_dynamic_metadata.py --games maze --skins 1 2 --difficulties easy
```

## Command-Line Arguments

### `--games`
Select game types (multiple allowed)

**Options:** `maze`, `irregular_maze`, `maze3d`, `sokoban`, `trapfield`

```bash
python test_dynamic_metadata.py --games maze sokoban
```

### `--skins`
Select skin IDs (multiple allowed)

**Options:** `1`, `2`, `3`, `4`, `5` (varies by game type)

**Skin counts:**
- maze: 5 skins
- irregular_maze: 4 skins
- maze3d: 4 skins
- sokoban: 5 skins
- trapfield: 4 skins

```bash
python test_dynamic_metadata.py --skins 1 2 3
```

### `--difficulties`
Select difficulty levels (multiple allowed)

**Options:** `easy`, `medium`, `hard`

```bash
python test_dynamic_metadata.py --difficulties easy hard
```

### `--splits`
Select dataset splits (default: train eval)

**Options:** `train`, `eval`

```bash
python test_dynamic_metadata.py --splits train
```

### `--merge`
Merge all matching data into a single metadata.csv

```bash
python test_dynamic_metadata.py --games maze --skins 1 2 --merge
```

### `--dataset-root`
Specify dataset root directory (default: project_root/downloaded_dataset)

```bash
python test_dynamic_metadata.py --dataset-root /path/to/dataset
```

### `--skins-root`
Specify skins configuration directory (default: project_root/skins)

```bash
python test_dynamic_metadata.py --skins-root /path/to/skins
```

## Usage Examples

### Generate all data
```bash
python test_dynamic_metadata.py
```
**Output:** 132 metadata.csv files (66 train + 66 eval)

### Generate specific game
```bash
python test_dynamic_metadata.py --games maze
```
**Output:** 30 files (5 skins × 3 difficulties × 2 splits)

### Generate specific combination
```bash
python test_dynamic_metadata.py --games maze --skins 1 --difficulties easy
```
**Output:** 2 files (train/maze_1_easy and eval/maze_1_easy)

### Merge multiple games
```bash
python test_dynamic_metadata.py --games maze irregular_maze --merge
```
**Output:** 2 merged files (one for train, one for eval)

### Cross-skin training
```bash
python test_dynamic_metadata.py --games maze --skins 1 2 3 --merge --splits train
```
**Output:** 1 merged file containing all train data for maze skins 1, 2, 3

### Regenerate specific skins
```bash
python test_dynamic_metadata.py --games irregular_maze --skins 1 2 3
```
**Output:** 18 files (3 skins × 3 difficulties × 2 splits)

## Output Structure

### Separate Mode (default)
```
downloaded_dataset/
└── metadata/
    ├── train/
    │   ├── maze_1_easy/
    │   │   └── metadata.csv
    │   ├── maze_1_medium/
    │   │   └── metadata.csv
    │   └── ...
    └── eval/
        ├── maze_1_easy/
        │   └── metadata.csv
        └── ...
```

### Merge Mode
```
downloaded_dataset/
└── metadata/
    ├── train/
    │   └── maze_sokoban_1_2_easy/
    │       └── metadata.csv
    └── eval/
        └── maze_sokoban_1_2_easy/
            └── metadata.csv
```

## Metadata CSV Format

Each CSV file contains 3 columns:

| Column | Description | Example |
|--------|-------------|---------|
| `video` | Video file path (relative to downloaded_dataset/) | `train/maze/1/easy/videos/easy_0001_0.mp4` |
| `prompt` | Dynamically generated prompt | `Create a 2D animation...` |
| `input_image` | Input image path (relative to downloaded_dataset/) | `train/maze/1/easy/images/easy_0001.png` |

## Dynamic Prompt System

Prompts are automatically generated based on skin configurations in `skins/{game_type}/{skin_id}/description.json`.

**Example:** For maze skin 1:
```json
{
  "visual_description": {
    "player": "red circle",
    "goal": "green square",
    "wall": "light blue square",
    "floor": "white square"
  }
}
```

**Generated prompt:**
```
Create a 2D animation based on the provided image of a maze.
The red circle slides smoothly along the white square path,
stopping perfectly on the green square...
```

Different skins produce different prompts automatically.

## Notes

- All paths in metadata.csv are relative to `downloaded_dataset/` directory
- Game type `irregular_maze` maps to `pathfinder` skin directory
- If skin description is not found, a warning is displayed and the combination is skipped
- Use `--merge` mode for training across multiple skins or game types
