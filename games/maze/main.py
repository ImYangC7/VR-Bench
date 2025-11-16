from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

try:
    from . import constants
    from .generators import data_gen
    from .templates.base_template import BaseTemplate
    from .utils import file_utils
except ImportError:
    import constants
    from generators import data_gen
    from templates.base_template import BaseTemplate
    from utils import file_utils

# Configure output directory and maze counts here
DEFAULT_OUTPUT_DIR = "maze_dataset_py"
DEFAULT_COUNTS = {
    9: 1,   # number of 9x9 mazes
    11: 1,  # number of 11x11 mazes
    13: 1,  # number of 13x13 mazes
}


def main(assets_folder: Optional[str] = None) -> int:
    counts: List[int] = [DEFAULT_COUNTS.get(size, 0) for size in constants.ALLOWED_SIZES]
    if any(count < 0 for count in counts):
        raise ValueError("Counts must be non-negative integers")

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    images_dir = output_dir / constants.IMAGES_DIR
    states_dir = output_dir / constants.STATES_DIR
    video_dir = output_dir / constants.VIDEOS_DIR
    data_file = output_dir / constants.DATA_PATH

    file_utils.setup_output_directories(
        str(output_dir), str(images_dir), str(states_dir), str(video_dir)
    )

    start_id = 0
    templates: List[BaseTemplate] = []

    for size, label, count in zip(constants.ALLOWED_SIZES, constants.SIZE_LABELS, counts):
        if count <= 0:
            continue
        print(f"Generating {count} {label} mazes...")
        templates.extend(
            data_gen.generate_data(
                start_id, count, size, str(images_dir), str(states_dir), str(video_dir),
                assets_folder=assets_folder
            )
        )
        start_id += count

    data_gen.save_data_to_json(templates, str(data_file))

    print(f"Data generation completed. Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    assets_folder = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(assets_folder))
