"""
测试 pymaze 换皮肤功能

使用方法:
    python -m pymaze.test_skin                    # 使用默认皮肤
    python -m pymaze.test_skin custom_assets      # 使用自定义皮肤
"""

import sys
from pathlib import Path

try:
    from . import constants
    from .generators import data_gen
    from .utils import file_utils
except ImportError:
    import constants
    from generators import data_gen
    from utils import file_utils


def main():
    assets_folder = sys.argv[1] if len(sys.argv) > 1 else None
    
    if assets_folder:
        print(f"Using custom skin: {assets_folder}")
    else:
        print("Using default skin")
    
    output_dir = Path("test_maze_output")
    images_dir = output_dir / "images"
    states_dir = output_dir / "states"
    video_dir = output_dir / "videos"
    
    file_utils.setup_output_directories(
        str(output_dir), str(images_dir), str(states_dir), str(video_dir)
    )
    
    print("Generating 1 test maze (9x9)...")
    data_gen.generate_data(
        0, 1, 9, str(images_dir), str(states_dir), str(video_dir),
        assets_folder=assets_folder
    )
    
    print(f"Done! Check output in: {output_dir}")


if __name__ == "__main__":
    main()

