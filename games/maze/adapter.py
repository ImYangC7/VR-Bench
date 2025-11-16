"""
Maze 游戏适配器
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import imageio.v2 as imageio

from core.game_adapter import GameAdapter


class MazeAdapter(GameAdapter):
    """Maze 游戏适配器"""

    def get_game_name(self) -> str:
        return "maze"

    def generate_level(
        self,
        difficulty_config: Dict[str, Any],
        assets_folder: str,
        **kwargs
    ) -> Optional[Any]:
        """生成 Maze 关卡"""
        from games.maze.generators.maze_gen import generate_maze
        from games.maze.default_textures import ensure_default_textures

        try:
            maze_size = difficulty_config.get('maze_size', 9)
            # generate_maze 需要 rows 和 cols 两个参数
            maze = generate_maze(maze_size, maze_size)

            # 如果没有指定素材文件夹或素材不完整，使用默认纹理
            if not assets_folder or not Path(assets_folder).exists():
                assets_folder = str(ensure_default_textures())

            # 附加素材文件夹信息
            # maze 是 List[List[int]]，不能直接添加属性，需要包装
            class MazeWrapper:
                def __init__(self, maze_data, assets):
                    self.data = maze_data
                    self._assets_folder = assets

                def __getitem__(self, key):
                    return self.data[key]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

            return MazeWrapper(maze, assets_folder)
        except Exception as e:
            logging.error(f"Failed to generate Maze level: {e}")
            return None
    
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty_name: str,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """保存 Maze 关卡"""
        from games.maze.generators.video_gen import create_solution_video
        from games.maze.generators.state_gen import save_state

        # 创建子目录
        images_dir = output_dir / "images"
        videos_dir = output_dir / "videos"
        states_dir = output_dir / "states"

        images_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        states_dir.mkdir(parents=True, exist_ok=True)

        # 文件名
        video_filename = f"{difficulty_name}_{level_id:04d}.mp4"
        image_filename = f"{difficulty_name}_{level_id:04d}.png"
        state_filename = f"{difficulty_name}_{level_id:04d}.json"

        video_path = videos_dir / video_filename
        image_path = images_dir / image_filename
        state_path = states_dir / state_filename

        result = {
            'video': None,
            'image': None,
            'state': None
        }

        try:
            # 获取素材文件夹和原始 maze 数据
            assets_folder = getattr(level, '_assets_folder', None)
            maze_data = getattr(level, 'data', level)  # 如果是包装类，获取 data；否则直接使用

            # 生成视频
            try:
                from games.maze.utils import maze_utils
                import random
                import imageio

                # 使用 DFS 求解
                solver_rng = random.Random(level_id)
                path = maze_utils.dfs_solve_maze(maze_data, [], rng=solver_rng)

                if path:
                    # 生成视频
                    create_solution_video(
                        maze_data,
                        path,
                        cell_size=64,
                        save_path=str(video_path),
                        frame_duration_ms=300,
                        assets_folder=assets_folder
                    )
                    result['video'] = video_filename

                    # 从视频提取第一帧作为图片
                    try:
                        reader = imageio.get_reader(str(video_path))
                        first_frame = reader.get_data(0)
                        reader.close()
                        imageio.imwrite(str(image_path), first_frame)
                        result['image'] = image_filename
                    except Exception as e:
                        logging.warning(f"Failed to extract first frame: {e}")
            except Exception as e:
                logging.warning(f"Failed to create video: {e}")

            # 保存状态文件
            try:
                save_state(maze_data, str(state_path))
                result['state'] = state_filename
            except Exception as e:
                logging.warning(f"Failed to save state: {e}")
                
        except Exception as e:
            logging.error(f"Failed to save Maze level: {e}")
        
        return result
    
    def get_level_hash(self, level: Any) -> str:
        """获取关卡哈希"""
        import hashlib

        # 获取原始 maze 数据
        maze_data = getattr(level, 'data', level)

        # 使用网格状态生成哈希
        grid_str = str(maze_data)
        return hashlib.md5(grid_str.encode()).hexdigest()
    
    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """检查是否重复"""
        level_hash = self.get_level_hash(level)
        return level_hash in existing_hashes
    
    def validate_difficulty_config(self, difficulty_config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_fields = ['count', 'maze_size']
        return all(field in difficulty_config for field in required_fields)
    
    def get_required_texture_files(self) -> list:
        """返回需要的纹理文件"""
        return ['floor', 'wall', 'player', 'target']

