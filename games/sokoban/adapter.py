"""
Sokoban 游戏适配器
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import imageio.v2 as imageio

from core.game_adapter import GameAdapter
from games.sokoban import generate_textured_random_board


class SokobanAdapter(GameAdapter):
    """Sokoban 游戏适配器"""
    
    def get_game_name(self) -> str:
        return "sokoban"
    
    def generate_level(
        self,
        difficulty_config: Dict[str, Any],
        assets_folder: str,
        **kwargs
    ) -> Optional[Any]:
        """生成 Sokoban 关卡"""
        try:
            board_size = difficulty_config.get('board_size', 5)
            num_boxes = difficulty_config.get('num_boxes', 1)
            check_solvable = difficulty_config.get('check_solvable', True)
            max_attempts = difficulty_config.get('max_attempts', 50)
            
            board = generate_textured_random_board(
                size=board_size,
                num_boxes=num_boxes,
                check_solvable=check_solvable,
                max_attempts=max_attempts,
                assets_folder=assets_folder
            )
            return board
        except Exception as e:
            logging.error(f"Failed to generate Sokoban level: {e}")
            return None
    
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty_name: str,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """保存 Sokoban 关卡"""
        from main import create_solution_video
        
        fps = kwargs.get('fps', 2)
        add_grid = kwargs.get('add_grid', False)
        
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
            # 生成视频
            video_result = create_solution_video(level, video_path, fps=fps, add_grid=add_grid)
            if video_result:
                result['video'] = video_filename
                
                # 从视频提取第一帧
                try:
                    reader = imageio.get_reader(str(video_path))
                    first_frame = reader.get_data(0)
                    reader.close()
                    imageio.imwrite(str(image_path), first_frame)
                    result['image'] = image_filename
                except Exception as e:
                    logging.warning(f"Failed to extract first frame: {e}")
            
            # 保存状态文件（使用统一格式）
            try:
                from core.schema import UnifiedState, Grid, Entity, RenderConfig

                old_state = level.get_full_state()
                grid_data = old_state["grid"]
                height = old_state["grid_size"]["height"]
                width = old_state["grid_size"]["width"]
                cell_size = 64

                player_x = old_state["player"]["x"]
                player_y = old_state["player"]["y"]

                boxes = [
                    Entity.from_grid_pos(box["y"], box["x"], cell_size)
                    for box in old_state["boxes"]
                ]

                targets = old_state["targets"]
                goal = targets[0] if targets else {"x": 0, "y": 0}

                state = UnifiedState(
                    version="1.0",
                    game_type="sokoban",
                    grid=Grid(data=grid_data, height=height, width=width),
                    player=Entity.from_grid_pos(player_y, player_x, cell_size),
                    goal=Entity.from_grid_pos(goal["y"], goal["x"], cell_size),
                    boxes=boxes,
                    render=RenderConfig.from_grid_size(height, width, cell_size),
                    metadata={"symbols": old_state["symbols"]}
                )

                state.save(str(state_path))
                result['state'] = state_filename
            except Exception as e:
                logging.warning(f"Failed to save state: {e}")
                
        except Exception as e:
            logging.error(f"Failed to save Sokoban level: {e}")
        
        return result
    
    def get_level_hash(self, level: Any) -> str:
        """获取关卡哈希"""
        import hashlib
        import numpy as np
        
        # 使用网格状态生成哈希
        grid_bytes = level.grid.tobytes()
        return hashlib.md5(grid_bytes).hexdigest()
    
    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """检查是否重复"""
        level_hash = self.get_level_hash(level)
        return level_hash in existing_hashes
    
    def validate_difficulty_config(self, difficulty_config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_fields = ['count', 'board_size', 'num_boxes']
        return all(field in difficulty_config for field in required_fields)
    
    def get_required_texture_files(self) -> list:
        """返回需要的纹理文件"""
        return ['floor', 'wall', 'box', 'target', 'player']

