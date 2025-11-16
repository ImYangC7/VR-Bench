"""PathFinder game adapter for unified data generation."""

import sys
import logging
import json
import hashlib
import imageio
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.game_adapter import GameAdapter
from core.schema import UnifiedState, Grid, Entity, RenderConfig
from .generator import generate_pathfinder_board
from .renderer import render_pathfinder_board, render_solution_video


class PathFinderAdapter(GameAdapter):
    """PathFinder 游戏适配器"""
    
    def get_game_name(self) -> str:
        return "pathfinder"
    
    def generate_level(
        self,
        difficulty_config: Dict[str, Any],
        assets_folder: str,
        **kwargs
    ) -> Optional[Any]:
        """生成 PathFinder 关卡"""
        try:
            difficulty = difficulty_config.get('difficulty', 'medium')
            image_size = difficulty_config.get('image_size', 512)
            max_attempts = difficulty_config.get('max_attempts', 50)
            
            board = generate_pathfinder_board(
                difficulty=difficulty,
                image_size=image_size,
                max_attempts=max_attempts
            )
            
            return board
        except Exception as e:
            logging.error(f"Failed to generate PathFinder level: {e}")
            return None
    
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty_name: str,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """保存 PathFinder 关卡"""
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

        # 获取纹理文件夹路径
        assets_folder = kwargs.get('assets_folder', None)
        # 如果 assets_folder 是空字符串，转换为 None
        if assets_folder == "":
            assets_folder = None

        result = {
            'video': None,
            'image': None,
            'state': None
        }

        try:
            # 生成视频
            try:
                # PathFinder 使用自己的 fps 配置（constants.py 中的 FRAMES_PER_SECOND）
                # 忽略外部传入的 fps 参数
                video_result = render_solution_video(
                    level,
                    str(video_path),
                    assets_folder=assets_folder  # 传递纹理文件夹
                )

                if video_result:
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

            # 保存状态文件（使用 UnifiedState 格式）
            try:
                # PathFinder 使用像素坐标系统
                start_x, start_y = level.start_point
                end_x, end_y = level.end_point
                road_width = level.road_width

                # 使用 from_pixel_pos 创建 Entity（无需 grid_pos）
                player_entity = Entity.from_pixel_pos(int(start_x), int(start_y), road_width)
                goal_entity = Entity.from_pixel_pos(int(end_x), int(end_y), road_width)

                # PathFinder 没有 grid，grid=None
                state = UnifiedState(
                    version="1.0",
                    game_type="pathfinder",
                    player=player_entity,
                    goal=goal_entity,
                    render=RenderConfig(
                        cell_size=level.road_width,
                        image_width=level.image_size,
                        image_height=level.image_size
                    ),
                    grid=None,
                    boxes=[],
                    metadata={
                        'segments': [[pt for pt in seg.control_points] for seg in level.segments],
                        'start_point': list(level.start_point),
                        'end_point': list(level.end_point),
                        'solution_segments': level.solution_segments,
                        'solution_path': [list(pt) for pt in level.solution_path],
                        'road_width': level.road_width
                    }
                )
                state.save(str(state_path))
                result['state'] = state_filename
            except Exception as e:
                logging.warning(f"Failed to save state: {e}")

        except Exception as e:
            logging.error(f"Failed to save PathFinder level: {e}")

        return result
    
    def get_level_hash(self, level: Any) -> str:
        """获取关卡哈希值用于去重"""
        # 使用路径段和起终点的组合生成哈希
        state = level.to_dict()

        # 创建一个唯一标识
        unique_str = f"{state['segments']}{state['start_point']}{state['end_point']}{state['solution_segments']}"

        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """检查是否重复"""
        level_hash = self.get_level_hash(level)
        return level_hash in existing_hashes
    
    def validate_difficulty_config(self, difficulty_config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_fields = ['count', 'difficulty']
        return all(field in difficulty_config for field in required_fields)
    
    def get_required_texture_files(self) -> list:
        """返回需要的纹理文件"""
        # PathFinder 需要：起点、终点、道路纹理
        return ['start', 'end', 'road']

