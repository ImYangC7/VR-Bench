"""
TrapField 游戏适配器
实现 GameAdapter 接口
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.game_adapter import GameAdapter
from .generator import generate_trapfield, solve_trapfield
from .renderer import TrapFieldRenderer, create_solution_video
from .config import REQUIRED_TEXTURES
from . import constants

try:
    import imageio
except ImportError:
    imageio = None


class TrapFieldAdapter(GameAdapter):
    """TrapField 游戏适配器"""

    def get_game_name(self) -> str:
        """返回游戏名称"""
        return "trapfield"

    def get_required_texture_files(self) -> list[str]:
        """返回需要的纹理文件列表"""
        return REQUIRED_TEXTURES
    
    def generate_level(self, config: Dict[str, Any], assets_folder: str = "") -> Optional[Any]:
        """
        生成关卡
        
        Args:
            config: 配置字典，包含 grid_size 和 trap_density
            assets_folder: 资源文件夹路径
        
        Returns:
            生成的网格数据
        """
        grid_size = config.get('grid_size', 7)
        trap_density = config.get('trap_density', 0.3)
        max_attempts = config.get('max_attempts', 50)
        
        for attempt in range(max_attempts):
            try:
                grid = generate_trapfield(grid_size, trap_density)
                
                # 验证是否可解
                path = solve_trapfield(grid)
                if path and len(path) >= 3:  # 至少需要3步
                    return grid
                
            except Exception as e:
                logging.debug(f"Generation attempt {attempt + 1} failed: {e}")
                continue
        
        return None
    
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        保存关卡到文件
        
        Args:
            level: 关卡数据（网格）
            output_dir: 输出目录
            level_id: 关卡ID
            difficulty: 难度
            **kwargs: 额外参数（fps, add_grid等）
        
        Returns:
            生成的文件字典 {'video': 文件名, 'image': 文件名, 'state': 文件名}
        """
        grid = level
        assets_folder = kwargs.get('assets_folder', '')
        
        # 创建子目录
        states_dir = output_dir / 'states'
        videos_dir = output_dir / 'videos'
        images_dir = output_dir / 'images'
        
        states_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名
        state_filename = f'{difficulty}_{level_id:04d}.json'
        video_filename = f'{difficulty}_{level_id:04d}.mp4'
        image_filename = f'{difficulty}_{level_id:04d}.png'
        
        state_path = states_dir / state_filename
        video_path = videos_dir / video_filename
        image_path = images_dir / image_filename
        
        result = {
            'video': None,
            'image': None,
            'state': None
        }
        
        try:
            # 保存状态文件（使用统一格式）
            from core.schema import UnifiedState, Grid, Entity, RenderConfig
            from .generator import find_position

            player_pos = find_position(grid, constants.PLAYER_CELL)
            goal_pos = find_position(grid, constants.GOAL_CELL)

            height = len(grid)
            width = len(grid[0]) if grid else 0
            cell_size = constants.CELL_SIZE

            state = UnifiedState(
                version="1.0",
                game_type="trapfield",
                grid=Grid.from_2d_list(grid),
                player=Entity.from_grid_pos(player_pos[0], player_pos[1], cell_size),
                goal=Entity.from_grid_pos(goal_pos[0], goal_pos[1], cell_size),
                boxes=[],
                render=RenderConfig.from_grid_size(height, width, cell_size),
                metadata={}
            )

            state.save(str(state_path))
            result['state'] = state_filename
            
            # 生成视频
            try:
                path = solve_trapfield(grid)
                
                if path:
                    # 生成视频
                    create_solution_video(
                        grid,
                        path,
                        cell_size=64,
                        save_path=str(video_path),
                        frame_duration_ms=300,
                        assets_folder=assets_folder
                    )
                    result['video'] = video_filename
                    
                    # 从视频提取第一帧作为图片
                    if imageio:
                        try:
                            reader = imageio.get_reader(str(video_path))
                            first_frame = reader.get_data(0)
                            reader.close()
                            imageio.imwrite(str(image_path), first_frame)
                            result['image'] = image_filename
                        except Exception as e:
                            logging.warning(f"Failed to extract first frame: {e}")
                            # 如果提取失败，直接渲染静态图片
                            renderer = TrapFieldRenderer(assets_folder=assets_folder, cell_size=64)
                            renderer.render_grid(grid, str(image_path))
                            result['image'] = image_filename
                    else:
                        # 如果没有 imageio，直接渲染静态图片
                        renderer = TrapFieldRenderer(assets_folder=assets_folder, cell_size=64)
                        renderer.render_grid(grid, str(image_path))
                        result['image'] = image_filename
                        
            except Exception as e:
                logging.warning(f"Failed to create video: {e}")
                # 如果视频生成失败，至少保存静态图片
                try:
                    renderer = TrapFieldRenderer(assets_folder=assets_folder, cell_size=64)
                    renderer.render_grid(grid, str(image_path))
                    result['image'] = image_filename
                except Exception as e2:
                    logging.error(f"Failed to create image: {e2}")
        
        except Exception as e:
            logging.error(f"Failed to save level: {e}")
        
        return result
    
    def get_level_hash(self, level: Any) -> str:
        """
        计算关卡的哈希值（用于去重）

        Args:
            level: 关卡数据（网格）

        Returns:
            哈希字符串
        """
        grid = level
        # 将网格转换为字符串并计算哈希
        grid_str = json.dumps(grid, sort_keys=True)
        return hashlib.md5(grid_str.encode()).hexdigest()

    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """
        检查关卡是否重复

        Args:
            level: 关卡数据（网格）
            existing_hashes: 已存在的哈希集合

        Returns:
            是否重复
        """
        level_hash = self.get_level_hash(level)
        return level_hash in existing_hashes

