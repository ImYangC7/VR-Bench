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
            
            # 使用generate_video方法生成视频
            if result['state']:
                assets_folder = kwargs.get('assets_folder', None)
                if self.generate_video(str(state_path), str(video_path), assets_folder=assets_folder):
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
    
    def generate_video(
        self,
        state_path: str,
        output_path: str,
        assets_folder: Optional[str] = None,
        **kwargs
    ) -> bool:
        """从state文件生成视频"""
        try:
            from core.schema import UnifiedState
            from games.sokoban.textured_board import TexturedSokobanBoard
            from games.sokoban.renderer import get_shared_renderer
            from games.sokoban.board import Solution
            import numpy as np
            import imageio.v2 as imageio
            
            state = UnifiedState.load(state_path)
            grid_array = np.array(state.grid.data, dtype=int)
            
            renderer = get_shared_renderer(assets_folder) if assets_folder else None
            
            board = TexturedSokobanBoard(
                grid=grid_array,
                player_x=state.player.grid_pos.col,
                player_y=state.player.grid_pos.row,
                renderer=renderer
            )
            
            solver = Solution()
            grid_chars = board.to_solver_grid()
            total_moves, _ = solver.minPushBox(grid_chars)
            
            if total_moves == -1:
                logging.warning(f"No solution found for {state_path}")
                return False
            
            moves = solver.get_solution_path()
            if not moves:
                return False
            
            original_state = board.save_state()
            frames = []
            frames_per_step = 12
            texture_size = board.renderer.texture_size
            
            try:
                for move in moves:
                    start_row, start_col = int(move.start_pos[0]), int(move.start_pos[1])
                    end_row, end_col = int(move.end_pos[0]), int(move.end_pos[1])
                    is_push = move.is_push
                    
                    pre_move_grid = board.grid.copy()
                    pre_move_player_x = board.player_x
                    pre_move_player_y = board.player_y
                    
                    box_start_row = box_start_col = box_end_row = box_end_col = None
                    if is_push:
                        box_start_row, box_start_col = int(move.box_start[0]), int(move.box_start[1])
                        box_end_row, box_end_col = int(move.box_end[0]), int(move.box_end[1])
                    
                    direction_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
                    dx = end_col - start_col
                    dy = end_row - start_row
                    direction = direction_map.get((dx, dy))
                    if direction is not None:
                        board.make_move(direction)
                    
                    for frame_idx in range(frames_per_step):
                        progress = frame_idx / frames_per_step
                        temp_grid = pre_move_grid.copy()
                        
                        player_pixel_x = (pre_move_player_x + (end_col - start_col) * progress) * texture_size
                        player_pixel_y = (pre_move_player_y + (end_row - start_row) * progress) * texture_size
                        
                        box_pixel_x = box_pixel_y = None
                        if is_push and box_start_row is not None:
                            box_pixel_x = (box_start_col + (box_end_col - box_start_col) * progress) * texture_size
                            box_pixel_y = (box_start_row + (box_end_row - box_start_row) * progress) * texture_size
                            
                            if temp_grid[box_start_row, box_start_col] in [2, 4]:
                                temp_grid[box_start_row, box_start_col] = 3 if temp_grid[box_start_row, box_start_col] == 4 else 0
                        
                        if temp_grid[pre_move_player_y, pre_move_player_x] in [5, 6]:
                            temp_grid[pre_move_player_y, pre_move_player_x] = 3 if temp_grid[pre_move_player_y, pre_move_player_x] == 6 else 0
                        
                        frame = board._render_grid_to_image(temp_grid, add_grid=False)
                        
                        if is_push and box_pixel_x is not None:
                            box_texture = board.renderer.handler.get_texture('box')
                            if box_texture:
                                frame.paste(box_texture, (int(box_pixel_x), int(box_pixel_y)), box_texture)
                        
                        player_texture = board.renderer.handler.get_texture('player')
                        if player_texture:
                            frame.paste(player_texture, (int(player_pixel_x), int(player_pixel_y)), player_texture)
                        
                        frames.append(np.array(frame))
                
                frames.append(np.array(board.render_to_image(add_grid=False)))
                
                if frames:
                    fps = kwargs.get('fps', 24)
                    with imageio.get_writer(str(output_path), format="FFMPEG", mode="I", fps=fps,
                                           codec="libx264", pixelformat="yuv420p", macro_block_size=1) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                
                return Path(output_path).exists()
            finally:
                board.load_state(original_state)
                
        except Exception as e:
            logging.error(f"Failed to generate Sokoban video: {e}")
            return False

