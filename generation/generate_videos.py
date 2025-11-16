#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schema import UnifiedState
from generation.path_finder import find_optimal_paths

Coordinate = Tuple[int, int]

logging.basicConfig(level=logging.INFO, format='%(message)s')


def generate_maze_video(
    state_path: str,
    path: List[Coordinate],
    output_path: str,
    assets_folder: str = None
):
    from games.maze.generators.video_gen import create_solution_video

    state = UnifiedState.load(state_path)

    if not assets_folder:
        from games.maze.default_textures import ensure_default_textures
        assets_folder = str(ensure_default_textures())

    create_solution_video(
        state.grid.data,
        path,
        cell_size=state.render.cell_size,
        save_path=output_path,
        frame_duration_ms=300,
        assets_folder=assets_folder
    )


def generate_trapfield_video(
    state_path: str,
    path: List[Coordinate],
    output_path: str,
    assets_folder: str = None
):
    from games.trapfield.renderer import create_solution_video

    state = UnifiedState.load(state_path)

    if not assets_folder:
        assets_folder = str(Path(__file__).parent.parent / 'games' / 'trapfield' / 'assets')

    create_solution_video(
        state.grid.data,
        path,
        cell_size=state.render.cell_size,
        save_path=output_path,
        frame_duration_ms=300,
        assets_folder=assets_folder
    )


def generate_pathfinder_video(state: UnifiedState, node_path: List, output_path: str, assets_folder: str = None):
    """生成PathFinder视频"""
    from games.pathfinder.board import PathFinderBoard, PathSegment
    from games.pathfinder.renderer import render_solution_video

    segments_data = state.metadata.get('segments', [])
    road_width = state.metadata.get('road_width', 35)

    board_size = state.render.image_width if hasattr(state.render, 'image_width') else 800

    segments = []
    for segment_data in segments_data:
        if len(segment_data) >= 2:
            segments.append(PathSegment(segment_data))

    start_point = state.player.pixel_pos if state.player.pixel_pos else (0, 0)
    end_point = state.goal.pixel_pos if state.goal.pixel_pos else (0, 0)

    solution_path_tuples = [(node.x, node.y) for node in node_path]

    solution_segments = []
    segment_map = {}
    for idx, segment in enumerate(segments_data):
        if len(segment) < 2:
            continue
        p1, p2 = tuple(segment[0]), tuple(segment[-1])
        segment_map[(p1, p2)] = idx
        segment_map[(p2, p1)] = idx

    for i in range(len(node_path) - 1):
        node1, node2 = node_path[i], node_path[i + 1]
        edge = ((node1.x, node1.y), (node2.x, node2.y))
        if edge in segment_map:
            solution_segments.append(segment_map[edge])

    board = PathFinderBoard(
        segments=segments,
        start_point=start_point,
        end_point=end_point,
        solution_segments=solution_segments,
        solution_path=solution_path_tuples,
        image_size=board_size,
        road_width=road_width
    )
    render_solution_video(board, output_path, fps=24, use_gpu=True, assets_folder=assets_folder)


def generate_sokoban_video(state_path: str, path: List[Coordinate], output_path: str, assets_folder: str = None):
    """生成Sokoban视频"""
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
        return

    moves = solver.get_solution_path()
    if not moves:
        return

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
            with imageio.get_writer(str(output_path), format="FFMPEG", mode="I", fps=24,
                                   codec="libx264", pixelformat="yuv420p", macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(frame)
    finally:
        board.load_state(original_state)


def extract_first_frame(video_path: str, image_path: str):
    """从视频提取第一帧"""
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
        imageio.imwrite(image_path, first_frame)
        return True
    except:
        return False


def process_single_path(args):
    """处理单条路径"""
    state_path, game_type, path_id, path, video_path, skip_existing, assets_folder = args

    try:
        if skip_existing and Path(video_path).exists():
            return (path_id, 'skipped', None)

        state = UnifiedState.load(str(state_path))

        if game_type == 'maze':
            generate_maze_video(str(state_path), path, str(video_path), assets_folder)
        elif game_type == 'trapfield':
            generate_trapfield_video(str(state_path), path, str(video_path), assets_folder)
        elif game_type == 'pathfinder':
            generate_pathfinder_video(state, path, str(video_path), assets_folder)
        elif game_type == 'sokoban':
            generate_sokoban_video(str(state_path), path, str(video_path), assets_folder)

        if Path(video_path).exists():
            return (path_id, 'success', None)
        else:
            return (path_id, 'failed', '视频生成失败')

    except Exception as e:
        return (path_id, 'failed', str(e))


def process_single_state(
    state_path: Path,
    game_type: str,
    skip_existing: bool = False,
    verbose: bool = True,
    num_workers: int = 4,
    assets_folder: str = None
) -> dict:
    """处理单个state文件，生成所有最优路径的视频"""
    stats = {'total_paths': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    try:
        state = UnifiedState.load(str(state_path))
        paths = find_optimal_paths(state, game_type)

        if game_type == 'pathfinder':
            paths = [p[0] for p in paths]

        stats['total_paths'] = len(paths)

        if not paths:
            if verbose:
                logging.warning(f"⚠️  {state_path.name}: 未找到路径")
            return stats

        state_dir = state_path.parent
        videos_dir = state_dir.parent / 'videos'
        images_dir = state_dir.parent / 'images'
        videos_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        base_name = state_path.stem
        tasks = [(state_path, game_type, i, path, str(videos_dir / f"{base_name}_{i}.mp4"), skip_existing, assets_folder)
                 for i, path in enumerate(paths)]

        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_path, tasks)

        first_video = videos_dir / f"{base_name}_0.mp4"
        if first_video.exists():
            extract_first_frame(str(first_video), str(images_dir / f"{base_name}.png"))

        for path_id, status, error_msg in results:
            if status == 'success':
                stats['success'] += 1
            elif status == 'skipped':
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
                if verbose:
                    logging.error(f"  ❌ {base_name}_{path_id}: {error_msg}")

    except Exception as e:
        if verbose:
            logging.error(f"❌ {state_path.name}: {e}")

    return stats


def batch_process_dataset(
    dataset_root: str,
    skip_existing: bool = False,
    verbose: bool = True,
    num_workers: int = None,
    parallel_states: int = 1,
    assets_folder: str = None
):
    """批量处理dataset目录"""
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    dataset_path = Path(dataset_root)
    state_files = []

    for state_file in dataset_path.rglob('states/*.json'):
        path_str = str(state_file).lower()
        if 'pathfinder' in path_str or 'irregular_maze' in path_str:
            game_type = 'pathfinder'
        elif 'maze' in path_str and '3d' not in path_str:
            game_type = 'maze'
        elif 'trapfield' in path_str:
            game_type = 'trapfield'
        else:
            continue
        state_files.append((state_file, game_type))

    logging.info(f"找到 {len(state_files)} 个state文件")
    logging.info(f"并行: {parallel_states} states, 每个{num_workers} workers")
    if assets_folder:
        logging.info(f"皮肤: {assets_folder}\n")
    else:
        logging.info(f"皮肤: 默认\n")

    by_game = {}
    for state_file, game_type in state_files:
        by_game.setdefault(game_type, []).append(state_file)

    total_stats = {'total_states': 0, 'total_paths': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    for game_type in sorted(by_game.keys()):
        state_paths = by_game[game_type]
        logging.info(f"\n{'='*60}")
        logging.info(f"{game_type.upper()}: {len(state_paths)} files")
        logging.info(f"{'='*60}\n")

        if parallel_states > 1:
            process_func = partial(process_single_state, game_type=game_type,
                                 skip_existing=skip_existing, verbose=False, num_workers=num_workers,
                                 assets_folder=assets_folder)
            with Pool(processes=parallel_states) as pool:
                results = pool.map(process_func, sorted(state_paths))

            for i, stats in enumerate(results):
                for k in total_stats:
                    total_stats[k] += stats.get(k.replace('total_states', 'total_paths') if k == 'total_states' else k, 0)
                total_stats['total_states'] += 1
                if verbose:
                    s = stats
                    logging.info(f"✅ {sorted(state_paths)[i].name}: {s['total_paths']}路径 "
                               f"{s['success']}成功 {s['skipped']}跳过 {s['failed']}失败")
        else:
            for state_path in sorted(state_paths):
                if verbose:
                    logging.info(f"📄 {state_path.relative_to(dataset_path)}")
                stats = process_single_state(state_path, game_type, skip_existing, verbose, num_workers, assets_folder)
                total_stats['total_states'] += 1
                for k in ['total_paths', 'success', 'failed', 'skipped']:
                    total_stats[k] += stats[k]

    logging.info(f"\n{'='*60}")
    logging.info("完成")
    logging.info(f"{'='*60}")
    logging.info(f"States: {total_stats['total_states']}, Paths: {total_stats['total_paths']}")
    logging.info(f"成功: {total_stats['success']}, 跳过: {total_stats['skipped']}, 失败: {total_stats['failed']}")
    logging.info(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量生成GT视频')
    parser.add_argument('dataset', help='Dataset根目录')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在文件')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'每个state的workers (默认: {max(1, cpu_count()//2)})')
    parser.add_argument('--parallel-states', type=int, default=1, help='并行处理的states数')
    parser.add_argument('--skin', type=str, default=None,
                       help='皮肤文件夹路径 (例如: skins/maze/5)')
    args = parser.parse_args()

    batch_process_dataset(args.dataset, args.skip_existing, not args.quiet,
                         args.workers, args.parallel_states, args.skin)

if __name__ == '__main__':
    main()

