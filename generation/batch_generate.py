"""
统一的并发数据集生成工具
支持多种游戏类型，通过适配器模式实现
"""

import yaml
import logging
import multiprocessing
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game_adapter import GameAdapter, LevelDeduplicator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path: str = "config.yaml") -> dict:
    # 如果是相对路径，从项目根目录查找
    config_file = Path(config_path)
    if not config_file.is_absolute() and not config_file.exists():
        # 尝试从脚本的父目录（项目根目录）查找
        config_file = Path(__file__).parent.parent / config_path

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_game_adapter(game_type: str) -> GameAdapter:
    """根据游戏类型获取对应的适配器"""
    if game_type == "sokoban":
        from games.sokoban.adapter import SokobanAdapter
        return SokobanAdapter()
    elif game_type == "maze":
        from games.maze.adapter import MazeAdapter
        return MazeAdapter()
    elif game_type == "pathfinder":
        from games.pathfinder.adapter import PathFinderAdapter
        return PathFinderAdapter()
    elif game_type == "trapfield":
        from games.trapfield.adapter import TrapFieldAdapter
        return TrapFieldAdapter()
    elif game_type == "maze3d":
        from games.maze3d.adapter import Maze3DAdapter
        return Maze3DAdapter()
    else:
        raise ValueError(f"Unknown game type: {game_type}")


def find_skin_folders(skins_root: Path, adapter: GameAdapter) -> List[Path]:
    """查找有效的皮肤文件夹"""
    if not skins_root.exists():
        logging.warning(f"Skins root folder not found: {skins_root}")
        return []

    skin_folders = []
    required_files = adapter.get_required_texture_files()

    for item in skins_root.iterdir():
        if item.is_dir():
            has_all = all(
                any((item / f"{name}{ext}").exists() for ext in ['.png', '.jpg', '.jpeg'])
                for name in required_files
            )
            if has_all:
                skin_folders.append(item)

    return skin_folders


def generate_level_with_retry(
    adapter: GameAdapter,
    difficulty_config: Dict[str, Any],
    deduplicator: LevelDeduplicator,
    max_retries: int,
    assets_folder: str
) -> tuple:
    """使用适配器生成关卡，支持重试和去重"""
    duplicate_count = 0

    for retry in range(max_retries):
        try:
            level = adapter.generate_level(difficulty_config, assets_folder)

            if level is None:
                continue

            level_hash = adapter.get_level_hash(level)

            if not deduplicator.is_duplicate(level_hash):
                deduplicator.add_hash(level_hash)
                return level, duplicate_count, None
            else:
                duplicate_count += 1

        except Exception as e:
            logging.warning(f"Generation attempt {retry + 1} failed: {e}")

    return None, duplicate_count, f"Failed after {max_retries} retries"


def generate_single_task(args):
    """生成单个任务（一个皮肤的一个难度）"""
    game_type, skin_folder, difficulty_name, difficulty_config, output_root, gen_config, colors = args

    # 获取游戏适配器
    adapter = get_game_adapter(game_type)

    # 处理皮肤文件夹（可能为 None）
    if skin_folder is None:
        skin_name = "default"
        assets_folder = ""
    else:
        skin_name = skin_folder.name
        assets_folder = str(skin_folder)

    output_dir = Path(output_root) / skin_name / difficulty_name
    output_dir.mkdir(parents=True, exist_ok=True)

    count = difficulty_config['count']
    fps = gen_config.get('fps', 2)
    add_grid = gen_config.get('add_grid', False)
    max_retries = gen_config.get('max_duplicate_retries', 100)

    deduplicator = LevelDeduplicator()

    logging.info(f"[{game_type}][{skin_name}/{difficulty_name}] Start generating {count} levels")

    generated = 0
    failed = 0
    level_records = []

    for i in range(count):
        level, duplicate_count, error = generate_level_with_retry(
            adapter,
            difficulty_config,
            deduplicator,
            max_retries,
            assets_folder
        )

        if level is None:
            logging.error(f"[{skin_name}/{difficulty_name}] Level {i+1}: {error}")
            failed += 1
            level_records.append({
                'level_id': i + 1,
                'files': {},
                'status': 'failed',
                'duplicate_retries': duplicate_count,
                'error': error
            })
            continue

        try:
            # 使用适配器保存关卡
            files = adapter.save_level(
                level,
                output_dir,
                i + 1,
                difficulty_name,
                fps=fps,
                add_grid=add_grid,
                assets_folder=assets_folder,
                colors=colors
            )

            # 检查是否成功生成了文件
            if files.get('video') or files.get('image'):
                generated += 1
                file_list = ', '.join([f for f in files.values() if f])
                logging.info(f"[{skin_name}/{difficulty_name}] {i+1}/{count}: {file_list} (duplicates: {duplicate_count})")
                level_records.append({
                    'level_id': i + 1,
                    'files': files,
                    'status': 'success',
                    'duplicate_retries': duplicate_count,
                    'level_hash': adapter.get_level_hash(level)
                })
            else:
                logging.warning(f"[{skin_name}/{difficulty_name}] No files generated for level {i+1}")
                failed += 1
                level_records.append({
                    'level_id': i + 1,
                    'files': files,
                    'status': 'no_files',
                    'duplicate_retries': duplicate_count
                })

        except Exception as e:
            logging.error(f"[{skin_name}/{difficulty_name}] Save failed for level {i+1}: {e}")
            failed += 1
            level_records.append({
                'level_id': i + 1,
                'files': {},
                'status': 'save_failed',
                'duplicate_retries': duplicate_count,
                'error': str(e)
            })

    # 保存结果记录
    result_file = output_dir / 'results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'skin': skin_name,
            'difficulty': difficulty_name,
            'total': count,
            'generated': generated,
            'failed': failed,
            'total_duplicates': sum(r['duplicate_retries'] for r in level_records),
            'levels': level_records
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"[{skin_name}/{difficulty_name}] Done: {generated}/{count} (failed: {failed})")

    return {
        'skin': skin_name,
        'difficulty': difficulty_name,
        'generated': generated,
        'failed': failed,
        'total': count
    }


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    game_type = config.get('game_type', 'sokoban')
    skins_root_str = config.get('skins_root')
    output_root = Path(config['output_root'])
    difficulties = config['difficulties']
    gen_config = config['generation']
    max_workers = config.get('parallel', {}).get('max_workers', 4)
    colors = config.get('colors', None)  # 读取颜色配置

    # 获取游戏适配器
    try:
        adapter = get_game_adapter(game_type)
    except ValueError as e:
        logging.error(str(e))
        return

    # 检查是否需要皮肤
    required_textures = adapter.get_required_texture_files()

    if required_textures and skins_root_str:
        # 需要皮肤的游戏
        skins_root = Path(skins_root_str)
        skin_folders = find_skin_folders(skins_root, adapter)

        if not skin_folders:
            logging.error(f"No valid skin folders found in {skins_root}")
            return

        logging.info(f"Game: {game_type}")
        logging.info(f"Found {len(skin_folders)} skins: {[s.name for s in skin_folders]}")
        logging.info(f"Difficulties: {list(difficulties.keys())}")

        # 每个皮肤×每个难度 = 一个任务
        tasks = []
        for skin_folder in skin_folders:
            for diff_name, diff_config in difficulties.items():
                merged_config = {**gen_config, **diff_config}
                tasks.append((game_type, skin_folder, diff_name, merged_config, output_root, gen_config, colors))
    else:
        # 不需要皮肤的游戏（如 PathFinder）
        logging.info(f"Game: {game_type} (no skins required)")
        logging.info(f"Difficulties: {list(difficulties.keys())}")

        # 每个难度 = 一个任务，使用虚拟皮肤文件夹
        tasks = []
        for diff_name, diff_config in difficulties.items():
            merged_config = {**gen_config, **diff_config}
            # 使用 None 作为皮肤文件夹
            tasks.append((game_type, None, diff_name, merged_config, output_root, gen_config, colors))

    logging.info(f"Total tasks: {len(tasks)}, Max workers: {max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single_task, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                logging.error(f"Task failed: {task[1].name}/{task[2]} - {e}")

    logging.info("\n=== Summary ===")
    total_generated = sum(r['generated'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    logging.info(f"Total: {total_generated} generated, {total_failed} failed")

    by_skin = {}
    for r in results:
        by_skin.setdefault(r['skin'], []).append(r)

    for skin_name, skin_results in by_skin.items():
        logging.info(f"\n[{skin_name}]:")
        for r in skin_results:
            logging.info(f"  {r['difficulty']}: {r['generated']}/{r['total']} (failed: {r['failed']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='并发生成游戏数据集')
    parser.add_argument('config', nargs='?', default='config.yaml', help='配置文件路径 (默认: config.yaml)')
    args = parser.parse_args()

    multiprocessing.freeze_support()
    main(args.config)

