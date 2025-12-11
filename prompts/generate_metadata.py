# -*- coding: utf-8 -*-
"""
数据集 metadata 生成工具。

支持灵活选择多种迷宫、皮肤和难度的组合生成 metadata.csv 文件。
使用动态 prompt 系统，根据皮肤配置生成对应的 prompt。
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

from . import get_dynamic_prompt, GAME_ALIASES

DATASET_ROOT = "downloaded_dataset"

# skins 目录相对于此文件的路径
SKINS_ROOT = Path(__file__).parent.parent / "skins"


def get_image_video_pairs(difficulty_dir: Path, split_name: str, game_type: str, skin_id: str, difficulty: str) -> list:
    """获取图片和视频配对，返回相对路径。"""
    images_dir = difficulty_dir / "images"
    videos_dir = difficulty_dir / "videos"

    if not images_dir.exists() or not videos_dir.exists():
        return []

    image_files = sorted(images_dir.glob("*.png"))
    pairs = []

    for image_path in image_files:
        base_name = image_path.stem
        video_files = []

        # 查找带数字后缀的视频（如 xxx_0.mp4, xxx_1.mp4）
        for video_path in sorted(videos_dir.glob(f"{base_name}_*.mp4")):
            suffix = video_path.stem[len(base_name) + 1:]
            if suffix.isdigit():
                video_files.append(video_path)

        # 如果没有找到，查找不带后缀的视频
        if not video_files:
            video_path = videos_dir / f"{base_name}.mp4"
            if video_path.exists():
                video_files.append(video_path)

        if video_files:
            for video_path in video_files:
                video_rel = f"{split_name}/{game_type}/{skin_id}/{difficulty}/videos/{video_path.name}"
                image_rel = f"{split_name}/{game_type}/{skin_id}/{difficulty}/images/{image_path.name}"
                pairs.append({'video': video_rel, 'image': image_rel})

    return pairs


def collect_all_data(
    dataset_root: Path,
    split_name: str,
    game_types: Optional[List[str]] = None,
    skin_ids: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    skins_root: Optional[Path] = None,
) -> dict:
    """收集数据并按 {game_type}_{skin_id}_{difficulty} 分组，使用动态 prompt。"""
    split_dir = dataset_root / split_name

    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return {}

    if difficulties is None:
        difficulties = ['easy', 'medium', 'hard']

    if skins_root is None:
        skins_root = SKINS_ROOT

    grouped_data = defaultdict(list)

    for game_dir in sorted(split_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        game_type = game_dir.name

        # 过滤游戏类型
        if game_types and game_type not in game_types:
            continue

        for skin_dir in sorted(game_dir.iterdir()):
            if not skin_dir.is_dir():
                continue

            skin_id = skin_dir.name

            # 过滤皮肤ID
            if skin_ids and skin_id not in skin_ids:
                continue

            # 为每个 (game_type, skin_id) 组合生成动态 prompt
            try:
                prompt = get_dynamic_prompt(game_type, skin_id, skins_root)
            except ValueError as e:
                print(f"  Warning: {e}")
                prompt = ""

            for difficulty in difficulties:
                difficulty_dir = skin_dir / difficulty
                if not difficulty_dir.exists():
                    continue

                pairs = get_image_video_pairs(difficulty_dir, split_name, game_type, skin_id, difficulty)
                group_key = f"{game_type}_{skin_id}_{difficulty}"

                for pair in pairs:
                    grouped_data[group_key].append({
                        'video': pair['video'],
                        'image': pair['image'],
                        'prompt': prompt
                    })

    return grouped_data


def write_metadata_csv(output_dir: Path, data: list) -> int:
    """写入 metadata.csv 文件。"""
    if not data:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'prompt', 'input_image'])

        for item in data:
            writer.writerow([item['video'], item['prompt'], item['image']])

    return len(data)


def process_split(
    dataset_root: Path,
    split_name: str,
    metadata_root: Path,
    game_types: Optional[List[str]] = None,
    skin_ids: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    merge_mode: str = 'separate',
    skins_root: Optional[Path] = None,
):
    """处理单个数据集分割。"""
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name} split")
    print('=' * 60)

    grouped_data = collect_all_data(
        dataset_root, split_name, game_types, skin_ids, difficulties, skins_root
    )

    if not grouped_data:
        print(f"No data found for {split_name}")
        return

    output_split_dir = metadata_root / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    total_groups = 0

    if merge_mode == 'merge':
        # 合并模式：所有数据合并到一个 metadata.csv
        all_data = []
        for data in grouped_data.values():
            all_data.extend(data)

        # 生成合并后的文件名
        parts = []
        if game_types:
            parts.append('_'.join(sorted(game_types)))
        if skin_ids:
            parts.append('_'.join(sorted(skin_ids)))
        if difficulties:
            parts.append('_'.join(sorted(difficulties)))

        merged_name = '_'.join(parts) if parts else 'all'
        merged_dir = output_split_dir / merged_name

        count = write_metadata_csv(merged_dir, all_data)
        print(f"  {merged_name}: {count} entries (merged)")
        total_entries = count
        total_groups = 1
    else:
        # 分离模式：每个组合单独生成 metadata.csv
        for group_key, data in sorted(grouped_data.items()):
            group_dir = output_split_dir / group_key
            count = write_metadata_csv(group_dir, data)

            if count > 0:
                print(f"  {group_key}: {count} entries")
                total_entries += count
                total_groups += 1

    print(f"\nTotal groups: {total_groups}")
    print(f"Total entries: {total_entries}")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='生成数据集 metadata.csv 文件，支持灵活选择游戏类型、皮肤和难度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 生成所有数据（默认）:
   python generate_metadata.py

2. 只生成 maze 和 maze3d 的数据:
   python generate_metadata.py --games maze maze3d

3. 只生成皮肤 1 和 2 的数据:
   python generate_metadata.py --skins 1 2

4. 只生成 easy 和 medium 难度:
   python generate_metadata.py --difficulties easy medium

5. 组合条件（maze 游戏，皮肤 1，easy 难度）:
   python generate_metadata.py --games maze --skins 1 --difficulties easy

6. 合并模式（将所有符合条件的数据合并到一个 metadata.csv）:
   python generate_metadata.py --games maze sokoban --merge

7. 只处理 train 数据集:
   python generate_metadata.py --splits train

8. 复杂组合（多游戏、多皮肤、多难度，合并）:
   python generate_metadata.py --games maze irregular_maze --skins 1 2 3 --difficulties easy hard --merge
        """
    )

    parser.add_argument(
        '--games',
        nargs='+',
        choices=['maze', 'irregular_maze', 'maze3d', 'sokoban', 'trapfield'],
        help='选择游戏类型（可多选）'
    )

    parser.add_argument(
        '--skins',
        nargs='+',
        help='选择皮肤ID（可多选，如: 1 2 3）'
    )

    parser.add_argument(
        '--difficulties',
        nargs='+',
        choices=['easy', 'medium', 'hard'],
        help='选择难度（可多选）'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'eval'],
        default=['train', 'eval'],
        help='选择数据集分割（默认: train eval）'
    )

    parser.add_argument(
        '--merge',
        action='store_true',
        help='合并模式：将所有符合条件的数据合并到一个 metadata.csv'
    )

    parser.add_argument(
        '--dataset-root',
        default=DATASET_ROOT,
        help=f'数据集根目录（默认: {DATASET_ROOT}）'
    )

    parser.add_argument(
        '--skins-root',
        default=None,
        help='皮肤目录根路径（默认: VR-Bench/skins）'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    skins_root = Path(args.skins_root) if args.skins_root else SKINS_ROOT

    print("=" * 60)
    print("Generating metadata structure (with dynamic prompts)")
    print(f"Dataset root: {dataset_root.absolute()}")
    print(f"Skins root: {skins_root.absolute()}")
    print("=" * 60)

    # 显示过滤条件
    if args.games:
        print(f"Game types: {', '.join(args.games)}")
    if args.skins:
        print(f"Skin IDs: {', '.join(args.skins)}")
    if args.difficulties:
        print(f"Difficulties: {', '.join(args.difficulties)}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Mode: {'merge' if args.merge else 'separate'}")
    print("=" * 60)

    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        return

    if not skins_root.exists():
        print(f"Warning: Skins root not found: {skins_root}")
        print("Dynamic prompts may fail for some game types.")

    metadata_root = dataset_root / "metadata"
    metadata_root.mkdir(exist_ok=True)

    merge_mode = 'merge' if args.merge else 'separate'

    for split in args.splits:
        process_split(
            dataset_root,
            split,
            metadata_root,
            game_types=args.games,
            skin_ids=args.skins,
            difficulties=args.difficulties,
            merge_mode=merge_mode,
            skins_root=skins_root,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Metadata files created in: {metadata_root.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

