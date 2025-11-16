#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

from evaluation.vlm_eval.vlm_client import VLMClient
from evaluation.vlm_eval.vlm_evaluator import VLMEvaluator
from evaluation.vlm_eval.executors.maze_executor import MazeExecutor
from evaluation.vlm_eval.executors.sokoban_executor import SokobanExecutor
from evaluation.vlm_eval.executors.trapfield_executor import TrapFieldExecutor
from evaluation.vlm_eval.executors.pathfinder_executor import PathfinderExecutor
from evaluation.vlm_eval.executors.maze3d_executor import Maze3DExecutor
from dotenv import load_dotenv
# 强制使用项目 .env 文件的值，覆盖系统环境变量
load_dotenv(override=True)


def create_executor(game: str, assets_folder: str = None):
    if game == 'maze':
        return MazeExecutor(assets_folder=assets_folder)
    elif game == 'sokoban':
        return SokobanExecutor(assets_folder=assets_folder)
    elif game == 'trapfield':
        return TrapFieldExecutor(assets_folder=assets_folder)
    elif game == 'pathfinder':
        return PathfinderExecutor(assets_folder=assets_folder)
    elif game in ['maze3d', '3dmaze' ,'maze3d_new']:
        return Maze3DExecutor(assets_folder=assets_folder)
    else:
        raise ValueError(f"Unsupported game: {game}")


def evaluate_single_model(game: str, dataset: str, model_config: Dict[str, Any],
                         output_base: str, workers: int, max_levels: int,
                         assets_folder: str = None) -> Dict[str, Any]:
    # 在子进程中重新加载环境变量，强制使用项目 .env 文件的值
    load_dotenv(override=True)

    model_name = model_config['name']
    output_dir = Path(output_base) / model_name

    logging.info(f"[{model_name}] Starting evaluation")

    vlm_client = VLMClient(
        model=model_name,
        base_url=model_config.get('base_url'),
        max_tokens=model_config.get('max_tokens', 10000),
        temperature=model_config.get('temperature', 0.0)
    )

    executor = create_executor(game, assets_folder)
    evaluator = VLMEvaluator(vlm_client, executor)

    summary = evaluator.evaluate_dataset(
        dataset_dir=dataset,
        output_dir=str(output_dir),
        max_workers=workers,
        max_levels=max_levels
    )

    logging.info(f"[{model_name}] Complete - SR: {summary['avg_sr']:.4f}, PR: {summary['avg_pr']:.4f}, MR: {summary['avg_mr']:.4f}")

    return {
        'model': model_name,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(description='VLM Game Evaluation')
    parser.add_argument('config', type=str, help='Config file path')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    game = config['game']
    dataset = config['dataset']
    output_base = config['output']
    models = config['models']
    workers = config.get('workers', 10)
    max_levels = config.get('max_levels', -1)
    assets_folder = config.get('assets_folder')

    logging.info(f"Game: {game}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Models: {[m['name'] for m in models]}")
    logging.info(f"Workers per model: {workers}")
    logging.info(f"Total parallel tasks: {len(models) * workers}")

    results = []
    with ProcessPoolExecutor(max_workers=len(models)) as executor:
        futures = []
        for model_config in models:
            future = executor.submit(
                evaluate_single_model,
                game, dataset, model_config, output_base,
                workers, max_levels, assets_folder
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Model evaluation failed: {e}")

    logging.info("\n=== Final Results ===")
    for result in results:
        model = result['model']
        summary = result['summary']
        logging.info(f"{model}: SR={summary['avg_sr']:.5f}, PR={summary['avg_pr']:.5f}, MR={summary['avg_mr']:.5f}, Step={summary['avg_step']:.5f}")


if __name__ == '__main__':
    main()
