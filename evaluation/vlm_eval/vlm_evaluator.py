import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from evaluation.vlm_eval.vlm_client import VLMClient
from evaluation.vlm_eval.game_executor import GameExecutor
from evaluation.vlm_eval.action_utils import parse_actions
from evaluation.vlm_eval.action_metrics import calculate_sr, calculate_pr, calculate_mr, calculate_step


@dataclass
class EvaluationResult:
    level_id: str
    success: bool
    metrics: Dict[str, float]
    predicted_actions: List[Dict[str, Any]]
    optimal_actions: List[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]
    initial_screenshot: Optional[str] = None
    error: Optional[str] = None


class VLMEvaluator:
    def __init__(self, vlm_client: VLMClient, game_executor: GameExecutor, max_retries: int = 3):
        self.vlm_client = vlm_client
        self.game_executor = game_executor
        self.max_retries = max_retries

    def evaluate_level(self, state_path: str, image_path: str, output_dir: Optional[str] = None) -> EvaluationResult:
        level_id = Path(state_path).stem

        try:
            state = self.game_executor.load_state(state_path)
            optimal_actions = self.game_executor.get_optimal_solution(state)

            system_prompt = self.game_executor.get_system_prompt()
            user_prompt = self.game_executor.get_user_prompt()

            # 对于 PathFinder 游戏，需要渲染带标注的图像
            game_type = self.game_executor.get_game_type()
            if game_type == 'pathfinder':
                # 创建临时目录保存带标注的图像
                if output_dir:
                    labeled_image_dir = Path(output_dir) / level_id
                    labeled_image_dir.mkdir(parents=True, exist_ok=True)
                    labeled_image_path = labeled_image_dir / f"{level_id}_labeled.png"
                else:
                    import tempfile
                    labeled_image_path = Path(tempfile.mkdtemp()) / f"{level_id}_labeled.png"

                # 渲染带标注的图像
                self.game_executor.render_state(state, str(labeled_image_path))
                image_path = str(labeled_image_path)

            predicted_actions = None
            for attempt in range(self.max_retries):
                try:
                    response = self.vlm_client.query(system_prompt, user_prompt, image_path)
                    predicted_actions = parse_actions(response, game_type)
                    break
                except Exception as e:
                    error_msg = f"Retry {attempt + 1}/{self.max_retries}: {type(e).__name__}: {e}"
                    if attempt == self.max_retries - 1:
                        logging.error(f"Failed after {self.max_retries} retries")
                        logging.error(f"Full traceback:\n{traceback.format_exc()}")
                        raise Exception(f"Failed to parse actions after {self.max_retries} retries: {e}")
                    logging.warning(error_msg)
                    logging.debug(f"Full traceback:\n{traceback.format_exc()}")

            execution_trace = []
            current_state = state
            is_win = False

            # 保存初始状态截图
            initial_screenshot_path = None
            if output_dir:
                trace_dir = Path(output_dir) / level_id / 'trace'
                trace_dir.mkdir(parents=True, exist_ok=True)
                initial_image_path = trace_dir / "step_000_initial.png"
                self.game_executor.render_state(current_state, str(initial_image_path))
                initial_screenshot_path = str(initial_image_path)

            for step_idx, action in enumerate(predicted_actions):
                new_state, success, msg = self.game_executor.execute_action(current_state, action)

                # 无论成功与否都保存截图
                screenshot_path = None
                if output_dir:
                    trace_dir = Path(output_dir) / level_id / 'trace'
                    trace_dir.mkdir(parents=True, exist_ok=True)

                    # 根据成功与否添加不同的后缀
                    status_suffix = "" if success else "_failed"
                    trace_image_path = trace_dir / f"step_{step_idx + 1:03d}{status_suffix}.png"
                    self.game_executor.render_state(new_state, str(trace_image_path))
                    screenshot_path = str(trace_image_path)

                # 构建 trace entry
                trace_entry = {
                    'step': step_idx + 1,  # 从1开始计数
                    'action': action,
                    'success': success,
                    'message': msg,
                    'screenshot': screenshot_path
                }

                # 只有当 player 存在且有 grid_pos 时才添加位置信息
                if hasattr(new_state, 'player') and new_state.player is not None:
                    if hasattr(new_state.player, 'grid_pos') and new_state.player.grid_pos is not None:
                        trace_entry['position'] = {
                            'row': new_state.player.grid_pos.row,
                            'col': new_state.player.grid_pos.col
                        }

                execution_trace.append(trace_entry)

                if not success:
                    break

                current_state = new_state

                if self.game_executor.check_win(current_state):
                    is_win = True
                    break

            sr = calculate_sr(is_win)
            pr = calculate_pr(predicted_actions, optimal_actions)
            mr = calculate_mr(predicted_actions, optimal_actions)
            step = calculate_step(predicted_actions, optimal_actions, is_win=is_win)

            result = EvaluationResult(
                level_id=level_id,
                success=True,
                metrics={'sr': sr, 'pr': pr, 'mr': mr, 'step': step},
                predicted_actions=predicted_actions,
                optimal_actions=optimal_actions,
                execution_trace=execution_trace,
                initial_screenshot=initial_screenshot_path
            )

            if output_dir:
                result_file = Path(output_dir) / level_id / 'result.json'
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(asdict(result), f, indent=2)

            return result

        except Exception as e:
            logging.error(f"Failed to evaluate {level_id}: {e}")
            return EvaluationResult(
                level_id=level_id,
                success=False,
                metrics={'sr': 0.0, 'pr': 0.0, 'mr': 0.0, 'step': 0.0},
                predicted_actions=[],
                optimal_actions=[],
                execution_trace=[],
                error=str(e)
            )

    def _evaluate_single_difficulty(self, dataset_path: Path, output_path: Path, max_workers: int, max_levels: int):
        output_path.mkdir(parents=True, exist_ok=True)

        state_files = sorted(dataset_path.glob('states/*.json'))

        # 只取后24个case
        if len(state_files) > 24:
            state_files = state_files[-24:]
            logging.info(f"选择后24个case进行评估 (总共 {len(state_files)} 个)")

        if max_levels > 0:
            state_files = state_files[:max_levels]

        results = []

        def evaluate_single(state_file):
            image_file = dataset_path / 'images' / f"{state_file.stem}.png"
            if not image_file.exists():
                logging.warning(f"Image not found: {image_file}")
                return None
            return self.evaluate_level(str(state_file), str(image_file), str(output_path))

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(evaluate_single, sf) for sf in state_files]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                        step_str = f"{result.metrics['step']:.2f}" if result.metrics['step'] is not None else "N/A"
                        logging.info(f"{result.level_id}: SR={result.metrics['sr']:.2f}, PR={result.metrics['pr']:.2f}, MR={result.metrics['mr']:.2f}, Step={step_str}")
        else:
            for state_file in state_files:
                result = evaluate_single(state_file)
                if result:
                    results.append(result)
                    step_str = f"{result.metrics['step']:.2f}" if result.metrics['step'] is not None else "N/A"
                    logging.info(f"{result.level_id}: SR={result.metrics['sr']:.2f}, PR={result.metrics['pr']:.2f}, MR={result.metrics['mr']:.2f}, Step={step_str}")

        return results

    def evaluate_dataset(self, dataset_dir: str, output_dir: str, max_workers: int = 4, max_levels: int = -1) -> Dict[str, Any]:
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        difficulty_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and (d / 'states').exists()]

        if difficulty_dirs:
            all_results = []
            for diff_dir in sorted(difficulty_dirs):
                logging.info(f"Evaluating difficulty: {diff_dir.name}")
                diff_results = self._evaluate_single_difficulty(diff_dir, output_path / diff_dir.name, max_workers, max_levels)
                all_results.extend(diff_results)
            results = all_results
        else:
            results = self._evaluate_single_difficulty(dataset_path, output_path, max_workers, max_levels)

        # 只计算成功 case 的 step
        successful_steps = [r.metrics['step'] for r in results if r.metrics['step'] is not None]

        summary = {
            'total': len(results),
            'avg_sr': sum(r.metrics['sr'] for r in results) / len(results) if results else 0.0,
            'avg_pr': sum(r.metrics['pr'] for r in results) / len(results) if results else 0.0,
            'avg_mr': sum(r.metrics['mr'] for r in results) / len(results) if results else 0.0,
            'avg_step': sum(successful_steps) / len(successful_steps) if successful_steps else 0.0,
            'successful_cases': len(successful_steps),
            'results': [asdict(r) for r in results]
        }

        summary_file = output_path / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(
            f"\nSummary: SR={summary['avg_sr']:.2f}, PR={summary['avg_pr']:.2f}, "
            f"MR={summary['avg_mr']:.2f}, Step={summary['avg_step']:.2f} "
            f"(based on {summary['successful_cases']} successful cases)"
        )

        return summary
