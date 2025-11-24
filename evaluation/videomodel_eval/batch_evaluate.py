#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Tuple

from core.schema import UnifiedState
from evaluation.videomodel_eval.extractor import TrajectoryExtractor
from evaluation.videomodel_eval.evaluator import TrajectoryEvaluator
from evaluation.videomodel_eval.utils import (
    get_video_info,
    draw_trajectory_comparison,
    find_matching_gt_videos,
    find_state_file
)


def evaluate_single_video(args_tuple: Tuple) -> Dict[str, Any]:
    """
    评估单个视频

    Args:
        args_tuple: (gen_video_path, dataset_dir, threshold, num_samples, result_dir, fidelity_pixel_threshold, frame_step, tracker_type, search_margin)

    Returns:
        评估结果字典
    """
    gen_video_path, dataset_dir, threshold, num_samples, result_dir, fidelity_pixel_threshold, frame_step, tracker_type, search_margin = args_tuple
    gen_path = Path(gen_video_path)
    dataset_path = Path(dataset_dir)
    
    # 查找对应的 state 文件
    state_file = find_state_file(dataset_path, gen_path.name)
    if not state_file:
        return {
            'video': gen_path.name,
            'status': 'error',
            'error': 'State file not found'
        }
    
    # 查找匹配的 GT 视频
    gt_videos = find_matching_gt_videos(dataset_path, gen_path.name)
    if not gt_videos:
        return {
            'video': gen_path.name,
            'status': 'error',
            'error': 'No matching GT videos found'
        }
    
    try:
        # 加载 state
        state = UnifiedState.load(state_file)

        # 检测是否是推箱子游戏（有 boxes）
        is_sokoban = bool(state.boxes)

        # 获取 Generated 视频信息，作为统一规格
        gen_w, gen_h, gen_fps = get_video_info(str(gen_path))

        # 统一规格：尺寸使用原始尺寸
        unified_w, unified_h = gen_w, gen_h

        # 提取 Generated 轨迹（使用 frame_step 控制采样密度）
        gen_extractor = TrajectoryExtractor(state_file, tracker_type=tracker_type, search_margin=search_margin)
        gen_traj, gen_frame, gen_w_actual, gen_h_actual = gen_extractor.extract(
            str(gen_path),
            target_size=(unified_w, unified_h),
            frame_step=frame_step
        )

        # 如果是推箱子，额外提取箱子轨迹用于 SR 计算
        gen_box_traj = None
        if is_sokoban:
            gen_box_traj, _, _, _ = gen_extractor.extract_box(
                str(gen_path),
                target_size=(unified_w, unified_h),
                frame_step=frame_step
            )

        # 创建评估器（使用传入的参数）
        evaluator = TrajectoryEvaluator(
            eps_ratio=threshold,
            num_samples=num_samples,
            fidelity_pixel_threshold=fidelity_pixel_threshold
        )

        # 对每个 GT 视频进行评估，选择最佳匹配
        best_result = None
        best_score = -float('inf')  # 改为负无穷，接受任何得分
        best_gt_video = None
        best_gt_traj = None
        best_gt_frame = None

        for gt_video in gt_videos:
            try:
                # 提取 GT 轨迹（使用 frame_step 控制采样密度）
                gt_extractor = TrajectoryExtractor(state_file, tracker_type=tracker_type, search_margin=search_margin)
                gt_traj, gt_frame, gt_w_actual, gt_h_actual = gt_extractor.extract(
                    gt_video,
                    target_size=(unified_w, unified_h),
                    frame_step=frame_step
                )

                # 评估（传递视频尺寸、生成视频路径和箱子轨迹）
                # 注意：只传递 gen_box_traj，不需要 gt_box_traj
                result = evaluator.evaluate(
                    gt_traj, gen_traj, unified_w, unified_h,
                    state=state,
                    gen_video_path=str(gen_path),
                    gen_box_traj=gen_box_traj
                )

                # 计算综合得分 (PR - |Step|)
                # Step 可能为 None（当 SR=0 时）
                step_value = result['step']
                if step_value is not None:
                    score = result['pr'] - abs(step_value)
                else:
                    score = result['pr']

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_gt_video = gt_video
                    best_gt_traj = gt_traj
                    best_gt_frame = gt_frame
            except Exception as e:
                # GT视频处理失败，跳过该视频
                import traceback
                print(f"    Warning: Failed to process GT video {Path(gt_video).name}: {str(e)}")
                continue

        # 移除 best_result 为 None 的检查，即使得分很低也记录
        if best_result is None:
            # 这种情况理论上不应该发生（除非 gt_videos 为空，但前面已经检查过了）
            return {
                'video': gen_path.name,
                'status': 'error',
                'error': 'Failed to evaluate any GT video'
            }
        
        # 生成可视化图片
        img_dir = Path(result_dir) / 'trajectories'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        frame_to_use = gen_frame if gen_frame is not None else best_gt_frame
        if frame_to_use is not None:
            metrics = {
                'pr': best_result['pr'],
                'step': best_result['step'],
                'sr': best_result.get('sr', 0.0),
                'em': best_result['em'],
                'fidelity': best_result.get('fidelity', 0.0)
            }

            player_bbox = state.get_player_bbox()
            goal_bbox = state.get_goal_bbox()

            # 如果是推箱子，获取箱子 bbox
            box_bbox = None
            if is_sokoban and state.boxes:
                box_bbox = state.boxes[0].bbox

            img_path = img_dir / f"{gen_path.stem}.png"

            # 使用重采样后的轨迹进行可视化（归一化坐标需要转回像素坐标）
            gt_resampled_norm = best_result.get('gt_resampled')
            gen_resampled_norm = best_result.get('gen_resampled')

            if gt_resampled_norm is not None:
                gt_traj_vis = gt_resampled_norm.copy()
                gt_traj_vis[:, 0] *= unified_w
                gt_traj_vis[:, 1] *= unified_h
            else:
                gt_traj_vis = best_gt_traj

            if gen_resampled_norm is not None:
                gen_traj_vis = gen_resampled_norm.copy()
                gen_traj_vis[:, 0] *= unified_w
                gen_traj_vis[:, 1] *= unified_h
            else:
                gen_traj_vis = gen_traj

            # 如果是推箱子，准备箱子轨迹用于可视化
            gen_box_traj_vis = None
            if is_sokoban and gen_box_traj is not None:
                gen_box_traj_vis = gen_box_traj.copy()

            draw_trajectory_comparison(
                frame_to_use,
                gt_traj_vis,
                gen_traj_vis,
                str(img_path),
                gen_path.name,
                metrics,
                goal_center=None,
                player_bbox=player_bbox,
                goal_bbox=goal_bbox,
                render_config=state.render,
                box_bbox=box_bbox,
                gt_box_traj=None,
                gen_box_traj=gen_box_traj_vis
            )

        # 返回结果
        # step 可能为 None（当 SR<1 时）
        step_value = best_result['step']
        return {
            'video': gen_path.name,
            'status': 'success',
            'pr': float(best_result['pr']),
            'step': float(step_value) if step_value is not None else None,
            'em': float(best_result['em']),
            'sr': float(best_result.get('sr', 0.0)),
            'fidelity': float(best_result.get('fidelity', 0.0)),
            'is_perfect': bool(best_result['is_perfect']),
            'gt_length': float(best_result['gt_length']),
            'gen_length': float(best_result['gen_length']),
            'best_gt_video': Path(best_gt_video).name,
            'num_gt_videos': len(gt_videos),
            'score': float(best_score)
        }
        
    except Exception as e:
        import traceback
        return {
            'video': gen_path.name,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def evaluate_difficulty(dataset_dir: str, output_dir: str, result_dir: str,
                       threshold: float, num_samples: int, workers: int, difficulty: str,
                       fidelity_pixel_threshold: int = 5, frame_step: int = 1,
                       tracker_type: str = 'template', search_margin: int = 50) -> Dict[str, Any]:
    """评估单个难度的所有视频"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 Generated 视频（匹配该难度）
    gen_videos = sorted([v for v in output_path.rglob('*.mp4') if v.stem.startswith(difficulty)])

    if not gen_videos:
        print(f"  No {difficulty} videos found")
        return None

    print(f"  Found {len(gen_videos)} videos")

    # 准备任务
    tasks = [
        (str(v), str(dataset_path), threshold, num_samples, str(result_path),
         fidelity_pixel_threshold, frame_step, tracker_type, search_margin)
        for v in gen_videos
    ]
    
    # 统计数据
    all_prs = []
    all_steps = []
    all_perfect = []
    all_srs = []
    all_fidelities = []
    results_list = []
    completed = 0

    # 并行处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_single_video, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results_list.append(result)
            completed += 1

            if result['status'] == 'success':
                all_prs.append(result['pr'])
                # Step 只在 SR=1 时有值，其他情况为 None
                step_value = result['step']
                if step_value is not None:
                    all_steps.append(step_value)
                all_perfect.append(result['is_perfect'])
                all_srs.append(result.get('sr', 0.0))
                all_fidelities.append(result.get('fidelity', 0.0))

                # 计算 score 时处理 None
                if step_value is not None:
                    score = result.get('score', result['pr'] - abs(step_value))
                else:
                    score = result.get('score', result['pr'])

                num_gt = result.get('num_gt_videos', 1)
                best_gt = result.get('best_gt_video', 'N/A')
                sr = result.get('sr', 0.0)

                # 打印时处理 None
                step_str = f"{step_value:.3f}" if step_value is not None else "N/A"
                fidelity_value = result.get('fidelity', 0.0)
                print(f"  [{completed}/{len(gen_videos)}] {result['video']}: "
                      f"PR={result['pr']:.3f}, Step={step_str}, SR={sr:.0f}, "
                      f"Fidelity={fidelity_value:.3f}, Score={score:.3f}, Perfect={result['is_perfect']}, "
                      f"Best_GT={best_gt}, Num_GT={num_gt}")
            else:
                print(f"  [{completed}/{len(gen_videos)}] {result['video']}: ERROR - {result['error']}")

    # 计算 EM（Perfect Path 比例）和 SR（Success Rate）
    em = sum(all_perfect) / len(all_perfect) if all_perfect else 0.0
    sr = sum(all_srs) / len(all_srs) if all_srs else 0.0

    # 统计数据
    # avg_step 只计算 SR=1 的样本（all_steps 中只包含非 None 的值）
    summary = {
        'difficulty': difficulty,
        'total_videos': len(gen_videos),
        'successful': len(all_prs),
        'failed': len(gen_videos) - len(all_prs),
        'avg_pr': float(np.mean(all_prs)) if all_prs else 0.0,
        'avg_step': float(np.mean(all_steps)) if all_steps else 0.0,  # 只包含 SR=1 的样本
        'step_count': len(all_steps),  # 有多少个样本计算了 step（即 SR=1 的数量）
        'em': float(em),
        'sr': float(sr),
        'fidelity': float(np.mean(all_fidelities)) if all_fidelities else 0.0,
        'perfect_count': sum(all_perfect),
        'perfect_ratio': float(em),
        'success_count': int(sum(all_srs)),
        'success_rate': float(sr)
    }

    # 保存结果
    results_file = result_path / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results_list}, f, indent=2, ensure_ascii=False)

    print(f"  PR={summary['avg_pr']:.3f}, Step={summary['avg_step']:.3f} (n={summary['step_count']}), "
          f"EM={summary['em']:.3f}, SR={summary['sr']:.3f}, Fidelity={summary.get('fidelity', 0.0):.3f}, "
          f"Perfect={summary['perfect_count']}/{summary['successful']}")
    print(f"  Results: {results_file}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量评估视频轨迹')
    parser.add_argument('dataset_dir', help='数据集根目录（包含 easy/medium/hard 子目录）')
    parser.add_argument('output_dir', help='Generated 视频目录')
    parser.add_argument('result_dir', help='结果输出目录')
    parser.add_argument('--threshold', type=float, default=0.01, help='匹配阈值（相对对角线比例）')
    parser.add_argument('--num-samples', type=int, default=100, help='重采样点数，默认100')
    parser.add_argument('--workers', type=int, default=4, help='并行工作进程数')
    parser.add_argument('--difficulty', type=str, default=None, help='指定难度（easy/medium/hard），不指定则评估所有')
    parser.add_argument('--gpu', action='store_true', help='启用GPU加速（需要安装CuPy）')
    parser.add_argument('--fidelity-pixel-threshold', type=int, default=5,
                        help='保真度像素差异阈值（默认5，即±5灰度值）')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='轨迹采样步长（1=每帧采样，2=每2帧采样一次，默认1）')
    parser.add_argument('--tracker-type', type=str, default='template',
                        choices=['csrt', 'template', 'optical_flow'],
                        help='追踪器类型（csrt=OpenCV CSRT, template=模板匹配, optical_flow=光流法，默认template）')
    parser.add_argument('--search-margin', type=int, default=50,
                        help='模板匹配搜索边距（0=全图搜索，>0=局部搜索范围，默认50像素）')
    args = parser.parse_args()

    # 如果指定了--gpu，设置环境变量
    if args.gpu:
        import os
        os.environ['USE_GPU'] = '1'
        print("GPU加速模式已启用")

    difficulties = [args.difficulty] if args.difficulty else ['easy', 'medium', 'hard']

    print("=" * 60)
    print("Video Trajectory Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Threshold: {args.threshold} (ratio)")
    print(f"Num Samples: {args.num_samples}")
    print(f"Frame Step: {args.frame_step} (1=每帧采样)")
    print(f"Tracker Type: {args.tracker_type}")
    print(f"Search Margin: {args.search_margin} (0=全图搜索)")
    print()

    all_summaries = []

    for difficulty in difficulties:
        print(f"Evaluating {difficulty}...")
        dataset_path = Path(args.dataset_dir) / difficulty

        if not dataset_path.exists():
            print(f"  Skipping (directory not found)")
            continue

        result_path = Path(args.result_dir) / difficulty
        summary = evaluate_difficulty(
            str(dataset_path),
            args.output_dir,
            str(result_path),
            args.threshold,
            args.num_samples,
            args.workers,
            difficulty,
            args.fidelity_pixel_threshold,
            args.frame_step,
            args.tracker_type,
            args.search_margin
        )

        if summary:
            all_summaries.append(summary)
        print()

    # 总体统计
    if all_summaries:
        print("=" * 60)
        print("Overall Summary")
        print("=" * 60)
        total_videos = sum(s['total_videos'] for s in all_summaries)
        total_successful = sum(s['successful'] for s in all_summaries)
        total_perfect = sum(s['perfect_count'] for s in all_summaries)
        total_success = sum(s.get('success_count', 0) for s in all_summaries)
        avg_pr = np.mean([s['avg_pr'] for s in all_summaries])
        avg_step = np.mean([s['avg_step'] for s in all_summaries])
        avg_em = np.mean([s['em'] for s in all_summaries])
        avg_sr = np.mean([s.get('sr', 0.0) for s in all_summaries])

        print(f"Total videos: {total_videos}")
        print(f"Successful: {total_successful}")
        print(f"Perfect paths: {total_perfect}")
        print(f"Success paths (reached goal): {total_success}")
        print(f"Average PR: {avg_pr:.3f}")
        print(f"Average Step: {avg_step:.3f}")
        print(f"Average EM: {avg_em:.3f}")
        print(f"Average SR: {avg_sr:.3f}")


if __name__ == '__main__':
    main()

