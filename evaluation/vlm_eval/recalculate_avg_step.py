#!/usr/bin/env python3
"""
重新计算 VLM 评估结果中的 avg_step 指标

旧方法：对所有 case 都计算 step
新方法：只对 SR=1（成功的 case）计算 step

使用方法：
    python evaluation/vlm_eval/recalculate_avg_step.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def recalculate_summary(summary_path: Path) -> Dict[str, Any]:
    """重新计算单个 summary.json 的 avg_step"""
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    results = summary.get('results', [])
    
    # 只计算 SR=1.0 的 case 的 step
    successful_steps = []
    for result in results:
        metrics = result.get('metrics', {})
        sr = metrics.get('sr', 0.0)
        step = metrics.get('step')
        
        # 只统计成功的 case (SR=1.0) 且 step 不为 None
        if sr == 1.0 and step is not None:
            successful_steps.append(step)
    
    # 计算新的 avg_step
    old_avg_step = summary.get('avg_step', 0.0)
    old_successful_cases = summary.get('successful_cases', 0)
    
    new_avg_step = sum(successful_steps) / len(successful_steps) if successful_steps else 0.0
    new_successful_cases = len(successful_steps)
    
    # 更新 summary
    summary['avg_step'] = new_avg_step
    summary['successful_cases'] = new_successful_cases
    
    return {
        'path': str(summary_path),
        'old_avg_step': old_avg_step,
        'new_avg_step': new_avg_step,
        'old_successful_cases': old_successful_cases,
        'new_successful_cases': new_successful_cases,
        'changed': abs(old_avg_step - new_avg_step) > 1e-6 or old_successful_cases != new_successful_cases,
        'updated_summary': summary
    }


def process_directory(base_dir: Path) -> List[Dict[str, Any]]:
    """处理目录下的所有 summary.json 文件"""
    
    results = []
    
    # 查找所有 summary.json 文件
    summary_files = list(base_dir.rglob('summary.json'))
    
    if not summary_files:
        print(f"  ⚠ 未找到 summary.json 文件")
        return results
    
    print(f"  找到 {len(summary_files)} 个 summary.json 文件")
    
    for summary_file in sorted(summary_files):
        try:
            result = recalculate_summary(summary_file)
            results.append(result)
            
            # 显示相对路径
            rel_path = summary_file.relative_to(base_dir)
            
            if result['changed']:
                print(f"  ✓ {rel_path}")
                print(f"    旧: avg_step={result['old_avg_step']:.4f}, successful_cases={result['old_successful_cases']}")
                print(f"    新: avg_step={result['new_avg_step']:.4f}, successful_cases={result['new_successful_cases']}")
            else:
                print(f"  - {rel_path} (无变化)")
                
        except Exception as e:
            print(f"  ✗ 处理 {summary_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def save_updated_summaries(results: List[Dict[str, Any]], dry_run: bool = False):
    """保存更新后的 summary.json 文件"""
    
    changed_count = sum(1 for r in results if r['changed'])
    
    if changed_count == 0:
        print("\n没有需要更新的文件")
        return
    
    if dry_run:
        print(f"\n[DRY RUN] 将更新 {changed_count} 个文件（实际未保存）")
        return
    
    print(f"\n正在保存 {changed_count} 个更新的文件...")
    
    for result in results:
        if result['changed']:
            summary_path = Path(result['path'])
            with open(summary_path, 'w') as f:
                json.dump(result['updated_summary'], f, indent=2)
            print(f"  ✓ 已保存: {summary_path}")
    
    print(f"\n✅ 成功更新 {changed_count} 个文件")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='重新计算 VLM 评估结果的 avg_step')
    parser.add_argument('--dry-run', action='store_true', help='只显示变化，不实际保存')
    parser.add_argument('--dirs', nargs='+',
                       default=['vlm_eval_results/maze', 'vlm_eval_results/sokoban', 'vlm_eval_results/trapfield'],
                       help='要处理的目录列表')
    args = parser.parse_args()
    
    print("=" * 70)
    print("重新计算 VLM 评估结果的 avg_step")
    print("=" * 70)
    print(f"新方法: 只对 SR=1.0 的成功 case 计算 avg_step")
    print()
    
    all_results = []
    
    for dir_path in args.dirs:
        base_dir = Path(dir_path)
        
        if not base_dir.exists():
            print(f"⚠ 目录不存在: {base_dir}")
            continue
        
        print(f"处理目录: {base_dir}")
        results = process_directory(base_dir)
        all_results.extend(results)
        print()
    
    # 统计
    total_files = len(all_results)
    changed_files = sum(1 for r in all_results if r['changed'])
    
    print("=" * 70)
    print(f"统计: 共处理 {total_files} 个文件，其中 {changed_files} 个需要更新")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN 模式] 未实际保存文件")
        print("如需保存，请去掉 --dry-run 参数重新运行")
    else:
        # 保存更新
        save_updated_summaries(all_results, dry_run=False)


if __name__ == '__main__':
    main()

