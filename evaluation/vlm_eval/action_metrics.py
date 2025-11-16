from typing import List, Dict, Any, Union


def calculate_sr(is_win: bool) -> float:
    return 1.0 if is_win else 0.0


def calculate_pr(
    pred_actions: List[Dict[str, Any]],
    opt_actions: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> float:
    if isinstance(opt_actions, list) and opt_actions and isinstance(opt_actions[0], list):
        return max(_calculate_pr_single(pred_actions, opt) for opt in opt_actions)
    return _calculate_pr_single(pred_actions, opt_actions)


def _calculate_pr_single(
    pred_actions: List[Dict[str, Any]],
    opt_actions: List[Dict[str, Any]]
) -> float:
    if not opt_actions:
        return 1.0 if not pred_actions else 0.0

    # 检查是否是 PathFinder 游戏（通过检查 action 类型）
    if pred_actions and pred_actions[0].get('action') == 'path':
        # PathFinder: 计算字母数组的连续匹配数
        pred_path = pred_actions[0].get('path', [])
        opt_path = opt_actions[0].get('path', [])

        if not opt_path:
            return 1.0 if not pred_path else 0.0

        # 计算从头开始连续匹配的节点数
        matched = 0
        for p_node, o_node in zip(pred_path, opt_path):
            if p_node == o_node:
                matched += 1
            else:
                break

        return matched / len(opt_path)
    else:
        # 其他游戏: 计算动作序列的连续匹配数
        matched = 0
        for p, o in zip(pred_actions, opt_actions):
            if p == o:
                matched += 1
            else:
                break

        return matched / len(opt_actions)


def calculate_mr(
    pred_actions: List[Dict[str, Any]],
    opt_actions: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> float:
    if isinstance(opt_actions, list) and opt_actions and isinstance(opt_actions[0], list):
        return 1.0 if any(pred_actions == opt for opt in opt_actions) else 0.0
    return 1.0 if pred_actions == opt_actions else 0.0


def calculate_step(
    pred_actions: List[Dict[str, Any]],
    opt_actions: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    is_win: bool = False
) -> float:
    """Calculate step metric: (predicted_length / optimal_length) - 1

    Only calculates for successful cases (is_win=True).
    Returns None for failed cases.

    For PathFinder game, calculates based on sequence length instead of action count.
    """
    if not is_win:
        return None

    # 检查是否是 PathFinder 游戏（通过检查 action 类型）
    if pred_actions and pred_actions[0].get('action') == 'path':
        # PathFinder: 使用字母数组的长度
        pred_length = len(pred_actions[0].get('path', []))

        if isinstance(opt_actions, list) and opt_actions and isinstance(opt_actions[0], list):
            opt_length = min(len(opt[0].get('path', [])) for opt in opt_actions)
        else:
            opt_length = len(opt_actions[0].get('path', []))

        if opt_length == 0:
            return 0.0

        return pred_length / opt_length - 1.0
    else:
        # 其他游戏: 使用动作数量
        if isinstance(opt_actions, list) and opt_actions and isinstance(opt_actions[0], list):
            opt_length = min(len(opt) for opt in opt_actions)
        else:
            opt_length = len(opt_actions)

        if opt_length == 0:
            return 0.0

        return len(pred_actions) / opt_length - 1.0
