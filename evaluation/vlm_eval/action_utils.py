import json
import re
from typing import List, Dict, Any


def parse_actions(response: str, game_type: str = 'default') -> List[Dict[str, Any]]:
    if game_type in ['maze', 'trapfield']:
        return _parse_path_actions(response)
    elif game_type == 'sokoban':
        return _parse_sokoban_actions(response)
    elif game_type == 'pathfinder':
        return _parse_pathfinder_actions(response)
    elif game_type in ['maze3d', '3dmaze']:
        return _parse_maze3d_actions(response)

    attempts = [
        lambda: json.loads(response),
        lambda: _parse_markdown_json(response),
        lambda: _extract_json_array(response)
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, list):
                return result
        except:
            continue

    raise ValueError(f"Failed to parse JSON from response: {response[:200]}")


def _parse_markdown_json(text: str) -> List[Dict[str, Any]]:
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No markdown JSON found")


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON array found")


def _parse_path_actions(response: str) -> List[Dict[str, Any]]:
    attempts = [
        lambda: json.loads(response),
        lambda: _parse_markdown_json(response),
        lambda: _extract_json_object(response)
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict) and 'path' in result:
                path = result['path']
                if isinstance(path, list):
                    return [{'action': 'move', 'direction': d} for d in path]
        except:
            continue

    raise ValueError(f"Failed to parse path from response: {response[:200]}")


def _parse_sokoban_actions(response: str) -> List[Dict[str, Any]]:
    """Parse Sokoban actions - only move actions (up/down/left/right)"""
    attempts = [
        lambda: json.loads(response),
        lambda: _parse_markdown_json(response),
        lambda: _extract_json_object(response)
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict) and 'actions' in result:
                actions = result['actions']
                if isinstance(actions, list):
                    parsed = []
                    for a in actions:
                        # 只支持 move 操作，push 会自动发生
                        parsed.append({'action': 'move', 'direction': a})
                    return parsed
        except:
            continue

    raise ValueError(f"Failed to parse sokoban actions from response: {response[:200]}")


def _parse_pathfinder_actions(response: str) -> List[Dict[str, Any]]:
    """Parse PathFinder actions - letter array representing the path

    Expected format: {"path": ["A", "C", "D"]}
    """
    attempts = [
        lambda: json.loads(response),
        lambda: _parse_markdown_json(response),
        lambda: _extract_json_object(response)
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict) and 'path' in result:
                path = result['path']
                if isinstance(path, list):
                    # Validate all elements are strings
                    if all(isinstance(item, str) for item in path):
                        return [{'action': 'path', 'path': path}]
        except:
            continue

    raise ValueError(f"Failed to parse pathfinder actions from response: {response[:200]}")


def _parse_maze3d_actions(response: str) -> List[Dict[str, Any]]:
    """Parse 3D Maze actions - direction array

    Expected format: {"path": ["up", "forward_right", "forward_left", ...]}
    Valid directions: forward_left, forward_right, backward_left, backward_right, up, down
    """
    valid_directions = {
        'forward_left', 'forward_right', 'backward_left', 'backward_right', 'up', 'down'
    }

    attempts = [
        lambda: json.loads(response),
        lambda: _parse_markdown_json(response),
        lambda: _extract_json_object(response)
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict) and 'path' in result:
                path = result['path']
                if isinstance(path, list):
                    # Validate all elements are valid direction strings
                    if all(isinstance(item, str) and item in valid_directions for item in path):
                        return [{'action': 'move', 'direction': direction} for direction in path]
        except:
            continue

    raise ValueError(f"Failed to parse maze3d actions from response: {response[:200]}")


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON object found")
