"""
TrapField 游戏配置
"""

from .constants import EMPTY_CELL, TRAP_CELL, PLAYER_CELL, GOAL_CELL

# 需要的纹理文件
REQUIRED_TEXTURES = ['floor', 'trap', 'player', 'goal']

# 单元格到图层的映射
CELL_LAYER_MAP = {
    EMPTY_CELL: 'floor',
    TRAP_CELL: 'trap',
    PLAYER_CELL: 'player',
    GOAL_CELL: 'goal'
}

