"""
TrapField 游戏常量定义
"""

# 单元格类型
EMPTY_CELL = 0      # 空地（可以走）
TRAP_CELL = 1       # 陷阱（踩上去游戏结束）
PLAYER_CELL = 2     # 玩家起点
GOAL_CELL = 3       # 目标终点

# 渲染配置
CELL_SIZE = 64      # 每个单元格的像素大小

# 难度配置
DIFFICULTY_CONFIG = {
    'easy': {
        'grid_size': 7,      # 7x7 网格
        'trap_density': 0.2, # 20% 陷阱密度
    },
    'medium': {
        'grid_size': 11,     # 11x11 网格
        'trap_density': 0.3, # 30% 陷阱密度
    },
    'hard': {
        'grid_size': 15,     # 15x15 网格
        'trap_density': 0.35,# 35% 陷阱密度
    }
}

