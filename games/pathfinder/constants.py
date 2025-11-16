"""Constants for PathFinder game."""

# 难度配置 - 通过图片尺寸、道路宽度、节点密度、支路数量区分难度
DIFFICULTY_CONFIG = {
    'easy': {
        'image_size': 1024,        # 小图
        'road_width': 60,         # 宽道路
        'node_spacing_ratio': 0.3,  # 节点间距占图片尺寸的比例（18% = 稀疏）
        'extra_paths': 1,         # 额外支路数量（少）
        'min_solution_nodes': 4,  # 解决方案最少节点数（短路径）
        'connectivity_ratio': 0.15,  # 连通率（0-1）：0.15 = 稀疏道路，看起来像真实道路
    },
    'medium': {
        'image_size': 1024,        # 小图
        'road_width': 50,         # 宽道路
        'node_spacing_ratio': 0.2,  # 节点间距占图片尺寸的比例（18% = 稀疏）
        'extra_paths': 2,         # 额外支路数量（少）
        'min_solution_nodes': 6,  # 解决方案最少节点数（短路径）
        'connectivity_ratio': 0.2,  # 连通率：0.2 = 中等密度
    },
    'hard': {
        'image_size': 1024,       # 大图
        'road_width': 36,         # 窄道路
        'node_spacing_ratio': 0.15,  # 节点间距占图片尺寸的比例（12% = 密集）
        'extra_paths': 3,         # 额外支路数量（多）
        'min_solution_nodes': 7,  # 解决方案最少节点数（长路径）
        'connectivity_ratio': 0.25,  # 连通率：0.25 = 较密集（但仍然像道路）
    }
}

# 渲染配置
DEFAULT_IMAGE_SIZE = 500   # 默认图片尺寸（如果不指定难度）
ROAD_WIDTH = 35            # 道路宽度（更细）
NODE_RADIUS = 20           # 起点/终点半径
START_COLOR = (255, 0, 0)    # 起点颜色（红色）
END_COLOR = (0, 255, 0)      # 终点颜色（绿色）
ROAD_COLOR = (255, 255, 255) # 道路颜色（白色）
BG_COLOR = (0, 0, 0)         # 背景颜色（黑色）

# 曲线配置
CURVE_SEGMENTS = 400       # 曲线分段数（更平滑）
CURVE_CONTROL_POINTS = 3   # 每条曲线的控制点数量
CURVE_BEND_FACTOR = 0.25   # 曲线弯曲程度

# 边界留白
MARGIN = 80

# 视频配置
FRAMES_PER_SECOND = 24     # 帧率（与其他游戏保持一致）
MOVEMENT_SPEED = 1.0       # 移动速度（像素/帧）

