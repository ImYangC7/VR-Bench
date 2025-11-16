"""
Maze 游戏默认纹理生成器
生成简单的纯色方块作为默认纹理
"""

from PIL import Image, ImageDraw
from pathlib import Path
import logging


# 默认颜色配置（参考图片）
DEFAULT_COLORS = {
    'floor': (173, 216, 230),      # 浅蓝色 (light blue)
    'wall': (255, 255, 255),       # 白色 (white)
    'player': (0, 128, 0),         # 绿色 (green)
    'target': (255, 0, 0),         # 红色 (red)
}


def create_default_texture(color: tuple, size: int = 64, border: bool = True) -> Image.Image:
    """
    创建默认纹理
    
    Args:
        color: RGB 颜色元组
        size: 纹理大小（像素）
        border: 是否添加边框
        
    Returns:
        PIL Image 对象
    """
    img = Image.new('RGB', (size, size), color)
    
    if border:
        draw = ImageDraw.Draw(img)
        # 绘制浅灰色边框
        border_color = (200, 200, 200)
        draw.rectangle([0, 0, size-1, size-1], outline=border_color, width=1)
    
    return img


def create_player_texture(size: int = 64) -> Image.Image:
    """
    创建玩家纹理（绿色方块）
    
    Args:
        size: 纹理大小（像素）
        
    Returns:
        PIL Image 对象
    """
    # 创建透明背景
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制绿色方块（居中，占 60% 大小）
    margin = int(size * 0.2)
    color = DEFAULT_COLORS['player']
    draw.rectangle(
        [margin, margin, size - margin - 1, size - margin - 1],
        fill=color,
        outline=(0, 100, 0),  # 深绿色边框
        width=2
    )
    
    return img


def create_target_texture(size: int = 64) -> Image.Image:
    """
    创建目标纹理（红色圆点）
    
    Args:
        size: 纹理大小（像素）
        
    Returns:
        PIL Image 对象
    """
    # 创建透明背景
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制红色圆点（居中，占 50% 大小）
    margin = int(size * 0.25)
    color = DEFAULT_COLORS['target']
    draw.ellipse(
        [margin, margin, size - margin - 1, size - margin - 1],
        fill=color,
        outline=(200, 0, 0),  # 深红色边框
        width=2
    )
    
    return img


def generate_default_textures(output_dir: str | Path, size: int = 64):
    """
    生成所有默认纹理并保存到指定目录
    
    Args:
        output_dir: 输出目录路径
        size: 纹理大小（像素）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Generating default maze textures to {output_path}")
    
    # 生成地板纹理
    floor_img = create_default_texture(DEFAULT_COLORS['floor'], size, border=True)
    floor_img.save(output_path / 'floor.png')
    
    # 生成墙壁纹理
    wall_img = create_default_texture(DEFAULT_COLORS['wall'], size, border=True)
    wall_img.save(output_path / 'wall.png')
    
    # 生成玩家纹理
    player_img = create_player_texture(size)
    player_img.save(output_path / 'player.png')
    
    # 生成目标纹理
    target_img = create_target_texture(size)
    target_img.save(output_path / 'target.png')
    
    logging.info(f"✓ Generated 4 default textures: floor, wall, player, target")


def ensure_default_textures(assets_folder: str | Path = None) -> Path:
    """
    确保默认纹理存在，如果不存在则生成
    
    Args:
        assets_folder: 素材文件夹路径，如果为 None 则使用 assets/default_maze
        
    Returns:
        纹理文件夹路径
    """
    if assets_folder is None:
        assets_folder = Path(__file__).parent.parent.parent / 'assets' / 'default_maze'
    else:
        assets_folder = Path(assets_folder)
    
    # 检查是否所有纹理都存在
    required_textures = ['floor', 'wall', 'player', 'target']
    all_exist = all(
        any((assets_folder / f"{name}{ext}").exists() for ext in ['.png', '.jpg', '.jpeg'])
        for name in required_textures
    )
    
    if not all_exist:
        logging.info(f"Default textures not found, generating...")
        generate_default_textures(assets_folder)
    
    return assets_folder


if __name__ == '__main__':
    # 测试：生成默认纹理
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(__file__).parent.parent.parent / 'assets' / 'default_maze'
    generate_default_textures(output_dir, size=64)
    print(f"Default textures generated in: {output_dir}")

