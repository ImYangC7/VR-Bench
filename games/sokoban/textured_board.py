import os
from .board import SokobanBoard
from .renderer import get_shared_renderer

class TexturedSokobanBoard(SokobanBoard):
    """扩展SokobanBoard类，添加自定义贴图支持"""

    def __init__(self, grid, player_x, player_y, renderer=None):
        """初始化带贴图的推箱子游戏板

        Args:
            grid: 游戏网格
            player_x: 玩家X坐标
            player_y: 玩家Y坐标
            renderer: 渲染器，如果为None则使用共享渲染器
        """
        # 调用父类构造函数
        super().__init__(grid, player_x, player_y)

        # 初始化渲染器
        self.renderer = renderer if renderer else get_shared_renderer()

    def save_board(self, path: str, add_grid=False):
        """Render the board with textures to disk."""
        self.renderer.render_board(self, output_path=path, add_grid=add_grid)

    def render_to_image(self, add_grid=False):
        """Render the board to a PIL image without saving."""
        return self.renderer.render_board(self, output_path=None, add_grid=add_grid)

    def _render_grid_to_image(self, grid, add_grid=False):
        """Render a custom grid to a PIL image (for animation frames)."""
        from PIL import Image, ImageDraw

        # 获取纹理处理器和纹理大小
        handler = self.renderer.handler
        texture_size = self.renderer.texture_size

        height, width = grid.shape
        total_width = width * texture_size
        total_height = height * texture_size

        img = Image.new('RGB', (total_width, total_height), "#E0C9A6")

        # Layer 1: Floor
        if handler.has_texture('floor'):
            floor_tile = handler.get_texture('floor')
            for y in range(height):
                for x in range(width):
                    if grid[y, x] != 1:  # Not wall
                        img.paste(floor_tile, (x * texture_size, y * texture_size), floor_tile)

        # Layer 2: Walls and targets
        for y in range(height):
            for x in range(width):
                cell = grid[y, x]
                texture = None

                if cell == 1:  # Wall
                    texture = handler.get_texture('wall')
                elif cell in [3, 4, 6]:  # Target, box on target, player on target
                    texture = handler.get_texture('target')

                if texture:
                    img.paste(texture,
                            (x * texture_size, y * texture_size),
                            texture if texture.mode == 'RGBA' else None)

        # Layer 3: Boxes and players
        for y in range(height):
            for x in range(width):
                cell = grid[y, x]
                texture = None

                if cell in [2, 4]:  # Box or box on target
                    texture = handler.get_texture('box')
                elif cell in [5, 6]:  # Player or player on target
                    texture = handler.get_texture('player')

                if texture:
                    img.paste(texture,
                            (x * texture_size, y * texture_size),
                            texture if texture.mode == 'RGBA' else None)

        # Add grid if requested
        if add_grid:
            draw = ImageDraw.Draw(img)
            for i in range(height + 1):
                y_pos = i * texture_size
                draw.line([(0, y_pos), (total_width, y_pos)], fill="#000000", width=2)

            for i in range(width + 1):
                x_pos = i * texture_size
                draw.line([(x_pos, 0), (x_pos, total_height)], fill="#000000", width=2)

        return img

    @property
    def handler(self):
        """Get texture handler from renderer."""
        return self.renderer.handler

    @property
    def texture_size(self):
        """Get texture size from renderer."""
        return self.renderer.texture_size


# 全局共享的渲染器（避免重复加载纹理）
_shared_renderer_instance = None
_current_assets_folder = None

def get_shared_texture_handler(assets_folder=None):
    """获取全局共享的渲染器，如果不存在或素材文件夹改变则创建新的

    Args:
        assets_folder: 素材文件夹路径，如果为None则使用默认的assets文件夹
    """
    global _shared_renderer_instance, _current_assets_folder

    if assets_folder is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_folder = os.path.join(current_dir, "assets")

    if _shared_renderer_instance is None or _current_assets_folder != assets_folder:
        _shared_renderer_instance = get_shared_renderer(assets_folder=assets_folder)
        _current_assets_folder = assets_folder

    return _shared_renderer_instance

# 修改generate_random_board函数以支持贴图
def generate_textured_random_board(size, renderer=None, num_boxes=None, check_solvable=True, max_attempts=10, assets_folder=None):
    """生成带贴图的随机推箱子游戏板

    Args:
        size: 游戏板大小
        renderer: 渲染器，如果为None则使用全局共享的渲染器
        num_boxes: 箱子数量
        check_solvable: 是否检查可解性
        max_attempts: 最大尝试次数
        assets_folder: 素材文件夹路径，用于换皮肤

    Returns:
        TexturedSokobanBoard: 带贴图的游戏板
    """
    from .board import generate_random_board

    # 使用原始函数生成游戏板
    board = generate_random_board(size, num_boxes, check_solvable, max_attempts)

    # 如果没有提供渲染器，使用全局共享的渲染器
    if renderer is None:
        renderer = get_shared_texture_handler(assets_folder)

    # 创建带贴图的游戏板
    textured_board = TexturedSokobanBoard(board.grid.copy(), board.player_x, board.player_y, renderer)

    return textured_board

def setup_game_with_custom_textures(assets_folder=None):
    """设置带自定义贴图的游戏

    Args:
        assets_folder: 素材文件夹路径，可选

    Returns:
        TexturedSokobanBoard: 配置好贴图的游戏板
    """
    # 生成带贴图的随机游戏板
    board = generate_textured_random_board(6, num_boxes=2, assets_folder=assets_folder)

    return board

# 演示用法示例
if __name__ == "__main__":
    # 检查是否提供了截图路径作为命令行参数
    import sys
    screenshot_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 设置游戏
    board = setup_game_with_custom_textures(screenshot_path)
    
    # 保存渲染后的游戏板
    board.save_board("custom_textured_board.png")
    
    print("游戏已设置完成，请查看生成的图像文件。")