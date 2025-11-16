"""
PathFinder 纹理处理器
支持起点、终点图标和道路纹理
"""

from pathlib import Path
from typing import Optional
from PIL import Image


class PathFinderTextureHandler:
    """PathFinder 纹理处理器"""
    
    # 支持的图片格式
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg']
    
    # 必需的纹理
    REQUIRED_TEXTURES = ['start', 'end', 'road']
    
    def __init__(self, assets_folder: Optional[str] = None):
        """
        初始化纹理处理器
        
        Args:
            assets_folder: 纹理文件夹路径，如果为 None 则使用默认路径
        """
        if assets_folder is None:
            # 默认使用 games/pathfinder/assets
            assets_folder = Path(__file__).parent / "assets"
        
        self.assets_path = Path(assets_folder)
        self.textures = {}
        
        # 确保资源文件夹存在
        self.assets_path.mkdir(parents=True, exist_ok=True)
    
    def load_textures(self):
        """加载所有纹理"""
        for texture_name in self.REQUIRED_TEXTURES:
            texture = self._load_texture(texture_name)
            if texture:
                self.textures[texture_name] = texture
    
    def _load_texture(self, name: str) -> Optional[Image.Image]:
        """
        加载单个纹理
        
        Args:
            name: 纹理名称（不含扩展名）
            
        Returns:
            加载的纹理图片，如果不存在则返回 None
        """
        for ext in self.SUPPORTED_FORMATS:
            file_path = self.assets_path / f"{name}{ext}"
            if file_path.exists():
                try:
                    img = Image.open(file_path).convert("RGBA")
                    return img
                except Exception as e:
                    print(f"Failed to load texture {name}: {e}")
                    return None
        
        return None
    
    def get_texture(self, name: str) -> Optional[Image.Image]:
        """
        获取纹理
        
        Args:
            name: 纹理名称
            
        Returns:
            纹理图片，如果不存在则返回 None
        """
        return self.textures.get(name)
    
    def has_texture(self, name: str) -> bool:
        """
        检查是否有指定纹理
        
        Args:
            name: 纹理名称
            
        Returns:
            是否存在该纹理
        """
        return name in self.textures
    
    def get_start_icon(self, size: int) -> Optional[Image.Image]:
        """
        获取起点图标（调整到指定尺寸）
        
        Args:
            size: 图标尺寸（直径）
            
        Returns:
            调整尺寸后的起点图标
        """
        texture = self.get_texture('start')
        if texture:
            return self._resize_keep_aspect_ratio(texture, size)
        return None
    
    def get_end_icon(self, size: int) -> Optional[Image.Image]:
        """
        获取终点图标（调整到指定尺寸）
        
        Args:
            size: 图标尺寸（直径）
            
        Returns:
            调整尺寸后的终点图标
        """
        texture = self.get_texture('end')
        if texture:
            return self._resize_keep_aspect_ratio(texture, size)
        return None
    
    def get_road_texture(self) -> Optional[Image.Image]:
        """
        获取道路纹理（原始尺寸）
        
        Returns:
            道路纹理图片
        """
        return self.get_texture('road')
    
    @staticmethod
    def _resize_keep_aspect_ratio(img: Image.Image, target_size: int) -> Image.Image:
        """
        调整图片尺寸，保持长宽比
        
        Args:
            img: 原始图片
            target_size: 目标尺寸（正方形边长）
            
        Returns:
            调整后的图片
        """
        # 获取原始尺寸
        width, height = img.size
        
        # 计算缩放比例
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        
        # 调整尺寸
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建正方形画布，居中放置
        canvas = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y), resized)
        
        return canvas
    
    def validate_textures(self) -> tuple[bool, list]:
        """
        验证所有必需纹理是否存在
        
        Returns:
            (是否全部存在, 缺失的纹理列表)
        """
        missing = []
        for texture_name in self.REQUIRED_TEXTURES:
            if not self.has_texture(texture_name):
                missing.append(texture_name)
        
        return len(missing) == 0, missing


# 全局缓存
_texture_handlers = {}


def get_texture_handler(assets_folder: Optional[str] = None) -> PathFinderTextureHandler:
    """
    获取或创建缓存的纹理处理器
    
    Args:
        assets_folder: 纹理文件夹路径
        
    Returns:
        纹理处理器实例
    """
    if assets_folder is None:
        assets_folder = str(Path(__file__).parent / "assets")
    
    cache_key = str(assets_folder)
    
    if cache_key not in _texture_handlers:
        handler = PathFinderTextureHandler(assets_folder)
        handler.load_textures()
        _texture_handlers[cache_key] = handler
    
    return _texture_handlers[cache_key]

