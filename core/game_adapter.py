"""
游戏适配器基类
定义统一的接口，让不同游戏都能接入并发生成系统
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class GameAdapter(ABC):
    """游戏适配器基类"""
    
    @abstractmethod
    def get_game_name(self) -> str:
        """返回游戏名称"""
        pass
    
    @abstractmethod
    def generate_level(
        self,
        difficulty_config: Dict[str, Any],
        assets_folder: str,
        **kwargs
    ) -> Optional[Any]:
        """
        生成一个关卡
        
        Args:
            difficulty_config: 难度配置字典
            assets_folder: 素材文件夹路径
            **kwargs: 其他参数
            
        Returns:
            生成的关卡对象，失败返回 None
        """
        pass
    
    @abstractmethod
    def save_level(
        self,
        level: Any,
        output_dir: Path,
        level_id: int,
        difficulty_name: str,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """
        保存关卡（包括视频、图片等）
        
        Args:
            level: 关卡对象
            output_dir: 输出目录
            level_id: 关卡ID
            difficulty_name: 难度名称
            **kwargs: 其他参数（如 fps）
            
        Returns:
            包含文件信息的字典，例如:
            {
                'video': 'video_0001.mp4',
                'image': 'image_0001.png',
                'state': 'state_0001.json'
            }
            如果某个文件生成失败，对应值为 None
        """
        pass
    
    @abstractmethod
    def get_level_hash(self, level: Any) -> str:
        """
        获取关卡的哈希值（用于去重）
        
        Args:
            level: 关卡对象
            
        Returns:
            关卡的哈希字符串
        """
        pass
    
    @abstractmethod
    def is_duplicate(self, level: Any, existing_hashes: set) -> bool:
        """
        检查关卡是否重复
        
        Args:
            level: 关卡对象
            existing_hashes: 已存在的哈希集合
            
        Returns:
            True 如果重复，False 如果不重复
        """
        pass
    
    def validate_difficulty_config(self, difficulty_config: Dict[str, Any]) -> bool:
        """
        验证难度配置是否有效
        
        Args:
            difficulty_config: 难度配置字典
            
        Returns:
            True 如果配置有效，False 如果无效
        """
        # 默认实现：检查是否有 count 字段
        return 'count' in difficulty_config
    
    def get_required_texture_files(self) -> list:
        """
        返回游戏需要的纹理文件列表
        
        Returns:
            纹理文件名列表（不含扩展名）
        """
        return []
    
    def cleanup(self):
        """清理资源（可选）"""
        pass


class LevelDeduplicator:
    """关卡去重器（通用版本）"""
    
    def __init__(self):
        self.hashes = set()
    
    def add_hash(self, hash_value: str):
        """添加哈希值"""
        self.hashes.add(hash_value)
    
    def is_duplicate(self, hash_value: str) -> bool:
        """检查是否重复"""
        return hash_value in self.hashes
    
    def get_count(self) -> int:
        """获取已存储的哈希数量"""
        return len(self.hashes)

