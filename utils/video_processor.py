"""
视频处理工具
用于标准化视频帧率、帧数等
"""

from __future__ import annotations

import contextlib
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from .video_metadata import (
    VideoMetadataError,
    get_video_metadata,
    has_audio_stream,
)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, target_fps: float = 24):
        """
        初始化视频处理器
        
        Args:
            target_fps: 目标帧率 (默认 24)
        """
        self.target_fps = target_fps
        self._ensure_ffmpeg()
    
    def _ensure_ffmpeg(self):
        """确保 ffmpeg 可用"""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required but not found in PATH.")
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe is required but not found in PATH.")
    
    def resize_to_frames(
        self,
        input_path: Path | str,
        target_frames: int,
        output_path: Optional[Path | str] = None,
        overwrite: bool = True,
        retain_audio: bool = False,
    ) -> Path:
        """
        调整视频到指定帧数
        
        Args:
            input_path: 输入视频路径
            target_frames: 目标帧数
            output_path: 输出路径 (None 表示原地替换)
            overwrite: 是否覆盖已存在的文件
            retain_audio: 是否保留音频
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        target_duration = target_frames / self.target_fps
        
        return self._retime_video(
            input_path,
            target_duration,
            output_path,
            overwrite,
            retain_audio,
        )
    
    def normalize_duration(
        self,
        input_path: Path | str,
        target_duration: float,
        output_path: Optional[Path | str] = None,
        overwrite: bool = True,
        retain_audio: bool = False,
    ) -> Path:
        """
        调整视频到指定时长
        
        Args:
            input_path: 输入视频路径
            target_duration: 目标时长 (秒)
            output_path: 输出路径 (None 表示原地替换)
            overwrite: 是否覆盖已存在的文件
            retain_audio: 是否保留音频
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        
        return self._retime_video(
            input_path,
            target_duration,
            output_path,
            overwrite,
            retain_audio,
        )
    
    def _retime_video(
        self,
        input_path: Path,
        target_duration: float,
        output_path: Optional[Path | str],
        overwrite: bool,
        retain_audio: bool,
    ) -> Path:
        """内部方法：重新计时视频"""
        # 获取原始时长
        try:
            duration_s, _, _ = get_video_metadata(input_path)
        except VideoMetadataError as exc:
            raise RuntimeError(f"Cannot read metadata: {exc}") from exc
        
        # 计算速度因子
        video_factor = target_duration / duration_s
        audio_speed = duration_s / target_duration
        
        # 构建 ffmpeg 滤镜
        vf = f"setpts={video_factor:.12f}*PTS,fps={self.target_fps}"
        af = self._atempo_filters(audio_speed) if retain_audio and has_audio_stream(input_path) else None
        
        # 确定输出路径
        if output_path is None:
            # 原地替换：先写临时文件
            temp_output = input_path.with_name(f".__temp__{uuid.uuid4().hex}{input_path.suffix}")
            final_output = input_path
            in_place = True
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_output = output_path
            final_output = output_path
            in_place = False
        
        # 构建 ffmpeg 命令
        cmd: List[str] = [
            "ffmpeg",
            "-y" if (overwrite or in_place) else "-n",
            "-i", str(input_path),
            "-vf", vf,
        ]
        
        if retain_audio:
            if af:
                cmd.extend(["-af", af])
        else:
            cmd.append("-an")
        
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ])
        
        if retain_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        
        cmd.extend(["-r", f"{self.target_fps:.6g}", str(temp_output)])
        
        # 执行 ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            if in_place:
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    temp_output.unlink()
            raise RuntimeError(result.stderr.strip() or f"ffmpeg failed on {input_path}")
        
        # 原地替换
        if in_place:
            try:
                temp_output.replace(input_path)
            finally:
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    temp_output.unlink()
        
        return final_output
    
    @staticmethod
    def _atempo_filters(speed: float) -> Optional[str]:
        """生成 atempo 滤镜链 (ffmpeg 限制单个 atempo 在 0.5-2.0 之间)"""
        if math.isclose(speed, 1.0, rel_tol=0.0, abs_tol=1e-3):
            return None

        filters: List[str] = []
        remaining = speed

        while remaining > 2.0 + 1e-6:
            filters.append("atempo=2.0")
            remaining /= 2.0

        while remaining < 0.5 - 1e-6:
            filters.append("atempo=0.5")
            remaining /= 0.5

        filters.append(f"atempo={remaining:.6f}")
        return ",".join(filters)


# 便捷函数

def normalize_video(
    input_path: Path | str,
    target_duration: float,
    output_path: Optional[Path | str] = None,
    target_fps: float = 24,
    overwrite: bool = True,
    retain_audio: bool = False,
) -> Path:
    """
    标准化视频到指定时长和帧率

    Args:
        input_path: 输入视频路径
        target_duration: 目标时长 (秒)
        output_path: 输出路径 (None 表示原地替换)
        target_fps: 目标帧率 (默认 24)
        overwrite: 是否覆盖已存在的文件
        retain_audio: 是否保留音频

    Returns:
        输出文件路径
    """
    processor = VideoProcessor(target_fps=target_fps)
    return processor.normalize_duration(
        input_path,
        target_duration,
        output_path,
        overwrite,
        retain_audio,
    )


def resize_video_to_frames(
    input_path: Path | str,
    target_frames: int,
    output_path: Optional[Path | str] = None,
    target_fps: float = 24,
    overwrite: bool = True,
    retain_audio: bool = False,
) -> Path:
    """
    调整视频到指定帧数

    Args:
        input_path: 输入视频路径
        target_frames: 目标帧数
        output_path: 输出路径 (None 表示原地替换)
        target_fps: 目标帧率 (默认 24)
        overwrite: 是否覆盖已存在的文件
        retain_audio: 是否保留音频

    Returns:
        输出文件路径

    Example:
        >>> resize_video_to_frames('input.mp4', 192, 'output.mp4')
        PosixPath('output.mp4')
    """
    processor = VideoProcessor(target_fps=target_fps)
    return processor.resize_to_frames(
        input_path,
        target_frames,
        output_path,
        overwrite,
        retain_audio,
    )


def get_video_info(video_path: Path | str) -> Tuple[float, int, float]:
    """
    获取视频信息

    Args:
        video_path: 视频路径

    Returns:
        (duration_seconds, frame_count, fps)

    Example:
        >>> duration, frames, fps = get_video_info('video.mp4')
        >>> print(f"Duration: {duration:.2f}s, Frames: {frames}, FPS: {fps}")
    """
    duration, frames, fps = get_video_metadata(Path(video_path))
    return duration, int(frames), fps

