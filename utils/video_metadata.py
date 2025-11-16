"""
视频元数据提取工具
支持 ffprobe 和 OpenCV 两种方式
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Tuple


class VideoMetadataError(RuntimeError):
    """视频元数据提取失败"""
    pass


def get_video_metadata(path: Path) -> Tuple[float, float, float]:
    """
    获取视频元数据
    
    Args:
        path: 视频文件路径
        
    Returns:
        (duration_s, frame_count, fps)
        
    Raises:
        VideoMetadataError: 无法提取元数据
    """
    try:
        return _metadata_with_ffprobe(path)
    except FileNotFoundError:
        pass
    except VideoMetadataError:
        raise
    except Exception as exc:
        raise VideoMetadataError(f"ffprobe error for {path}: {exc}") from exc

    try:
        return _metadata_with_cv2(path)
    except ImportError:
        raise VideoMetadataError(
            "Neither ffprobe (from ffmpeg) nor OpenCV (cv2) is available"
        )
    except Exception as exc:
        raise VideoMetadataError(f"OpenCV error for {path}: {exc}") from exc


def _metadata_with_ffprobe(path: Path) -> Tuple[float, float, float]:
    """使用 ffprobe 提取元数据"""
    if not shutil.which("ffprobe"):
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-show_entries", "stream=nb_frames,avg_frame_rate",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoMetadataError(result.stderr.strip() or "ffprobe failed")

    data = json.loads(result.stdout)
    try:
        duration_s = float(data["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise VideoMetadataError("Duration unavailable") from exc

    stream = data.get("streams", [{}])[0]
    raw_frames = stream.get("nb_frames")
    avg_frame_rate = stream.get("avg_frame_rate")
    fps = _fps_from_rate(avg_frame_rate)

    if raw_frames in (None, "N/A"):
        frames = duration_s * fps
    else:
        try:
            frames = float(raw_frames)
        except ValueError as exc:
            raise VideoMetadataError("Invalid frame count") from exc

    return duration_s, frames, fps


def _metadata_with_cv2(path: Path) -> Tuple[float, float, float]:
    """使用 OpenCV 提取元数据"""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoMetadataError("Unable to open video with OpenCV")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise VideoMetadataError("Frame rate unavailable via OpenCV")

    if not math.isfinite(frame_count) or frame_count <= 0:
        frame_count = float(_count_frames_with_cv2(capture))

    frames = frame_count
    capture.release()

    duration_s = frames / fps
    return duration_s, frames, fps


def _count_frames_with_cv2(capture) -> int:
    """手动计数帧数"""
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    return frames


def _fps_from_rate(avg_frame_rate: str | None) -> float:
    """从帧率字符串解析 FPS"""
    if not avg_frame_rate or avg_frame_rate in ("0/0", "0"):
        raise VideoMetadataError("Frame rate unavailable")

    try:
        rate = Fraction(avg_frame_rate)
    except (ZeroDivisionError, ValueError) as exc:
        raise VideoMetadataError("Invalid frame rate") from exc

    return float(rate)


def has_audio_stream(path: Path) -> bool:
    """检查视频是否有音频流"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())

