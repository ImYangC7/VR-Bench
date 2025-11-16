from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable, Optional


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_directory(path: str | Path) -> None:
    directory = Path(path)
    if not directory.exists():
        return
    for entry in directory.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def setup_output_directories(
    output_dir: str,
    images_dir: str,
    states_dir: str,
    video_dir: Optional[str] = None,
) -> None:
    for folder in (output_dir, images_dir, states_dir):
        ensure_directory(folder)
    clean_directory(images_dir)
    clean_directory(states_dir)

    if video_dir is not None:
        ensure_directory(video_dir)
        clean_directory(video_dir)
