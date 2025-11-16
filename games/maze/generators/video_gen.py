from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from ..renderer import get_shared_renderer

Coordinate = Tuple[int, int]


class PillowNotInstalledError(RuntimeError):
    pass


def create_solution_video(
    maze: Sequence[Sequence[int]],
    path: Iterable[Coordinate],
    cell_size: int,
    save_path: str,
    frame_duration_ms: int = 300,
    assets_folder: Optional[str] = None,
) -> None:
    renderer = get_shared_renderer(assets_folder)
    renderer.render_video(maze, path, save_path, frame_duration_ms)
