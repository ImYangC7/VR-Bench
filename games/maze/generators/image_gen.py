from __future__ import annotations

from typing import Optional, Sequence

from ..renderer import get_shared_renderer


class PillowNotInstalledError(RuntimeError):
    pass


def draw_maze(maze: Sequence[Sequence[int]], cell_size: int, save_path: str,
              assets_folder: Optional[str] = None) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise PillowNotInstalledError(
            "Pillow is required to render maze images. Install it with 'pip install pillow'."
        ) from exc

    renderer = get_shared_renderer(assets_folder)
    renderer.render_maze(maze, save_path)
