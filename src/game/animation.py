"""Animation manager and geometric frame data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pyglet

from src.game import config as cfg


@dataclass
class FrameData:
    """A single animation frame: list of (color, vertices) for drawing."""

    color: tuple[int, int, int, int]
    vertices: tuple[tuple[float, float], ...]


# ── Hero animation frames (geometric shapes) ──
def _make_rect(w: float, h: float, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[tuple[float, float], ...]:
    ox, oy = offset
    return (
        (ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)
    )


def _make_hero_idle_frame(frame_idx: int) -> FrameData:
    colors = [(180, 60, 40, 255), (190, 65, 42, 255), (175, 58, 38, 255), (185, 62, 40, 255)]
    # Simple 5x10 rect, slight bobbing
    bob = (frame_idx % 2) * 0.3
    w, h = cfg.HERO_WIDTH * cfg.CELL_SCALE, cfg.HERO_HEIGHT * cfg.CELL_SCALE
    verts = _make_rect(w, h - bob, (0.0, bob))
    return FrameData(color=colors[frame_idx % 4], vertices=verts)


def _make_hero_walk_frame(frame_idx: int) -> FrameData:
    colors = [(200, 80, 50, 255), (210, 85, 55, 255), (190, 75, 48, 255), (205, 82, 52, 255)]
    lean = (frame_idx % 2) * 1.5
    w, h = cfg.HERO_WIDTH * cfg.CELL_SCALE, cfg.HERO_HEIGHT * cfg.CELL_SCALE
    verts = _make_rect(w, h, (lean, 0.0))
    return FrameData(color=colors[frame_idx % 4], vertices=verts)


def _make_hero_jump_frame(frame_idx: int) -> FrameData:
    w, h = cfg.HERO_WIDTH * cfg.CELL_SCALE, cfg.HERO_HEIGHT * cfg.CELL_SCALE
    verts = _make_rect(w, h, (0.0, 2.0))
    return FrameData(color=(220, 100, 60, 255), vertices=verts)


def _make_hero_cast_frame(frame_idx: int) -> FrameData:
    w, h = cfg.HERO_WIDTH * cfg.CELL_SCALE, cfg.HERO_HEIGHT * cfg.CELL_SCALE
    # Slightly wider glow during cast
    glow = (frame_idx % 2) * 2.0
    verts = _make_rect(w + glow, h, (-glow / 2, 0.0))
    return FrameData(color=(240, 200, 100, 255), vertices=verts)


def _make_hero_chant_frame(frame_idx: int) -> FrameData:
    w, h = cfg.HERO_WIDTH * cfg.CELL_SCALE, cfg.HERO_HEIGHT * cfg.CELL_SCALE
    # Channeling pose: slightly narrowed, upward glow
    narrow = (frame_idx % 2) * 0.5
    verts = _make_rect(w - narrow, h, (narrow / 2, 0.0))
    return FrameData(color=(230, 180, 80, 255), vertices=verts)


HERO_ANIMATIONS: dict[str, list[FrameData]] = {
    "idle": [_make_hero_idle_frame(i) for i in range(4)],
    "walk": [_make_hero_walk_frame(i) for i in range(4)],
    "jump": [_make_hero_jump_frame(i) for i in range(4)],
    "chant": [_make_hero_chant_frame(i) for i in range(4)],
    "cast": [_make_hero_cast_frame(i) for i in range(4)],
}


@dataclass
class AnimationManager:
    """Manages per-entity animation state."""

    state: str = "idle"
    frame: int = 0
    timer: float = 0.0

    def update(self, dt: float, state: str) -> FrameData | None:
        """Update animation and return current frame data."""
        if state != self.state:
            self.state = state
            self.frame = 0
            self.timer = 0.0

        self.timer += dt
        fps = cfg.ENTITY_ANIM_FPS
        if self.timer >= 1.0 / fps:
            self.timer = 0.0
            self.frame = (self.frame + 1) % 4

        frames = HERO_ANIMATIONS.get(self.state)
        if not frames:
            return None
        return frames[self.frame % len(frames)]


class SimpleSpriteBatch:
    """Batch-based sprite renderer for pyglet."""

    def __init__(self) -> None:
        self.batch = pyglet.graphics.Batch()
        self._shapes: list[pyglet.shapes.ShapeBase] = []

    def clear(self) -> None:
        self._shapes.clear()
        self.batch = pyglet.graphics.Batch()

    def add_rect(self, x: float, y: float, w: float, h: float, color: tuple[int, int, int, int]) -> None:
        shape = pyglet.shapes.Rectangle(x, y, w, h, color=color[:3] + (color[3] // 2,))
        shape.opacity = color[3]
        self._shapes.append(shape)

    def draw(self) -> None:
        for shape in self._shapes:
            shape.draw()
