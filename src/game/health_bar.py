"""HP / MP bar rendering.

Note: actual bar drawing is handled by GameRenderer.draw_ui() using ModernGL
to avoid shader conflicts between pyglet and ModernGL. This module is kept
for API compatibility.
"""

from __future__ import annotations


class HealthBar:
    """Placeholder — bars are drawn by GameRenderer.draw_ui()."""

    def __init__(self, x: int = 10, y: int = 10, width: int = 150, height: int = 12) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, hp: float, max_hp: float, mp: float, max_mp: float) -> None:
        pass  # Drawing handled by GameRenderer.draw_ui()
