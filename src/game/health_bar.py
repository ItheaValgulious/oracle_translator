"""HP / MP bar rendering."""

from __future__ import annotations

import pyglet

from src.game import config as cfg


class HealthBar:
    """Simple HP and MP bars rendered with pyglet shapes."""

    def __init__(self, x: int = 10, y: int = 10, width: int = 150, height: int = 12) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # Pre-allocate shapes (reused every frame)
        self._hp_bg = pyglet.shapes.Rectangle(x, y + height + 2, width, height, color=(60, 0, 0))
        self._hp_fill = pyglet.shapes.Rectangle(x, y + height + 2, width, height, color=(220, 30, 30))
        self._mp_bg = pyglet.shapes.Rectangle(x, y, width, height, color=(0, 0, 60))
        self._mp_fill = pyglet.shapes.Rectangle(x, y, width, height, color=(30, 30, 220))
        self._label = pyglet.text.Label(
            "", x=x + 4, y=y + height // 2, font_size=9, color=(255, 255, 255, 255),
        )

    def draw(self, hp: float, max_hp: float, mp: float, max_mp: float) -> None:
        hp_ratio = max(0.0, min(1.0, hp / max_hp))
        mp_ratio = max(0.0, min(1.0, mp / max_mp))

        self._hp_bg.draw()
        self._hp_fill.width = int(self.width * hp_ratio)
        self._hp_fill.draw()

        self._mp_bg.draw()
        self._mp_fill.width = int(self.width * mp_ratio)
        self._mp_fill.draw()

        self._label.text = f"HP: {int(hp)}/{int(max_hp)}  MP: {int(mp)}/{int(max_mp)}"
        self._label.draw()
