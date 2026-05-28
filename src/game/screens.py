"""Game screens: Title, Game, Options, Console."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

import pyglet
from pyglet.window import key

from src.game import config as cfg
from src.game.spell_system import SPELL_CATALOG

if TYPE_CHECKING:
    from src.game.app import GameApp


class BaseScreen:
    """Base class for game screens."""

    def __init__(self, app: "GameApp") -> None:
        self.app = app

    def on_draw(self) -> None:
        raise NotImplementedError

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        pass

    def update(self, dt: float) -> None:
        pass


class TitleScreen(BaseScreen):
    """Title screen with start/options/quit."""

    def __init__(self, app: "GameApp") -> None:
        super().__init__(app)
        self.title_label = pyglet.text.Label(
            "Oracle Translator", font_size=36, x=app.width // 2, y=app.height * 0.7,
            anchor_x="center", anchor_y="center", color=(240, 220, 180, 255),
        )
        self.sub_label = pyglet.text.Label(
            "Press ENTER to start", font_size=14, x=app.width // 2, y=app.height * 0.5,
            anchor_x="center", anchor_y="center", color=(180, 180, 180, 255),
        )
        self.hint_label = pyglet.text.Label(
            "ESC to quit  |  O for options", font_size=10, x=app.width // 2, y=app.height * 0.3,
            anchor_x="center", anchor_y="center", color=(140, 140, 140, 255),
        )

    def on_draw(self) -> None:
        if not hasattr(self, '_draw_count'):
            self._draw_count = 0
        self._draw_count += 1
        if self._draw_count <= 3:
            log.info("[TitleScreen] on_draw #%d", self._draw_count)
        self.app.ctx.clear(0.04, 0.05, 0.07, 1.0)
        self.title_label.draw()
        self.sub_label.draw()
        self.hint_label.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.O:
            self.app.change_screen("options")
        elif symbol == key.ESCAPE:
            pyglet.app.exit()


class GameScreen(BaseScreen):
    """Main gameplay screen."""

    def __init__(self, app: "GameApp") -> None:
        super().__init__(app)
        self.paused = False
        self.show_console = False
        self.console = ConsoleOverlay(app)

    def on_draw(self) -> None:
        if self.app.world is None:
            return
        self.app.renderer.draw(
            self.app.world,
            self.app.hero,
            self.app.world.camera_x,
            self.app.world.camera_y,
            self.app.view_mode,
            dt=self.app._last_dt,
        )
        if self.show_console:
            self.console.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.ESCAPE:
            self.app.change_screen("title")
            return
        if self.show_console:
            self.console.on_key_press(symbol, modifiers)
            return
        if symbol == key.C:
            self.show_console = not self.show_console
            return

    def update(self, dt: float) -> None:
        if self.paused or self.app.world is None:
            return
        self.app.update_game(dt)


CELL_SCALE_OPTIONS = [4, 6, 8]

class OptionsScreen(BaseScreen):
    """Options screen with seed input and cell scale selection."""

    def __init__(self, app: "GameApp") -> None:
        super().__init__(app)
        self._scale_idx = CELL_SCALE_OPTIONS.index(cfg.CELL_SCALE) if cfg.CELL_SCALE in CELL_SCALE_OPTIONS else 0
        self.title_label = pyglet.text.Label(
            "Options", font_size=28, x=app.width // 2, y=app.height * 0.8,
            anchor_x="center", anchor_y="center", color=(240, 220, 180, 255),
        )
        self.seed_label = pyglet.text.Label(
            f"Seed: {app.seed}", font_size=14, x=app.width // 2, y=app.height * 0.5,
            anchor_x="center", anchor_y="center", color=(200, 200, 200, 255),
        )
        self.scale_label = pyglet.text.Label(
            font_size=14, x=app.width // 2, y=app.height * 0.4,
            anchor_x="center", anchor_y="center", color=(200, 200, 200, 255),
        )
        self.controls_label = pyglet.text.Label(
            "WASD: Move  SPACE: Chant  1-8: Select spell  C: Console",
            font_size=10, x=app.width // 2, y=app.height * 0.3,
            anchor_x="center", anchor_y="center", color=(160, 160, 160, 255),
        )
        self.back_hint = pyglet.text.Label(
            "ESC: Back  LEFT/RIGHT: Change cell scale",
            font_size=12, x=app.width // 2, y=app.height * 0.2,
            anchor_x="center", anchor_y="center", color=(140, 140, 140, 255),
        )
        self._update_labels()

    def _update_labels(self) -> None:
        scale = CELL_SCALE_OPTIONS[self._scale_idx]
        self.scale_label.text = f"Cell Scale: {scale}  (LEFT/RIGHT to change)"

    def on_draw(self) -> None:
        self.app.ctx.clear(0.04, 0.05, 0.07, 1.0)
        self.title_label.draw()
        self.seed_label.draw()
        self.scale_label.draw()
        self.controls_label.draw()
        self.back_hint.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.ESCAPE:
            self.app.change_screen("title")
        elif symbol == key.LEFT:
            self._scale_idx = (self._scale_idx - 1) % len(CELL_SCALE_OPTIONS)
            self._apply_scale()
        elif symbol == key.RIGHT:
            self._scale_idx = (self._scale_idx + 1) % len(CELL_SCALE_OPTIONS)
            self._apply_scale()

    def _apply_scale(self) -> None:
        scale = CELL_SCALE_OPTIONS[self._scale_idx]
        self.app.resize_window(scale)
        self._update_labels()


class ConsoleOverlay:
    """Modal overlay for spell selection. Sets hero.selected_spell_idx."""

    def __init__(self, app: "GameApp") -> None:
        self.app = app
        self.selected = 0

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Navigate spell menu and set selected spell index."""
        if symbol == key.UP:
            self.selected = max(0, self.selected - 1)
        elif symbol == key.DOWN:
            self.selected = min(len(SPELL_CATALOG) - 1, self.selected + 1)
        elif symbol == key.ENTER:
            # Select this spell and close console
            self.app.hero.selected_spell_idx = self.selected
            game_screen = self.app.screens.get("game")
            if isinstance(game_screen, GameScreen):
                game_screen.show_console = False
        elif symbol == key.C or symbol == key.ESCAPE:
            game_screen = self.app.screens.get("game")
            if isinstance(game_screen, GameScreen):
                game_screen.show_console = False

    def draw(self) -> None:
        w = self.app.width
        h = self.app.height
        pyglet.shapes.Rectangle(0, 0, w, h, color=(0, 0, 0)).draw()
        pyglet.text.Label(
            "-- Spell Menu --", font_size=18, x=w // 2, y=h - 40,
            anchor_x="center", color=(240, 220, 180, 255),
        ).draw()
        for i, spell in enumerate(SPELL_CATALOG):
            color = (255, 255, 0, 255) if i == self.selected else (200, 200, 200, 255)
            prefix = "> " if i == self.selected else "  "
            pyglet.text.Label(
                f"{prefix}{i + 1}. {spell['name']} (MP: {spell['mp']})",
                font_size=12, x=w // 2 - 100, y=h - 70 - i * 20,
                color=color,
            ).draw()