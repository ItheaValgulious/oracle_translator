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
        if self._draw_count <= 3 and log.isEnabledFor(logging.DEBUG):
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
        self.app.ctx.clear(0.04, 0.05, 0.07, 1.0)
        cam_x = self.app.world.camera_x
        cam_y = self.app.world.camera_y
        self.app.renderer.draw(
            self.app.world,
            self.app.hero,
            cam_x,
            cam_y,
            self.app.view_mode,
            dt=self.app._last_dt,
            enemies=self.app.enemies,
            projectiles=self.app.projectiles,
        )
        if self.app.debug_overlay_enabled:
            self.app.renderer.draw_perf_overlay(self.app)
        if self.app.entity_manager.debug_collision:
            debug = self.app.entity_manager.last_debug
            if debug is not None:
                self.app.renderer.draw_debug_collision(cam_x, cam_y, debug)
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
    """Modal overlay for spell selection + debug actions."""

    def __init__(self, app: "GameApp") -> None:
        self.app = app
        self.selected = 0
        # Menu items: (label, action_callback)
        self._items: list[tuple[str, callable]] = []
        self._rebuild_items()

    def _rebuild_items(self) -> None:
        self._items = []
        # Spell entries
        for i, spell in enumerate(SPELL_CATALOG):
            self._items.append((f"{i + 1}. {spell['name']} (MP: {spell['mp']})", lambda idx=i: self._select_spell(idx)))
        # Separator
        self._items.append(("", None))
        # Debug actions
        self._items.append(("Spawn Enemy A (nearby)", lambda: self._spawn("A")))
        self._items.append(("Spawn Enemy B (nearby)", lambda: self._spawn("B")))
        self._items.append(("Spawn Enemy C/Boss (nearby)", lambda: self._spawn("C")))
        self._items.append(("Heal Hero +50 HP", lambda: self._heal(50)))
        self._items.append(("", None))
        # Teleport to terrain types
        bw = cfg.BIOME_WIDTH
        self._items.append(("Teleport: Plains (center)", lambda: self._teleport_biome(bw // 2)))
        self._items.append(("Teleport: Hillside (center)", lambda: self._teleport_biome(bw + bw // 2)))
        self._items.append(("Teleport: Alpine (center)", lambda: self._teleport_biome(2 * bw + bw // 2)))
        self._items.append(("Teleport: Underground (center)", lambda: self._teleport_biome(3 * bw + bw // 2)))

    def _select_spell(self, idx: int) -> None:
        self.app.hero.selected_spell_idx = idx
        game_screen = self.app.screens.get("game")
        if isinstance(game_screen, GameScreen):
            game_screen.show_console = False

    def _spawn(self, enemy_type: str) -> None:
        from src.game.enemy import EnemyA, EnemyB, EnemyC
        if self.app.world is None:
            return
        sx = self.app.hero.x + 30.0
        sy = self.app.hero.y
        eid = f"debug_{enemy_type}_{len(self.app.enemies)}"
        if enemy_type == "B":
            # Flying enemy: spawn above hero at hover height
            sy = self.app.hero.y - cfg.ENEMY_B_HOVER_HEIGHT
            enemy = EnemyB.create(eid, sx, sy, "plains")
        elif enemy_type == "C":
            enemy = EnemyC.create(eid, sx, sy, "underground")
        else:
            enemy = EnemyA.create(eid, sx, sy, "plains")
        self.app.enemies[eid] = enemy
        self.app.entity_manager.register_enemy(enemy)
        self.app.entity_manager.register_entity_shapes(self.app.world)

    def _heal(self, amount: float) -> None:
        self.app.hero.heal(amount)

    def _teleport(self, x: float, y: float) -> None:
        self.app.move_hero_to(x, y)

    def _teleport_biome(self, x: float) -> None:
        """Teleport hero to ground level at the given world x."""
        if not self.app.teleport_to_surface_x(x):
            if self.app.terrain_gen is None:
                return
            ground_y = self.app.terrain_gen.ground_height_at(int(x))
            self._teleport(x, ground_y - 1 - cfg.HERO_HEIGHT)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Navigate menu and execute actions."""
        if symbol == key.UP:
            self.selected = max(0, self.selected - 1)
        elif symbol == key.DOWN:
            self.selected = min(len(self._items) - 1, self.selected + 1)
        elif symbol == key.ENTER:
            _, action = self._items[self.selected]
            if action is not None:
                action()
            # Close console after action (for debug items, keep open for spells)
            if self.selected < len(SPELL_CATALOG):
                game_screen = self.app.screens.get("game")
                if isinstance(game_screen, GameScreen):
                    game_screen.show_console = False
        elif symbol == key.C or symbol == key.ESCAPE:
            game_screen = self.app.screens.get("game")
            if isinstance(game_screen, GameScreen):
                game_screen.show_console = False

    def draw(self) -> None:
        """Draw console overlay using bitmap font (no pyglet labels after GPU compute)."""
        w = self.app.width
        h = self.app.height
        from array import array as _array
        renderer = self.app.renderer
        verts = _array("f")
        line_h = 27.0  # pixel_size * (FONT_HEIGHT+1) approx
        # Title
        title = "-- Menu (UP/DOWN/ENTER) --"
        tw = len(title) * 6 * 3.0  # (FONT_WIDTH+1) * pixel_size per char
        renderer._append_text(verts, (w - tw) / 2, h - 50.0, title, (240, 220, 180), pixel_size=3.0)
        # Spells section
        renderer._append_text(verts, 60.0, h - 85.0, "Spells:", (180, 180, 180), pixel_size=3.0)
        for i in range(min(len(SPELL_CATALOG), len(self._items))):
            label, _ = self._items[i]
            is_selected = i == self.selected
            color = (255, 255, 0) if is_selected else (200, 200, 200)
            prefix = "> " if is_selected else "  "
            renderer._append_text(verts, 80.0, h - 85.0 - (i + 1) * line_h, f"{prefix}{label}", color, pixel_size=3.0)
        # Debug section
        debug_start = len(SPELL_CATALOG) + 1
        dy = h - 85.0 - (len(SPELL_CATALOG) + 1) * line_h - 15.0
        renderer._append_text(verts, 60.0, dy, "Debug:", (180, 140, 100), pixel_size=3.0)
        for i in range(debug_start, len(self._items)):
            label, _ = self._items[i]
            if not label:
                continue
            is_selected = i == self.selected
            color = (255, 200, 50) if is_selected else (160, 140, 120)
            prefix = "> " if is_selected else "  "
            row = i - debug_start
            renderer._append_text(verts, 80.0, dy - (row + 1) * line_h, f"{prefix}{label}", color, pixel_size=3.0)
        # Dark overlay (drawn first so it appears behind text)
        dark_verts = _array("f")
        renderer._append_rect(dark_verts, 0, 0, float(w), float(h), (0, 0, 0), opacity=200)
        renderer._flush_overlay(dark_verts)
        # Text on top
        renderer._flush_overlay(verts)
