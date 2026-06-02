"""Game screens: Title, Game, Options, Console."""

from __future__ import annotations

import logging
from time import perf_counter
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

    def on_draw(self) -> None:
        self.app.clear_screen()
        renderer = self.app.renderer
        w = float(self.app.width)
        h = float(self.app.height)
        from array import array as _array
        bg_verts = _array("f")
        text_verts = _array("f")
        renderer._append_rect(bg_verts, w * 0.18, h * 0.2, w * 0.82, h * 0.82, (8, 10, 14), opacity=210)
        renderer._append_rect(bg_verts, w * 0.2, h * 0.22, w * 0.8, h * 0.8, (20, 24, 30), opacity=220)
        title = "Oracle Translator"
        title_x = (w - len(title) * 6 * 4.0) * 0.5
        renderer._append_text(text_verts, title_x, h * 0.68, title, (240, 220, 180), pixel_size=4.0)
        subtitle = "Press ENTER to start"
        subtitle_x = (w - len(subtitle) * 6 * 2.5) * 0.5
        renderer._append_text(text_verts, subtitle_x, h * 0.52, subtitle, (200, 205, 215), pixel_size=2.5)
        hint = "O: Options   ESC: Quit"
        hint_x = (w - len(hint) * 6 * 2.0) * 0.5
        renderer._append_text(text_verts, hint_x, h * 0.36, hint, (150, 155, 165), pixel_size=2.0)
        renderer._flush_overlay(bg_verts)
        renderer._flush_overlay(text_verts)

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
        if bool(getattr(self.app, "gpu_owner_active", False)):
            return
        self.app.clear_screen()
        status_fn = getattr(self.app, "gpu_world_status_snapshot", None)
        world_status = status_fn(block=False) if callable(status_fn) else None
        camera = None if world_status is None else world_status.get("camera")
        if camera is not None:
            cam_x = int(camera[0])
            cam_y = int(camera[1])
        else:
            cam_x = self.app.world.camera_x
            cam_y = self.app.world.camera_y
        frame_payload = None
        frame_fn = getattr(self.app, "gpu_frame_payload_snapshot", None)
        if callable(frame_fn):
            frame_payload = frame_fn()
        owner_active = bool(getattr(self.app, "gpu_owner_active", False))
        world_started_at = perf_counter()
        if not owner_active or frame_payload is not None:
            self.app.renderer.draw(
                self.app.world,
                self.app.hero,
                cam_x,
                cam_y,
                self.app.view_mode,
                dt=self.app._last_dt,
                enemies=self.app.enemies,
                projectiles=self.app.projectiles,
                frame_payload=frame_payload,
            )
        world_finished_at = perf_counter()
        self.app._set_perf_ms("draw_world", (world_finished_at - world_started_at) * 1000.0)
        if self.app.debug_overlay_enabled:
            overlay_started_at = perf_counter()
            self.app.renderer.draw_perf_overlay(self.app)
            overlay_finished_at = perf_counter()
            self.app._set_perf_ms("draw_debug_overlay", (overlay_finished_at - overlay_started_at) * 1000.0)
        else:
            self.app._set_perf_ms("draw_debug_overlay", 0.0)
        if self.app.entity_manager.debug_collision:
            debug = self.app.entity_manager.last_debug
            if debug is not None:
                collision_started_at = perf_counter()
                self.app.renderer.draw_debug_collision(cam_x, cam_y, debug)
                collision_finished_at = perf_counter()
                self.app._set_perf_ms("draw_collision_overlay", (collision_finished_at - collision_started_at) * 1000.0)
            else:
                self.app._set_perf_ms("draw_collision_overlay", 0.0)
        else:
            self.app._set_perf_ms("draw_collision_overlay", 0.0)
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


CELL_SCALE_OPTIONS = list(cfg.CELL_SCALE_OPTIONS)

class OptionsScreen(BaseScreen):
    """Options screen with seed input and cell scale selection."""

    def __init__(self, app: "GameApp") -> None:
        super().__init__(app)
        self._scale_idx = CELL_SCALE_OPTIONS.index(cfg.CELL_SCALE) if cfg.CELL_SCALE in CELL_SCALE_OPTIONS else 0
        self._scale_text = ""
        self._update_labels()

    def _update_labels(self) -> None:
        scale = CELL_SCALE_OPTIONS[self._scale_idx]
        self._scale_text = f"Cell Scale: {scale}"

    def on_draw(self) -> None:
        self.app.clear_screen()
        renderer = self.app.renderer
        w = float(self.app.width)
        h = float(self.app.height)
        from array import array as _array
        bg_verts = _array("f")
        text_verts = _array("f")
        renderer._append_rect(bg_verts, w * 0.14, h * 0.12, w * 0.86, h * 0.86, (10, 12, 16), opacity=215)
        renderer._append_rect(bg_verts, w * 0.16, h * 0.14, w * 0.84, h * 0.84, (20, 24, 30), opacity=225)
        title = "Options"
        renderer._append_text(text_verts, (w - len(title) * 6 * 3.5) * 0.5, h * 0.74, title, (240, 220, 180), pixel_size=3.5)
        seed_text = f"Seed: {self.app.seed}"
        renderer._append_text(text_verts, (w - len(seed_text) * 6 * 2.5) * 0.5, h * 0.58, seed_text, (210, 210, 210), pixel_size=2.5)
        renderer._append_text(text_verts, (w - len(self._scale_text) * 6 * 2.5) * 0.5, h * 0.48, self._scale_text, (210, 210, 210), pixel_size=2.5)
        controls = "WASD Move  SPACE Chant  1-8 Spell  C Console"
        renderer._append_text(text_verts, (w - len(controls) * 6 * 1.7) * 0.5, h * 0.34, controls, (170, 170, 170), pixel_size=1.7)
        hint = "LEFT/RIGHT Change Scale   ESC Back"
        renderer._append_text(text_verts, (w - len(hint) * 6 * 2.0) * 0.5, h * 0.24, hint, (150, 150, 160), pixel_size=2.0)
        renderer._flush_overlay(bg_verts)
        renderer._flush_overlay(text_verts)

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
        self.app._submit_gpu_entity_shape_registration()

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
