"""Pyglet presentation path for GPU-owner rendered frames.

This renderer intentionally avoids ``moderngl``. The GPU owner thread
renders the authoritative world frame and returns CPU RGBA bytes; the
main thread presents those bytes and draws lightweight UI overlays.
"""

from __future__ import annotations

import time
from array import array
from collections import defaultdict
from typing import Any

import pyglet

from src.engine.gpu_owner import GpuFramePayload
from src.game import config as cfg
from src.game.enemy import EnemyB, EnemyC
from src.game.entity_manager import DebugCollisionInfo
from src.game.hero import Hero
from src.game.projectile import Arrow, Fireball


class OwnerFrameRenderer:
    """Display-only renderer used by the owner-created world path."""

    def __init__(self, window_width: int, window_height: int) -> None:
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self._overlay_commands: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        self._render_fps = 0.0
        self._render_count = 0
        self._render_started_at = time.perf_counter()

    def resize(self, window_width: int, window_height: int) -> None:
        self.window_width = int(window_width)
        self.window_height = int(window_height)

    def _record_render(self) -> None:
        self._render_count += 1
        now = time.perf_counter()
        elapsed = now - self._render_started_at
        if elapsed >= 1.0:
            self._render_fps = self._render_count / elapsed
            self._render_count = 0
            self._render_started_at = now

    def _commands_for(self, vertices: array) -> list[tuple[Any, ...]]:
        return self._overlay_commands[id(vertices)]

    def _append_rect(
        self,
        vertices: array,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        *,
        opacity: int = 255,
    ) -> None:
        del vertices[:]
        self._commands_for(vertices).append(
            ("rect", float(x0), float(y0), float(x1), float(y1), tuple(color), int(opacity))
        )

    def _append_circle(
        self,
        vertices: array,
        cx: float,
        cy: float,
        radius: float,
        color: tuple[int, int, int],
        *,
        opacity: int = 255,
    ) -> None:
        del vertices[:]
        self._commands_for(vertices).append(
            ("circle", float(cx), float(cy), float(radius), tuple(color), int(opacity))
        )

    def _append_text(
        self,
        vertices: array,
        x: float,
        y: float,
        text: str,
        color: tuple[int, int, int],
        *,
        pixel_size: float = 2.0,
    ) -> float:
        del vertices[:]
        text_value = str(text)
        self._commands_for(vertices).append(
            ("text", float(x), float(y), text_value, tuple(color), float(pixel_size))
        )
        return float(x) + len(text_value) * 6.0 * float(pixel_size)

    def _flush_overlay(self, vertices: array, mode: object = None) -> None:
        del mode
        commands = self._overlay_commands.pop(id(vertices), [])
        if not commands:
            return
        batch = pyglet.graphics.Batch()
        drawables: list[Any] = []
        for command in commands:
            kind = command[0]
            if kind == "rect":
                _, x0, y0, x1, y1, color, opacity = command
                rect = pyglet.shapes.Rectangle(
                    x0,
                    y0,
                    max(1.0, x1 - x0),
                    max(1.0, y1 - y0),
                    color=color,
                    batch=batch,
                )
                rect.opacity = max(0, min(255, int(opacity)))
                drawables.append(rect)
            elif kind == "circle":
                _, cx, cy, radius, color, opacity = command
                circle = pyglet.shapes.Circle(cx, cy, max(1.0, radius), color=color, batch=batch)
                circle.opacity = max(0, min(255, int(opacity)))
                drawables.append(circle)
            elif kind == "text":
                _, x, y, text, color, pixel_size = command
                label = pyglet.text.Label(
                    text,
                    x=x,
                    y=y,
                    anchor_x="left",
                    anchor_y="baseline",
                    font_name="Consolas",
                    font_size=max(6, int(round(pixel_size * 5.5))),
                    color=(int(color[0]), int(color[1]), int(color[2]), 255),
                    batch=batch,
                )
                drawables.append(label)
        batch.draw()
        drawables.clear()

    def draw_grid_payload(self, payload: GpuFramePayload) -> None:
        tex_width = max(1, int(payload.width))
        tex_height = max(1, int(payload.height))
        origin_x, origin_y, scale_x, scale_y = payload.uv_rect
        crop_x = max(0, min(tex_width - 1, int(origin_x * tex_width)))
        crop_y = max(0, min(tex_height - 1, int(origin_y * tex_height)))
        crop_width = max(1, min(tex_width - crop_x, int(round(scale_x * tex_width))))
        crop_height = max(1, min(tex_height - crop_y, int(round(scale_y * tex_height))))
        image = pyglet.image.ImageData(tex_width, tex_height, "RGBA", payload.rgba, pitch=tex_width * 4)
        region = image.get_region(crop_x, crop_y, crop_width, crop_height)
        region.blit(0, 0, width=self.window_width, height=self.window_height)

    def _draw_rects(self, rects: list[tuple[float, float, float, float, tuple[int, int, int], int]]) -> None:
        if not rects:
            return
        verts = array("f")
        for x0, y0, x1, y1, color, opacity in rects:
            self._append_rect(verts, x0, y0, x1, y1, color, opacity=opacity)
        self._flush_overlay(verts)

    def draw_hero(self, hero: Hero, camera_x: int, camera_y: int, dt: float) -> None:
        del dt
        cs = cfg.CELL_SCALE
        sx = (hero.left - camera_x) * cs
        sy = self.window_height - (hero.top - camera_y) * cs
        color = (240, 70, 52) if hero.state != "chant" else (120, 190, 255)
        self._draw_rects([(sx, sy, sx + hero.width * cs, sy + hero.height * cs, color, 255)])

    def draw_enemies(self, enemies: dict, camera_x: int, camera_y: int) -> None:
        cs = cfg.CELL_SCALE
        rects: list[tuple[float, float, float, float, tuple[int, int, int], int]] = []
        for enemy in enemies.values():
            if not enemy.is_alive:
                continue
            sx = (enemy.left - camera_x) * cs
            sy = self.window_height - (enemy.top - camera_y) * cs
            color: tuple[int, int, int] = (50, 100, 220)
            if isinstance(enemy, EnemyB):
                color = (160, 40, 200)
            elif isinstance(enemy, EnemyC):
                color = (220, 30, 30)
            if enemy.damage_flash_timer > 0.0:
                color = (255, 255, 255)
            rects.append((sx, sy, sx + enemy.width * cs, sy + enemy.height * cs, color, 230))
        self._draw_rects(rects)

    def draw_projectiles(self, projectiles: list, camera_x: int, camera_y: int) -> None:
        cs = cfg.CELL_SCALE
        verts = array("f")
        for projectile in projectiles:
            if not projectile.is_alive:
                continue
            sx = (projectile.x - camera_x) * cs
            sy = self.window_height - (projectile.y - camera_y) * cs - cs
            if isinstance(projectile, Arrow):
                size = cs * 2
                self._append_rect(
                    verts,
                    sx - size * 0.5,
                    sy - size * 0.5,
                    sx + size * 0.5,
                    sy + size * 0.5,
                    (180, 120, 60),
                    opacity=220,
                )
            elif isinstance(projectile, Fireball):
                self._append_circle(verts, sx, sy, cs * 2, (255, 140, 0), opacity=230)
        self._flush_overlay(verts)

    def draw(
        self,
        world: object,
        hero: Hero,
        camera_x: int,
        camera_y: int,
        view_mode: object,
        *,
        dt: float,
        enemies: dict,
        projectiles: list,
        frame_payload: GpuFramePayload | None = None,
    ) -> None:
        del world, view_mode
        if frame_payload is not None:
            self.draw_grid_payload(frame_payload)
        self.draw_hero(hero, camera_x, camera_y, dt)
        self.draw_enemies(enemies, camera_x, camera_y)
        self.draw_projectiles(projectiles, camera_x, camera_y)
        self._record_render()

    def draw_perf_overlay(self, app: object) -> None:
        lines = [
            f"S {float(getattr(app, 'sim_fps', 0.0)):.1f}",
            f"R {float(getattr(self, '_render_fps', 0.0)):.1f}",
        ]
        perf = app.debug_perf_snapshot() if hasattr(app, "debug_perf_snapshot") else {}
        for key in ("main_tick_total_last_ms", "world_step_submit_cpu_last_ms", "snapshot_ready_to_consume_delay_last_ms"):
            if key in perf:
                lines.append(f"{key}: {float(perf[key]):.2f}ms")
        verts = array("f")
        y = self.window_height - 24.0
        for line in lines:
            self._append_text(verts, 12.0, y, line, (230, 230, 210), pixel_size=2.0)
            y -= 18.0
        self._flush_overlay(verts)

    def draw_debug_collision(self, camera_x: int, camera_y: int, debug: DebugCollisionInfo) -> None:
        cs = cfg.CELL_SCALE
        rects: list[tuple[float, float, float, float, tuple[int, int, int], int]] = []
        for wx, wy in (
            list(debug.ground_cells)
            + list(debug.ceiling_cells)
            + list(debug.left_wall_cells)
            + list(debug.right_wall_cells)
        ):
            sx = (wx - camera_x) * cs
            sy = self.window_height - (wy - camera_y + 1) * cs
            rects.append((sx, sy, sx + cs, sy + cs, (255, 255, 0), 180))
        self._draw_rects(rects)
