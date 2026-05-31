"""Debug HTTP server for inspecting game state at runtime.

Provides endpoints for:
- /status, /hero, /enemies, /projectiles: game state queries
- /cells, /gpu_cells, /gpu_pressure: grid inspection
- /press?key=...: key input dispatch (main-thread safe)
- /screenshot: PNG framebuffer capture (main-thread safe)
"""

from __future__ import annotations

import json
import struct
import threading
import zlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.game.app import GameApp


class DebugHandler(BaseHTTPRequestHandler):
    """HTTP handler that reads game state."""

    app: GameApp  # set by factory

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        try:
            if path == "/status":
                data = self._get_status()
            elif path == "/hero":
                data = self._get_hero()
            elif path == "/enemies":
                data = self._get_enemies()
            elif path == "/projectiles":
                data = self._get_projectiles()
            elif path == "/fps":
                data = self._get_fps()
            elif path == "/snapshots":
                data = self._get_snapshots()
            elif path.startswith("/cells"):
                data = self._get_cells()
            elif path.startswith("/gpu_cells"):
                data = self._get_gpu_cells()
            elif path.startswith("/gpu_debug"):
                data = self._get_gpu_debug()
            elif path.startswith("/gpu_apply_pressure"):
                data = self._gpu_apply_pressure()
            elif path.startswith("/gpu_inject_probe"):
                data = self._gpu_inject_probe()
            elif path.startswith("/gpu_pressure_timeline"):
                data = self._gpu_pressure_timeline()
            elif path.startswith("/gpu_pressure"):
                data = self._get_gpu_pressure()
            elif path.startswith("/press"):
                data = self._press_key()
            elif path == "/screenshot":
                return self._send_screenshot()
            elif path.startswith("/teleport_surface"):
                data = self._teleport_surface()
            elif path.startswith("/nudge_surface"):
                data = self._nudge_surface()
            elif path.startswith("/teleport"):
                data = self._teleport()
            elif path.startswith("/explode"):
                data = self._explode()
            elif path.startswith("/spawn"):
                data = self._spawn_enemy()
            elif path.startswith("/heal"):
                data = self._heal_hero()
            else:
                data = {
                    "error": "unknown endpoint",
                    "endpoints": [
                        "/status", "/hero", "/enemies", "/projectiles",
                        "/fps",
                        "/snapshots",
                        "/cells?x=..&y=..&w=..&h=..",
                        "/gpu_cells?x=..&y=..&w=..&h=..",
                        "/gpu_pressure?x=..&y=..&w=..&h=..",
                        "/gpu_debug",
                        "/gpu_apply_pressure",
                        "/gpu_inject_probe?x=..&y=..&radius=..&pressure=..",
                        "/gpu_pressure_timeline?x=..&y=..&radius=..&frames=..",
                        "/press?key=ENTER", "/screenshot",
                        "/teleport?x=..&y=..",
                        "/teleport_surface?x=..",
                        "/nudge_surface?dx=..",
                        "/explode?x=..&y=..",
                        "/spawn?type=A|B|C",
                        "/heal?amount=50",
                    ],
                }
        except Exception as exc:
            data = {"error": str(exc)}
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    # ── State queries ──────────────────────────────────────────────

    def _get_status(self) -> dict[str, Any]:
        app = self.app
        paging = None
        if app.world is not None:
            chunk_stats = app.world.chunk_cache.snapshot_stats()
            paging = {
                "shift_ms": round(app.world.shift_time_last_ms(), 3),
                "incoming_load_ms": round(app.world.incoming_load_time_last_ms(), 3),
                "evict_stage_ms": round(app.world.stage_time_last_ms(), 3),
                "overlap_ms": round(app.world.overlap_copy_time_last_ms(), 3),
                "overlap_fx_ms": round(app.world.overlap_transient_copy_time_last_ms(), 3),
                "clear_fx_ms": round(app.world.incoming_transient_clear_time_last_ms(), 3),
                "anchor_build_ms": round(app.world.anchor_build_time_last_ms(), 3),
                "anchor_upload_ms": round(app.world.anchor_upload_time_last_ms(), 3),
                "shift_count": app.world.paging_stats.shift_count,
                "pending_writebacks": app.world.pending_writeback_count,
                "last_shift_cache_hits": app.world.paging_stats.last_shift_cache_hits,
                "last_shift_empty_hits": app.world.paging_stats.last_shift_empty_hits,
                "last_shift_inflight_wait_hits": app.world.paging_stats.last_shift_inflight_wait_hits,
                "last_shift_disk_loads": app.world.paging_stats.last_shift_disk_loads,
                "last_shift_generates": app.world.paging_stats.last_shift_generates,
                "last_shift_saves": app.world.paging_stats.last_shift_saves,
                "last_shift_disk_load_ms": round(app.world.paging_stats.last_shift_disk_load_seconds * 1000.0, 3),
                "last_shift_generate_ms": round(app.world.paging_stats.last_shift_generate_seconds * 1000.0, 3),
                "last_shift_save_ms": round(app.world.paging_stats.last_shift_save_seconds * 1000.0, 3),
                "chunk_cached": chunk_stats.cached_chunks,
                "chunk_prefetch_queued": chunk_stats.prefetch_queued,
                "chunk_prefetch_inflight": chunk_stats.prefetch_inflight,
                "chunk_disk_load_count": chunk_stats.disk_load_count,
                "chunk_generate_count": chunk_stats.generate_count,
                "chunk_save_count": chunk_stats.save_count,
                "chunk_disk_load_last_ms": round(chunk_stats.disk_load_last_seconds * 1000.0, 3),
                "chunk_generate_last_ms": round(chunk_stats.generate_last_seconds * 1000.0, 3),
                "chunk_save_last_ms": round(chunk_stats.save_last_seconds * 1000.0, 3),
            }
        return {
            "screen": app.current_screen,
            "world_loaded": app.world is not None,
            "hero_state": app.hero.state if app.hero else None,
            "camera": (app.world.camera_x, app.world.camera_y) if app.world else None,
            "active_origin": (app.world.active_origin_x, app.world.active_origin_y) if app.world else None,
            "enemy_count": len(app.enemies),
            "projectile_count": len(app.projectiles),
            "active_pressure_bursts": len(app.active_pressure_bursts),
            "local_snapshots": {
                entity_id: {
                    "origin": (snapshot.origin_x, snapshot.origin_y),
                    "size": (snapshot.width, snapshot.height),
                    "tick_id": snapshot.tick_id,
                    "age_frames": snapshot.age_frames,
                }
                for entity_id, snapshot in app.entity_manager._latest_snapshots.items()
            },
            "paging": paging,
            "perf": app.debug_perf_snapshot() if hasattr(app, "debug_perf_snapshot") else {},
        }

    def _get_hero(self) -> dict[str, Any]:
        h = self.app.hero
        return {
            "x": h.x, "y": h.y,
            "top": h.top, "bottom": h.bottom,
            "left": h.left, "right": h.right,
            "vel_x": h.vel_x, "vel_y": h.vel_y,
            "on_ground": h.on_ground,
            "state": h.state,
            "hp": h.hp, "mp": h.mp,
            "facing_right": h.facing_right,
        }

    def _get_enemies(self) -> dict[str, Any]:
        enemies = {}
        for eid, e in self.app.enemies.items():
            enemies[eid] = {
                "type": type(e).__name__,
                "x": e.x, "y": e.y,
                "hp": e.hp, "max_hp": e.max_hp,
                "state": e.state,
                "is_alive": e.is_alive,
                "facing_right": e.facing_right,
                "biome": e.biome,
            }
        return {"count": len(enemies), "enemies": enemies}

    def _get_projectiles(self) -> dict[str, Any]:
        projs = []
        for p in self.app.projectiles:
            projs.append({
                "type": type(p).__name__,
                "x": p.x, "y": p.y,
                "vel_x": p.vel_x, "vel_y": p.vel_y,
                "is_alive": p.is_alive,
                "age": round(p.age, 3),
            })
        return {"count": len(projs), "projectiles": projs}

    def _get_fps(self) -> dict[str, Any]:
        app = self.app
        render_fps = None
        if getattr(app, "renderer", None) is not None:
            render_fps = getattr(app.renderer, "_render_fps", None)
        return {
            "sim_fps": round(float(app.sim_fps), 3) if hasattr(app, "sim_fps") else None,
            "render_fps": round(float(render_fps), 3) if render_fps is not None else None,
        }

    def _get_snapshots(self) -> dict[str, Any]:
        snapshots = {}
        for entity_id, snapshot in self.app.entity_manager._latest_snapshots.items():
            snapshots[entity_id] = {
                "origin": [snapshot.origin_x, snapshot.origin_y],
                "size": [snapshot.width, snapshot.height],
                "tick_id": snapshot.tick_id,
                "age_frames": snapshot.age_frames,
            }
        return {"count": len(snapshots), "snapshots": snapshots}

    # ── Cell inspection ────────────────────────────────────────────

    def _get_cells(self) -> dict[str, Any]:
        return self._get_gpu_cells()

    def _get_gpu_cells(self) -> dict[str, Any]:
        """Read cells directly from GPU textures (bypasses CPU grid)."""
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = int(v)
        ax = app.world.active_origin_x
        ay = app.world.active_origin_y
        x = params.get("x", int(app.hero.x))
        y = params.get("y", int(app.hero.y + app.hero.height))
        w = params.get("w", 5)
        h = params.get("h", 10)
        # Convert world coords to local grid coords
        lx = x - ax
        ly = y - ay
        cells_2d = app.world.gpu_simulator.readback_cells_region(lx, ly, w, h)
        cells = []
        for row_idx, row_cells in enumerate(cells_2d):
            for col_idx, cell in enumerate(row_cells):
                wx = x + col_idx
                wy = y + row_idx
                cells.append({
                    "wx": wx, "wy": wy,
                    "family": cell.family_id,
                    "variant": cell.variant_id,
                    "integrity": round(cell.integrity, 4),
                    "temperature": round(cell.temperature, 2),
                })
        return {"x": x, "y": y, "w": w, "h": h, "source": "gpu", "cells": cells}

    def _get_gpu_pressure(self) -> dict[str, Any]:
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = int(v)
        x = params.get("x", int(app.hero.x) - 4)
        y = params.get("y", int(app.hero.y + app.hero.height * 0.5) - 4)
        w = params.get("w", 9)
        h = params.get("h", 9)
        pressure_rows = app.world.readback_pressure_region_world(x, y, w, h)
        samples = []
        for row_idx, row_values in enumerate(pressure_rows):
            for col_idx, value in enumerate(row_values):
                samples.append({
                    "wx": x + col_idx,
                    "wy": y + row_idx,
                    "pressure": round(float(value), 3),
                })
        return {"x": x, "y": y, "w": w, "h": h, "source": "gpu_pressure", "samples": samples}

    def _get_gpu_debug(self) -> dict[str, Any]:
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        gpu = app.world.gpu_simulator
        return {
            "step_index": int(gpu.step_index),
            "front_index": int(gpu.front_index),
            "pressure_front_index": int(gpu.pressure_front_index),
            "source_force_front_index": int(gpu.source_force_front_index),
            "wave_force_front_index": int(gpu.wave_force_front_index),
            "pending_pressure_injections": [
                {
                    "center_x": int(cx),
                    "center_y": int(cy),
                    "inner_radius": int(inner_radius),
                    "outer_radius": int(outer_radius),
                    "pressure": float(pressure),
                }
                for cx, cy, inner_radius, outer_radius, pressure in getattr(gpu, "_pending_pressure_injections", [])
            ],
        }

    def _gpu_apply_pressure(self) -> dict[str, Any]:
        import pyglet
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_apply(dt):
            try:
                injections = app.world.gpu_simulator.debug_apply_pressure_injections_now()
                result["applied"] = [
                    {
                        "center_x": int(cx),
                        "center_y": int(cy),
                        "inner_radius": int(inner_radius),
                        "outer_radius": int(outer_radius),
                        "pressure": float(pressure),
                    }
                    for cx, cy, inner_radius, outer_radius, pressure in injections
                ]
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_apply, 0)
        event.wait(timeout=5)
        return result

    def _gpu_inject_probe(self) -> dict[str, Any]:
        import pyglet
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        tx = int(params.get("x", app.hero.x))
        ty = int(params.get("y", app.hero.y + app.hero.height * 0.5))
        radius = max(1, int(params.get("radius", 6)))
        pressure = float(params.get("pressure", 120.0))
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_probe(dt):
            try:
                app.world.inject_pressure_world(tx, ty, radius, pressure)
                injections = app.world.gpu_simulator.debug_apply_pressure_injections_now()
                pressure_rows = app.world.readback_pressure_region_world(tx - radius, ty - radius, radius * 2 + 1, radius * 2 + 1)
                flat = [float(value) for row in pressure_rows for value in row]
                result["probe"] = {
                    "x": tx,
                    "y": ty,
                    "radius": radius,
                    "pressure": pressure,
                    "applied_count": len(injections),
                    "max_pressure": max(flat) if flat else 0.0,
                    "nonzero_samples": sum(1 for value in flat if value > 0.0),
                }
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_probe, 0)
        event.wait(timeout=5)
        return result

    def _gpu_pressure_timeline(self) -> dict[str, Any]:
        import pyglet
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        tx = int(params.get("x", app.hero.x))
        ty = int(params.get("y", app.hero.y + app.hero.height * 0.5))
        radius = max(1, int(params.get("radius", 12)))
        frames = max(1, min(12, int(params.get("frames", 6))))
        event = threading.Event()
        result: dict[str, Any] = {}

        def _sample() -> dict[str, Any]:
            rows = app.world.readback_pressure_region_world(tx - radius, ty - radius, radius * 2 + 1, radius * 2 + 1)
            flat = [float(value) for row in rows for value in row]
            return {
                "step_index": int(app.world.gpu_simulator.step_index),
                "max_pressure": max(flat) if flat else 0.0,
                "nonzero_samples": sum(1 for value in flat if value > 0.0),
            }

        def _do_timeline(dt):
            samples: list[dict[str, Any]] = []

            def _tick_sample(inner_dt):
                samples.append(_sample())
                if len(samples) >= frames:
                    result["timeline"] = {
                        "x": tx,
                        "y": ty,
                        "radius": radius,
                        "frames": frames,
                        "samples": samples,
                    }
                    event.set()
                    return
                pyglet.clock.schedule_once(_tick_sample, 0)

            _tick_sample(0)

        pyglet.clock.schedule_once(_do_timeline, 0)
        event.wait(timeout=5)
        return result

    # ── Key input ──────────────────────────────────────────────────

    def _press_key(self) -> dict[str, Any]:
        import pyglet
        from pyglet.window import key as pyglet_key
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        key_name = params.get("key", "ENTER").upper()
        key_map = {
            "ENTER": pyglet_key.ENTER,
            "SPACE": pyglet_key.SPACE,
            "W": pyglet_key.W, "A": pyglet_key.A,
            "S": pyglet_key.S, "D": pyglet_key.D,
            "UP": pyglet_key.UP, "DOWN": pyglet_key.DOWN,
            "LEFT": pyglet_key.LEFT, "RIGHT": pyglet_key.RIGHT,
            "ESCAPE": pyglet_key.ESCAPE,
            "F3": pyglet_key.F3,
            "F4": pyglet_key.F4,
            "F5": pyglet_key.F5,
        }
        symbol = key_map.get(key_name, pyglet_key.ENTER)
        app = self.app
        # Dispatch on main thread — OpenGL operations (e.g. _init_game_world)
        # must run on the thread that owns the GL context.
        event = threading.Event()
        def _do_press(dt):
            app.on_key_press(symbol, 0)
            event.set()
        pyglet.clock.schedule_once(_do_press, 0)
        event.wait(timeout=5)
        # Schedule key release after a short delay
        def _do_release(dt):
            app.on_key_release(symbol, 0)
        pyglet.clock.schedule_once(_do_release, 0.1)
        return {"pressed": key_name, "symbol": symbol, "screen": app.current_screen}

    # ── Debug actions ──────────────────────────────────────────────

    def _teleport(self) -> dict[str, Any]:
        """Teleport hero to (x, y) world coordinates."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        tx = params.get("x", self.app.hero.x)
        ty = params.get("y", self.app.hero.y)
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_teleport(dt):
            try:
                app.move_hero_to(tx, ty)
                result["teleport"] = {"x": tx, "y": ty}
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_teleport, 0)
        event.wait(timeout=20)
        return result

    def _teleport_surface(self) -> dict[str, Any]:
        """Teleport hero to the nearest valid ground spawn near x."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        tx = params.get("x", self.app.hero.x)
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_teleport(dt):
            try:
                if app.teleport_to_surface_x(tx):
                    result["teleport_surface"] = {"x": float(app.hero.x), "y": float(app.hero.y)}
                else:
                    result["error"] = f"no surface spawn near x={tx}"
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_teleport, 0)
        event.wait(timeout=10)
        return result

    def _nudge_surface(self) -> dict[str, Any]:
        """Move hero horizontally while snapping back to a valid surface spawn.

        Unlike teleport, this does not force the camera, so page shifts still
        happen through the normal pan_camera path on subsequent ticks.
        """
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        dx = params.get("dx", 0.0)
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_nudge(dt):
            try:
                if app.terrain_gen is None:
                    result["error"] = "no terrain generator"
                    event.set()
                    return
                target_x = float(app.hero.x + dx)
                spawn = app.terrain_gen.find_spawn_point_near(
                    int(target_x),
                    entity_width=app.hero.width,
                    entity_height=app.hero.height,
                    search_radius=max(64, int(abs(dx)) + 32),
                )
                if spawn is None:
                    result["error"] = f"no surface spawn near x={target_x}"
                    event.set()
                    return
                spawn_x, spawn_y = spawn
                app.hero.x = spawn_x
                app.hero.y = spawn_y
                app.hero.vel_x = 0.0
                app.hero.vel_y = 0.0
                app.hero.on_ground = False
                hero_entity = app.entity_manager._entities.get("hero")
                if hero_entity is not None:
                    hero_entity.x = spawn_x
                    hero_entity.y = spawn_y
                result["nudge_surface"] = {"x": float(spawn_x), "y": float(spawn_y), "dx": float(dx)}
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_nudge, 0)
        event.wait(timeout=15)
        return result

    def _explode(self) -> dict[str, Any]:
        """Trigger an explosion at a world position for debugging."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        app = self.app
        tx = params.get("x", app.hero.x)
        ty = params.get("y", app.hero.y + app.hero.height * 0.5)
        pressure = params.get("pressure")
        radius = params.get("radius")
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_explode(dt):
            try:
                if app.world is None:
                    result["error"] = "world not loaded"
                else:
                    app._trigger_explosion(
                        int(tx),
                        int(ty),
                        pressure=None if pressure is None else float(pressure),
                        radius=None if radius is None else int(radius),
                    )
                    result["explosion"] = {
                        "x": int(tx),
                        "y": int(ty),
                        "pressure": pressure,
                        "radius": radius,
                    }
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_explode, 0)
        event.wait(timeout=5)
        return result

    def _spawn_enemy(self) -> dict[str, Any]:
        """Spawn an enemy near the hero."""
        import pyglet
        from src.game.enemy import EnemyA, EnemyB, EnemyC
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        enemy_type = params.get("type", "A").upper()
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_spawn(dt):
            try:
                if app.world is None:
                    result["error"] = "world not loaded"
                    event.set()
                    return
                # Spawn near hero
                sx = app.hero.x + 30.0
                sy = app.hero.y
                eid = f"debug_{enemy_type}_{len(app.enemies)}"
                if enemy_type == "B":
                    enemy = EnemyB.create(eid, sx, sy, "plains")
                elif enemy_type == "C":
                    enemy = EnemyC.create(eid, sx, sy, "underground")
                else:
                    enemy = EnemyA.create(eid, sx, sy, "plains")
                app.enemies[eid] = enemy
                app.entity_manager.register_enemy(enemy)
                result["spawned"] = {"id": eid, "type": enemy_type, "x": sx, "y": sy}
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_spawn, 0)
        event.wait(timeout=5)
        return result

    def _heal_hero(self) -> dict[str, Any]:
        """Heal the hero by a given amount."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        amount = params.get("amount", 50.0)
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_heal(dt):
            try:
                app.hero.heal(amount)
                result["healed"] = {"amount": amount, "hp": app.hero.hp}
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_heal, 0)
        event.wait(timeout=5)
        return result

    # ── Screenshot ─────────────────────────────────────────────────

    def _send_screenshot(self) -> None:
        """Capture framebuffer as PNG on the main thread and send it."""
        import ctypes
        import ctypes.wintypes
        import pyglet
        event = threading.Event()
        result: dict[str, Any] = {}

        def _capture(dt):
            try:
                app = self.app
                w, h = app.width, app.height
                # Use ModernGL's ctx.screen.read which works on the current context
                raw = app.ctx.screen.read(components=3, alignment=1)
                # GL returns bottom-to-top RGB, flip vertically for PNG
                row_stride = w * 3
                rgb_rows = bytearray()
                for y in range(h - 1, -1, -1):
                    offset = y * row_stride
                    rgb_rows.extend(raw[offset:offset + row_stride])
                png_bytes = _rgb_to_png(w, h, bytes(rgb_rows))
                result["width"] = w
                result["height"] = h
                result["png"] = png_bytes
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_capture, 0)
        event.wait(timeout=10)

        if "error" in result:
            body = json.dumps(result).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return

        png_data = result["png"]
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_data)))
        self.end_headers()
        self.wfile.write(png_data)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress request logging


# ── PNG encoding (no Pillow dependency) ────────────────────────────

def _rgb_to_png(width: int, height: int, rgb: bytes) -> bytes:
    """Minimal PNG encoder for RGB framebuffer data."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # Filter each row with filter type 0 (None)
    raw_rows = b""
    row_bytes = width * 3
    for y in range(height):
        offset = y * row_bytes
        raw_rows += b"\x00" + rgb[offset : offset + row_bytes]

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    compressed = zlib.compress(raw_rows, 6)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


# ── Server wrapper ─────────────────────────────────────────────────

class DebugServer:
    """HTTP debug server for automated testing."""

    def __init__(self, app: GameApp, port: int = 9123) -> None:
        self._app = app
        self._port = port
        self._server: HTTPServer | None = None

    def start(self) -> None:
        handler = type("H", (DebugHandler,), {"app": self._app})
        self._server = HTTPServer(("127.0.0.1", self._port), handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
