"""Debug HTTP server for inspecting game state at runtime.

Provides endpoints for:
- /status, /hero, /enemies, /projectiles: game state queries
- /cells, /gpu_cells, /gpu_pressure: grid inspection
- /press?key=...&action=tap|down|up: key input dispatch (main-thread safe)
- /paint?x=..&y=..&radius=..&family=..&variant=..: generic paint helper
- /ignite?x=..&y=..&radius=..: fire paint helper for stress/debug
- /screenshot: PNG framebuffer capture (main-thread safe)
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
import zlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.game.app import GameApp

log = logging.getLogger(__name__)


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
            elif path.startswith("/spawn_projectile"):
                data = self._spawn_projectile()
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
            elif path.startswith("/benchmark"):
                data = self._get_benchmark()
            elif path.startswith("/teleport_surface"):
                data = self._teleport_surface()
            elif path.startswith("/nudge_surface"):
                data = self._nudge_surface()
            elif path.startswith("/experiment_camera"):
                data = self._experiment_camera()
            elif path.startswith("/experiment_camera_clear"):
                data = self._experiment_camera_clear()
            elif path.startswith("/evict_disk_chunks"):
                data = self._evict_disk_chunks()
            elif path.startswith("/teleport"):
                data = self._teleport()
            elif path.startswith("/explode"):
                data = self._explode()
            elif path.startswith("/paint"):
                data = self._paint()
            elif path.startswith("/ignite"):
                data = self._ignite()
            elif path.startswith("/spawn"):
                data = self._spawn_enemy()
            elif path.startswith("/heal"):
                data = self._heal_hero()
            elif path.startswith("/shutdown"):
                data = self._shutdown_game()
            else:
                data = {
                    "error": f"unknown endpoint: {path}",
                    "endpoints": [
                        "/status", "/hero", "/enemies", "/projectiles",
                        "/spawn_projectile?type=arrow|fireball",
                        "/fps",
                        "/snapshots",
                        "/cells?x=..&y=..&w=..&h=..",
                        "/gpu_cells?x=..&y=..&w=..&h=..",
                        "/gpu_pressure?x=..&y=..&w=..&h=..",
                        "/gpu_debug",
                        "/gpu_apply_pressure",
                        "/gpu_inject_probe?x=..&y=..&radius=..&pressure=..",
                        "/gpu_pressure_timeline?x=..&y=..&radius=..&frames=..",
                        "/benchmark",
                        "/press?key=ENTER&action=tap|down|up", "/screenshot",
                        "/teleport?x=..&y=..",
                        "/teleport_surface?x=..",
                        "/nudge_surface?dx=..",
                        "/experiment_camera?x=..&y=..&freeze=1",
                        "/experiment_camera_clear",
                        "/evict_disk_chunks?start_x=..&end_x=..&margin_x=..&margin_y=..",
                        "/explode?x=..&y=..",
                        "/paint?x=..&y=..&radius=..&family=stone&variant=stone_platform",
                        "/ignite?x=..&y=..&radius=..",
                        "/spawn?type=A|B|C",
                        "/heal?amount=50",
                        "/shutdown",
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
        world_status = None
        status_fn = getattr(app, "gpu_world_status_snapshot", None)
        if callable(status_fn):
            world_status = status_fn(block=True, timeout=1.0)
        paging = None
        if world_status is not None:
            paging_status = dict(world_status.get("paging") or {})
            chunk_status = dict(world_status.get("chunk") or {})
            paging = {
                "shift_ms": paging_status.get("shift_ms", 0.0),
                "incoming_load_ms": paging_status.get("incoming_load_ms", 0.0),
                "evict_stage_ms": paging_status.get("evict_stage_ms", 0.0),
                "overlap_ms": paging_status.get("overlap_ms", 0.0),
                "overlap_fx_ms": paging_status.get("overlap_fx_ms", 0.0),
                "clear_fx_ms": paging_status.get("clear_fx_ms", 0.0),
                "anchor_build_ms": paging_status.get("anchor_build_ms", 0.0),
                "anchor_upload_ms": paging_status.get("anchor_upload_ms", 0.0),
                "anchor_total_ms": paging_status.get("anchor_total_ms", 0.0),
                "prefetch_submit_ms": paging_status.get("prefetch_submit_ms", 0.0),
                "prefetch_service_ms": paging_status.get("prefetch_service_ms", 0.0),
                "residency_service_ms": paging_status.get("residency_service_ms", 0.0),
                "shift_count": paging_status.get("shift_count", 0),
                # Renamed for clarity (Phase 1): this counts GPU staged regions
                # awaiting CPU readback, NOT disk-save backlog.
                "gpu_writeback_queue_depth": paging_status.get("gpu_writeback_queue_depth", 0),
                "active_chunk_patch_queue_depth": paging_status.get("active_chunk_patch_queue_depth", 0),
                # Legacy alias retained so callers transitioning to the new
                # name keep working during the migration.
                "pending_writebacks": paging_status.get("pending_writebacks", 0),
                "incoming_ready_parts": paging_status.get("incoming_ready_parts", 0),
                "incoming_empty_parts": paging_status.get("incoming_empty_parts", 0),
                "incoming_pending_parts": paging_status.get("incoming_pending_parts", 0),
                "last_shift_cache_hits": paging_status.get("last_shift_cache_hits", 0),
                "last_shift_empty_hits": paging_status.get("last_shift_empty_hits", 0),
                "last_shift_inflight_wait_hits": paging_status.get("last_shift_inflight_wait_hits", 0),
                "last_shift_disk_loads": paging_status.get("last_shift_disk_loads", 0),
                "last_shift_generates": paging_status.get("last_shift_generates", 0),
                "last_shift_saves": paging_status.get("last_shift_saves", 0),
                "last_shift_disk_load_ms": paging_status.get("last_shift_disk_load_ms", 0.0),
                "last_shift_generate_ms": paging_status.get("last_shift_generate_ms", 0.0),
                "last_shift_save_ms": paging_status.get("last_shift_save_ms", 0.0),
                "chunk_cached": chunk_status.get("cached", 0),
                "chunk_clean_resident": chunk_status.get("clean_resident", 0),
                "chunk_dirty_resident": chunk_status.get("dirty_resident", 0),
                "chunk_prefetch_queued": chunk_status.get("prefetch_queued", 0),
                "chunk_prefetch_inflight": chunk_status.get("prefetch_inflight", 0),
                "chunk_queued_read": chunk_status.get("queued_read", 0),
                "chunk_queued_write": chunk_status.get("queued_write", 0),
                "chunk_queued_generate": chunk_status.get("queued_generate", 0),
                "chunk_inflight_io": chunk_status.get("inflight_io", 0),
                "chunk_inflight_generation": chunk_status.get("inflight_generation", 0),
                "chunk_worker_fallback": chunk_status.get("worker_fallback", 0),
                "chunk_sync_blocking_fetch": chunk_status.get("sync_blocking_fetch", 0),
                "chunk_disk_load_count": chunk_status.get("disk_load_count", 0),
                "chunk_generate_count": chunk_status.get("generate_count", 0),
                "chunk_save_count": chunk_status.get("save_count", 0),
                "chunk_disk_load_last_ms": chunk_status.get("disk_load_last_ms", 0.0),
                "chunk_generate_last_ms": chunk_status.get("generate_last_ms", 0.0),
                "chunk_save_last_ms": chunk_status.get("save_last_ms", 0.0),
            }
        gpu_world_mutations = getattr(app, "_gpu_world_mutations", None)
        gpu_world_mutation_pending = int(getattr(gpu_world_mutations, "pending_count", 0))
        owner_for_commands = getattr(app, "_gpu_owner_for_commands", None)
        gpu_owner = owner_for_commands() if callable(owner_for_commands) else None
        gpu_owner_stats = getattr(gpu_owner, "stats", None)
        # Snapshot freshness/age: surface per-entity tick_id and frame age so
        # the F3 overlay and external profiling can see snapshot delivery lag.
        snapshot_freshness: dict[str, dict[str, int]] = {}
        snapshot_entity_ids = set(app.entity_manager._latest_snapshots)
        snapshot_entity_ids.update(app._snapshot_registry.entity_ids())
        current_tick = int(((world_status or {}).get("gpu") or {}).get("step_index", 0) or 0)
        for entity_id in sorted(snapshot_entity_ids):
            snapshot = app.entity_manager.latest_snapshot_for(entity_id, current_tick=current_tick)
            if snapshot is None:
                continue
            snapshot_freshness[entity_id] = {
                "tick_id": int(snapshot.tick_id),
                "age_frames": int(snapshot.age_frames),
            }
        # GPU step timing
        gpu_timings: dict[str, Any] = dict((world_status or {}).get("gpu") or {})
        if gpu_timings:
            gpu_timings["tick_rate"] = round(float(getattr(app, "gpu_tick_rate", 0.0)), 3)
        return {
            "screen": app.current_screen,
            "game_console_visible": bool(
                getattr(getattr(app, "_game_screen", None), "show_console", False)
            ),
            "world_loaded": world_status is not None,
            "gpu_owner_active": gpu_owner is not None,
            "gpu_owner_created_world": bool(getattr(app, "_gpu_owner_created_world", False)),
            "gpu_owner_commands_dispatched": int(getattr(gpu_owner_stats, "commands_dispatched", 0)),
            "gpu_owner_commands_completed": int(getattr(gpu_owner_stats, "commands_completed", 0)),
            "gpu_owner_commands_failed": int(getattr(gpu_owner_stats, "commands_failed", 0)),
            "gpu_owner_queue_depth": int(getattr(gpu_owner, "queue_depth", 0)),
            "debug_overlay_enabled": bool(getattr(app, "debug_overlay_enabled", False)),
            "owner_present_overlay_line_count": int(getattr(app, "_last_owner_present_overlay_line_count", 0)),
            "owner_present_enemy_count": int(getattr(app, "_last_owner_present_enemy_count", 0)),
            "owner_present_projectile_count": int(getattr(app, "_last_owner_present_projectile_count", 0)),
            "hero_state": app.hero.state if app.hero else None,
            "camera": (world_status or {}).get("camera"),
            "active_origin": (world_status or {}).get("active_origin"),
            "enemy_count": len(app.enemies),
            "projectile_count": len(app.projectiles),
            "active_pressure_bursts": len(app.active_pressure_bursts),
            "gpu_world_mutation_pending": gpu_world_mutation_pending,
            "gpu_world_mutation_last_flush_count": int(getattr(app, "_last_gpu_world_mutation_flush_count", 0)),
            "local_snapshots": {
                entity_id: {
                    "origin": (snapshot.origin_x, snapshot.origin_y),
                    "size": (snapshot.width, snapshot.height),
                    "tick_id": snapshot.tick_id,
                    "age_frames": snapshot.age_frames,
                }
                for entity_id, snapshot in app.entity_manager._latest_snapshots.items()
            },
            "pending_local_snapshots": int(
                ((world_status or {}).get("snapshot") or {}).get("pending", 0)
            ),
            "queued_local_snapshot_requests": len(getattr(app, "_queued_local_snapshot_requests", []) or []),
            "snapshot_freshness": snapshot_freshness,
            "paging": paging,
            "gpu": gpu_timings,
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
            "input_left": h.input_left,
            "input_right": h.input_right,
            "input_jump": h.input_jump,
            "input_chant_held": h.input_chant_held,
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
        render_fps = getattr(app, "visible_render_fps", None)
        if callable(render_fps):
            render_fps = render_fps()
        return {
            "sim_fps": round(float(app.sim_fps), 3) if hasattr(app, "sim_fps") else None,
            "render_fps": round(float(render_fps), 3) if render_fps is not None else None,
        }

    def _run_on_gpu_owner_if_available(self, fn, *, timeout: float = 5.0, high_priority: bool = False):
        owner_for_commands = getattr(self.app, "_gpu_owner_for_commands", None)
        if not callable(owner_for_commands) or owner_for_commands() is None:
            return None
        run_on_gpu_world = getattr(self.app, "run_on_gpu_world", None)
        if not callable(run_on_gpu_world):
            return None
        return run_on_gpu_world(fn, timeout=timeout, high_priority=high_priority)

    def _get_snapshots(self) -> dict[str, Any]:
        snapshots = {}
        current_tick = int(getattr(self.app.gpu_world_mirror_snapshot(block=False), "gpu_step_index", 0) or 0)
        entity_ids = set(self.app.entity_manager._latest_snapshots)
        entity_ids.update(self.app._snapshot_registry.entity_ids())
        for entity_id in sorted(entity_ids):
            snapshot = self.app.entity_manager.latest_snapshot_for(entity_id, current_tick=current_tick)
            if snapshot is None:
                continue
            snapshots[entity_id] = {
                "origin": [snapshot.origin_x, snapshot.origin_y],
                "size": [snapshot.width, snapshot.height],
                "tick_id": snapshot.tick_id,
                "age_frames": snapshot.age_frames,
            }
        return {"count": len(snapshots), "snapshots": snapshots}

    # ── Cell inspection ────────────────────────────────────────────

    def _get_benchmark(self) -> dict[str, Any]:
        """Return current performance snapshot for benchmarking."""
        try:
            app = self.app
            perf = app.debug_perf_snapshot() if hasattr(app, "debug_perf_snapshot") else {}
            world_status = None
            status_fn = getattr(app, "gpu_world_status_snapshot", None)
            if callable(status_fn):
                world_status = status_fn(block=True, timeout=1.0)
            gpu_info: dict[str, Any] = dict((world_status or {}).get("gpu") or {})
            result = {
                "perf": perf,
                "gpu_info": gpu_info,
                "sim_fps": round(float(app.sim_fps), 3) if hasattr(app, "sim_fps") else None,
                "tick_rate": getattr(app, "_tick_call_rate", None),
                "sim_accumulator": getattr(app, "_sim_accumulator", None),
                "last_dt": getattr(app, "_last_dt", None),
                "tick_count": getattr(app, "_tick_call_count", None),
                "render_fps": (
                    round(float(app.visible_render_fps), 3)
                    if getattr(app, "visible_render_fps", None) is not None
                    else None
                ),
            }
            # Include recent tick history for debugging
            history = getattr(app, "_debug_tick_history", [])
            if history:
                result["tick_history_last_10"] = history[-10:]
            return result
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _get_cells(self) -> dict[str, Any]:
        return self._get_gpu_cells()

    def _get_gpu_cells(self) -> dict[str, Any]:
        """Read cells directly from GPU textures (bypasses CPU grid)."""
        app = self.app
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = int(v)
        x = params.get("x", int(app.hero.x))
        y = params.get("y", int(app.hero.y + app.hero.height))
        w = params.get("w", 5)
        h = params.get("h", 10)

        def _read_cells(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            ax = world.active_origin_x
            ay = world.active_origin_y
            lx = x - ax
            ly = y - ay
            cells_2d = world.gpu_simulator.readback_cells_region(lx, ly, w, h)
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

        owner_result = self._run_on_gpu_owner_if_available(_read_cells, high_priority=True)
        if owner_result is not None:
            return owner_result
        return _read_cells(app.world)

    def _get_gpu_pressure(self) -> dict[str, Any]:
        app = self.app
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

        def _read_pressure(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            pressure_rows = world.readback_pressure_region_world(x, y, w, h)
            samples = []
            for row_idx, row_values in enumerate(pressure_rows):
                for col_idx, value in enumerate(row_values):
                    samples.append({
                        "wx": x + col_idx,
                        "wy": y + row_idx,
                        "pressure": round(float(value), 3),
                    })
            return {"x": x, "y": y, "w": w, "h": h, "source": "gpu_pressure", "samples": samples}

        owner_result = self._run_on_gpu_owner_if_available(_read_pressure, high_priority=True)
        if owner_result is not None:
            return owner_result
        return _read_pressure(app.world)

    def _get_gpu_debug(self) -> dict[str, Any]:
        app = self.app

        def _read_debug(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            gpu = world.gpu_simulator
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

        owner_result = self._run_on_gpu_owner_if_available(_read_debug, high_priority=True)
        if owner_result is not None:
            return owner_result
        return _read_debug(app.world)

    def _gpu_apply_pressure(self) -> dict[str, Any]:
        import pyglet
        app = self.app

        def _apply_now(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            injections = world.gpu_simulator.debug_apply_pressure_injections_now()
            return {
                "applied": [
                    {
                        "center_x": int(cx),
                        "center_y": int(cy),
                        "inner_radius": int(inner_radius),
                        "outer_radius": int(outer_radius),
                        "pressure": float(pressure),
                    }
                    for cx, cy, inner_radius, outer_radius, pressure in injections
                ]
            }

        owner_result = self._run_on_gpu_owner_if_available(_apply_now, high_priority=True)
        if owner_result is not None:
            return owner_result

        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_apply(dt):
            try:
                result.update(_apply_now(app.world))
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_apply, 0)
        event.wait(timeout=5)
        return result

    def _gpu_inject_probe(self) -> dict[str, Any]:
        import pyglet
        app = self.app
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

        def _probe_now(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            world.inject_pressure_world(tx, ty, radius, pressure)
            injections = world.gpu_simulator.debug_apply_pressure_injections_now()
            pressure_rows = world.readback_pressure_region_world(tx - radius, ty - radius, radius * 2 + 1, radius * 2 + 1)
            flat = [float(value) for row in pressure_rows for value in row]
            return {
                "probe": {
                    "x": tx,
                    "y": ty,
                    "radius": radius,
                    "pressure": pressure,
                    "applied_count": len(injections),
                    "max_pressure": max(flat) if flat else 0.0,
                    "nonzero_samples": sum(1 for value in flat if value > 0.0),
                }
            }

        owner_result = self._run_on_gpu_owner_if_available(_probe_now, high_priority=True)
        if owner_result is not None:
            return owner_result

        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_probe(dt):
            try:
                result.update(_probe_now(app.world))
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_probe, 0)
        event.wait(timeout=5)
        return result

    def _gpu_pressure_timeline(self) -> dict[str, Any]:
        import pyglet
        app = self.app
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

        def _sample(world) -> dict[str, Any]:
            if world is None or world.gpu_simulator is None:
                return {"error": "no world or no GPU"}
            rows = world.readback_pressure_region_world(tx - radius, ty - radius, radius * 2 + 1, radius * 2 + 1)
            flat = [float(value) for row in rows for value in row]
            return {
                "step_index": int(world.gpu_simulator.step_index),
                "max_pressure": max(flat) if flat else 0.0,
                "nonzero_samples": sum(1 for value in flat if value > 0.0),
            }

        run_on_gpu_world = getattr(app, "run_on_gpu_world", None)
        owner_for_commands = getattr(app, "_gpu_owner_for_commands", None)
        owner = owner_for_commands() if callable(owner_for_commands) else None
        if owner is not None and callable(run_on_gpu_world):
            import time
            samples: list[dict[str, Any]] = []
            for _ in range(frames):
                sample = run_on_gpu_world(_sample, timeout=1.0, high_priority=True)
                if isinstance(sample, dict) and "error" in sample:
                    return sample
                samples.append(sample)
                time.sleep(0.05)
            return {
                "timeline": {
                    "x": tx,
                    "y": ty,
                    "radius": radius,
                    "frames": frames,
                    "samples": samples,
                }
            }

        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_timeline(dt):
            samples: list[dict[str, Any]] = []

            def _tick_sample(inner_dt):
                samples.append(_sample(app.world))
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
        hold_seconds = max(0.0, min(60.0, float(params.get("hold", "0.1"))))
        action = str(params.get("action", "tap")).strip().lower() or "tap"
        if action not in {"tap", "down", "up"}:
            action = "tap"
        key_map = {
            "ENTER": pyglet_key.ENTER,
            "SPACE": pyglet_key.SPACE,
            "W": pyglet_key.W, "A": pyglet_key.A,
            "S": pyglet_key.S, "D": pyglet_key.D,
            "C": pyglet_key.C,
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
        def _do_release(dt):
            app.on_key_release(symbol, 0)
            event.set()
        if action == "up":
            pyglet.clock.schedule_once(_do_release, 0)
            event.wait(timeout=5)
        else:
            pyglet.clock.schedule_once(_do_press, 0)
            event.wait(timeout=5)
            if action == "tap":
                pyglet.clock.schedule_once(_do_release, hold_seconds)
        return {
            "pressed": key_name,
            "action": action,
            "symbol": symbol,
            "screen": app.current_screen,
            "hold_seconds": hold_seconds,
            "keys_pressed_count": len(getattr(app, "_keys_pressed", set()) or ()),
        }

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

    def _experiment_camera(self) -> dict[str, Any]:
        """Drive paging experiments from explicit camera intent, not hero physics."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}
        mirror = app.gpu_world_mirror_snapshot(block=True)
        default_x = int(getattr(mirror, "camera_x", 0) if mirror is not None else 0)
        default_y = int(getattr(mirror, "camera_y", 0) if mirror is not None else 0)
        target_x = int(float(params.get("x", default_x)))
        target_y = int(float(params.get("y", default_y)))
        freeze = str(params.get("freeze", "1")).strip().lower() not in {"0", "false", "no", "off"}

        def _do_set(dt):
            try:
                app.set_experiment_camera_target(target_x, target_y, freeze_gameplay=freeze)
                result["experiment_camera"] = {
                    "x": target_x,
                    "y": target_y,
                    "freeze_gameplay": freeze,
                }
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_set, 0)
        event.wait(timeout=5)
        return result

    def _experiment_camera_clear(self) -> dict[str, Any]:
        """Clear experiment camera override and resume normal gameplay-driven follow."""
        import pyglet
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_clear(dt):
            try:
                app.clear_experiment_camera_target()
                result["experiment_camera_clear"] = True
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_clear, 0)
        event.wait(timeout=5)
        return result

    def _evict_disk_chunks(self) -> dict[str, Any]:
        """Evict clean disk-backed chunks along a camera path for profiling."""
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        app = self.app
        current_camera_x = 0
        current_camera_y = 0
        mirror_fn = getattr(app, "gpu_world_mirror_snapshot", None)
        mirror = mirror_fn(block=True) if callable(mirror_fn) else getattr(app, "world", None)
        if mirror is not None:
            current_camera_x = int(getattr(mirror, "camera_x", 0))
            current_camera_y = int(getattr(mirror, "camera_y", 0))
        start_x = int(params.get("start_x", current_camera_x))
        end_x = int(params.get("end_x", start_x))
        camera_y = int(params.get("camera_y", current_camera_y))
        margin_x = int(params.get("margin_x", 0))
        margin_y = int(params.get("margin_y", 0))

        def _evict(world):
            evict = getattr(world, "evict_clean_disk_backed_chunks_for_camera_path", None)
            if not callable(evict):
                return {"error": "world does not support disk-backed cache eviction"}
            return evict(
                start_camera_x=start_x,
                end_camera_x=end_x,
                camera_y=camera_y,
                margin_x=margin_x,
                margin_y=margin_y,
            )

        owner_result = self._run_on_gpu_owner_if_available(_evict, timeout=10.0)
        if owner_result is not None:
            return {
                "evict_disk_chunks": owner_result,
                "start_x": start_x,
                "end_x": end_x,
                "camera_y": camera_y,
            }
        world = getattr(app, "world", None)
        if world is None:
            return {"error": "world not loaded"}
        return {
            "evict_disk_chunks": _evict(world),
            "start_x": start_x,
            "end_x": end_x,
            "camera_y": camera_y,
        }

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
                    app._flush_gpu_world_mutations()
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

    def _ignite(self) -> dict[str, Any]:
        """Paint a fire patch at a world position for debug stress tests."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = float(v)
        app = self.app
        tx = int(params.get("x", app.hero.x))
        ty = int(params.get("y", app.hero.y + app.hero.height * 0.5))
        radius = max(1, int(params.get("radius", 4)))
        temperature = float(params.get("temperature", 650.0))
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_ignite(dt):
            try:
                if app.world is None:
                    result["error"] = "world not loaded"
                else:
                    app._queue_world_paint(
                        tx,
                        ty,
                        radius,
                        "fire",
                        "fire",
                        overrides={"temperature": temperature, "vel_y": -1.0},
                    )
                    app._flush_gpu_world_mutations()
                    result["ignite"] = {
                        "x": tx,
                        "y": ty,
                        "radius": radius,
                        "temperature": temperature,
                    }
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_ignite, 0)
        event.wait(timeout=5)
        return result

    def _paint(self) -> dict[str, Any]:
        """Paint a generic material patch at a world position for debug verification."""
        import pyglet
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        app = self.app
        tx = int(float(params.get("x", app.hero.x)))
        ty = int(float(params.get("y", app.hero.y + app.hero.height * 0.5)))
        radius = max(1, int(float(params.get("radius", "1"))))
        family = str(params.get("family", "stone")).strip() or "stone"
        variant = str(params.get("variant", "stone_platform")).strip() or "stone_platform"
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_paint(dt):
            try:
                if app.world is None:
                    result["error"] = "world not loaded"
                else:
                    app._queue_world_paint(tx, ty, radius, family, variant)
                    app._flush_gpu_world_mutations()
                    result["paint"] = {
                        "x": tx,
                        "y": ty,
                        "radius": radius,
                        "family": family,
                        "variant": variant,
                    }
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_do_paint, 0)
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

    def _spawn_projectile(self) -> dict[str, Any]:
        """Spawn a projectile near the hero for render/debug validation."""
        import pyglet
        from src.game.projectile import Arrow, Fireball

        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        projectile_type = params.get("type", "arrow").strip().lower()
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_spawn(dt):
            try:
                if app.world is None:
                    result["error"] = "world not loaded"
                    event.set()
                    return
                start_x = float(app.hero.x + 10.0)
                start_y = float(app.hero.y + app.hero.height * 0.5)
                target_x = float(start_x + 48.0)
                target_y = float(start_y - 6.0)
                if projectile_type == "fireball":
                    projectile = Fireball.create(start_x, start_y, target_x, target_y)
                else:
                    projectile = Arrow.create(start_x, start_y, target_x, target_y, facing_right=True)
                app.projectiles.append(projectile)
                result["spawned"] = {
                    "type": projectile_type,
                    "x": projectile.x,
                    "y": projectile.y,
                    "vel_x": projectile.vel_x,
                    "vel_y": projectile.vel_y,
                }
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

    def _shutdown_game(self) -> dict[str, Any]:
        """Flush world state on the main thread, then ask pyglet to exit."""
        import pyglet
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _do_shutdown(dt):
            try:
                app._close_game_world()
                result["closed"] = True
                result["gpu_owner_active"] = bool(getattr(app, "gpu_owner_active", False))
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                event.set()
                pyglet.clock.schedule_once(lambda _dt: pyglet.app.exit(), 0)

        pyglet.clock.schedule_once(_do_shutdown, 0)
        if not event.wait(timeout=60):
            return {"error": "shutdown timed out"}
        return result

    # ── Screenshot ─────────────────────────────────────────────────

    def _send_screenshot(self) -> None:
        """Capture the latest display RGB as PNG and send it."""
        import ctypes
        import ctypes.wintypes
        import pyglet
        app = self.app
        event = threading.Event()
        result: dict[str, Any] = {}

        def _build_png() -> None:
            w, h = app.width, app.height
            raw = app.read_framebuffer_rgb()
            # GL-compatible RGB bytes are bottom-to-top; flip vertically for PNG.
            row_stride = w * 3
            rgb_rows = bytearray(row_stride * h)
            dst = 0
            for y in range(h - 1, -1, -1):
                offset = y * row_stride
                rgb_rows[dst:dst + row_stride] = raw[offset:offset + row_stride]
                dst += row_stride
            png_bytes = _rgb_to_png(w, h, bytes(rgb_rows))
            result["width"] = w
            result["height"] = h
            result["png"] = png_bytes

        owner_active = bool(getattr(app, "gpu_owner_active", False))
        log.debug("[debug] screenshot request owner_active=%s", owner_active)
        if owner_active:
            try:
                started_at = time.perf_counter()
                w, h, raw = app.owner_frame_view_rgb_snapshot(timeout=0.5)
                log.debug("[debug] owner screenshot readback %.3fms", (time.perf_counter() - started_at) * 1000.0)
                row_stride = w * 3
                rgb_rows = bytearray(row_stride * h)
                dst = 0
                for y in range(h - 1, -1, -1):
                    offset = y * row_stride
                    rgb_rows[dst:dst + row_stride] = raw[offset:offset + row_stride]
                    dst += row_stride
                result["width"] = w
                result["height"] = h
                result["png"] = _rgb_to_png(w, h, bytes(rgb_rows))
                log.debug("[debug] owner screenshot png %.3fms", (time.perf_counter() - started_at) * 1000.0)
            except Exception as exc:
                log.debug("[debug] owner screenshot not ready: %s", exc)
                result["error"] = str(exc)
            return self._write_screenshot_result(result)

        def _capture(dt):
            try:
                _build_png()
            except Exception as exc:
                result["error"] = str(exc)
            event.set()

        pyglet.clock.schedule_once(_capture, 0)
        if not event.wait(timeout=10):
            result["error"] = "screenshot capture timed out"
        return self._write_screenshot_result(result)

    def _write_screenshot_result(self, result: dict[str, Any]) -> None:
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

    # Filter each row with filter type 0 (None).
    row_bytes = width * 3
    raw_rows = bytearray((row_bytes + 1) * height)
    dst = 0
    for y in range(height):
        offset = y * row_bytes
        raw_rows[dst] = 0
        raw_rows[dst + 1:dst + 1 + row_bytes] = rgb[offset : offset + row_bytes]
        dst += row_bytes + 1

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    compressed = zlib.compress(bytes(raw_rows), 6)
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
