"""Game App: state machine (TITLE/GAME/OPTIONS/CONSOLE)."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time as _time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pyglet

log = logging.getLogger(__name__)
from pyglet.window import key

from src.engine.gpu_owner import (
    CameraFollowCommand,
    CameraTeleportCommand,
    ChunkPrefetchCommand,
    CreateWorldCommand,
    EntityGpuSyncCommand,
    GpuFramePayload,
    GpuCommand,
    GpuCommandType,
    GpuOwnedWorldRuntime,
    GpuOptionsCommand,
    GpuOwnerThread,
    GpuWorldCommandHandlers,
    GpuWorldMutationBuffer,
    RenderActorState,
    RenderFrameCommand,
    SnapshotRequestCommand,
    SnapshotServiceCommand,
    WorldCallableCommand,
    WorldStatusCommand,
    build_world_status,
)
from src.engine.materials import build_material_registry
from src.engine.render import DebugViewMode
from src.engine.snapshot_collision import build_probe_for_envelope
from src.engine.snapshot_mailbox import SnapshotMailboxRegistry
from src.engine.world import ActiveWorldWindow, WorldChunkStore, WorldRect
from src.game import config as cfg
from src.game.enemy import EnemyA, EnemyB, EnemyC
from src.game.entity_manager import Entity, EntityManager
from src.game.hero import Hero
from src.game.projectile import Arrow, Fireball, ProjectileBase
from src.game.owner_frame_renderer import OwnerFrameRenderer
from src.game.screens import GameScreen, OptionsScreen, TitleScreen
from src.game.spell_system import (
    SPELL_CATALOG, ActiveStream, expand_model_socket,
    execute_magic_socket, inject_stream_tick,
)
from src.game.stt import SpeechToText
from src.game.terrain import TerrainGenerator


@dataclass
class PressureBurst:
    x: int
    y: int
    radius: int
    pressure: float
    ticks_remaining: int


@dataclass(frozen=True)
class GpuWorldMirror:
    camera_x: int
    camera_y: int
    active_origin_x: int
    active_origin_y: int
    active_width: int
    active_height: int
    viewport_width: int
    viewport_height: int
    world_width: int
    world_height: int
    gpu_step_index: int = 0
    gpu_simulator: object | None = None

    @classmethod
    def from_status(cls, status: dict[str, Any]) -> "GpuWorldMirror":
        camera_x, camera_y = status.get("camera", (0, 0))
        active_origin_x, active_origin_y = status.get("active_origin", (0, 0))
        active_width, active_height = status.get("active_size", (cfg.VIEWPORT_WIDTH, cfg.VIEWPORT_HEIGHT))
        viewport_width, viewport_height = status.get("viewport_size", (cfg.VIEWPORT_WIDTH, cfg.VIEWPORT_HEIGHT))
        world_width, world_height = status.get("world_size", (cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT))
        gpu_status = dict(status.get("gpu") or {})
        return cls(
            camera_x=int(camera_x),
            camera_y=int(camera_y),
            active_origin_x=int(active_origin_x),
            active_origin_y=int(active_origin_y),
            active_width=int(active_width),
            active_height=int(active_height),
            viewport_width=int(viewport_width),
            viewport_height=int(viewport_height),
            world_width=int(world_width),
            world_height=int(world_height),
            gpu_step_index=int(gpu_status.get("step_index", 0) or 0),
        )

    @classmethod
    def from_world(cls, world: Any) -> "GpuWorldMirror":
        gpu = getattr(world, "gpu_simulator", None)
        return cls(
            camera_x=int(getattr(world, "camera_x", 0)),
            camera_y=int(getattr(world, "camera_y", 0)),
            active_origin_x=int(getattr(world, "active_origin_x", 0)),
            active_origin_y=int(getattr(world, "active_origin_y", 0)),
            active_width=int(getattr(world, "active_width", cfg.VIEWPORT_WIDTH)),
            active_height=int(getattr(world, "active_height", cfg.VIEWPORT_HEIGHT)),
            viewport_width=int(getattr(world, "viewport_width", cfg.VIEWPORT_WIDTH)),
            viewport_height=int(getattr(world, "viewport_height", cfg.VIEWPORT_HEIGHT)),
            world_width=int(getattr(world, "world_width", cfg.WORLD_WIDTH)),
            world_height=int(getattr(world, "world_height", cfg.WORLD_HEIGHT)),
            gpu_step_index=int(getattr(gpu, "step_index", 0) or 0),
            gpu_simulator=gpu,
        )


def _fit_cell_scale_to_screen(
    requested_scale: int,
    *,
    screen_width: int,
    screen_height: int,
) -> int:
    requested = max(1, int(requested_scale))
    fitting_scales = [
        int(scale)
        for scale in cfg.CELL_SCALE_OPTIONS
        if (
            int(scale) <= requested
            and cfg.VIEWPORT_WIDTH * int(scale) <= int(screen_width)
            and cfg.VIEWPORT_HEIGHT * int(scale) <= int(screen_height)
        )
    ]
    if fitting_scales:
        return max(fitting_scales)
    fallback = min(requested, max(1, min(int(screen_width) // cfg.VIEWPORT_WIDTH, int(screen_height) // cfg.VIEWPORT_HEIGHT)))
    return max(1, int(fallback))


def _best_launch_cell_scale(requested_scale: int) -> int:
    requested = max(1, int(requested_scale))
    try:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
    except Exception:
        return requested
    return _fit_cell_scale_to_screen(
        requested,
        screen_width=int(screen.width),
        screen_height=int(screen.height),
    )


class GameApp(pyglet.window.Window):
    """Main game application with screen state machine."""

    def __init__(self, seed: int = 42, cell_scale: int = cfg.CELL_SCALE) -> None:
        requested_cell_scale = max(1, int(cell_scale))
        fitted_cell_scale = _best_launch_cell_scale(requested_cell_scale)
        self._cell_scale = fitted_cell_scale
        cfg.CELL_SCALE = fitted_cell_scale
        window_width = cfg.VIEWPORT_WIDTH * fitted_cell_scale
        window_height = cfg.VIEWPORT_HEIGHT * fitted_cell_scale
        super().__init__(
            width=window_width,
            height=window_height,
            caption="Oracle Translator",
            resizable=False,
            vsync=False,
        )
        try:
            self.activate()
        except Exception:
            pass
        if fitted_cell_scale != requested_cell_scale:
            log.warning(
                "[app] requested cell_scale=%d does not fit the current display, using cell_scale=%d",
                requested_cell_scale,
                fitted_cell_scale,
            )
        self.seed = seed
        self.registry = build_material_registry()
        self._snapshot_registry = SnapshotMailboxRegistry()
        self.world: ActiveWorldWindow | GpuWorldMirror | None = None
        self.terrain_gen: TerrainGenerator | None = None
        self.hero = Hero()
        self.entity_manager = EntityManager(hero=self.hero, _snapshot_registry=self._snapshot_registry)
        self._gpu_owner_thread = None
        self._gpu_owner_created_world = False
        self._gpu_gl_lock = threading.RLock()
        self._gpu_world_mutations = GpuWorldMutationBuffer()
        self._last_gpu_world_mutation_flush_count = 0
        self._latest_world_snapshots: dict[str, object] = {}
        self._queued_local_snapshot_requests: deque[SnapshotRequestCommand] = deque()
        self._pending_gpu_snapshot_service_handle = None
        self._pending_gpu_world_status_handle = None
        self._latest_gpu_world_status: dict[str, Any] | None = None
        self._pending_gpu_frame_handle = None
        self._latest_gpu_frame_payload: GpuFramePayload | None = None
        self._pending_gpu_present_handle = None
        self._pending_gpu_world_step_handle = None
        self._pending_gpu_camera_follow_handle = None
        self._pending_gpu_background_io_handle = None
        self._pending_gpu_screenshot_handle = None
        self._last_owner_present_overlay_line_count = 0
        self._last_owner_present_enemy_count = 0
        self._last_owner_present_projectile_count = 0
        self._owner_input_events: queue.SimpleQueue[tuple[str, int, int]] = queue.SimpleQueue()
        self._experiment_camera_override: tuple[int, int] | None = None
        self._experiment_freeze_gameplay: bool = False
        self.view_mode = DebugViewMode.MATERIAL
        self._camera_target_x = 0
        self._camera_target_y = 0
        self.active_streams: list[ActiveStream] = []
        self.active_pressure_bursts: list[PressureBurst] = []
        self.enemies: dict[str, EnemyA | EnemyB | EnemyC] = {}
        self.projectiles: list[ProjectileBase] = []

        # STT (speech-to-text)
        self.stt = SpeechToText()
        self._chant_started = False

        # Display renderer. The default owner-created world path must not
        # create a main-thread ModernGL context; it presents owner-rendered
        # RGBA frames through pyglet. Legacy fallback creates GameRenderer
        # lazily only when explicitly requested or when owner creation fails.
        self.ctx = None
        self.renderer = self._create_display_renderer(window_width, window_height)

        # Screens
        self.screens: dict[str, pyglet.window.Window] = {}
        self.current_screen: str | None = None
        self._title_screen = TitleScreen(self)
        self._options_screen = OptionsScreen(self)
        self._game_screen = GameScreen(self)
        self.screens = {
            "title": self._title_screen,
            "options": self._options_screen,
            "game": self._game_screen,
        }
        self.change_screen("title")

        # Input
        self._keys_pressed: set[int] = set()
        self._last_dt = 1.0 / 60.0
        self._sim_accumulator = 0.0
        self._sim_fps = 0.0
        self._sim_fps_count = 0
        self._sim_fps_started_at = perf_counter()
        self._sim_step_timestamps: deque[float] = deque(maxlen=240)
        self._gpu_tick_timestamps: deque[float] = deque(maxlen=240)
        self._debug_overlay_enabled = False
        self._perf_samples: dict[str, deque[float]] = {}
        self._perf_last_ms: dict[str, float] = {}

        # Pump the event loop at 120Hz so rendering/input stay responsive
        # even though simulation itself remains fixed at 60Hz.
        pyglet.clock.schedule_interval(self._tick, self._TICK_PUMP_DT)

        # Debug HTTP server
        from src.game.debug_server import DebugServer
        self._debug = DebugServer(self, port=9123)
        self._debug.start()

    def _snapshot_request_queue(self) -> deque[SnapshotRequestCommand]:
        queue = getattr(self, "_queued_local_snapshot_requests", None)
        if queue is None:
            queue = deque()
            self._queued_local_snapshot_requests = queue
        return queue

    def _drop_queued_snapshot_requests(self, entity_id: str) -> None:
        queue = self._snapshot_request_queue()
        if not queue:
            return
        self._queued_local_snapshot_requests = deque(
            request for request in queue if request.entity_id != entity_id
        )

    def _owner_result_if_ready(self, handle: object) -> object | None:
        result_if_ready = getattr(handle, "result_if_ready", None)
        if callable(result_if_ready):
            return result_if_ready()
        done = getattr(handle, "done", None)
        wait = getattr(handle, "wait", None)
        if not callable(done) or not callable(wait):
            return None
        if not done():
            return None
        return wait(timeout=0.0)

    def _submit_gpu_snapshot_requests(self) -> int:
        if self.world is None:
            return 0
        queue = self._snapshot_request_queue()
        owner = self._gpu_owner_for_commands()
        submit = None if owner is None else getattr(owner, "submit", None)
        if not callable(submit):
            raise RuntimeError("GPU owner is required for snapshot submission")
        handle = getattr(self, "_pending_gpu_snapshot_service_handle", None)
        if handle is not None:
            result = self._owner_result_if_ready(handle)
            if result is None:
                return 0
            self._pending_gpu_snapshot_service_handle = None
            if not bool(getattr(result, "ok", False)):
                if getattr(result, "error", ""):
                    log.warning("[app] GPU owner snapshot service failed: %s", result.error)
                return 0
            value = getattr(result, "value", None)
            if isinstance(value, dict):
                self._set_perf_ms("snapshot_ready_to_consume_delay", float(value.get("ready_delay_ms_max", 0.0)))
        if not queue:
            return 0
        requests: list[SnapshotRequestCommand] = []
        seen_entity_ids: set[str] = set()
        while queue:
            request = queue.popleft()
            if request.entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(request.entity_id)
            requests.append(request)
        if not requests:
            return 0
        submit_started_at = _time.perf_counter()
        self._pending_gpu_snapshot_service_handle = submit(
            GpuCommand(
                op=GpuCommandType.SERVICE_SNAPSHOTS,
                payload=SnapshotServiceCommand(requests=tuple(requests)),
            )
        )
        self._set_perf_ms("snapshot_submit", (_time.perf_counter() - submit_started_at) * 1000.0)
        return len(requests)

    def _consume_ready_local_snapshots(self) -> None:
        self._submit_gpu_snapshot_requests()

    def _queue_local_snapshot(self, entity_id: str, min_x: int, min_y: int, max_x: int, max_y: int) -> None:
        if self.world is None:
            return
        queue = self._snapshot_request_queue()
        if any(request.entity_id == entity_id for request in queue):
            return
        width = max(1, int(max_x) - int(min_x))
        height = max(1, int(max_y) - int(min_y))
        queue.append(
            SnapshotRequestCommand(
                entity_id=entity_id,
                world_x=int(min_x),
                world_y=int(min_y),
                width=width,
                height=height,
                submitted_at=_time.perf_counter(),
            )
        )

    @staticmethod
    def _actor_snapshot_bounds(x: float, y: float, width: float, height: float, *, margin: int = 3) -> tuple[int, int, int, int]:
        left = int(x - width / 2.0) - margin
        right = int(x + width / 2.0) + margin + 1
        bottom = int(y) - margin
        top = int(y + height) + margin + 2
        return left, bottom, right, top

    def _refresh_local_entity_snapshots(self, *, force: bool = False) -> None:
        if self.world is None:
            return
        self._consume_ready_local_snapshots()
        if not force:
            self._snapshot_refresh_counter = getattr(self, "_snapshot_refresh_counter", 0) + 1
            if self._snapshot_refresh_counter % 4 != 0:
                return

        hero_bounds = self._actor_snapshot_bounds(
            self.hero.x,
            self.hero.y,
            cfg.HERO_WIDTH,
            cfg.HERO_HEIGHT,
        )
        self._queue_local_snapshot("hero", *hero_bounds)

        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive or not self._enemy_is_near_active_world(enemy):
                continue
            bounds = self._actor_snapshot_bounds(enemy.x, enemy.y, enemy.width, enemy.height)
            self._queue_local_snapshot(eid, *bounds)

        for projectile in self.projectiles:
            if not projectile.is_alive:
                continue
            if not self._projectile_is_in_world_bounds(projectile):
                continue
            if not self._projectile_is_near_active_world(projectile, margin=24):
                continue
            entity_id = f"projectile_{id(projectile)}"
            lead_x = int(projectile.x + projectile.vel_x * 0.15)
            lead_y = int(projectile.y + projectile.vel_y * 0.15)
            min_x = min(int(projectile.x), lead_x) - 2
            max_x = max(int(projectile.x), lead_x) + 3
            min_y = min(int(projectile.y), lead_y) - 2
            max_y = max(int(projectile.y), lead_y) + 3
            self._queue_local_snapshot(entity_id, min_x, min_y, max_x, max_y)
        self._submit_gpu_snapshot_requests()

    def _prune_projectile_snapshots(self) -> None:
        active_ids = {f"projectile_{id(projectile)}" for projectile in self.projectiles if projectile.is_alive}
        snapshot_ids = set(self.entity_manager._latest_snapshots)
        snapshot_ids.update(
            entity_id
            for entity_id in self._snapshot_registry.entity_ids()
            if entity_id.startswith("projectile_")
        )
        stale_ids = [
            entity_id
            for entity_id in snapshot_ids
            if entity_id.startswith("projectile_") and entity_id not in active_ids
        ]
        for entity_id in stale_ids:
            self.entity_manager._latest_snapshots.pop(entity_id, None)
            self.entity_manager._last_feedback.pop(entity_id, None)
            self._snapshot_registry.drop(entity_id)
            self._drop_queued_snapshot_requests(entity_id)

    def _rect_overlaps_active_world(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        *,
        margin: int = 0,
    ) -> bool:
        mirror = self.gpu_world_mirror_snapshot(block=False)
        if mirror is None:
            return False
        active_left = mirror.active_origin_x - int(margin)
        active_top = mirror.active_origin_y - int(margin)
        active_right = mirror.active_origin_x + mirror.active_width + int(margin)
        active_bottom = mirror.active_origin_y + mirror.active_height + int(margin)
        return (
            right >= active_left
            and left <= active_right
            and top >= active_top
            and bottom <= active_bottom
        )

    def _enemy_is_near_active_world(self, enemy: EnemyA | EnemyB | EnemyC) -> bool:
        return self._rect_overlaps_active_world(enemy.left, enemy.bottom, enemy.right, enemy.top)

    def _projectile_is_near_active_world(self, projectile: ProjectileBase, *, margin: int = 0) -> bool:
        return self._rect_overlaps_active_world(
            projectile.x,
            projectile.y,
            projectile.x,
            projectile.y,
            margin=margin,
        )

    def _projectile_is_in_world_bounds(self, projectile: ProjectileBase, *, margin: int = 32) -> bool:
        return (
            -margin <= projectile.x < cfg.WORLD_WIDTH + margin
            and -margin <= projectile.y < cfg.WORLD_HEIGHT + margin
        )

    def _submit_gpu_chunk_prefetch(
        self,
        rect: WorldRect,
        *,
        margin_x: int,
        margin_y: int,
        prioritize: bool,
    ) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            owner.fire_and_forget(
                GpuCommand(
                    op=GpuCommandType.SCHEDULE_PREFETCH,
                    payload=ChunkPrefetchCommand(
                        rect=rect,
                        margin_x=int(margin_x),
                        margin_y=int(margin_y),
                        prioritize=bool(prioritize),
                    ),
                )
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for chunk prefetch")

    def _gpu_owner_for_commands(self):
        owner = getattr(self, "_gpu_owner_thread", None)
        if owner is None:
            return None
        if not bool(getattr(owner, "is_running", True)):
            return None
        if not hasattr(owner, "fire_and_forget"):
            return None
        return owner

    @property
    def gpu_owner_active(self) -> bool:
        return self._gpu_owner_for_commands() is not None

    def _uses_gpu_owner_world(self) -> bool:
        return self._gpu_owner_for_commands() is not None

    def _create_display_renderer(self, window_width: int, window_height: int):
        self._owner_created_world_requested()
        return OwnerFrameRenderer(window_width, window_height)

    def _create_legacy_renderer(self, window_width: int, window_height: int):
        del window_width, window_height
        raise RuntimeError("legacy main-thread ModernGL renderer has been removed")

    def _ensure_legacy_display_renderer(self) -> None:
        raise RuntimeError("legacy main-thread ModernGL renderer has been removed")

    def _cache_gpu_frame_payload(self, payload: GpuFramePayload) -> GpuFramePayload:
        self._latest_gpu_frame_payload = payload
        status = dict(payload.status or {})
        status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
        self._latest_gpu_world_status = status
        return payload

    def gpu_frame_payload_snapshot(self) -> GpuFramePayload | None:
        owner = self._gpu_owner_for_commands()
        if owner is None:
            self._pending_gpu_frame_handle = None
            return None
        handle = getattr(self, "_pending_gpu_frame_handle", None)
        if handle is not None:
            result = self._owner_result_if_ready(handle)
            if result is not None:
                self._pending_gpu_frame_handle = None
                if bool(getattr(result, "ok", False)) and isinstance(getattr(result, "value", None), GpuFramePayload):
                    self._cache_gpu_frame_payload(result.value)
                elif getattr(result, "error", ""):
                    log.warning("[app] GPU owner render frame failed: %s", result.error)
        if self._pending_gpu_frame_handle is None:
            submit = getattr(owner, "submit", None)
            if callable(submit):
                self._pending_gpu_frame_handle = submit(
                    GpuCommand(
                        op=GpuCommandType.RENDER_FRAME,
                        payload=RenderFrameCommand(
                            view_mode=self.view_mode,
                            readback_rgba=True,
                        ),
                    )
                )
        return getattr(self, "_latest_gpu_frame_payload", None)

    def _render_actor_snapshot(self, entity_id: str, kind: str, actor: object) -> RenderActorState:
        return RenderActorState(
            entity_id=entity_id,
            kind=kind,
            x=float(getattr(actor, "x", 0.0)),
            y=float(getattr(actor, "y", 0.0)),
            width=float(getattr(actor, "width", 1.0)),
            height=float(getattr(actor, "height", 1.0)),
            facing_right=bool(getattr(actor, "facing_right", True)),
            state=str(getattr(actor, "state", "idle")),
            hp=float(getattr(actor, "hp", getattr(self.hero, "hp", 1.0))),
            max_hp=float(getattr(actor, "max_hp", getattr(self.hero, "max_hp", cfg.HERO_MAX_HP))),
            mp=float(getattr(actor, "mp", getattr(self.hero, "mp", 0.0))),
            is_alive=bool(getattr(actor, "is_alive", True)),
            damage_flash_timer=float(getattr(actor, "damage_flash_timer", 0.0)),
        )

    def _owner_present_overlay_lines(self) -> tuple[str, ...]:
        if not self.debug_overlay_enabled:
            return ()
        perf = self.debug_perf_snapshot()
        status = dict(getattr(self, "_latest_gpu_world_status", None) or {})
        chunk = dict(status.get("chunk") or {})
        paging = dict(status.get("paging") or {})
        gpu = dict(status.get("gpu") or {})
        current_tick = int(gpu.get("step_index", 0) or 0)
        freshness = self._snapshot_registry.freshness_report(current_tick=current_tick)
        active_snapshot_ages = [
            age
            for entity_id, age in freshness.items()
            if age >= 0
            and (
                entity_id == "hero"
                or entity_id in self.enemies
                or entity_id.startswith("projectile_")
            )
        ]
        snapshot_age_min = min(active_snapshot_ages) if active_snapshot_ages else -1
        snapshot_age_max = max(active_snapshot_ages) if active_snapshot_ages else -1
        return (
            (
                "Main "
                f"tick {perf.get('main_tick_total_last_ms', 0.0):.1f}/{perf.get('main_tick_total_avg_ms', 0.0):.1f} ms "
                f"sim {perf.get('main_tick_sim_last_ms', 0.0):.1f}/{perf.get('main_tick_sim_avg_ms', 0.0):.1f} "
                f"cam {perf.get('main_tick_camera_last_ms', 0.0):.1f}/{perf.get('main_tick_camera_avg_ms', 0.0):.1f}"
            ),
            (
                "CPU submit "
                f"step {perf.get('world_step_submit_cpu_last_ms', 0.0):.1f}/{perf.get('world_step_submit_cpu_avg_ms', 0.0):.1f} ms "
                f"snap {perf.get('snapshot_submit_last_ms', 0.0):.1f}/{perf.get('snapshot_ready_to_consume_delay_last_ms', 0.0):.1f} "
                f"q/p {int(perf.get('snapshot_queued_request_count', 0.0))}/{int(perf.get('snapshot_pending_token_count', 0.0))}"
            ),
            (
                "Snapshot "
                f"age {snapshot_age_min}/{snapshot_age_max} f "
                f"tick {current_tick} "
                f"hero {int(freshness.get('hero', -1))}"
            ),
            (
                "Actors "
                f"ai {perf.get('update_game_enemy_ai_last_ms', 0.0):.1f}/{perf.get('update_game_enemy_ai_avg_ms', 0.0):.1f} "
                f"proj {perf.get('update_game_projectiles_last_ms', 0.0):.1f}/{perf.get('update_game_projectiles_avg_ms', 0.0):.1f} "
                f"mut {perf.get('gpu_world_mutation_flush_last_ms', 0.0):.1f}/{perf.get('gpu_world_mutation_flush_avg_ms', 0.0):.1f}"
            ),
            (
                "Paging "
                f"shift {float(paging.get('shift_ms', 0.0)):.1f} ms "
                f"load {int(paging.get('last_shift_disk_loads', 0))}/{float(paging.get('last_shift_disk_load_ms', 0.0)):.1f} "
                f"gen {int(paging.get('last_shift_generates', 0))}/{float(paging.get('last_shift_generate_ms', 0.0)):.1f} "
                f"save {int(paging.get('last_shift_saves', 0))}/{float(paging.get('last_shift_save_ms', 0.0)):.1f}"
            ),
            (
                "Chunks "
                f"resident {int(chunk.get('cached', 0))} clean {int(chunk.get('clean_resident', 0))} dirty {int(chunk.get('dirty_resident', 0))} "
                f"queued r/w/g {int(chunk.get('queued_read', 0))}/{int(chunk.get('queued_write', 0))}/{int(chunk.get('queued_generate', 0))} "
                f"wb {int(paging.get('gpu_writeback_queue_depth', paging.get('pending_writebacks', 0)))} "
                f"fallback {int(chunk.get('worker_fallback', 0))}"
            ),
        )

    def _owner_present_console_lines(self) -> tuple[str, ...]:
        screens = getattr(self, "screens", None)
        if not isinstance(screens, dict):
            return ()
        game_screen = screens.get("game")
        if game_screen is None or not bool(getattr(game_screen, "show_console", False)):
            return ()
        console = getattr(game_screen, "console", None)
        items = tuple(getattr(console, "_items", ()) or ())
        selected = int(getattr(console, "selected", 0) or 0)
        spell_count = len(SPELL_CATALOG)
        lines = ["[Console] UP/DOWN ENTER ESC", "Spells:"]
        for index, item in enumerate(items):
            label = str(item[0]) if item else ""
            if not label:
                if index <= spell_count:
                    if lines and lines[-1] != "":
                        lines.append("")
                    if "Debug:" not in lines:
                        lines.append("Debug:")
                continue
            prefix = "> " if index == selected else "  "
            lines.append(f"{prefix}{label}")
        if "Debug:" not in lines and len(items) > spell_count:
            lines.append("")
            lines.append("Debug:")
        return tuple(lines)

    def _submit_gpu_present_frame(self) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is None or self.current_screen != "game":
            return
        handle = getattr(self, "_pending_gpu_present_handle", None)
        if handle is not None:
            result = self._owner_result_if_ready(handle)
            if result is None:
                return
            self._pending_gpu_present_handle = None
            if bool(getattr(result, "ok", False)) and isinstance(getattr(result, "value", None), dict):
                status = dict(result.value.get("status") or {})
                status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
                self._latest_gpu_world_status = status
                self._last_owner_present_overlay_line_count = int(result.value.get("overlay_line_count", 0) or 0)
                self._last_owner_present_enemy_count = int(result.value.get("enemy_overlay_count", 0) or 0)
                self._last_owner_present_projectile_count = int(result.value.get("projectile_overlay_count", 0) or 0)
            elif getattr(result, "error", ""):
                log.warning("[app] GPU owner present failed: %s", result.error)
        enemies = tuple(
            self._render_actor_snapshot(eid, enemy.__class__.__name__, enemy)
            for eid, enemy in self.enemies.items()
            if getattr(enemy, "is_alive", False)
        )
        projectiles = tuple(
            self._render_actor_snapshot(f"projectile_{index}", projectile.__class__.__name__, projectile)
            for index, projectile in enumerate(self.projectiles)
            if getattr(projectile, "is_alive", False)
        )
        collision_debug = (
            self.entity_manager.last_debug
            if self.entity_manager.debug_collision and self.entity_manager.last_debug is not None
            else None
        )
        overlay_lines = self._owner_present_overlay_lines()
        self._last_owner_present_overlay_line_count = 1 + len(overlay_lines)
        self._last_owner_present_enemy_count = len(enemies)
        self._last_owner_present_projectile_count = len(projectiles)
        self._pending_gpu_present_handle = owner.submit(
            GpuCommand(
                op=GpuCommandType.RENDER_FRAME,
                payload=RenderFrameCommand(
                    view_mode=self.view_mode,
                    readback_rgba=False,
                    present=True,
                    window_size=(int(self.width), int(self.height)),
                    hero=self._render_actor_snapshot("hero", "Hero", self.hero),
                    enemies=enemies,
                    projectiles=projectiles,
                    dt=float(getattr(self, "_last_dt", 1.0 / 60.0)),
                    sim_fps=float(getattr(self, "sim_fps", 0.0)),
                    gpu_tick_rate=float(getattr(self, "gpu_tick_rate", 0.0)),
                    overlay_lines=overlay_lines,
                    collision_debug=collision_debug,
                    console_lines=self._owner_present_console_lines(),
                ),
            )
        )

    def gpu_world_status_snapshot(
        self,
        *,
        block: bool = False,
        timeout: float = 0.25,
    ) -> dict[str, Any] | None:
        owner = self._gpu_owner_for_commands()
        if owner is None:
            if self.world is None:
                return None
            status = build_world_status(self.world)
            status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
            self._latest_gpu_world_status = status
            return status
        handle = getattr(self, "_pending_gpu_world_status_handle", None)
        if handle is None:
            submit = getattr(owner, "submit", None)
            if callable(submit):
                handle = submit(
                    GpuCommand(
                        op=GpuCommandType.WORLD_STATUS,
                        payload=WorldStatusCommand(include_gpu_debug=True),
                    )
                )
                self._pending_gpu_world_status_handle = handle
        if handle is not None:
            result = handle.wait(timeout=timeout) if block else self._owner_result_if_ready(handle)
            if result is not None:
                if bool(getattr(result, "ok", False)) and isinstance(getattr(result, "value", None), dict):
                    self._pending_gpu_world_status_handle = None
                    status = dict(result.value)
                    status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
                    self._latest_gpu_world_status = status
                    return status
                if getattr(result, "error", "") != "timeout":
                    self._pending_gpu_world_status_handle = None
        return getattr(self, "_latest_gpu_world_status", None)

    def gpu_world_mirror_snapshot(self, *, block: bool = False) -> GpuWorldMirror | None:
        if self._gpu_owner_for_commands() is not None:
            status = self.gpu_world_status_snapshot(block=block)
            if status is not None:
                return GpuWorldMirror.from_status(status)
        if self.world is None:
            return None
        return GpuWorldMirror.from_world(self.world)

    def run_on_gpu_world(self, fn, *, timeout: float = 5.0, high_priority: bool = False):
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            submit = getattr(owner, "submit", None)
            submit_priority = getattr(owner, "submit_high_priority", None)
            submit_fn = submit_priority if high_priority and callable(submit_priority) else submit
            if callable(submit_fn):
                result = submit_fn(
                    GpuCommand(
                        op=GpuCommandType.RUN_WORLD_CALLABLE,
                        payload=WorldCallableCommand(fn),
                    )
                ).wait(timeout=timeout)
                if not result.ok:
                    return {"error": result.error}
                return result.value
        del fn, timeout
        return {"error": "GPU owner is not active"}

    @staticmethod
    def _owner_created_world_requested() -> bool:
        value = os.environ.get("ORACLE_TRANSLATOR_OWNER_CREATED_WORLD", "").strip().lower()
        if value in {"0", "false", "no", "off"}:
            raise RuntimeError(
                "ORACLE_TRANSLATOR_OWNER_CREATED_WORLD cannot disable the GPU owner runtime; "
                "legacy main-thread GPU ownership has been removed"
            )
        return True

    def _create_gpu_owned_world(
        self,
        *,
        store: WorldChunkStore,
        registry,
        cam_x: int,
        cam_y: int,
    ) -> bool:
        owner = GpuOwnerThread()
        runtime = GpuOwnedWorldRuntime()
        runtime.register(owner)
        owner.start()

        def make_context():
            import moderngl

            render_window = pyglet.window.Window(
                width=int(self.width),
                height=int(self.height),
                caption="Oracle Translator",
                resizable=False,
                vsync=False,
            )
            render_window.switch_to()
            ctx = moderngl.create_context()
            pyglet.app.windows.discard(render_window)

            @render_window.event
            def on_key_press(symbol, modifiers):
                self._push_owner_input_event("press", symbol, modifiers)

            @render_window.event
            def on_key_release(symbol, modifiers):
                self._push_owner_input_event("release", symbol, modifiers)

            @render_window.event
            def on_close():
                self._push_owner_input_event("press", key.ESCAPE, 0)

            return {
                "ctx": ctx,
                "window": render_window,
                "release_on_close": True,
            }

        def make_world(context_bundle):
            return ActiveWorldWindow(
                store,
                registry,
                viewport_width=cfg.VIEWPORT_WIDTH,
                viewport_height=cfg.VIEWPORT_HEIGHT,
                ctx=context_bundle["ctx"],
                initial_camera_x=cam_x,
                initial_camera_y=cam_y,
                chunk_save_dir=cfg.chunk_storage_root(),
                chunk_cache_prefetch_x=cfg.CHUNK_CACHE_PREFETCH_X,
                chunk_cache_prefetch_y=cfg.CHUNK_CACHE_PREFETCH_Y,
                enable_chunk_worker_processes=bool(getattr(self, "_enable_chunk_worker_processes", True)),
                enable_gl_sync=False,
            )

        create_result = owner.submit(
            GpuCommand(
                op=GpuCommandType.CREATE_WORLD,
                payload=CreateWorldCommand(
                    context_factory=make_context,
                    world_factory=make_world,
                    snapshot_registry=self._snapshot_registry,
                ),
            )
        ).wait(timeout=30.0)
        if not create_result.ok:
            owner.shutdown()
            raise RuntimeError(f"owner-created GPU world failed: {create_result.error}")
        self._gpu_owner_thread = owner
        self._gpu_owner_created_world = True
        status_result = owner.submit(
            GpuCommand(
                op=GpuCommandType.WORLD_STATUS,
                payload=WorldStatusCommand(include_gpu_debug=True),
            )
        ).wait(timeout=5.0)
        if status_result.ok and isinstance(status_result.value, dict):
            status = dict(status_result.value)
            status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
            self._latest_gpu_world_status = status
            self.world = GpuWorldMirror.from_status(status)
        else:
            self.world = GpuWorldMirror(
                camera_x=int(cam_x),
                camera_y=int(cam_y),
                active_origin_x=int(cam_x),
                active_origin_y=int(cam_y),
                active_width=cfg.VIEWPORT_WIDTH,
                active_height=cfg.VIEWPORT_HEIGHT,
                viewport_width=cfg.VIEWPORT_WIDTH,
                viewport_height=cfg.VIEWPORT_HEIGHT,
                world_width=cfg.WORLD_WIDTH,
                world_height=cfg.WORLD_HEIGHT,
            )
        return True

    def _start_gpu_owner_thread(self) -> bool:
        if self.world is None:
            return False
        owner = getattr(self, "_gpu_owner_thread", None)
        if owner is not None and bool(getattr(owner, "is_running", False)):
            return True
        owner = GpuOwnerThread(
            before_command=self._activate_gpu_gl_context,
            command_lock=self._gpu_gl_context_guard(),
        )
        GpuWorldCommandHandlers(self.world).register(owner)
        owner.start()
        probe = owner.run_on_owner(self._probe_gpu_owner_gl_migration, timeout=5.0)
        if not probe.ok:
            log.warning("[app] GPU owner startup probe failed: %s", probe.error)
            owner.shutdown()
            self._gpu_owner_thread = None
            return False
        self._gpu_owner_thread = owner
        return True

    def _probe_gpu_owner_gl_migration(self) -> bool:
        world = getattr(self, "world", None)
        gpu = None if world is None else getattr(world, "gpu_simulator", None)
        if gpu is None:
            return True
        stage_region = getattr(gpu, "stage_region", None)
        release_staged_region = getattr(gpu, "release_staged_region", None)
        if not callable(stage_region) or not callable(release_staged_region):
            return True
        staged = stage_region(0, 0, 1, 1)
        release_staged_region(staged)
        return True

    def _stop_gpu_owner_thread(self, *, timeout: float = 2.0) -> None:
        owner = getattr(self, "_gpu_owner_thread", None)
        self._gpu_owner_thread = None
        if owner is None:
            return
        shutdown = getattr(owner, "shutdown", None)
        if callable(shutdown):
            shutdown(timeout=timeout)

    def _close_game_world(self) -> None:
        world = getattr(self, "world", None)
        owner = self._gpu_owner_for_commands()
        if world is None and owner is None:
            return
        if owner is not None:
            submit = getattr(owner, "submit", None)
            if callable(submit):
                result = submit(GpuCommand(op=GpuCommandType.CLOSE_WORLD)).wait(timeout=60.0)
                if not result.ok:
                    log.warning("[app] GPU owner world close failed: %s", result.error)
                    close = getattr(world, "close", None)
                    if callable(close):
                        close()
            else:
                close = getattr(world, "close", None)
                if callable(close):
                    close()
            self._stop_gpu_owner_thread(timeout=10.0)
            self.world = None
            self._pending_gpu_frame_handle = None
            self._latest_gpu_frame_payload = None
            self._pending_gpu_present_handle = None
            self._pending_gpu_world_step_handle = None
            self._pending_gpu_camera_follow_handle = None
            self._pending_gpu_background_io_handle = None
            self._pending_gpu_screenshot_handle = None
            self._gpu_owner_created_world = False
            try:
                self.set_visible(True)
            except Exception:
                pass
            return
        close = getattr(world, "close", None)
        if callable(close):
            close()
        self.world = None
        self._pending_gpu_frame_handle = None
        self._latest_gpu_frame_payload = None
        self._pending_gpu_present_handle = None
        self._pending_gpu_world_step_handle = None
        self._pending_gpu_camera_follow_handle = None
        self._pending_gpu_background_io_handle = None
        self._pending_gpu_screenshot_handle = None
        self._gpu_owner_created_world = False
        self._stop_gpu_owner_thread()
        try:
            self.set_visible(True)
        except Exception:
            pass

    def _gpu_gl_context_guard(self):
        lock = getattr(self, "_gpu_gl_lock", None)
        if lock is None:
            return nullcontext()
        return lock

    def _activate_gpu_gl_context(self) -> None:
        context = getattr(self, "context", None)
        canvas = getattr(self, "canvas", None)
        attach = getattr(context, "attach", None)
        if callable(attach) and canvas is not None:
            try:
                attach(canvas)
            except Exception:
                pass
        set_current = getattr(context, "set_current", None)
        if callable(set_current):
            try:
                set_current()
                return
            except Exception:
                pass
        switch_to = getattr(self, "switch_to", None)
        if callable(switch_to):
            try:
                switch_to()
            except Exception:
                pass

    def _world_mutation_sink(self) -> GpuWorldMutationBuffer:
        sink = getattr(self, "_gpu_world_mutations", None)
        if sink is None:
            sink = GpuWorldMutationBuffer()
            self._gpu_world_mutations = sink
        return sink

    def _queue_world_paint(
        self,
        world_x: int,
        world_y: int,
        radius: int,
        family_id: str | None,
        variant_id: str | None,
        *,
        overrides: dict | None = None,
    ) -> None:
        self._world_mutation_sink().paint_world(
            world_x,
            world_y,
            radius,
            family_id,
            variant_id,
            overrides=overrides,
        )

    def _queue_world_pressure(self, world_x: int, world_y: int, radius: int, pressure: float) -> None:
        self._world_mutation_sink().inject_pressure_world(world_x, world_y, radius, pressure)

    def _queue_world_pressure_ring(
        self,
        world_x: int,
        world_y: int,
        inner_radius: int,
        outer_radius: int,
        pressure: float,
    ) -> None:
        self._world_mutation_sink().inject_pressure_ring_world(
            world_x,
            world_y,
            inner_radius,
            outer_radius,
            pressure,
        )

    def _flush_gpu_world_mutations(self) -> int:
        started_at = _time.perf_counter()
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            commands = self._world_mutation_sink().drain()
            count = len(commands)
            if commands:
                owner.fire_and_forget(
                    GpuCommand(op=GpuCommandType.APPLY_WORLD_MUTATIONS, payload=commands)
                )
        else:
            if self.world is None:
                return 0
            raise RuntimeError("GPU owner is required for world mutations")
        self._last_gpu_world_mutation_flush_count = count
        self._set_perf_ms("gpu_world_mutation_flush", (_time.perf_counter() - started_at) * 1000.0)
        return count

    def _submit_gpu_world_tick(self, dt: float) -> None:
        owner = self._gpu_owner_for_commands()
        if self.world is None and owner is None:
            return
        if owner is None:
            raise RuntimeError("GPU owner is required for world stepping")
        step_handle = getattr(self, "_pending_gpu_world_step_handle", None)
        if step_handle is not None:
            result = self._owner_result_if_ready(step_handle)
            if result is None:
                return
            self._pending_gpu_world_step_handle = None
            if bool(getattr(result, "ok", False)):
                self._record_gpu_tick()
            elif getattr(result, "error", ""):
                log.warning("[app] GPU owner world step failed: %s", result.error)
        self._flush_gpu_world_mutations()
        self._submit_gpu_entity_state_update()
        self._submit_gpu_entity_mask_update()
        self._pending_gpu_world_step_handle = owner.submit(
            GpuCommand(op=GpuCommandType.STEP_WORLD, payload={"dt": float(dt)})
        )

    def _push_owner_input_event(self, kind: str, symbol: int, modifiers: int) -> None:
        events = getattr(self, "_owner_input_events", None)
        if events is not None:
            events.put((str(kind), int(symbol), int(modifiers)))

    def _drain_owner_input_events(self) -> None:
        events = getattr(self, "_owner_input_events", None)
        if events is None:
            return
        while True:
            try:
                kind, symbol, modifiers = events.get_nowait()
            except queue.Empty:
                break
            if kind == "press":
                self.on_key_press(symbol, modifiers)
            elif kind == "release":
                self.on_key_release(symbol, modifiers)

    def change_screen(self, name: str) -> None:
        self.current_screen = name
        if name == "game":
            try:
                self.activate()
            except Exception:
                pass

    def resize_window(self, new_cell_scale: int) -> None:
        """Resize the window for a new cell scale. Called from OptionsScreen."""
        self._cell_scale = new_cell_scale
        cfg.CELL_SCALE = new_cell_scale
        window_width = cfg.VIEWPORT_WIDTH * new_cell_scale
        window_height = cfg.VIEWPORT_HEIGHT * new_cell_scale
        self.set_size(window_width, window_height)
        resize = getattr(self.renderer, "resize", None)
        if callable(resize):
            resize(window_width, window_height)
        else:
            raise RuntimeError("display renderer does not support resizing")

    def clear_screen(self, red: float = 0.04, green: float = 0.05, blue: float = 0.07, alpha: float = 1.0) -> None:
        started_at = perf_counter()
        del red, green, blue, alpha
        self.clear()
        self._set_perf_ms("draw_ctx_clear", (perf_counter() - started_at) * 1000.0)

    def read_framebuffer_rgb(self) -> bytes:
        if self._gpu_owner_for_commands() is not None:
            payload = getattr(self, "_latest_gpu_frame_payload", None)
            if isinstance(payload, GpuFramePayload):
                return self._rgb_from_gpu_frame_payload(payload)
            raise RuntimeError("GPU owner framebuffer payload is not ready")
        raise RuntimeError("GPU owner framebuffer payload is not ready")

    def _rgb_from_gpu_frame_payload(self, payload: GpuFramePayload) -> bytes:
        """Build bottom-to-top RGB screen bytes from an owner-rendered frame."""
        out_width = max(1, int(getattr(self, "width", payload.width)))
        out_height = max(1, int(getattr(self, "height", payload.height)))
        tex_width = max(1, int(payload.width))
        tex_height = max(1, int(payload.height))
        origin_x, origin_y, scale_x, scale_y = payload.uv_rect
        x_indices = [
            max(0, min(tex_width - 1, int((origin_x + ((x + 0.5) / out_width) * scale_x) * tex_width)))
            for x in range(out_width)
        ]
        rgb = bytearray(out_width * out_height * 3)
        dst = 0
        for y in range(out_height):
            sample_v = origin_y + ((y + 0.5) / out_height) * scale_y
            src_y = max(0, min(tex_height - 1, int(sample_v * tex_height)))
            row_offset = src_y * tex_width * 4
            for src_x in x_indices:
                src = row_offset + src_x * 4
                rgb[dst:dst + 3] = payload.rgba[src:src + 3]
                dst += 3
        return bytes(rgb)

    def owner_frame_view_rgb_snapshot(self, *, timeout: float = 10.0) -> tuple[int, int, bytes]:
        legacy_payload = getattr(self, "_latest_gpu_frame_payload", None)
        if isinstance(legacy_payload, GpuFramePayload):
            tex_width = max(1, int(legacy_payload.width))
            tex_height = max(1, int(legacy_payload.height))
            origin_x, origin_y, scale_x, scale_y = legacy_payload.uv_rect
            crop_x = max(0, min(tex_width - 1, int(origin_x * tex_width)))
            crop_y = max(0, min(tex_height - 1, int(origin_y * tex_height)))
            crop_width = max(1, min(tex_width - crop_x, int(round(scale_x * tex_width))))
            crop_height = max(1, min(tex_height - crop_y, int(round(scale_y * tex_height))))
            rgb = bytearray(crop_width * crop_height * 3)
            dst = 0
            for y in range(crop_height):
                row_offset = (crop_y + y) * tex_width * 4
                for x in range(crop_width):
                    src = row_offset + (crop_x + x) * 4
                    rgb[dst:dst + 3] = legacy_payload.rgba[src:src + 3]
                    dst += 3
            return crop_width, crop_height, bytes(rgb)
        owner = self._gpu_owner_for_commands()
        submit = getattr(owner, "submit", None)
        submit_priority = getattr(owner, "submit_high_priority", None)
        submit_fn = submit_priority if callable(submit_priority) else submit
        if not callable(submit_fn):
            raise RuntimeError("GPU owner framebuffer payload is not ready")
        handle = getattr(self, "_pending_gpu_screenshot_handle", None)
        if handle is None:
            enemies = tuple(
                self._render_actor_snapshot(eid, enemy.__class__.__name__, enemy)
                for eid, enemy in self.enemies.items()
                if getattr(enemy, "is_alive", False)
            )
            projectiles = tuple(
                self._render_actor_snapshot(f"projectile_{index}", projectile.__class__.__name__, projectile)
                for index, projectile in enumerate(self.projectiles)
                if getattr(projectile, "is_alive", False)
            )
            collision_debug = (
                self.entity_manager.last_debug
                if self.entity_manager.debug_collision and self.entity_manager.last_debug is not None
                else None
            )
            handle = submit_fn(
                GpuCommand(
                    op=GpuCommandType.RENDER_FRAME,
                    payload=RenderFrameCommand(
                        view_mode=self.view_mode,
                        readback_rgba=False,
                        readback_present_rgb=True,
                        present=True,
                        window_size=(int(self.width), int(self.height)),
                        hero=self._render_actor_snapshot("hero", "Hero", self.hero),
                        enemies=enemies,
                        projectiles=projectiles,
                        dt=float(getattr(self, "_last_dt", 1.0 / 60.0)),
                        sim_fps=float(getattr(self, "sim_fps", 0.0)),
                        gpu_tick_rate=float(getattr(self, "gpu_tick_rate", 0.0)),
                        overlay_lines=self._owner_present_overlay_lines(),
                        collision_debug=collision_debug,
                        console_lines=self._owner_present_console_lines(),
                    ),
                )
            )
            self._pending_gpu_screenshot_handle = handle
        result = handle.wait(timeout=max(0.0, float(timeout)))
        if not result.ok or not isinstance(getattr(result, "value", None), dict):
            if getattr(result, "error", "") != "timeout":
                self._pending_gpu_screenshot_handle = None
            raise RuntimeError(result.error or "GPU owner framebuffer payload is not ready")
        self._pending_gpu_screenshot_handle = None
        value = result.value
        screen_rgb = value.get("screen_rgb")
        if not isinstance(screen_rgb, (bytes, bytearray)):
            raise RuntimeError("GPU owner presented framebuffer RGB is not ready")
        width = max(1, int(value.get("width", getattr(self, "width", 1))))
        height = max(1, int(value.get("height", getattr(self, "height", 1))))
        status = dict(value.get("status") or {})
        if status:
            status["tick_rate"] = round(float(getattr(self, "gpu_tick_rate", 0.0)), 3)
            self._latest_gpu_world_status = status
        self._last_owner_present_overlay_line_count = int(value.get("overlay_line_count", 0) or 0)
        self._last_owner_present_enemy_count = int(value.get("enemy_overlay_count", 0) or 0)
        self._last_owner_present_projectile_count = int(value.get("projectile_overlay_count", 0) or 0)
        return width, height, bytes(screen_rgb)

    def _submit_gpu_entity_mask_update(self) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            mirror = self.gpu_world_mirror_snapshot(block=False)
            if mirror is None:
                return
            rects = self.entity_manager.entity_mask_payload(mirror)
            owner.fire_and_forget(
                GpuCommand(
                    op=GpuCommandType.SYNC_ENTITY_STATE,
                    payload=EntityGpuSyncCommand(mask_rects=tuple(rects)),
                )
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for entity mask upload")

    def _submit_gpu_entity_shape_registration(self) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            owner.fire_and_forget(
                GpuCommand(
                    op=GpuCommandType.SYNC_ENTITY_STATE,
                    payload=EntityGpuSyncCommand(
                        shapes=tuple(self.entity_manager.entity_shape_payloads()),
                    ),
                )
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for entity shape registration")

    def _submit_gpu_entity_state_update(self) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            mirror = self.gpu_world_mirror_snapshot(block=False)
            if mirror is None:
                return
            states = self.entity_manager.entity_state_payload(mirror)
            if states:
                owner.fire_and_forget(
                    GpuCommand(
                        op=GpuCommandType.SYNC_ENTITY_STATE,
                        payload=EntityGpuSyncCommand(
                            states=tuple(states),
                            origin_x=int(mirror.active_origin_x),
                            origin_y=int(mirror.active_origin_y),
                        ),
                    )
                )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for entity state upload")

    def _submit_gpu_options(self, *, skip_liquid_physics: bool | None = None) -> None:
        owner = self._gpu_owner_for_commands()
        command = GpuOptionsCommand(skip_liquid_physics=skip_liquid_physics)
        if owner is not None:
            owner.fire_and_forget(
                GpuCommand(op=GpuCommandType.SET_GPU_OPTIONS, payload=command)
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for GPU option updates")

    def _submit_gpu_camera_follow(self, target_x: int, target_y: int, *, dt: float) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            handle = getattr(self, "_pending_gpu_camera_follow_handle", None)
            if handle is not None:
                result = self._owner_result_if_ready(handle)
                if result is None:
                    return
                self._pending_gpu_camera_follow_handle = None
                if not bool(getattr(result, "ok", False)) and getattr(result, "error", ""):
                    log.warning("[app] GPU owner camera follow failed: %s", result.error)
            self._pending_gpu_camera_follow_handle = owner.submit(
                GpuCommand(
                    op=GpuCommandType.CAMERA_FOLLOW,
                    payload=CameraFollowCommand(
                        target_x=int(target_x),
                        target_y=int(target_y),
                        dt=float(dt),
                    ),
                )
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for camera follow")

    def _submit_gpu_camera_teleport(self, target_x: int, target_y: int) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            owner.fire_and_forget(
                GpuCommand(
                    op=GpuCommandType.CAMERA_TELEPORT,
                    payload=CameraTeleportCommand(
                        target_x=int(target_x),
                        target_y=int(target_y),
                        submit_chunks=8,
                    ),
                )
            )
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for camera teleport")

    def _submit_gpu_background_io_service(self) -> None:
        owner = self._gpu_owner_for_commands()
        if owner is not None:
            handle = getattr(self, "_pending_gpu_background_io_handle", None)
            if handle is not None:
                result = self._owner_result_if_ready(handle)
                if result is None:
                    return
                self._pending_gpu_background_io_handle = None
                if not bool(getattr(result, "ok", False)) and getattr(result, "error", ""):
                    log.warning("[app] GPU owner background IO service failed: %s", result.error)
            self._pending_gpu_background_io_handle = owner.submit(GpuCommand(op=GpuCommandType.SERVICE_BACKGROUND_IO))
            return
        if self.world is not None:
            raise RuntimeError("GPU owner is required for background IO service")

    def _init_game_world(self) -> None:
        """Initialize a fresh game world."""
        if self.world is not None:
            self._close_game_world()
        t0 = perf_counter()
        log.info("[app] _init_game_world: seed=%d world=%dx%d viewport=%dx%d chunk_size=%d",
                 self.seed, cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                 cfg.VIEWPORT_WIDTH, cfg.VIEWPORT_HEIGHT, cfg.CHUNK_SIZE)
        terrain_gen = TerrainGenerator(self.seed, self.registry)
        self.terrain_gen = terrain_gen
        store = WorldChunkStore(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                                chunk_size=cfg.CHUNK_SIZE, seed=self.seed,
                                chunk_generator=terrain_gen.generate_chunk)
        hero_x = float(cfg.VIEWPORT_WIDTH // 2)
        spawn = terrain_gen.find_spawn_point_near(
            int(hero_x),
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=max(128, cfg.VIEWPORT_WIDTH // 2),
        )
        if spawn is None:
            ground_y = terrain_gen.ground_height_at(int(hero_x))
            hero_y = ground_y - 1 - cfg.HERO_HEIGHT
        else:
            hero_x, hero_y = spawn
            ground_y = hero_y + 1 + cfg.HERO_HEIGHT
        log.info("[app] hero spawn: x=%.1f y=%.1f ground_y=%.1f", hero_x, hero_y, ground_y)
        self.hero.reset(hero_x, hero_y)
        # Initialize camera at hero position to avoid loading empty world center
        cam_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        cam_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        owner_started = False
        self._gpu_owner_created_world = False
        self._owner_created_world_requested()
        owner_started = self._create_gpu_owned_world(
            store=store,
            registry=self.registry,
            cam_x=cam_x,
            cam_y=cam_y,
        )
        log.info("[app] world initialized in %.1fms, camera=(%d,%d) hero=(%.1f,%.1f)",
                 (perf_counter() - t0) * 1000,
                 self.world.camera_x, self.world.camera_y,
                 self.hero.x, self.hero.y)
        if owner_started:
            log.info("[app] GPU owner thread started for world commands")
            try:
                self.set_visible(False)
            except Exception:
                pass
        self._world_mutation_sink().clear()
        self._last_gpu_world_mutation_flush_count = 0
        self._snapshot_request_queue().clear()
        self._pending_gpu_snapshot_service_handle = None
        self._sim_fps = 0.0
        self._sim_fps_count = 0
        self._sim_fps_started_at = perf_counter()
        self._sim_step_timestamps = deque(maxlen=240)
        self._gpu_tick_timestamps = deque(maxlen=240)
        self.active_streams.clear()
        self.active_pressure_bursts.clear()
        self.enemies.clear()
        self.projectiles.clear()
        # Register hero entity for GPU query / GPU occupancy mask / GPU placeholder.
        self.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=self.hero.x,
            y=self.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))
        # Spawn enemies from terrain seed distribution
        enemy_idx = 0
        for sp in terrain_gen.spawn_points():
            eid = f"enemy_{sp['type']}_{enemy_idx}"
            enemy_idx += 1
            if sp["type"] == "A":
                enemy = EnemyA.create(eid, sp["x"], sp["y"], sp["biome"])
            elif sp["type"] == "B":
                enemy = EnemyB.create(eid, sp["x"], sp["y"], sp["biome"])
            elif sp["type"] == "C":
                enemy = EnemyC.create(eid, sp["x"], sp["y"], sp["biome"])
            else:
                continue
            self.enemies[eid] = enemy
            self.entity_manager.register_enemy(enemy)
        if self.world is not None:
            self._submit_gpu_entity_mask_update()
            self._submit_gpu_entity_shape_registration()
            self._refresh_local_entity_snapshots(force=True)
            # Optimize: skip liquid physics for stone-only scenes.
            self._submit_gpu_options(skip_liquid_physics=True)

    def move_hero_to(self, x: float, y: float) -> None:
        self.hero.x = x
        self.hero.y = y
        self.hero.vel_x = 0.0
        self.hero.vel_y = 0.0
        self.hero.on_ground = False
        hero_entity = self.entity_manager._entities.get("hero")
        if hero_entity is not None:
            hero_entity.x = x
            hero_entity.y = y
        mirror = self.gpu_world_mirror_snapshot(block=False)
        if mirror is not None:
            target_x = int(x) - mirror.viewport_width // 2
            target_y = int(y) - mirror.viewport_height // 2
            self._submit_gpu_camera_teleport(target_x, target_y)

    def set_experiment_camera_target(self, camera_x: int, camera_y: int, *, freeze_gameplay: bool = False) -> None:
        self._experiment_camera_override = (int(camera_x), int(camera_y))
        self._experiment_freeze_gameplay = bool(freeze_gameplay)

    def clear_experiment_camera_target(self) -> None:
        self._experiment_camera_override = None
        self._experiment_freeze_gameplay = False

    def teleport_to_surface_x(self, world_x: float) -> bool:
        if self.terrain_gen is None:
            return False
        spawn = self.terrain_gen.find_spawn_point_near(
            int(world_x),
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=max(256, cfg.BIOME_WIDTH // 5),
        )
        if spawn is None:
            return False
        self.move_hero_to(*spawn)
        return True

    def cast_spell_by_index(self, idx: int) -> None:
        """Cast a spell from the catalog by index."""
        if not SPELL_CATALOG or idx < 0 or idx >= len(SPELL_CATALOG):
            return
        spell = SPELL_CATALOG[idx]
        if self.world is None:
            return
        if not self.hero.consume_mp(spell["mp"]):
            return
        self.hero.selected_spell_idx = idx
        magic = expand_model_socket(spell, self.hero.x, self.hero.y, self.hero.facing_right)
        result = execute_magic_socket(magic, self._world_mutation_sink(), self.registry)
        if result is not None and isinstance(result, ActiveStream):
            self.active_streams.append(result)

    def _handle_hero_input(self, dt: float) -> None:
        """Read keyboard input and update hero."""
        self.hero.input_left = key.A in self._keys_pressed or key.LEFT in self._keys_pressed
        self.hero.input_right = key.D in self._keys_pressed or key.RIGHT in self._keys_pressed
        self.hero.input_jump = key.W in self._keys_pressed or key.UP in self._keys_pressed
        self.hero.input_chant_held = key.SPACE in self._keys_pressed

    def _mock_slm_to_spell_index(self, text: str) -> int:
        """Mock SLM: return the currently selected spell index.

        In the real pipeline this would be:
            text -> SLM inference -> Model Socket -> expand -> execute
        For now, just return the selected spell from the catalog.
        """
        return self.hero.selected_spell_idx

    def update_game(self, dt: float) -> None:
        """Update game logic for one tick."""
        if self.world is None:
            return
        if bool(getattr(self, "_experiment_freeze_gameplay", False)):
            freeze_started_at = _time.perf_counter()
            t1 = _time.perf_counter()
            self._submit_gpu_world_tick(dt)
            t2 = _time.perf_counter()
            self._refresh_local_entity_snapshots()
            t3 = _time.perf_counter()
            self._set_perf_ms("update_game_poll", 0.0)
            self._set_perf_ms("update_game_apply", 0.0)
            self._set_perf_ms("update_game_enemy_ai", 0.0)
            self._set_perf_ms("update_game_tick", 0.0)
            self._set_perf_ms("update_game_world_step_submit", (t2 - t1) * 1000.0)
            self._set_perf_ms("update_game_world_step", (t2 - t1) * 1000.0)
            self._set_perf_ms("update_game_schedule_feedback", 0.0)
            self._set_perf_ms("update_game_schedule", 0.0)
            self._set_perf_ms("update_game_snapshot", (t3 - t2) * 1000.0)
            self._set_perf_ms("update_game_streams", 0.0)
            self._set_perf_ms("update_game_projectiles", 0.0)
            self._set_perf_ms("update_game_cleanup", 0.0)
            self._set_perf_ms("update_game_post_step_gpu_mutations", 0.0)
            self._set_perf_ms("update_game_total", (_time.perf_counter() - freeze_started_at) * 1000.0)
            self._record_sim_fps()
            return
        if self.hero.state == "chant" and not self._chant_started:
            self._chant_started = True
            if self.stt.available:
                self.stt.start()
        was_chanting = self.hero.state == "chant"
        t0 = _time.perf_counter()
        t1 = _time.perf_counter()
        self.entity_manager.read_feedback_and_update(self.world, dt)
        t2 = _time.perf_counter()
        self._process_enemy_ai(dt)
        t3 = _time.perf_counter()
        self._service_pressure_bursts()
        t4 = _time.perf_counter()
        self._submit_gpu_world_tick(dt)
        t5 = _time.perf_counter()
        t6 = _time.perf_counter()
        self._refresh_local_entity_snapshots()
        t7 = _time.perf_counter()
        self._set_perf_ms("update_game_poll", (t1 - t0) * 1000.0)
        self._set_perf_ms("update_game_apply", (t2 - t1) * 1000.0)
        self._set_perf_ms("update_game_enemy_ai", (t3 - t2) * 1000.0)
        self._set_perf_ms("update_game_tick", (t4 - t3) * 1000.0)
        self._set_perf_ms("update_game_world_step_submit", (t5 - t4) * 1000.0)
        self._set_perf_ms("update_game_world_step", (t5 - t4) * 1000.0)
        self._set_perf_ms("update_game_schedule_feedback", (t6 - t5) * 1000.0)
        self._set_perf_ms("update_game_schedule", (t6 - t5) * 1000.0)
        self._set_perf_ms("update_game_snapshot", (t7 - t6) * 1000.0)
        if was_chanting and self.hero.state == "cast":
            # Stop STT and get transcribed text
            stt_text = ""
            if self._chant_started and self.stt.available:
                stt_text = self.stt.stop()
            self._chant_started = False

            # Mock SLM: transcribed text -> spell index
            spell_idx = self._mock_slm_to_spell_index(stt_text)
            self.cast_spell_by_index(spell_idx)

        # Inject active stream ticks
        t8 = _time.perf_counter()
        remaining: list[ActiveStream] = []
        for stream in self.active_streams:
            if stream.remaining_ticks > 0:
                inject_stream_tick(
                    stream, self._world_mutation_sink(), self.registry,
                    self.hero.x, self.hero.y, self.hero.facing_right,
                )
                remaining.append(stream)
                # Spell-to-enemy damage
                self._damage_enemies_with_spell(stream)
        self.active_streams = remaining
        t9 = _time.perf_counter()

        # ── Enemy physics: movement, gravity, collision ──

        # ── Projectile update + GPU collision ──
        self._process_projectiles(dt)
        t10 = _time.perf_counter()

        # ── Clean up dead enemies ──
        self._cleanup_dead_enemies()
        t11 = _time.perf_counter()
        self._flush_gpu_world_mutations()
        t12 = _time.perf_counter()
        self._set_perf_ms("update_game_streams", (t9 - t8) * 1000.0)
        self._set_perf_ms("update_game_projectiles", (t10 - t9) * 1000.0)
        self._set_perf_ms("update_game_cleanup", (t11 - t10) * 1000.0)
        self._set_perf_ms("update_game_post_step_gpu_mutations", (t12 - t11) * 1000.0)
        self._set_perf_ms("update_game_total", (t12 - t0) * 1000.0)
        self._record_sim_fps()

    def _queue_pressure_burst(self, world_x: int, world_y: int, radius: int, pressure: float) -> None:
        self.active_pressure_bursts.append(
            PressureBurst(
                x=int(world_x),
                y=int(world_y),
                radius=int(radius),
                pressure=float(pressure),
                ticks_remaining=max(1, int(cfg.BLAST_PRESSURE_BURST_TICKS)),
            )
        )

    def _service_pressure_bursts(self) -> None:
        if self.world is None or not self.active_pressure_bursts:
            return
        remaining: list[PressureBurst] = []
        for burst in self.active_pressure_bursts:
            age = max(0, cfg.BLAST_PRESSURE_BURST_TICKS - burst.ticks_remaining)
            scale = cfg.BLAST_PRESSURE_DECAY ** age
            pulse = burst.pressure * scale
            if pulse <= 1.0:
                continue
            core_radius = max(2, burst.radius - age)
            core_pulse = pulse * float(cfg.BLAST_PRESSURE_CORE_SCALE)
            if core_pulse > 1.0:
                self._queue_world_pressure(burst.x, burst.y, core_radius, core_pulse)

            shell_mid_radius = (
                float(cfg.BLAST_RING_DISTANCE)
                + float(burst.radius)
                + float(age) * float(cfg.BLAST_PRESSURE_SHELL_SPEED)
            )
            shell_half_width = max(1.0, float(cfg.BLAST_PRESSURE_SHELL_WIDTH) * 0.5)
            shell_inner = max(1, int(round(shell_mid_radius - shell_half_width)))
            shell_outer = max(shell_inner + 1, int(round(shell_mid_radius + shell_half_width)))
            self._queue_world_pressure_ring(
                burst.x,
                burst.y,
                shell_inner,
                shell_outer,
                pulse * float(cfg.BLAST_PRESSURE_SHELL_SCALE),
            )

            trail_outer = max(1, shell_inner)
            trail_inner = max(0, trail_outer - max(1, int(cfg.BLAST_PRESSURE_SHELL_WIDTH)))
            trail_pulse = pulse * float(cfg.BLAST_SECONDARY_PRESSURE_SCALE)
            if trail_pulse > 1.0:
                self._queue_world_pressure_ring(
                    burst.x,
                    burst.y,
                    trail_inner,
                    trail_outer,
                    trail_pulse,
                )
            burst.ticks_remaining -= 1
            if burst.ticks_remaining > 0:
                remaining.append(burst)
        self.active_pressure_bursts = remaining

    def _trigger_explosion(self, world_x: int, world_y: int, *, pressure: float | None = None, radius: int | None = None) -> None:
        if self.world is None:
            return
        center_x = int(world_x)
        center_y = int(world_y)
        pressure_value = float(cfg.BLAST_PRESSURE if pressure is None else pressure)
        pressure_radius = int(cfg.BLAST_PRESSURE_RADIUS if radius is None else radius)
        self._queue_world_paint(center_x, center_y, int(cfg.BLAST_FIRE_RADIUS), "fire", "fire", overrides={"vel_y": -2.0})
        self._queue_world_paint(center_x, center_y, int(cfg.BLAST_GAS_RADIUS), "water", "steam", overrides={"temperature": 220.0})
        self._queue_world_pressure(center_x, center_y, pressure_radius, pressure_value)
        shell_half_width = max(1.0, float(cfg.BLAST_PRESSURE_SHELL_WIDTH) * 0.5)
        shell_inner = max(1, int(round(float(pressure_radius) - shell_half_width)))
        shell_outer = max(shell_inner + 1, int(round(float(pressure_radius) + shell_half_width)))
        self._queue_world_pressure_ring(
            center_x,
            center_y,
            shell_inner,
            shell_outer,
            pressure_value * float(cfg.BLAST_PRESSURE_SHELL_SCALE),
        )
        self._queue_pressure_burst(center_x, center_y, pressure_radius, pressure_value)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            vx = dx * cfg.BLAST_GAS_VELOCITY
            vy = dy * cfg.BLAST_GAS_VELOCITY
            px = center_x + int(dx * cfg.BLAST_RING_DISTANCE)
            py = center_y + int(dy * cfg.BLAST_RING_DISTANCE)
            self._queue_world_paint(px, py, int(cfg.BLAST_GAS_RADIUS), "water", "steam", overrides={"vel_x": vx, "vel_y": vy, "temperature": 180.0})
            self._queue_world_paint(px, py, 1, "stone", "stone_powder", overrides={"vel_x": dx * cfg.BLAST_DEBRIS_VELOCITY, "vel_y": dy * cfg.BLAST_DEBRIS_VELOCITY})

    def _process_enemy_ai(self, dt: float) -> None:
        """Process enemy AI: physics, movement, arrows, boss attacks."""
        mirror = self.gpu_world_mirror_snapshot(block=False)
        if mirror is None:
            return
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                continue
            if not self._enemy_is_near_active_world(enemy):
                continue
            # Physics/movement update with latest completed snapshot.
            current_tick = int(mirror.gpu_step_index)
            enemy_snapshot = self.entity_manager.latest_snapshot_for(eid, current_tick=current_tick)
            enemy.update(
                dt,
                local_snapshot=enemy_snapshot,
                world_width=mirror.world_width,
                hero_x=self.hero.x,
                hero_y=self.hero.y,
            )
            # Sync entity position
            entity = self.entity_manager._entities.get(eid)
            if entity is not None:
                entity.x = enemy.x
                entity.y = enemy.y
            # Attack actions
            if isinstance(enemy, EnemyA) and enemy.should_fire_arrow():
                arrow = Arrow.create(enemy.x, enemy.y, self.hero.x, self.hero.y, enemy.facing_right)
                self.projectiles.append(arrow)
                enemy.fire_arrow()
            elif isinstance(enemy, EnemyC):
                if enemy.should_spray_oil:
                    dx = cfg.BOSS_OIL_SPRAY_WIDTH if enemy.facing_right else -cfg.BOSS_OIL_SPRAY_WIDTH
                    spray_x = int(enemy.x + dx)
                    spray_y = int(enemy.y + enemy.height * 0.5)
                    if 0 <= spray_x < cfg.WORLD_WIDTH and 0 <= spray_y < cfg.WORLD_HEIGHT:
                        self._queue_world_paint(spray_x, spray_y, int(cfg.BOSS_OIL_SPRAY_WIDTH // 2), "tar", "tar_liquid")
                elif enemy.should_fire_fireball:
                    fb = Fireball.create(enemy.x, enemy.y, self.hero.x, self.hero.y)
                    self.projectiles.append(fb)
                elif enemy.should_collapse:
                    cx = int(enemy.x)
                    cy = int(enemy.y) - int(cfg.BOSS_COLLAPSE_HEIGHT)
                    if 0 <= cx < cfg.WORLD_WIDTH and 0 <= cy < cfg.WORLD_HEIGHT:
                        self._queue_world_paint(cx, cy, int(cfg.BOSS_COLLAPSE_WIDTH // 2), None, None)

    def _projectile_hero_overlap(self, projectile: ProjectileBase) -> bool:
        hero_left = self.hero.x - cfg.HERO_WIDTH / 2.0
        hero_right = self.hero.x + cfg.HERO_WIDTH / 2.0
        hero_bottom = self.hero.y
        hero_top = self.hero.y + cfg.HERO_HEIGHT
        return (
            projectile.x > hero_left
            and projectile.x < hero_right
            and projectile.y > hero_bottom
            and projectile.y < hero_top
        )

    def _projectile_snapshot_terrain_hit(
        self,
        projectile: ProjectileBase,
        previous_x: float,
        previous_y: float,
    ) -> tuple[int, int] | None:
        snapshot = self.entity_manager.latest_snapshot_for(f"projectile_{id(projectile)}")
        if snapshot is None:
            return None
        probe = build_probe_for_envelope(snapshot)
        dx = projectile.x - previous_x
        dy = projectile.y - previous_y
        steps = max(1, int(max(abs(dx), abs(dy)) * 2.0))
        for step_index in range(steps + 1):
            t = step_index / steps
            sample_x = previous_x + dx * t
            sample_y = previous_y + dy * t
            world_x = int(round(sample_x))
            world_y = int(round(sample_y))
            if not (0 <= world_x < cfg.WORLD_WIDTH and 0 <= world_y < cfg.WORLD_HEIGHT):
                return (world_x, world_y)
            if probe.is_solid(world_x, world_y):
                return (world_x, world_y)
        return None

    def _process_projectiles(self, dt: float) -> None:
        """Move projectiles, check GPU collision, handle hits."""
        if self.world is None:
            return
        previous_positions: list[tuple[float, float]] = []
        for p in self.projectiles:
            previous_positions.append((p.x, p.y))
            p.update(dt)
            if not self._projectile_is_in_world_bounds(p):
                p.is_alive = False
                continue
            if not self._projectile_is_near_active_world(p, margin=24):
                p.is_alive = False

        for i, p in enumerate(self.projectiles):
            if not p.is_alive:
                continue
            previous_x, previous_y = previous_positions[i]
            terrain_hit = self._projectile_snapshot_terrain_hit(p, previous_x, previous_y)
            if terrain_hit is not None:
                hit_x, hit_y = terrain_hit
                if isinstance(p, Arrow):
                    self._queue_world_paint(
                        int(hit_x), int(hit_y), 2, "stone", "stone_powder",
                        overrides={"vel_x": p.vel_x * 0.3, "vel_y": -2.0},
                    )
                elif isinstance(p, Fireball):
                    self._trigger_explosion(int(hit_x), int(hit_y))
                p.is_alive = False
                continue

            if self._projectile_hero_overlap(p):
                if isinstance(p, Arrow):
                    self._queue_world_paint(
                        int(p.x),
                        int(p.y),
                        1,
                        "iron",
                        "iron_grit",
                        overrides={"vel_x": p.vel_x, "vel_y": p.vel_y},
                    )
                elif isinstance(p, Fireball):
                    self._trigger_explosion(int(p.x), int(p.y))
                p.is_alive = False

        self.projectiles = [p for p in self.projectiles if p.is_alive]
        self._prune_projectile_snapshots()

    def _cleanup_dead_enemies(self) -> None:
        """Remove dead enemies, handle EnemyB explosion."""
        if self.world is None:
            return
        dead_ids: list[str] = []
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                dead_ids.append(eid)
                if isinstance(enemy, EnemyB) and enemy.should_explode:
                    # EnemyB explodes on death
                    self._trigger_explosion(int(enemy.x), int(enemy.y))
        for eid in dead_ids:
            self.enemies.pop(eid, None)
            self.entity_manager.unregister_enemy(eid)

    def _damage_enemies_with_spell(self, stream: ActiveStream) -> None:
        """Damage enemies within the spell's brush radius."""
        magic = stream.magic
        # Spell origin: same logic as inject_stream_tick
        if magic.force_strength > 0.0 or magic.carrier_velocity > 0.0:
            sx = self.hero.x + (1.0 if self.hero.facing_right else -1.0) * cfg.HERO_WIDTH
            sy = self.hero.y + cfg.HERO_HEIGHT * 0.5
        else:
            sx, sy = magic.origin
        radius = float(magic.brush_radius)
        damage = cfg.SPELL_BASE_DAMAGE * (0.5 + magic.powerness * 0.5)
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                continue
            dist = ((enemy.x - sx) ** 2 + (enemy.y - sy) ** 2) ** 0.5
            if dist <= radius + enemy.width * 0.5:
                enemy.take_damage(damage)

    def _record_sim_fps(self) -> None:
        """Track actual completed simulation ticks for the F3 overlay."""
        now = perf_counter()
        if not hasattr(self, "_sim_step_timestamps"):
            self._sim_step_timestamps = deque(maxlen=240)
        self._sim_step_timestamps.append(now)
        self._sim_fps_count += 1
        elapsed = now - self._sim_fps_started_at
        if elapsed >= 1.0:
            self._sim_fps = self._sim_fps_count / elapsed
            self._sim_fps_count = 0
            self._sim_fps_started_at = now

    def _record_gpu_tick(self) -> None:
        now = perf_counter()
        if not hasattr(self, "_gpu_tick_timestamps"):
            self._gpu_tick_timestamps = deque(maxlen=240)
        self._gpu_tick_timestamps.append(now)

    @property
    def sim_fps(self) -> float:
        timestamps = getattr(self, "_sim_step_timestamps", None)
        if timestamps is not None and len(timestamps) >= 2:
            window_elapsed = timestamps[-1] - timestamps[0]
            if window_elapsed > 0.0:
                return (len(timestamps) - 1) / window_elapsed
        elapsed = perf_counter() - self._sim_fps_started_at
        if self._sim_fps_count > 0 and elapsed > 0.0:
            return self._sim_fps_count / elapsed
        return self._sim_fps

    @property
    def gpu_tick_rate(self) -> float:
        timestamps = getattr(self, "_gpu_tick_timestamps", None)
        if timestamps is None or len(timestamps) < 2:
            return 0.0
        window_elapsed = timestamps[-1] - timestamps[0]
        if window_elapsed <= 0.0:
            return 0.0
        return (len(timestamps) - 1) / window_elapsed

    def _set_perf_ms(self, key: str, value_ms: float, *, history: int = 120) -> None:
        if not hasattr(self, "_perf_last_ms"):
            self._perf_last_ms = {}
        if not hasattr(self, "_perf_samples"):
            self._perf_samples = {}
        self._perf_last_ms[key] = float(value_ms)
        samples = self._perf_samples.get(key)
        if samples is None:
            samples = deque(maxlen=history)
            self._perf_samples[key] = samples
        samples.append(float(value_ms))

    def perf_last_ms(self, key: str) -> float:
        if not hasattr(self, "_perf_last_ms"):
            return 0.0
        return float(self._perf_last_ms.get(key, 0.0))

    def perf_avg_ms(self, key: str) -> float:
        if not hasattr(self, "_perf_samples"):
            return 0.0
        samples = self._perf_samples.get(key)
        if not samples:
            return 0.0
        return float(sum(samples) / len(samples))

    def debug_perf_snapshot(self) -> dict[str, float]:
        keys = (
            "main_tick_total",
            "main_tick_input",
            "main_tick_sim",
            "main_tick_camera",
            "main_tick_background_io",
            "draw_total",
            "draw_ctx_clear",
            "draw_world",
            "draw_debug_overlay",
            "draw_collision_overlay",
            "update_game_total",
            "update_game_poll",
            "update_game_apply",
            "update_game_tick",
            # update_game_world_step_submit is *CPU submit time* for
            # world.step() only — it does not include GPU execution.
            "update_game_world_step_submit",
            "update_game_schedule_feedback",
            "update_game_snapshot",
            "snapshot_submit",
            "snapshot_ready_to_consume_delay",
            "update_game_streams",
            "update_game_enemy_ai",
            "update_game_projectiles",
            "update_game_cleanup",
            "update_game_post_step_gpu_mutations",
            "gpu_world_mutation_flush",
            "update_game_ctx_clear",
        )
        snapshot: dict[str, float] = {}
        for key in keys:
            snapshot[f"{key}_last_ms"] = round(self.perf_last_ms(key), 3)
            snapshot[f"{key}_avg_ms"] = round(self.perf_avg_ms(key), 3)
        snapshot["update_game_world_step_last_ms"] = snapshot.get("update_game_world_step_submit_last_ms", 0.0)
        snapshot["update_game_world_step_avg_ms"] = snapshot.get("update_game_world_step_submit_avg_ms", 0.0)
        snapshot["update_game_schedule_last_ms"] = snapshot.get("update_game_schedule_feedback_last_ms", 0.0)
        snapshot["update_game_schedule_avg_ms"] = snapshot.get("update_game_schedule_feedback_avg_ms", 0.0)
        snapshot["world_step_submit_cpu_last_ms"] = snapshot.get("update_game_world_step_submit_last_ms", 0.0)
        snapshot["world_step_submit_cpu_avg_ms"] = snapshot.get("update_game_world_step_submit_avg_ms", 0.0)
        snapshot["gpu_world_mutation_pending"] = float(self._world_mutation_sink().pending_count)
        snapshot["gpu_world_mutation_last_flush_count"] = float(getattr(self, "_last_gpu_world_mutation_flush_count", 0))
        snapshot["snapshot_queued_request_count"] = float(len(self._snapshot_request_queue()))
        snapshot_status = dict((getattr(self, "_latest_gpu_world_status", None) or {}).get("snapshot") or {})
        snapshot["snapshot_pending_token_count"] = float(snapshot_status.get("pending", 0))
        return snapshot

    @property
    def debug_overlay_enabled(self) -> bool:
        return bool(getattr(self, "_debug_overlay_enabled", False))

    _MAX_DT = 1.0 / 15.0  # cap large pauses without forcing a long catch-up loop
    _SIM_DT = 1.0 / 60.0  # GPU simulation fixed step (60Hz)
    _TICK_PUMP_DT = 1.0 / 120.0
    _MAX_STEPS_PER_FRAME = 5

    def _tick(self, dt: float) -> None:
        """Main tick loop. Simulation runs at 60Hz while the pump runs at 120Hz."""
        tick_started_at = perf_counter()
        dt = min(dt, self._MAX_DT)
        self._last_dt = dt
        # Track _tick call frequency
        if not hasattr(self, "_tick_call_count"):
            self._tick_call_count = 0
            self._tick_call_start = tick_started_at
        self._tick_call_count += 1
        elapsed_since_start = tick_started_at - self._tick_call_start
        if elapsed_since_start >= 1.0:
            self._tick_call_rate = self._tick_call_count / elapsed_since_start
            self._tick_call_count = 0
            self._tick_call_start = tick_started_at
        # Debug: log accumulator and dt periodically
        if not hasattr(self, "_debug_tick_history"):
            self._debug_tick_history: list[dict[str, float]] = []
        self._debug_tick_history.append({"dt": dt, "acc_before": self._sim_accumulator})
        if len(self._debug_tick_history) > 300:
            self._debug_tick_history = self._debug_tick_history[-300:]
        self._drain_owner_input_events()
        if self.current_screen != "game":
            return
        if self.world is None:
            self._game_screen.update(dt)
            return
        # Always handle input every frame for responsiveness
        input_started_at = perf_counter()
        self._handle_hero_input(dt)
        input_finished_at = perf_counter()
        self._sim_accumulator += dt
        steps = 0
        sim_started_at = perf_counter()
        while self._sim_accumulator >= self._SIM_DT and steps < self._MAX_STEPS_PER_FRAME:
            self._sim_accumulator -= self._SIM_DT
            self._game_screen.update(self._SIM_DT)
            steps += 1
        if steps >= self._MAX_STEPS_PER_FRAME and self._sim_accumulator >= self._SIM_DT:
            self._sim_accumulator = 0.0
        sim_finished_at = perf_counter()
        # Camera follows hero every render tick, independent of sim cadence.
        camera_override = getattr(self, "_experiment_camera_override", None)
        if camera_override is not None:
            target_x = int(camera_override[0])
            target_y = int(camera_override[1])
        else:
            target_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
            target_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        camera_started_at = perf_counter()
        self._submit_gpu_camera_follow(target_x, target_y, dt=dt)
        camera_finished_at = perf_counter()
        bg_io_started_at = perf_counter()
        self._submit_gpu_background_io_service()
        present_started_at = perf_counter()
        self._submit_gpu_present_frame()
        tick_finished_at = perf_counter()
        self.invalid = not bool(getattr(self, "gpu_owner_active", False))
        self._set_perf_ms("main_tick_input", (input_finished_at - input_started_at) * 1000.0)
        self._set_perf_ms("main_tick_sim", (sim_finished_at - sim_started_at) * 1000.0)
        self._set_perf_ms("main_tick_camera", (camera_finished_at - camera_started_at) * 1000.0)
        self._set_perf_ms("main_tick_background_io", (present_started_at - bg_io_started_at) * 1000.0)
        self._set_perf_ms("main_tick_total", (tick_finished_at - tick_started_at) * 1000.0)

    def on_draw(self) -> None:
        screen = self.screens.get(self.current_screen)
        if screen is not None:
            draw_started_at = perf_counter()
            screen.on_draw()
            draw_finished_at = perf_counter()
            self._set_perf_ms("draw_total", (draw_finished_at - draw_started_at) * 1000.0)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.add(symbol)
        log.debug("[app] key_press: symbol=%d screen=%s", symbol, self.current_screen)

        # Global: ENTER on title -> init world + switch to game
        if self.current_screen == "title" and symbol == key.ENTER:
            log.info("[app] ENTER pressed on title, initializing game world...")
            try:
                self._init_game_world()
                self.change_screen("game")
                log.info("[app] switched to game screen")
            except Exception:
                import traceback
                log.error("[app] _init_game_world failed:\n%s", traceback.format_exc())
            return

        # F3: toggle perf/debug overlay
        if symbol == key.F3:
            self._debug_overlay_enabled = not self._debug_overlay_enabled
            return
        if symbol == key.F6:
            self.entity_manager.debug_collision = not self.entity_manager.debug_collision
            if not self.entity_manager.debug_collision:
                self.entity_manager.last_debug = None
            return
        if symbol == key.F4:
            self.view_mode = (
                DebugViewMode.MATERIAL
                if self.view_mode == DebugViewMode.TEMPERATURE
                else DebugViewMode.TEMPERATURE
            )
            return
        if symbol == key.F5:
            self.view_mode = (
                DebugViewMode.MATERIAL
                if self.view_mode == DebugViewMode.PRESSURE
                else DebugViewMode.PRESSURE
            )
            return

        # Spell hotkeys 1-8 set selected spell (don't cast immediately)
        if self.current_screen == "game" and key._1 <= symbol <= key._8:
            self.hero.selected_spell_idx = symbol - key._1

        screen = self.screens.get(self.current_screen)
        if screen is not None:
            screen.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.discard(symbol)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        screen = self.screens.get(self.current_screen)
        if screen is not None:
            screen.on_mouse_press(x, y, button, modifiers)

    def on_close(self) -> None:
        self._debug.stop()
        self._close_game_world()
        super().on_close()


def run_game(*, seed: int = 42, cell_scale: int = cfg.CELL_SCALE) -> None:
    app = GameApp(seed=seed, cell_scale=cell_scale)
    app.set_minimum_size(400, 300)
    # Match redraw scheduling to the 120Hz pump so Windows timer jitter
    # does not drag presentation cadence below the 60Hz sim target.
    pyglet.app.run(interval=1.0 / 120.0)
