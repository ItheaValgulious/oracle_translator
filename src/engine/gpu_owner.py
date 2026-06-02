"""Phase 6: dedicated GPU owner thread that serializes all GL/world ops.

This module provides the *channel infrastructure* that goal.md §1 calls
for: a single dedicated thread that owns world stepping, render
submission, paging upload, GPU writeback, and snapshot
submission/polling. The main thread communicates with it only through
explicit command/result channels.

The actual migration of every existing call site to dispatch through
this owner happens incrementally. Phase 6 ships the owner runtime plus
its public protocol so subsequent phases can wire callers over without
touching the threading model again.

Design:

- ``GpuOwnerThread`` runs a single worker thread.
- Each command is a ``GpuCommand`` (op type + payload). The thread runs
  registered handlers in submission order, FIFO.
- Each command returns a ``GpuResult``. Callers receive a Future-like
  ``CommandHandle`` they can ``.wait(timeout)`` on. ``fire_and_forget``
  submits without keeping a result handle.
- The thread itself never calls back into the main thread synchronously.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
import uuid
from array import array
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable, ContextManager, Optional

from src.engine.snapshot_mailbox import SnapshotEnvelope, SnapshotMailboxRegistry

log = logging.getLogger(__name__)


class GpuCommandType(enum.IntEnum):
    NOOP = 0
    STEP_WORLD = 1
    APPLY_PAGE = 2
    SUBMIT_SNAPSHOT = 3
    POLL_SNAPSHOT = 4
    RUN_CALLABLE = 5
    APPLY_WORLD_MUTATIONS = 6
    SYNC_ENTITY_STATE = 7
    RENDER_FRAME = 8
    SCHEDULE_PREFETCH = 9
    CAMERA_FOLLOW = 10
    CAMERA_TELEPORT = 11
    SERVICE_BACKGROUND_IO = 12
    CLOSE_WORLD = 13
    CREATE_WORLD = 14
    WORLD_STATUS = 15
    RUN_WORLD_CALLABLE = 16
    SET_GPU_OPTIONS = 17
    SERVICE_SNAPSHOTS = 18
    SHUTDOWN = 99


@dataclass
class GpuCommand:
    op: GpuCommandType
    command_id: str = ""
    payload: Any = None
    fn: Optional[Callable[..., Any]] = None


@dataclass
class GpuResult:
    command_id: str
    ok: bool
    value: Any = None
    error: str = ""


@dataclass(frozen=True)
class PaintWorldCommand:
    world_x: int
    world_y: int
    radius: int
    family_id: str | None
    variant_id: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectPressureWorldCommand:
    world_x: int
    world_y: int
    radius: int
    pressure: float


@dataclass(frozen=True)
class InjectPressureRingWorldCommand:
    world_x: int
    world_y: int
    inner_radius: int
    outer_radius: int
    pressure: float


@dataclass(frozen=True)
class SnapshotRequestCommand:
    entity_id: str
    world_x: int
    world_y: int
    width: int
    height: int
    submitted_at: float


@dataclass(frozen=True)
class SnapshotPollCommand:
    token: Any
    force_ready: bool = False


@dataclass(frozen=True)
class SnapshotServiceCommand:
    requests: tuple[SnapshotRequestCommand, ...] = ()


@dataclass(frozen=True)
class EntityGpuSyncCommand:
    shapes: tuple[tuple[str, int, int], ...] = ()
    states: tuple[tuple[str, int, int, bool, float], ...] = ()
    mask_rects: tuple[tuple[int, int, int, int, int], ...] = ()
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class ChunkPrefetchCommand:
    rect: Any
    margin_x: int = 0
    margin_y: int = 0
    prioritize: bool = False


@dataclass(frozen=True)
class CameraFollowCommand:
    target_x: int
    target_y: int
    dt: float


@dataclass(frozen=True)
class CameraTeleportCommand:
    target_x: int
    target_y: int
    submit_chunks: int = 8


@dataclass(frozen=True)
class RenderFrameCommand:
    view_mode: Any
    readback_rgba: bool = True
    readback_present_rgb: bool = False
    present: bool = False
    window_size: tuple[int, int] | None = None
    hero: Any = None
    enemies: tuple[Any, ...] = ()
    projectiles: tuple[Any, ...] = ()
    dt: float = 1.0 / 60.0
    sim_fps: float = 0.0
    gpu_tick_rate: float = 0.0
    overlay_lines: tuple[str, ...] = ()
    collision_debug: Any = None
    console_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuFramePayload:
    width: int
    height: int
    rgba: bytes
    uv_rect: tuple[float, float, float, float]
    status: dict[str, Any]
    view_mode: Any


@dataclass(frozen=True)
class RenderActorState:
    entity_id: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    facing_right: bool = True
    state: str = "idle"
    hp: float = 1.0
    max_hp: float = 1.0
    mp: float = 0.0
    is_alive: bool = True
    damage_flash_timer: float = 0.0

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class WorldStatusCommand:
    include_gpu_debug: bool = True


@dataclass(frozen=True)
class GpuOptionsCommand:
    skip_liquid_physics: bool | None = None


@dataclass(frozen=True)
class WorldCallableCommand:
    fn: Callable[[Any], Any]


@dataclass(frozen=True)
class CreateWorldCommand:
    world_factory: Callable[..., Any]
    context_factory: Callable[[], Any] | None = None
    pass_context: bool = True
    close_existing: bool = True
    snapshot_registry: SnapshotMailboxRegistry | None = None


@dataclass
class PendingSnapshotServiceState:
    request: SnapshotRequestCommand
    token: Any


WorldMutationCommand = (
    PaintWorldCommand
    | InjectPressureWorldCommand
    | InjectPressureRingWorldCommand
)


def build_world_status(world: Any, *, include_gpu_debug: bool = True) -> dict[str, Any]:
    """Build a serializable read-only status snapshot for command results."""
    status: dict[str, Any] = {
        "camera": (int(world.camera_x), int(world.camera_y)),
        "active_origin": (int(world.active_origin_x), int(world.active_origin_y)),
        "active_size": (int(world.active_width), int(world.active_height)),
        "viewport_size": (int(world.viewport_width), int(world.viewport_height)),
        "world_size": (int(world.world_width), int(world.world_height)),
    }
    chunk_cache = getattr(world, "chunk_cache", None)
    if chunk_cache is not None and hasattr(chunk_cache, "snapshot_stats"):
        chunk_stats = chunk_cache.snapshot_stats()
        status["chunk"] = {
            "cached": int(chunk_stats.cached_chunks),
            "clean_resident": int(chunk_stats.clean_resident_chunks),
            "dirty_resident": int(chunk_stats.dirty_resident_chunks),
            "prefetch_queued": int(chunk_stats.prefetch_queued),
            "prefetch_inflight": int(chunk_stats.prefetch_inflight),
            "queued_read": int(chunk_stats.queued_read_count),
            "queued_write": int(chunk_stats.queued_write_count),
            "queued_generate": int(chunk_stats.queued_generate_count),
            "inflight_io": int(chunk_stats.inflight_io_count),
            "inflight_generation": int(chunk_stats.inflight_generation_count),
            "worker_fallback": int(chunk_stats.worker_fallback_count),
            "sync_blocking_fetch": int(chunk_stats.sync_blocking_fetch_count),
            "disk_load_count": int(chunk_stats.disk_load_count),
            "generate_count": int(chunk_stats.generate_count),
            "save_count": int(chunk_stats.save_count),
            "disk_load_last_ms": round(float(chunk_stats.disk_load_last_seconds) * 1000.0, 3),
            "generate_last_ms": round(float(chunk_stats.generate_last_seconds) * 1000.0, 3),
            "save_last_ms": round(float(chunk_stats.save_last_seconds) * 1000.0, 3),
            "disk_load_total_ms": round(float(chunk_stats.disk_load_total_seconds) * 1000.0, 3),
            "generate_total_ms": round(float(chunk_stats.generate_total_seconds) * 1000.0, 3),
            "save_total_ms": round(float(chunk_stats.save_total_seconds) * 1000.0, 3),
        }
        if hasattr(chunk_cache, "chunk_residency_state"):
            chunk_states: dict[str, int] = {}
            try:
                active_origin_x = int(getattr(world, "active_origin_x", 0))
                active_origin_y = int(getattr(world, "active_origin_y", 0))
                active_width = int(getattr(world, "active_width", 0))
                active_height = int(getattr(world, "active_height", 0))
                chunk_size = int(getattr(chunk_cache, "chunk_size", 1) or 1)
                min_cx = max(0, active_origin_x) // chunk_size
                max_cx = max(0, max(active_origin_x, active_origin_x + active_width - 1)) // chunk_size
                min_cy = max(0, active_origin_y) // chunk_size
                max_cy = max(0, max(active_origin_y, active_origin_y + active_height - 1)) // chunk_size
                for chunk_y in range(min_cy, max_cy + 1):
                    for chunk_x in range(min_cx, max_cx + 1):
                        chunk_states[f"{chunk_x},{chunk_y}"] = int(chunk_cache.chunk_residency_state(chunk_x, chunk_y))
            except Exception:
                log.debug("[gpu-owner] failed to collect chunk residency states", exc_info=True)
            status["chunk_cache_states"] = chunk_states
    paging_stats = getattr(world, "paging_stats", None)
    if paging_stats is not None:
        status["paging"] = {
            "shift_ms": round(float(world.shift_time_last_ms()), 3),
            "incoming_load_ms": round(float(world.incoming_load_time_last_ms()), 3),
            "evict_stage_ms": round(float(world.stage_time_last_ms()), 3),
            "overlap_ms": round(float(world.overlap_copy_time_last_ms()), 3),
            "overlap_fx_ms": round(float(world.overlap_transient_copy_time_last_ms()), 3),
            "clear_fx_ms": round(float(world.incoming_transient_clear_time_last_ms()), 3),
            "anchor_build_ms": round(float(world.anchor_build_time_last_ms()), 3),
            "anchor_upload_ms": round(float(world.anchor_upload_time_last_ms()), 3),
            "shift_count": int(paging_stats.shift_count),
            "gpu_writeback_queue_depth": int(getattr(world, "gpu_writeback_queue_depth", 0)),
            "active_chunk_patch_queue_depth": int(getattr(world, "active_chunk_patch_queue_depth", 0)),
            "pending_writebacks": int(getattr(world, "pending_writeback_count", 0)),
            "last_shift_cache_hits": int(paging_stats.last_shift_cache_hits),
            "last_shift_empty_hits": int(paging_stats.last_shift_empty_hits),
            "last_shift_inflight_wait_hits": int(paging_stats.last_shift_inflight_wait_hits),
            "last_shift_disk_loads": int(paging_stats.last_shift_disk_loads),
            "last_shift_generates": int(paging_stats.last_shift_generates),
            "last_shift_saves": int(paging_stats.last_shift_saves),
            "last_shift_disk_load_ms": round(float(paging_stats.last_shift_disk_load_seconds) * 1000.0, 3),
            "last_shift_generate_ms": round(float(paging_stats.last_shift_generate_seconds) * 1000.0, 3),
            "last_shift_save_ms": round(float(paging_stats.last_shift_save_seconds) * 1000.0, 3),
        }
    gpu = getattr(world, "gpu_simulator", None)
    if include_gpu_debug and gpu is not None:
        status["gpu"] = {
            "step_index": int(getattr(gpu, "step_index", 0)),
            "has_support": bool(getattr(gpu, "_has_support", True)),
            "has_reactions": bool(getattr(gpu, "_has_reactions", True)),
            "has_gas": bool(getattr(gpu, "_has_gas", True)),
            "has_thermal_phase": bool(getattr(gpu, "_has_thermal_phase", True)),
            "has_collapse": bool(getattr(gpu, "_has_collapse", True)),
            "skip_liquid_physics": bool(getattr(gpu, "_skip_liquid_physics", False)),
            "step_timings": dict(getattr(gpu, "_last_step_timings", {}) or {}),
        }
    return status


def apply_world_mutation_command(world: Any, command: WorldMutationCommand) -> None:
    """Apply one queued world mutation on the GPU owner boundary."""
    if isinstance(command, PaintWorldCommand):
        world.paint_world(
            command.world_x,
            command.world_y,
            command.radius,
            command.family_id,
            command.variant_id,
            overrides=dict(command.overrides),
        )
        return
    if isinstance(command, InjectPressureWorldCommand):
        world.inject_pressure_world(
            command.world_x,
            command.world_y,
            command.radius,
            command.pressure,
        )
        return
    if isinstance(command, InjectPressureRingWorldCommand):
        world.inject_pressure_ring_world(
            command.world_x,
            command.world_y,
            command.inner_radius,
            command.outer_radius,
            command.pressure,
        )
        return
    raise TypeError(f"unsupported world mutation command: {type(command)!r}")


class GpuWorldMutationBuffer:
    """Collects GPU-world writes for explicit owner-boundary flushing.

    The gameplay layer can use this object as a minimal world-like sink
    because it exposes ``paint_world`` and pressure injection methods.
    """

    def __init__(self) -> None:
        self._commands: deque[WorldMutationCommand] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._commands)

    def clear(self) -> None:
        self._commands.clear()

    def paint_world(
        self,
        world_x: int,
        world_y: int,
        radius: int,
        family_id: str | None,
        variant_id: str | None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self._commands.append(
            PaintWorldCommand(
                world_x=int(world_x),
                world_y=int(world_y),
                radius=int(radius),
                family_id=family_id,
                variant_id=variant_id,
                overrides=dict(overrides or {}),
            )
        )

    def inject_pressure_world(self, world_x: int, world_y: int, radius: int, pressure: float) -> None:
        self._commands.append(
            InjectPressureWorldCommand(
                world_x=int(world_x),
                world_y=int(world_y),
                radius=int(radius),
                pressure=float(pressure),
            )
        )

    def inject_pressure_ring_world(
        self,
        world_x: int,
        world_y: int,
        inner_radius: int,
        outer_radius: int,
        pressure: float,
    ) -> None:
        self._commands.append(
            InjectPressureRingWorldCommand(
                world_x=int(world_x),
                world_y=int(world_y),
                inner_radius=int(inner_radius),
                outer_radius=int(outer_radius),
                pressure=float(pressure),
            )
        )

    def drain(self) -> list[WorldMutationCommand]:
        commands = list(self._commands)
        self._commands.clear()
        return commands

    def flush_to_world(self, world: Any) -> int:
        commands = self.drain()
        for command in commands:
            apply_world_mutation_command(world, command)
        return len(commands)


class GpuWorldCommandHandlers:
    """Typed command handlers for a world owned by ``GpuOwnerThread``."""

    def __init__(self, world: Any) -> None:
        self.world = world

    def register(self, owner: "GpuOwnerThread") -> None:
        owner.register_handler(GpuCommandType.APPLY_WORLD_MUTATIONS, self.apply_world_mutations)
        owner.register_handler(GpuCommandType.STEP_WORLD, self.step_world)
        owner.register_handler(GpuCommandType.SUBMIT_SNAPSHOT, self.submit_snapshot)
        owner.register_handler(GpuCommandType.POLL_SNAPSHOT, self.poll_snapshot)
        owner.register_handler(GpuCommandType.SYNC_ENTITY_STATE, self.sync_entity_state)
        owner.register_handler(GpuCommandType.RENDER_FRAME, self.render_frame)
        owner.register_handler(GpuCommandType.SCHEDULE_PREFETCH, self.schedule_prefetch)
        owner.register_handler(GpuCommandType.CAMERA_FOLLOW, self.camera_follow)
        owner.register_handler(GpuCommandType.CAMERA_TELEPORT, self.camera_teleport)
        owner.register_handler(GpuCommandType.SERVICE_BACKGROUND_IO, self.service_background_io)
        owner.register_handler(GpuCommandType.CLOSE_WORLD, self.close_world)
        owner.register_handler(GpuCommandType.WORLD_STATUS, self.world_status)
        owner.register_handler(GpuCommandType.RUN_WORLD_CALLABLE, self.run_world_callable)
        owner.register_handler(GpuCommandType.SET_GPU_OPTIONS, self.set_gpu_options)

    def apply_world_mutations(self, payload: Any) -> int:
        commands = list(payload or [])
        for command in commands:
            apply_world_mutation_command(self.world, command)
        return len(commands)

    def step_world(self, payload: Any) -> None:
        dt = float(payload.get("dt", 0.0)) if isinstance(payload, dict) else float(payload)
        self.world.step(dt)

    def submit_snapshot(self, payload: SnapshotRequestCommand) -> Any:
        return self.world.request_snapshot_cells_region_world(
            entity_id=payload.entity_id,
            world_x=payload.world_x,
            world_y=payload.world_y,
            width=payload.width,
            height=payload.height,
        )

    def poll_snapshot(self, payload: Any) -> Any:
        if isinstance(payload, SnapshotPollCommand):
            return self.world.poll_snapshot_cells_region_world(payload.token, force_ready=payload.force_ready)
        return self.world.poll_snapshot_cells_region_world(payload)

    def sync_entity_state(self, payload: EntityGpuSyncCommand) -> dict[str, int]:
        gpu = getattr(self.world, "gpu_simulator", None)
        if gpu is None:
            return {"shapes": 0, "states": 0, "mask_rects": 0}
        for entity_id, width, height in payload.shapes:
            gpu.register_entity_shape(entity_id, int(width), int(height))
        if payload.states:
            gpu.upload_entity_states(
                list(payload.states),
                origin_x=int(payload.origin_x),
                origin_y=int(payload.origin_y),
            )
        gpu.update_entity_mask(list(payload.mask_rects))
        return {
            "shapes": len(payload.shapes),
            "states": len(payload.states),
            "mask_rects": len(payload.mask_rects),
        }

    def render_frame(self, payload: Any) -> Any:
        readback_rgba = False
        if isinstance(payload, RenderFrameCommand):
            view_mode = payload.view_mode
            readback_rgba = bool(payload.readback_rgba)
        elif isinstance(payload, dict):
            view_mode = payload.get("view_mode")
            readback_rgba = bool(payload.get("readback_rgba", False))
        else:
            view_mode = payload
        frame = self.world.render(view_mode)
        if not readback_rgba:
            return frame
        read = getattr(frame, "read", None)
        if not callable(read):
            return frame
        size = getattr(frame, "size", None)
        if size is None:
            size = (int(self.world.active_width), int(self.world.active_height))
        uv_rect = self.world.visible_uv_rect()
        return GpuFramePayload(
            width=int(size[0]),
            height=int(size[1]),
            rgba=bytes(read(alignment=1)),
            uv_rect=tuple(float(value) for value in uv_rect),
            status=build_world_status(self.world, include_gpu_debug=True),
            view_mode=view_mode,
        )

    def schedule_prefetch(self, payload: ChunkPrefetchCommand) -> None:
        self.world.chunk_cache.schedule_prefetch_for_rect(
            payload.rect,
            margin_x=int(payload.margin_x),
            margin_y=int(payload.margin_y),
            prioritize=bool(payload.prioritize),
        )

    def camera_follow(self, payload: CameraFollowCommand) -> dict[str, int]:
        prefetch_camera_region = getattr(self.world, "prefetch_camera_region", None)
        if callable(prefetch_camera_region):
            prefetch_camera_region(int(payload.target_x), int(payload.target_y), submit_chunks=4)
        self.world.pan_camera(
            int(payload.target_x) - int(self.world.camera_x),
            int(payload.target_y) - int(self.world.camera_y),
        )
        self.world.mark_camera_activity(False, dt=float(payload.dt))
        return {
            "camera_x": int(self.world.camera_x),
            "camera_y": int(self.world.camera_y),
        }

    def camera_teleport(self, payload: CameraTeleportCommand) -> dict[str, int]:
        target_x = int(payload.target_x)
        target_y = int(payload.target_y)
        if (
            abs(target_x - int(self.world.camera_x)) >= int(self.world.active_width)
            or abs(target_y - int(self.world.camera_y)) >= int(self.world.active_height)
        ):
            prefetch_camera_region = getattr(self.world, "prefetch_camera_region", None)
            if callable(prefetch_camera_region):
                prefetch_camera_region(target_x, target_y, submit_chunks=int(payload.submit_chunks))
        self.world.set_camera(target_x, target_y)
        return {
            "camera_x": int(self.world.camera_x),
            "camera_y": int(self.world.camera_y),
        }

    def service_background_io(self, payload: Any = None) -> None:
        del payload
        self.world.service_background_io()

    def close_world(self, payload: Any = None) -> None:
        del payload
        self.world.close()

    def world_status(self, payload: Any = None) -> dict[str, Any]:
        include_gpu_debug = True
        if isinstance(payload, WorldStatusCommand):
            include_gpu_debug = bool(payload.include_gpu_debug)
        return build_world_status(self.world, include_gpu_debug=include_gpu_debug)

    def set_gpu_options(self, payload: GpuOptionsCommand) -> dict[str, bool]:
        gpu = getattr(self.world, "gpu_simulator", None)
        if gpu is None:
            return {}
        applied: dict[str, bool] = {}
        if isinstance(payload, GpuOptionsCommand) and payload.skip_liquid_physics is not None:
            setter = getattr(gpu, "set_skip_liquid_physics", None)
            if callable(setter):
                value = bool(payload.skip_liquid_physics)
                setter(value)
                applied["skip_liquid_physics"] = value
        return applied

    def run_world_callable(self, payload: WorldCallableCommand) -> Any:
        return payload.fn(self.world)


class GpuOwnedWorldRuntime:
    """Owner-thread world lifecycle handler.

    This runtime is the stricter migration path for goal.md: callers
    submit an explicit CREATE_WORLD command whose factories run on the
    GPU owner thread, so the GL context, world, and all GPU objects are
    created and later closed on the same thread.
    """

    def __init__(self) -> None:
        self.world: Any | None = None
        self.context: Any | None = None
        self._present_renderer: Any | None = None
        self._present_renderer_size: tuple[int, int] | None = None
        self._snapshot_registry: SnapshotMailboxRegistry | None = None
        self._pending_snapshots: dict[str, PendingSnapshotServiceState] = {}

    def register(self, owner: "GpuOwnerThread") -> None:
        owner.register_handler(GpuCommandType.CREATE_WORLD, self.create_world)
        owner.register_handler(GpuCommandType.APPLY_WORLD_MUTATIONS, self.apply_world_mutations)
        owner.register_handler(GpuCommandType.STEP_WORLD, self.step_world)
        owner.register_handler(GpuCommandType.SUBMIT_SNAPSHOT, self.submit_snapshot)
        owner.register_handler(GpuCommandType.POLL_SNAPSHOT, self.poll_snapshot)
        owner.register_handler(GpuCommandType.SYNC_ENTITY_STATE, self.sync_entity_state)
        owner.register_handler(GpuCommandType.RENDER_FRAME, self.render_frame)
        owner.register_handler(GpuCommandType.SCHEDULE_PREFETCH, self.schedule_prefetch)
        owner.register_handler(GpuCommandType.CAMERA_FOLLOW, self.camera_follow)
        owner.register_handler(GpuCommandType.CAMERA_TELEPORT, self.camera_teleport)
        owner.register_handler(GpuCommandType.SERVICE_BACKGROUND_IO, self.service_background_io)
        owner.register_handler(GpuCommandType.CLOSE_WORLD, self.close_world)
        owner.register_handler(GpuCommandType.WORLD_STATUS, self.world_status)
        owner.register_handler(GpuCommandType.RUN_WORLD_CALLABLE, self.run_world_callable)
        owner.register_handler(GpuCommandType.SET_GPU_OPTIONS, self.set_gpu_options)
        owner.register_handler(GpuCommandType.SERVICE_SNAPSHOTS, self.service_snapshots)

    def _require_world(self) -> Any:
        if self.world is None:
            raise RuntimeError("GPU owner world has not been created")
        return self.world

    def _handlers(self) -> GpuWorldCommandHandlers:
        return GpuWorldCommandHandlers(self._require_world())

    def create_world(self, payload: CreateWorldCommand) -> dict[str, Any]:
        if self.world is not None and payload.close_existing:
            self.close_world()
        if self.world is not None:
            raise RuntimeError("GPU owner world already exists")
        context = payload.context_factory() if payload.context_factory is not None else None
        try:
            if payload.pass_context:
                world = payload.world_factory(context)
            else:
                world = payload.world_factory()
        except Exception:
            if isinstance(context, dict):
                window = context.get("window")
                close = getattr(window, "close", None)
                if callable(close):
                    close()
                release_target = context.get("ctx")
            else:
                release_target = context
            release = getattr(release_target, "release", None)
            if callable(release):
                release()
            raise
        self.context = context
        self.world = world
        self._snapshot_registry = payload.snapshot_registry
        self._pending_snapshots.clear()
        return {
            "world_created": True,
            "context_created": context is not None,
        }

    def apply_world_mutations(self, payload: Any) -> int:
        return self._handlers().apply_world_mutations(payload)

    def step_world(self, payload: Any) -> None:
        return self._handlers().step_world(payload)

    def submit_snapshot(self, payload: SnapshotRequestCommand) -> Any:
        return self._handlers().submit_snapshot(payload)

    def poll_snapshot(self, payload: Any) -> Any:
        return self._handlers().poll_snapshot(payload)

    def sync_entity_state(self, payload: EntityGpuSyncCommand) -> dict[str, int]:
        return self._handlers().sync_entity_state(payload)

    def render_frame(self, payload: Any) -> Any:
        if isinstance(payload, RenderFrameCommand) and payload.present:
            return self.present_frame(payload)
        return self._handlers().render_frame(payload)

    def _context_item(self, name: str, default: Any = None) -> Any:
        context = self.context
        if isinstance(context, dict):
            return context.get(name, default)
        return default

    def _moderngl_context(self) -> Any:
        context = self.context
        if isinstance(context, dict):
            ctx = context.get("ctx")
            if ctx is not None:
                return ctx
        return context

    def _ensure_present_renderer(self, width: int, height: int) -> Any:
        size = (int(width), int(height))
        if self._present_renderer is not None and self._present_renderer_size == size:
            return self._present_renderer
        from src.game.renderer import GameRenderer

        renderer = GameRenderer(self._moderngl_context(), size[0], size[1])
        self._present_renderer = renderer
        self._present_renderer_size = size
        return renderer

    def _pump_owner_window_events(self) -> None:
        window = self._context_item("window")
        if window is None:
            return
        dispatch_pending = getattr(window, "dispatch_pending_events", None)
        previous_allow_dispatch = getattr(window, "_allow_dispatch_event", None)
        try:
            try:
                setattr(window, "_allow_dispatch_event", True)
            except Exception:
                pass
            if callable(dispatch_pending):
                dispatch_pending()
            try:
                from pyglet.window import win32 as pyglet_win32

                msg = pyglet_win32.MSG()
                while pyglet_win32._user32.PeekMessageW(
                    pyglet_win32.byref(msg),
                    0,
                    0,
                    0,
                    pyglet_win32.constants.PM_REMOVE,
                ):
                    pyglet_win32._user32.TranslateMessage(pyglet_win32.byref(msg))
                    pyglet_win32._user32.DispatchMessageW(pyglet_win32.byref(msg))
            except Exception:
                log.debug("[gpu-owner] owner window event pump skipped", exc_info=True)
        finally:
            if previous_allow_dispatch is not None:
                try:
                    setattr(window, "_allow_dispatch_event", previous_allow_dispatch)
                except Exception:
                    pass

    def _draw_owner_overlay_lines(self, renderer: Any, payload: RenderFrameCommand) -> None:
        lines = [
            f"R {float(getattr(renderer, '_render_fps', 0.0)):.1f} S {float(payload.sim_fps):.1f} GPU {float(payload.gpu_tick_rate):.1f}/s",
        ]
        lines.extend(str(line) for line in payload.overlay_lines)
        bg = array("f")
        text = array("f")
        line_h = 18.0
        top = float(getattr(renderer, "window_height", 0)) - 24.0
        max_chars = max((len(line) for line in lines), default=1)
        panel_w = min(float(getattr(renderer, "window_width", 0)) - 12.0, 10.0 + max_chars * 12.0)
        panel_h = 10.0 + len(lines) * line_h
        renderer._append_rect(bg, 6.0, top - panel_h + 6.0, 6.0 + panel_w, top + 12.0, (0, 0, 0), opacity=180)
        for index, line in enumerate(lines):
            renderer._append_text(text, 12.0, top - index * line_h, line, (255, 230, 120), pixel_size=2.0)
        renderer._flush_overlay(bg)
        renderer._flush_overlay(text)

    def _draw_owner_console_overlay(self, renderer: Any, payload: RenderFrameCommand) -> None:
        lines = tuple(str(line) for line in payload.console_lines)
        if not lines:
            return
        bg = array("f")
        text = array("f")
        window_width = float(getattr(renderer, "window_width", 0))
        window_height = float(getattr(renderer, "window_height", 0))
        line_h = 20.0
        max_chars = max((len(line) for line in lines), default=1)
        panel_w = min(window_width - 80.0, 40.0 + max_chars * 12.0)
        panel_h = 24.0 + len(lines) * line_h
        panel_x0 = 32.0
        panel_y0 = max(24.0, window_height - panel_h - 40.0)
        renderer._append_rect(bg, panel_x0, panel_y0, panel_x0 + panel_w, panel_y0 + panel_h, (0, 0, 0), opacity=208)
        y = panel_y0 + panel_h - 28.0
        for line in lines:
            color = (255, 220, 110) if line.startswith("> ") else (220, 220, 210)
            renderer._append_text(text, panel_x0 + 14.0, y, line, color, pixel_size=2.2)
            y -= line_h
        renderer._flush_overlay(bg)
        renderer._flush_overlay(text)

    def present_frame(self, payload: RenderFrameCommand) -> dict[str, Any]:
        world = self._require_world()
        ctx = self._moderngl_context()
        window = self._context_item("window")
        self._pump_owner_window_events()
        width, height = payload.window_size or (
            int(getattr(world, "viewport_width", 1)),
            int(getattr(world, "viewport_height", 1)),
        )
        width = max(1, int(width))
        height = max(1, int(height))
        screen = getattr(ctx, "screen", None)
        use = getattr(screen, "use", None)
        if callable(use):
            use()
        ctx.viewport = (0, 0, width, height)
        ctx.clear(0.04, 0.05, 0.07, 1.0)
        renderer = self._ensure_present_renderer(width, height)
        hero = payload.hero
        if hero is not None:
            renderer.draw(
                world,
                hero,
                int(getattr(world, "camera_x", 0)),
                int(getattr(world, "camera_y", 0)),
                payload.view_mode,
                dt=float(payload.dt),
                enemies={getattr(enemy, "entity_id", str(index)): enemy for index, enemy in enumerate(payload.enemies)},
                projectiles=list(payload.projectiles),
            )
        else:
            renderer.draw_grid(world, payload.view_mode)
        if payload.collision_debug is not None:
            renderer.draw_debug_collision(
                int(getattr(world, "camera_x", 0)),
                int(getattr(world, "camera_y", 0)),
                payload.collision_debug,
            )
        self._draw_owner_overlay_lines(renderer, payload)
        self._draw_owner_console_overlay(renderer, payload)
        window_context = self._context_item("window_context")
        flip = getattr(window_context, "flip", None)
        if callable(flip):
            flip()
        else:
            flip = getattr(window, "flip", None)
            if callable(flip):
                flip()
        screen_rgb = None
        if bool(payload.readback_present_rgb):
            read = getattr(screen, "read", None)
            if callable(read):
                screen_rgb = bytes(
                    read(
                        viewport=(0, 0, width, height),
                        components=3,
                        alignment=1,
                    )
                )
        status = build_world_status(world, include_gpu_debug=True)
        status["snapshot"] = {
            "pending": len(self._pending_snapshots),
        }
        return {
            "presented": True,
            "width": width,
            "height": height,
            "overlay_line_count": 1 + len(tuple(payload.overlay_lines)),
            "enemy_overlay_count": len(tuple(payload.enemies)),
            "projectile_overlay_count": len(tuple(payload.projectiles)),
            "screen_rgb": screen_rgb,
            "status": status,
        }

    def schedule_prefetch(self, payload: ChunkPrefetchCommand) -> None:
        return self._handlers().schedule_prefetch(payload)

    def camera_follow(self, payload: CameraFollowCommand) -> dict[str, int]:
        return self._handlers().camera_follow(payload)

    def camera_teleport(self, payload: CameraTeleportCommand) -> dict[str, int]:
        return self._handlers().camera_teleport(payload)

    def service_background_io(self, payload: Any = None) -> None:
        return self._handlers().service_background_io(payload)

    def close_world(self, payload: Any = None) -> dict[str, bool]:
        del payload
        world = self.world
        context = self.context
        self.world = None
        self.context = None
        self._present_renderer = None
        self._present_renderer_size = None
        if world is not None:
            world.close()
        if isinstance(context, dict):
            window = context.get("window")
            close = getattr(window, "close", None)
            if callable(close):
                try:
                    from pyglet import app as pyglet_app

                    pyglet_app.windows.add(window)
                except Exception:
                    log.debug("[gpu-owner] could not re-register owner window before close", exc_info=True)
                close()
        release_enabled = not isinstance(context, dict) or bool(context.get("release_on_close", True))
        release_target = context.get("ctx") if isinstance(context, dict) else context
        release = getattr(release_target, "release", None)
        if release_enabled and callable(release):
            release()
        return {
            "world_closed": world is not None,
            "context_released": bool(release_enabled and callable(release)),
        }

    def world_status(self, payload: Any = None) -> dict[str, Any]:
        include_gpu_debug = True
        if isinstance(payload, WorldStatusCommand):
            include_gpu_debug = bool(payload.include_gpu_debug)
        status = build_world_status(self._require_world(), include_gpu_debug=include_gpu_debug)
        status["snapshot"] = {
            "pending": len(self._pending_snapshots),
        }
        return status

    def set_gpu_options(self, payload: GpuOptionsCommand) -> dict[str, bool]:
        return self._handlers().set_gpu_options(payload)

    def run_world_callable(self, payload: WorldCallableCommand) -> Any:
        return payload.fn(self._require_world())

    def service_snapshots(self, payload: Any = None) -> dict[str, Any]:
        world = self._require_world()
        requests = ()
        if isinstance(payload, SnapshotServiceCommand):
            requests = payload.requests
        elif payload:
            requests = tuple(payload)
        submitted = 0
        completed = 0
        ready_delay_total_ms = 0.0
        ready_delay_max_ms = 0.0
        for request in requests:
            entity_id = str(request.entity_id)
            if entity_id in self._pending_snapshots:
                continue
            token = world.request_snapshot_cells_region_world(
                entity_id=entity_id,
                world_x=int(request.world_x),
                world_y=int(request.world_y),
                width=int(request.width),
                height=int(request.height),
            )
            if not hasattr(token, "submitted_at"):
                try:
                    setattr(token, "submitted_at", float(request.submitted_at))
                except Exception:
                    pass
            self._pending_snapshots[entity_id] = PendingSnapshotServiceState(
                request=request,
                token=token,
            )
            submitted += 1
        registry = self._snapshot_registry
        now = time.perf_counter
        for entity_id, state in list(self._pending_snapshots.items()):
            snapshot = world.poll_snapshot_cells_region_world(state.token, force_ready=False)
            if snapshot is None:
                continue
            self._pending_snapshots.pop(entity_id, None)
            completed += 1
            submitted_at = float(getattr(state.token, "submitted_at", state.request.submitted_at))
            ready_delay_ms = max(0.0, (now() - submitted_at) * 1000.0)
            ready_delay_total_ms += ready_delay_ms
            ready_delay_max_ms = max(ready_delay_max_ms, ready_delay_ms)
            if registry is not None:
                registry.submit(
                    SnapshotEnvelope(
                        entity_id=entity_id,
                        origin_x=int(getattr(snapshot, "world_x", state.request.world_x)),
                        origin_y=int(getattr(snapshot, "world_y", state.request.world_y)),
                        width=int(getattr(snapshot, "width", state.request.width)),
                        height=int(getattr(snapshot, "height", state.request.height)),
                        tick_id=int(getattr(snapshot, "issued_step", 0) or 0),
                        packed=snapshot,
                        submitted_at=submitted_at,
                    )
                )
        return {
            "submitted": submitted,
            "completed": completed,
            "pending": len(self._pending_snapshots),
            "ready_delay_ms_max": round(ready_delay_max_ms, 3),
            "ready_delay_ms_mean": round(
                ready_delay_total_ms / completed,
                3,
            ) if completed > 0 else 0.0,
        }


class CommandHandle:
    """Future-like handle returned to the main thread.

    The handle stays in scope at the call site; the owner thread resolves
    it via ``_set_result``. ``wait`` blocks the calling thread (not the
    owner) until the result is ready or the timeout elapses.
    """

    __slots__ = ("_event", "_result")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Optional[GpuResult] = None

    def _set_result(self, result: GpuResult) -> None:
        self._result = result
        self._event.set()

    def done(self) -> bool:
        return self._event.is_set()

    def result_if_ready(self) -> Optional[GpuResult]:
        if not self._event.is_set():
            return None
        assert self._result is not None
        return self._result

    def wait(self, timeout: float = 5.0) -> GpuResult:
        if not self._event.wait(timeout=timeout):
            return GpuResult(command_id="", ok=False, error="timeout")
        assert self._result is not None
        return self._result


@dataclass
class GpuOwnerStats:
    commands_dispatched: int = 0
    commands_completed: int = 0
    commands_failed: int = 0
    last_error: str = ""


class GpuOwnerThread:
    """Owns a single worker thread that runs registered command handlers.

    Handlers are registered via ``register_handler(op, fn)``. ``fn`` is
    called as ``fn(payload)`` on the owner thread. For ``RUN_CALLABLE``
    the command's own ``fn`` field is invoked with no args — this is the
    catch-all used by call sites that don't fit a dedicated op type.
    """

    def __init__(
        self,
        *,
        before_command: Callable[[], None] | None = None,
        command_lock: ContextManager[Any] | None = None,
    ) -> None:
        self._queue: "queue.PriorityQueue[tuple[int, int, GpuCommand, Optional[CommandHandle]]]" = queue.PriorityQueue()
        self._submit_sequence = count()
        self._handlers: dict[GpuCommandType, Callable[[Any], Any]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._stats = GpuOwnerStats()
        self._owner_thread_id: Optional[int] = None
        self._before_command = before_command
        self._command_lock = command_lock

    @property
    def stats(self) -> GpuOwnerStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def owner_thread_id(self) -> Optional[int]:
        return self._owner_thread_id

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def register_handler(self, op: GpuCommandType, fn: Callable[[Any], Any]) -> None:
        self._handlers[op] = fn

    def start(self) -> None:
        if self.is_running:
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-owner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        while not self._stopped.is_set():
            try:
                _priority, _seq, cmd, handle = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if cmd.op == GpuCommandType.SHUTDOWN:
                if handle is not None:
                    handle._set_result(GpuResult(cmd.command_id, ok=True))
                break
            try:
                lock = self._command_lock
                if lock is None:
                    value = self._execute_command(cmd)
                else:
                    with lock:
                        value = self._execute_command(cmd)
                self._stats.commands_completed += 1
                if handle is not None:
                    handle._set_result(GpuResult(cmd.command_id, ok=True, value=value))
            except Exception as exc:  # noqa: BLE001
                log.exception("[gpu-owner] command %s failed", cmd.op.name)
                self._stats.commands_failed += 1
                self._stats.last_error = str(exc)
                if handle is not None:
                    handle._set_result(GpuResult(cmd.command_id, ok=False, error=str(exc)))

    def _execute_command(self, cmd: GpuCommand) -> Any:
        if self._before_command is not None:
            self._before_command()
        if cmd.op == GpuCommandType.RUN_CALLABLE:
            return cmd.fn() if cmd.fn is not None else None
        handler = self._handlers.get(cmd.op)
        if handler is None:
            raise RuntimeError(f"no handler registered for op {cmd.op.name}")
        return handler(cmd.payload)

    def _enqueue(
        self,
        command: GpuCommand,
        handle: Optional[CommandHandle],
        *,
        priority: int,
    ) -> Optional[CommandHandle]:
        if not command.command_id:
            command.command_id = uuid.uuid4().hex
        self._stats.commands_dispatched += 1
        self._queue.put((int(priority), next(self._submit_sequence), command, handle))
        return handle

    def submit(self, command: GpuCommand) -> CommandHandle:
        """Submit a command and receive a handle to wait on its result."""
        handle = CommandHandle()
        self._enqueue(command, handle, priority=10)
        return handle

    def submit_high_priority(self, command: GpuCommand) -> CommandHandle:
        """Submit an urgent command ahead of normal best-effort work."""
        handle = CommandHandle()
        self._enqueue(command, handle, priority=0)
        return handle

    def fire_and_forget(self, command: GpuCommand) -> None:
        self._enqueue(command, None, priority=10)

    def run_on_owner(self, fn: Callable[[], Any], *, timeout: float = 5.0) -> GpuResult:
        """Convenience: run ``fn`` on the owner thread and wait."""
        handle = self.submit(GpuCommand(op=GpuCommandType.RUN_CALLABLE, fn=fn))
        return handle.wait(timeout=timeout)

    def shutdown(self, *, timeout: float = 2.0) -> None:
        if not self.is_running:
            return
        handle = self.submit(GpuCommand(op=GpuCommandType.SHUTDOWN))
        handle.wait(timeout=timeout)
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._owner_thread_id = None
