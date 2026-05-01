from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import moderngl
import pyglet

from engine.materials import build_material_registry
from engine.scenarios import populate_demo_scene
from engine.world import ActiveWorldWindow, WorldChunkStore


_GPU_WINDOWS: list[pyglet.window.Window] = []


def create_compute_context() -> moderngl.Context:
    try:
        return moderngl.create_standalone_context(require=430)
    except Exception:
        window = pyglet.window.Window(width=32, height=32, visible=False)
        window.switch_to()
        context = moderngl.create_context(require=430)
        _GPU_WINDOWS.append(window)
        return context


def run_benchmark() -> None:
    registry = build_material_registry()
    store = WorldChunkStore(2560, 1440)
    populate_demo_scene(store, registry)
    store.recompute_anchored_support(registry)
    ctx = create_compute_context()
    world = ActiveWorldWindow(
        store,
        registry,
        viewport_width=640,
        viewport_height=360,
        halo_cells=32,
        page_shift_cells=16,
        safety_margin_cells=16,
        idle_flush_service_interval_seconds=0.0,
        pending_writeback_limit=256,
        ctx=ctx,
    )

    shift_totals: list[float] = []
    shift_evict: list[float] = []
    shift_overlap: list[float] = []
    shift_overlap_fx: list[float] = []
    shift_incoming: list[float] = []
    shift_clear_fx: list[float] = []
    shift_anchor_build: list[float] = []
    shift_anchor_upload: list[float] = []
    frame_totals: list[float] = []
    frame_pan: list[float] = []
    frame_step: list[float] = []
    frame_io: list[float] = []
    frame_render: list[float] = []
    idle_flush_io: list[float] = []
    idle_flush_total: list[float] = []

    original_shift = world._shift_active_window

    def wrapped_shift(new_origin_x: int, new_origin_y: int) -> None:
        started = perf_counter()
        original_shift(new_origin_x, new_origin_y)
        shift_totals.append((perf_counter() - started) * 1000.0)
        shift_evict.append(world.stage_time_last_ms())
        shift_overlap.append(world.overlap_copy_time_last_ms())
        shift_overlap_fx.append(world.overlap_transient_copy_time_last_ms())
        shift_incoming.append(world.incoming_load_time_last_ms())
        shift_clear_fx.append(world.incoming_transient_clear_time_last_ms())
        shift_anchor_build.append(world.anchor_build_time_last_ms())
        shift_anchor_upload.append(world.anchor_upload_time_last_ms())

    world._shift_active_window = wrapped_shift  # type: ignore[method-assign]

    dt = 1.0 / 60.0
    for _ in range(96):
        frame_started_at = perf_counter()

        pan_started_at = perf_counter()
        world.pan_camera(8, 0)
        frame_pan.append((perf_counter() - pan_started_at) * 1000.0)

        step_started_at = perf_counter()
        world.step(dt)
        frame_step.append((perf_counter() - step_started_at) * 1000.0)

        io_started_at = perf_counter()
        world.service_background_io()
        frame_io.append((perf_counter() - io_started_at) * 1000.0)

        render_started_at = perf_counter()
        world.render()
        frame_render.append((perf_counter() - render_started_at) * 1000.0)

        frame_totals.append((perf_counter() - frame_started_at) * 1000.0)

    world.mark_camera_activity(False, dt=1.0)
    for _ in range(64):
        if world.pending_writeback_count <= 0:
            break
        flush_started_at = perf_counter()
        io_started_at = perf_counter()
        world.mark_camera_activity(False, dt=1.0)
        world.service_background_io()
        idle_flush_io.append((perf_counter() - io_started_at) * 1000.0)
        idle_flush_total.append((perf_counter() - flush_started_at) * 1000.0)

    def summarize(name: str, values: list[float]) -> str:
        if not values:
            return f"{name}: no samples"
        return (
            f"{name}: count={len(values)} avg={statistics.mean(values):.3f}ms "
            f"median={statistics.median(values):.3f}ms max={max(values):.3f}ms"
        )

    print(summarize("shift_total", shift_totals))
    print(summarize("evict_stage", shift_evict))
    print(summarize("overlap_copy", shift_overlap))
    print(summarize("overlap_transient_copy", shift_overlap_fx))
    print(summarize("incoming_load", shift_incoming))
    print(summarize("incoming_transient_clear", shift_clear_fx))
    print(summarize("anchor_build", shift_anchor_build))
    print(summarize("anchor_upload", shift_anchor_upload))
    print(summarize("frame_total", frame_totals))
    print(summarize("frame_pan", frame_pan))
    print(summarize("frame_step", frame_step))
    print(summarize("frame_io", frame_io))
    print(summarize("frame_render", frame_render))
    print(summarize("idle_flush_io", idle_flush_io))
    print(summarize("idle_flush_total", idle_flush_total))
    print("scene=populate_demo_scene")


if __name__ == "__main__":
    run_benchmark()
