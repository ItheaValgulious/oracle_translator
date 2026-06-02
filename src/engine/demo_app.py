"""Compatibility launcher for the legacy engine demo.

The old demo owned a separate ModernGL context and ``ActiveWorldWindow``
on the pyglet UI thread. Goal.md now requires a single GPU owner thread,
so this module keeps the historical ``run_demo`` API while delegating to
the owner-thread-backed game runtime.
"""

from __future__ import annotations

import logging

from .world import DEFAULT_HALO_CELLS, DEFAULT_PAGE_SHIFT_CELLS

DEFAULT_TICK_RATE_HZ = 60.0
DEFAULT_SIMULATION_SUBSTEPS = 2
DEFAULT_CAMERA_SPEED_CELLS_PER_SECOND = 240.0
MAX_CAMERA_MOTION_DT_SECONDS = 1.0 / 30.0
BACKGROUND_IO_SERVICE_INTERVAL_SECONDS = 0.5

log = logging.getLogger(__name__)


def run_demo(
    *,
    grid_width: int = 160,
    grid_height: int = 96,
    world_width: int | None = None,
    world_height: int | None = None,
    halo_cells: int = DEFAULT_HALO_CELLS,
    page_shift_cells: int = DEFAULT_PAGE_SHIFT_CELLS,
    cell_scale: int = 8,
    window_width: int | None = None,
    window_height: int | None = None,
    simulation_substeps: int = DEFAULT_SIMULATION_SUBSTEPS,
    liquid_brownian_enabled: bool = True,
    blocked_impulse_enabled: bool = True,
    directional_fallback_enabled: bool = True,
    vsync: bool = True,
) -> None:
    del (
        grid_width,
        grid_height,
        world_width,
        world_height,
        halo_cells,
        page_shift_cells,
        window_width,
        window_height,
        simulation_substeps,
        liquid_brownian_enabled,
        blocked_impulse_enabled,
        directional_fallback_enabled,
        vsync,
    )
    log.info("[demo_app] legacy demo entry now launches the owner-thread game runtime")
    from src.game.app import run_game

    run_game(cell_scale=int(cell_scale))
