from __future__ import annotations

from math import hypot

from .grid import Grid
from .types import MaterialRegistry, SimKind


NEIGHBOR_OFFSETS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)

GRAVITY_BIAS = {
    SimKind.EMPTY: (0.0, 0.0),
    SimKind.PLATFORM: (0.0, 0.0),
    SimKind.POWDER: (0.0, 1.2),
    SimKind.LIQUID: (0.0, 0.8),
    SimKind.GAS: (0.0, -0.6),
    SimKind.FIRE: (0.0, -0.25),
    SimKind.MOLTEN: (0.0, 0.75),
}

BLOCKED_GAIN = 0.65
BLOCKED_DECAY = 0.7
VELOCITY_DAMPING = 0.85


def _normalize(x: float, y: float) -> tuple[float, float]:
    length = hypot(x, y)
    if length == 0.0:
        return (0.0, 0.0)
    return (x / length, y / length)


def _direction_score(direction: tuple[int, int], desired: tuple[float, float]) -> float:
    ndx, ndy = _normalize(direction[0], direction[1])
    return ndx * desired[0] + ndy * desired[1]


def _sorted_candidates(desired_x: float, desired_y: float) -> list[tuple[int, int]]:
    desired = _normalize(desired_x, desired_y)
    if desired == (0.0, 0.0):
        desired = (0.0, 1.0)
    return sorted(NEIGHBOR_OFFSETS, key=lambda direction: _direction_score(direction, desired), reverse=True)


def _can_move(sim_kind: SimKind) -> bool:
    return sim_kind in {SimKind.POWDER, SimKind.LIQUID, SimKind.GAS, SimKind.FIRE, SimKind.MOLTEN}


def apply_motion(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.clear_scratch()
    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            current_variant = registry.variant(current.family_id, current.variant_id)
            if current_variant.sim_kind == SimKind.EMPTY:
                continue

            updated = current.copy()
            if not _can_move(current_variant.sim_kind):
                updated.blocked_x *= BLOCKED_DECAY
                updated.blocked_y *= BLOCKED_DECAY
                grid.set_cell(x, y, updated, use_scratch=True)
                continue

            gravity_x, gravity_y = GRAVITY_BIAS[current_variant.sim_kind]
            desired_x = current.vel_x + current.blocked_x * BLOCKED_GAIN + gravity_x * dt
            desired_y = current.vel_y + current.blocked_y * BLOCKED_GAIN + gravity_y * dt
            candidates = _sorted_candidates(desired_x, desired_y)

            moved = False
            realized_x = 0.0
            realized_y = 0.0
            for dx, dy in candidates:
                nx = x + dx
                ny = y + dy
                if not grid.in_bounds(nx, ny):
                    continue
                target = grid.get_cell(nx, ny)
                if not target.is_empty:
                    continue
                if not grid.get_cell(nx, ny, use_scratch=True).is_empty:
                    continue
                updated.vel_x = dx * VELOCITY_DAMPING
                updated.vel_y = dy * VELOCITY_DAMPING
                realized_x = dx
                realized_y = dy
                moved = True
                grid.set_cell(nx, ny, updated, use_scratch=True)
                break

            residual_x = desired_x - realized_x
            residual_y = desired_y - realized_y
            updated.blocked_x = current.blocked_x * BLOCKED_DECAY + residual_x * BLOCKED_GAIN
            updated.blocked_y = current.blocked_y * BLOCKED_DECAY + residual_y * BLOCKED_GAIN

            if not moved:
                updated.vel_x = desired_x * VELOCITY_DAMPING
                updated.vel_y = desired_y * VELOCITY_DAMPING
                grid.set_cell(x, y, updated, use_scratch=True)

    grid.swap_buffers()
