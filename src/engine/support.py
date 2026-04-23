from __future__ import annotations

from .grid import Grid
from .types import CellFlag, MaterialRegistry


SUPPORT_SOURCE_VALUE = 1.0
SUPPORT_DIFFUSION_RATE = 0.65
SUPPORT_DECAY_RATE = 0.15
SUPPORT_FAILURE_THRESHOLD = 0.25
INTEGRITY_DECAY_UNSUPPORTED = 0.18

NEIGHBORS_8 = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def apply_support(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.copy_cells_to_scratch()
    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            updated = grid.get_cell(x, y, use_scratch=True)
            variant = registry.variant(current.family_id, current.variant_id)

            if current.flags & CellFlag.FIXPOINT:
                updated.support_value = SUPPORT_SOURCE_VALUE
            elif variant.support_transmission:
                neighbor_values = []
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    neighbor = grid.get_cell(nx, ny)
                    neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                    if neighbor.flags & CellFlag.FIXPOINT or neighbor_variant.support_transmission:
                        neighbor_values.append(neighbor.support_value)
                neighbor_avg = sum(neighbor_values) / len(neighbor_values) if neighbor_values else 0.0
                retained = current.support_value * max(0.0, 1.0 - SUPPORT_DECAY_RATE * dt)
                updated.support_value = retained + (neighbor_avg - retained) * min(1.0, SUPPORT_DIFFUSION_RATE * dt)
            else:
                updated.support_value = 0.0

            if variant.support_bearing and updated.support_value < SUPPORT_FAILURE_THRESHOLD:
                missing_support = SUPPORT_FAILURE_THRESHOLD - updated.support_value
                updated.integrity = max(0.0, updated.integrity - missing_support * INTEGRITY_DECAY_UNSUPPORTED * dt)

    grid.swap_buffers()
