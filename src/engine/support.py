from __future__ import annotations

from .grid import Grid
from .types import CellFlag, MaterialRegistry


SUPPORT_TIMEOUT_SECONDS = 10.0
SUPPORT_SOURCE_VALUE = SUPPORT_TIMEOUT_SECONDS
SUPPORT_FAILURE_THRESHOLD = 0.0
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
    emitted_generation = grid.step_id + 1
    grid.copy_cells_to_scratch()
    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            updated = grid.get_cell(x, y, use_scratch=True)
            variant = registry.variant(current.family_id, current.variant_id)
            index = grid.index(x, y)

            incoming_generation = current.generation
            if current.flags & CellFlag.FIXPOINT or grid.external_support_anchors[index]:
                incoming_generation = emitted_generation
            elif variant.support_transmission:
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    neighbor = grid.get_cell(nx, ny)
                    if neighbor.flags & CellFlag.FIXPOINT:
                        incoming_generation = max(incoming_generation, emitted_generation)
                        continue
                    neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                    if neighbor_variant.support_transmission:
                        incoming_generation = max(incoming_generation, neighbor.generation)

            if incoming_generation > current.generation:
                updated.support_value = SUPPORT_SOURCE_VALUE
                updated.generation = incoming_generation
            elif variant.support_transmission:
                updated.support_value = max(0.0, current.support_value - dt)
            else:
                updated.support_value = 0.0
                updated.generation = 0

            if variant.support_bearing and updated.support_value <= SUPPORT_FAILURE_THRESHOLD:
                unsupported_dt = dt if current.support_value <= SUPPORT_FAILURE_THRESHOLD else max(0.0, dt - current.support_value)
                updated.integrity = max(0.0, updated.integrity - INTEGRITY_DECAY_UNSUPPORTED * unsupported_dt)

    grid.swap_buffers()
