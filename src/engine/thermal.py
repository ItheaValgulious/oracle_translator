from __future__ import annotations

from .grid import Grid
from .types import MaterialRegistry


BASE_TEMPERATURE_RELAXATION = 0.015
MAX_HEAT_EXCHANGE = 120.0


def apply_thermal(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.copy_cells_to_scratch()

    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            updated = grid.get_cell(x, y, use_scratch=True)
            variant = registry.variant(current.family_id, current.variant_id)
            updated.temperature += (variant.base_temperature - current.temperature) * BASE_TEMPERATURE_RELAXATION * dt

    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            current_variant = registry.variant(current.family_id, current.variant_id)
            current_index = grid.index(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                nx = x + dx
                ny = y + dy
                if not grid.in_bounds(nx, ny):
                    continue
                neighbor = grid.get_cell(nx, ny)
                neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                conductivity = (current_variant.thermal_conductivity + neighbor_variant.thermal_conductivity) * 0.5
                capacity = max((current_variant.heat_capacity + neighbor_variant.heat_capacity) * 0.5, 0.001)
                delta = (neighbor.temperature - current.temperature) * conductivity * dt / capacity
                delta = max(-MAX_HEAT_EXCHANGE, min(MAX_HEAT_EXCHANGE, delta))
                grid.scratch[current_index].temperature += delta
                grid.scratch[grid.index(nx, ny)].temperature -= delta

    grid.swap_buffers()
