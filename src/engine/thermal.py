from __future__ import annotations

from .atmosphere import AMBIENT_AIR_RESTORE_RATE, ambient_air_temperature_for_row
from .grid import Grid
from .types import MaterialRegistry, MatterState


MAX_HEAT_EXCHANGE = 120.0
THERMAL_CONDUCTION_RATE = 8.0


def _thermal_conduction_multiplier(current, current_variant, neighbor, neighbor_variant) -> float:
    multiplier = 1.0
    if current_variant.matter_state == MatterState.GAS and neighbor_variant.matter_state == MatterState.LIQUID:
        multiplier = max(multiplier, current_variant.liquid_contact_heat_exchange_multiplier)
    if neighbor_variant.matter_state == MatterState.GAS and current_variant.matter_state == MatterState.LIQUID:
        multiplier = max(multiplier, neighbor_variant.liquid_contact_heat_exchange_multiplier)
    if (
        current_variant.matter_state == MatterState.GAS
        and neighbor_variant.matter_state == MatterState.GAS
        and current.family_id == neighbor.family_id
        and current.variant_id == neighbor.variant_id
    ):
        multiplier = max(
            multiplier,
            current_variant.same_variant_heat_exchange_multiplier,
            neighbor_variant.same_variant_heat_exchange_multiplier,
        )
    return multiplier


def apply_thermal(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.copy_cells_to_scratch()
    ambient_base_temperature = registry.variant("empty", "empty").base_temperature

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
                conductivity *= _thermal_conduction_multiplier(current, current_variant, neighbor, neighbor_variant)
                capacity = max((current_variant.heat_capacity + neighbor_variant.heat_capacity) * 0.5, 0.001)
                delta = (neighbor.temperature - current.temperature) * conductivity * THERMAL_CONDUCTION_RATE * dt / capacity
                delta = max(-MAX_HEAT_EXCHANGE, min(MAX_HEAT_EXCHANGE, delta))
                grid.scratch[current_index].temperature += delta
                grid.scratch[grid.index(nx, ny)].temperature -= delta

    for y in range(grid.height):
        ambient_temperature = ambient_air_temperature_for_row(grid.height, y, ambient_base_temperature)
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            if not current.is_empty:
                continue
            scratch_cell = grid.scratch[grid.index(x, y)]
            scratch_cell.temperature += (ambient_temperature - scratch_cell.temperature) * AMBIENT_AIR_RESTORE_RATE * dt

    grid.swap_buffers()
