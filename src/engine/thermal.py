from __future__ import annotations

from .grid import Grid
from .types import MaterialRegistry, MatterState


MAX_HEAT_EXCHANGE = 120.0
THERMAL_CONDUCTION_RATE = 8.0
LIQUID_GAS_INTERFACE_CONDUCTION_MULTIPLIER = 30.0
CONDENSABLE_GAS_CLUSTER_CONDUCTION_MULTIPLIER = 80.0


def _is_steam(cell, variant) -> bool:
    return cell.family_id == "water" and variant.variant_id == "steam"


def _thermal_conduction_multiplier(current, current_variant, neighbor, neighbor_variant) -> float:
    current_is_condensable_gas = _is_steam(current, current_variant)
    neighbor_is_condensable_gas = _is_steam(neighbor, neighbor_variant)
    current_is_liquid = current_variant.matter_state == MatterState.LIQUID
    neighbor_is_liquid = neighbor_variant.matter_state == MatterState.LIQUID
    if (current_is_liquid and neighbor_is_condensable_gas) or (neighbor_is_liquid and current_is_condensable_gas):
        return LIQUID_GAS_INTERFACE_CONDUCTION_MULTIPLIER
    if current_is_condensable_gas and neighbor_is_condensable_gas:
        return CONDENSABLE_GAS_CLUSTER_CONDUCTION_MULTIPLIER
    return 1.0


def apply_thermal(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.copy_cells_to_scratch()

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

    grid.swap_buffers()
