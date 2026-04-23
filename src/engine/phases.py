from __future__ import annotations

from .grid import Grid
from .support import SUPPORT_FAILURE_THRESHOLD
from .types import CellFlag, CellState, MaterialFamily, MaterialRegistry


def _convert_cell(
    cell: CellState,
    registry: MaterialRegistry,
    target_family_id: str,
    target_variant_id: str,
) -> CellState:
    target_variant = registry.variant(target_family_id, target_variant_id)
    updated = cell.copy()
    updated.family_id = target_family_id
    updated.variant_id = target_variant_id
    updated.temperature = max(updated.temperature, target_variant.base_temperature)
    updated.age = 0.0
    if not target_variant.support_bearing:
        updated.flags &= ~CellFlag.FIXPOINT
        updated.support_value = 0.0
    return updated


def _cooled_variant(family: MaterialFamily, cell: CellState) -> str:
    if cell.support_value >= SUPPORT_FAILURE_THRESHOLD:
        return family.default_variant
    return family.collapse_target or family.default_variant


def apply_phase_transitions(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    del dt
    grid.copy_cells_to_scratch()

    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            family = registry.family(current.family_id)
            for rule in family.phase_map:
                if not rule.matches(current.variant_id, current.temperature):
                    continue
                target_family_id = rule.target_family_id or current.family_id
                target_variant_id = rule.target_variant
                if current.variant_id in {"magma", "molten_glass", "molten_iron"} and rule.below_temperature is not None:
                    target_family = registry.family(target_family_id)
                    target_variant_id = _cooled_variant(target_family, current)
                converted = _convert_cell(current, registry, target_family_id, target_variant_id)
                grid.set_cell(x, y, converted, use_scratch=True)
                break

    grid.swap_buffers()
