from __future__ import annotations

from collections.abc import Iterable

from .atmosphere import default_ambient_air_temperature_for_row
from .grid import Grid
from .materials import build_material_registry
from .motion import apply_motion
from .phases import apply_phase_transitions
from .reactions import apply_reactions
from .support import SUPPORT_SOURCE_VALUE, apply_support
from .thermal import apply_thermal
from .types import CellFlag, CellState, MaterialRegistry


def _collapse_cells(grid: Grid, registry: MaterialRegistry) -> None:
    grid.copy_cells_to_scratch()
    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            variant = registry.variant(current.family_id, current.variant_id)
            if not variant.support_bearing or current.integrity > 0.0:
                continue
            family = registry.family(current.family_id)
            collapse_target = family.collapse_target
            if collapse_target is None:
                grid.set_cell(
                    x,
                    y,
                    CellState(temperature=default_ambient_air_temperature_for_row(grid.height, y)),
                    use_scratch=True,
                )
                continue
            collapsed = current.copy()
            collapsed.variant_id = collapse_target
            collapsed.flags &= ~CellFlag.FIXPOINT
            collapsed.support_value = 0.0
            collapsed.generation = 0
            collapsed.integrity = 0.6
            collapsed.age = 0.0
            grid.set_cell(x, y, collapsed, use_scratch=True)
    grid.swap_buffers()


def step(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    apply_support(grid, registry, dt)
    apply_reactions(grid, registry, dt)
    apply_thermal(grid, registry, dt)
    apply_phase_transitions(grid, registry, dt)
    apply_motion(grid, registry, dt)
    _collapse_cells(grid, registry)
    grid.step_id += 1


def inject_cells(
    grid: Grid,
    brush_or_cells: dict[str, int] | Iterable[tuple[int, int]],
    family_id: str,
    variant_id: str,
    overrides: dict[str, object] | None = None,
    registry: MaterialRegistry | None = None,
) -> None:
    registry = registry or build_material_registry()
    variant = registry.variant(family_id, variant_id)
    base = CellState(
        family_id=family_id,
        variant_id=variant_id,
        temperature=variant.base_temperature,
    )
    overrides = overrides or {}
    for key, value in overrides.items():
        setattr(base, key, value)

    targets: list[tuple[int, int]] = []
    if isinstance(brush_or_cells, dict):
        center_x = int(brush_or_cells["x"])
        center_y = int(brush_or_cells["y"])
        radius = int(brush_or_cells.get("radius", 0))
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                if grid.in_bounds(x, y) and (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    targets.append((x, y))
    else:
        targets.extend(brush_or_cells)

    for x, y in targets:
        if not grid.in_bounds(x, y):
            continue
        cell = base.copy()
        if family_id == "empty" and variant_id == "empty" and "temperature" not in overrides:
            cell.temperature = default_ambient_air_temperature_for_row(grid.height, y)
        if cell.flags & CellFlag.FIXPOINT:
            cell.support_value = SUPPORT_SOURCE_VALUE
        grid.set_cell(x, y, cell)
