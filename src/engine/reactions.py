from __future__ import annotations

from .grid import Grid
from .types import CellState, MaterialRegistry, ReactionKind


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


def _spawn_fire(cell: CellState) -> CellState:
    return CellState(
        family_id="fire",
        variant_id="fire",
        temperature=600.0,
        integrity=1.0,
        generation=cell.generation + 1,
        age=0.0,
    )


def _empty_like(cell: CellState) -> CellState:
    return CellState(temperature=cell.temperature)


def apply_reactions(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    grid.copy_cells_to_scratch()

    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            variant = registry.variant(current.family_id, current.variant_id)
            updated = grid.get_cell(x, y, use_scratch=True)
            consumed_by_reaction = False

            if variant.reaction_kind == ReactionKind.HEAT_SOURCE:
                updated.age = current.age + dt
                updated.temperature += variant.reaction_energy * dt / max(variant.heat_capacity, 0.001)
                max_age = registry.family(current.family_id).reaction_profile.get("max_age", 6.0)
                if updated.age >= max_age:
                    grid.set_cell(x, y, _empty_like(updated), use_scratch=True)
                    continue

            if current.family_id == "tar":
                ignited = current.temperature >= (variant.ignite_temperature or 0.0)
                if not ignited:
                    for dx, dy in NEIGHBORS_8:
                        nx = x + dx
                        ny = y + dy
                        if not grid.in_bounds(nx, ny):
                            continue
                        neighbor = grid.get_cell(nx, ny)
                        if neighbor.family_id == "fire":
                            ignited = True
                            break
                if ignited:
                    grid.set_cell(x, y, _spawn_fire(current), use_scratch=True)
                    continue
                if current.temperature >= 140.0:
                    updated.family_id = "tar"
                    updated.variant_id = "tar_smoke"

            if variant.reaction_kind == ReactionKind.CORROSIVE:
                corroded_anything = False
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    neighbor = grid.get_cell(nx, ny)
                    neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                    if neighbor_variant.support_bearing:
                        target = grid.get_cell(nx, ny, use_scratch=True)
                        corrosion = variant.reaction_strength * dt / max(neighbor_variant.hardness, 0.05)
                        target.integrity = max(0.0, target.integrity - corrosion)
                        corroded_anything = True
                if corroded_anything and not variant.reaction_preserves_self:
                    consumed_by_reaction = True

            if current.family_id == "poison" and current.temperature >= (variant.decompose_temperature or 10_000.0):
                updated.family_id = "poison"
                updated.variant_id = "poison_gas"
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    if grid.get_cell(nx, ny).is_empty:
                        grid.set_cell(nx, ny, _spawn_fire(current), use_scratch=True)
                        break

            if variant.support_bearing and variant.integrity_decay_from_heat > 0.0:
                heat_start = variant.base_temperature + 60.0
                if variant.melt_temperature is not None:
                    heat_start = min(heat_start, variant.melt_temperature * 0.7)
                if current.temperature > heat_start:
                    target = grid.get_cell(x, y, use_scratch=True)
                    target.integrity = max(
                        0.0,
                        target.integrity - (current.temperature - heat_start) * variant.integrity_decay_from_heat * dt,
                    )

            if consumed_by_reaction:
                grid.set_cell(x, y, _empty_like(updated), use_scratch=True)

    grid.swap_buffers()
