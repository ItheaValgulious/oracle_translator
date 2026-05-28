from __future__ import annotations

from .grid import Grid
from .types import DAMAGE_MASK_LIVING, DAMAGE_MASK_TERRAIN, CellFlag, CellState, LifetimeMode, MaterialRegistry, MatterState
from src.game.config import GROW_INTERVAL, GROW_MIN_INTEGRITY


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


def _empty_like(cell: CellState) -> CellState:
    return CellState(temperature=cell.temperature)


def apply_reactions(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    """Parameter-driven reaction system. No branches on reaction_kind or family_id.
    All behavior is driven by VariantDef, MaterialFamily.reaction_profile, and per-cell
    spell overrides (spell_damage_mask, spell_convert_mode, spell_generation_limit)."""
    grid.copy_cells_to_scratch()

    for y in range(grid.height):
        for x in range(grid.width):
            current = grid.get_cell(x, y)
            variant = registry.variant(current.family_id, current.variant_id)
            updated = grid.get_cell(x, y, use_scratch=True)
            consumed = False
            family = registry.family(current.family_id)
            rp = family.reaction_profile

            # ── 1. Self-heating (reaction_energy > 0) ──
            # Cells with reaction_energy auto-heat and may age out.
            if variant.reaction_energy > 0.0:
                updated.age = current.age + dt
                updated.temperature += variant.reaction_energy * dt / max(variant.heat_capacity, 0.001)
                if variant.lifetime_mode == LifetimeMode.DECAY_WITH_AGE:
                    max_age = rp.get("max_age", 6.0)
                    if updated.age >= max_age:
                        grid.set_cell(x, y, _empty_like(updated), use_scratch=True)
                        continue

            # ── 2. Ignition (reaction_min_temperature > 0) ──
            # Any variant with reaction_min_temperature catches fire when hot enough or near a heat source.
            if variant.reaction_min_temperature > 0.0:
                ignited = current.temperature >= variant.reaction_min_temperature
                if not ignited:
                    for dx, dy in NEIGHBORS_8:
                        nx = x + dx
                        ny = y + dy
                        if not grid.in_bounds(nx, ny):
                            continue
                        neighbor = grid.get_cell(nx, ny)
                        neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                        # A neighbor is a heat source if it has reaction_energy > 0
                        if neighbor_variant.reaction_energy > 0.0:
                            ignited = True
                            break
                if ignited:
                    target_family = variant.ignite_target_family_id or "fire"
                    target_variant = variant.ignite_target_variant_id or "fire"
                    grid.set_cell(x, y, CellState(
                        family_id=target_family,
                        variant_id=target_variant,
                        temperature=600.0,
                        integrity=1.0,
                        generation=current.generation + 1,
                        age=0.0,
                    ), use_scratch=True)
                    continue

            # ── 3. Neighbor damage (reaction_strength > 0) ──
            # Cells with reaction_strength damage neighbors based on damage_mask.
            # Per-cell spell_damage_mask overrides variant.damage_mask when set.
            if variant.reaction_strength > 0.0:
                damage_mask = current.spell_damage_mask if current.spell_damage_mask else rp.get("damage_mask", [])
                convert_mode = current.spell_convert_mode if current.spell_convert_mode else "none"
                damaged_any = False
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    neighbor = grid.get_cell(nx, ny)
                    neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                    target = grid.get_cell(nx, ny, use_scratch=True)

                    # Damage terrain (support_bearing cells)
                    if "terrain" in damage_mask and neighbor_variant.support_bearing:
                        corrosion = variant.reaction_strength * dt / max(neighbor_variant.hardness, 0.05)
                        target.integrity = max(0.0, target.integrity - corrosion)
                        if convert_mode == "empty" and target.integrity <= 0.0:
                            grid.set_cell(nx, ny, _empty_like(target), use_scratch=True)
                        elif convert_mode == "self" and target.integrity <= 0.0:
                            grid.set_cell(nx, ny, CellState(
                                family_id=current.family_id,
                                variant_id=current.variant_id,
                                temperature=current.temperature,
                                generation=0,
                                age=0.0,
                            ), use_scratch=True)
                        damaged_any = True

                    # Damage living (entity placeholders)
                    if "living" in damage_mask and neighbor.family_id == "entity_placeholder":
                        target.integrity = max(0.0, target.integrity - variant.reaction_strength * dt)
                        damaged_any = True

                    # Damage water-family cells (freeze reaction)
                    if "water" in damage_mask and neighbor.family_id == "water":
                        corrosion = variant.reaction_strength * dt / max(neighbor_variant.hardness, 0.05)
                        target.integrity = max(0.0, target.integrity - corrosion)
                        if convert_mode == "self" and target.integrity <= 0.3:
                            grid.set_cell(nx, ny, CellState(
                                family_id=current.family_id,
                                variant_id=current.variant_id,
                                temperature=current.temperature,
                                integrity=1.0,
                                generation=0,
                                age=0.0,
                            ), use_scratch=True)
                        damaged_any = True

                if damaged_any and not variant.reaction_preserves_self:
                    consumed = True

            # ── 4. Heat integrity decay (integrity_decay_from_heat > 0) ──
            # Support-bearing cells lose integrity when overheated.
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

            # ── 6. Growth (reaction_profile.max_generation > 0 or spell_generation_limit > 0) ──
            # Organic cells grow upward over time.
            # Per-cell spell_generation_limit overrides family default when set (>=0).
            max_gen = current.spell_generation_limit if current.spell_generation_limit >= 0 else rp.get("max_generation", 0)
            if max_gen > 0 and current.integrity >= GROW_MIN_INTEGRITY and current.age >= GROW_INTERVAL and current.generation < max_gen:
                if y > 0:
                    above = grid.get_cell(x, y - 1)
                    if above.is_empty:
                        above_variant = registry.variant(above.family_id, above.variant_id)
                        if above_variant.matter_state != MatterState.LIQUID:
                            grid.set_cell(x, y - 1, CellState(
                                family_id=current.family_id,
                                variant_id=current.variant_id,
                                temperature=current.temperature,
                                integrity=1.0,
                                generation=current.generation + 1,
                                age=0.0,
                                flags=CellFlag.FIXPOINT if current.flags & CellFlag.FIXPOINT else CellFlag.NONE,
                            ), use_scratch=True)

            # ── 7. Placeholder neighbor damage (entity_placeholder cells) ──
            # Placeholders take damage from any neighbor with reaction_strength > 0
            # whose family damage_mask includes "living".
            if current.family_id == "entity_placeholder":
                for dx, dy in NEIGHBORS_8:
                    nx = x + dx
                    ny = y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    neighbor = grid.get_cell(nx, ny)
                    if neighbor.family_id == "empty":
                        continue
                    neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
                    if neighbor_variant.reaction_strength > 0.0:
                        neighbor_family = registry.family(neighbor.family_id)
                        neighbor_mask = neighbor_family.reaction_profile.get("damage_mask", [])
                        if "living" in neighbor_mask:
                            target = grid.get_cell(x, y, use_scratch=True)
                            target.integrity = max(0.0, target.integrity - neighbor_variant.reaction_strength * dt)

            if consumed:
                grid.set_cell(x, y, _empty_like(updated), use_scratch=True)

    grid.swap_buffers()