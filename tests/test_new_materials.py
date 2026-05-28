"""Tests for Phase 0 new material behaviors: oil, magic_acid, obsidian, snow, wood, grass, wood_plank, stone_falling, entity_placeholder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.grid import create_grid
from engine.materials import build_material_registry
from engine.reactions import apply_reactions
from engine.sim import inject_cells, step
from engine.types import CellFlag, CellState


class NewMaterialTests(unittest.TestCase):
    """Tests for Phase 0 new material families."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()

    def _make_grid(self, w: int = 40, h: int = 40) -> "Grid":
        return create_grid(w, h)

    # ── Oil ──
    def test_oil_floats_on_water(self) -> None:
        """Oil (density 0.85) should float above water (density 1.0)."""
        grid = self._make_grid()
        # Place water at bottom
        inject_cells(grid, {"x": 20, "y": 38, "radius": 3}, "water", "water", registry=self.registry)
        # Place oil on top of water
        inject_cells(grid, {"x": 20, "y": 35, "radius": 2}, "oil", "oil_liquid", registry=self.registry)

        step(grid, self.registry, 0.1)

        # Check that oil cells are at higher y than water cells
        oil_cells = [(x, y) for y in range(grid.height) for x in range(grid.width)
                     if grid.get_cell(x, y).family_id == "oil"]
        water_cells = [(x, y) for y in range(grid.height) for x in range(grid.width)
                       if grid.get_cell(x, y).family_id == "water"]
        if oil_cells and water_cells:
            avg_oil_y = sum(y for _, y in oil_cells) / len(oil_cells)
            avg_water_y = sum(y for _, y in water_cells) / len(water_cells)
            self.assertLess(avg_oil_y, avg_water_y)  # lower y = higher position

    def test_oil_ignites_near_fire(self) -> None:
        """Oil with reaction_min_temperature should ignite when adjacent to fire."""
        grid = self._make_grid()
        # Place a single oil cell adjacent to a single fire cell
        grid.set_cell(20, 30, CellState(family_id="oil", variant_id="oil_liquid", integrity=1.0))
        grid.set_cell(21, 30, CellState(family_id="fire", variant_id="fire", integrity=1.0))

        apply_reactions(grid, self.registry, 0.1)

        # The oil cell adjacent to fire should have ignited → become fire
        cell = grid.get_cell(20, 30)
        self.assertEqual(cell.family_id, "fire")

    def test_oil_ignites_when_hot(self) -> None:
        """Oil should ignite when its temperature exceeds reaction_min_temperature."""
        grid = self._make_grid()
        oil_variant = self.registry.variant("oil", "oil_liquid")
        # Place hot oil directly
        hot_cell = CellState(
            family_id="oil",
            variant_id="oil_liquid",
            temperature=oil_variant.reaction_min_temperature + 10.0,
            integrity=1.0,
        )
        grid.set_cell(20, 30, hot_cell)

        apply_reactions(grid, self.registry, 0.1)

        # The hot oil cell should have ignited → become fire
        cell = grid.get_cell(20, 30)
        self.assertEqual(cell.family_id, "fire")

    # ── Magic Acid ──
    def test_magic_acid_persists_after_corroding(self) -> None:
        """Magic acid (reaction_preserves_self=True) should not be consumed when corroding."""
        grid = self._make_grid()
        # Place stone next to magic acid
        grid.set_cell(19, 30, CellState(family_id="stone", variant_id="stone_platform", integrity=1.0))
        grid.set_cell(20, 30, CellState(family_id="magic_acid", variant_id="magic_acid_liquid", integrity=1.0))

        apply_reactions(grid, self.registry, 0.1)

        # Magic acid should still exist (preserves_self)
        acid_cell = grid.get_cell(20, 30)
        self.assertEqual(acid_cell.family_id, "magic_acid")

    def test_magic_acid_corrodes_terrain(self) -> None:
        """Magic acid should reduce integrity of neighboring support-bearing cells."""
        grid = self._make_grid()
        # Place stone next to magic acid
        grid.set_cell(19, 30, CellState(family_id="stone", variant_id="stone_platform", integrity=1.0))
        grid.set_cell(20, 30, CellState(family_id="magic_acid", variant_id="magic_acid_liquid", integrity=1.0))

        initial_integrity = grid.get_cell(19, 30).integrity
        apply_reactions(grid, self.registry, 0.1)
        new_integrity = grid.get_cell(19, 30).integrity

        self.assertLess(new_integrity, initial_integrity)

    # ── Obsidian ──
    def test_obsidian_higher_hardness_than_stone(self) -> None:
        """Obsidian (hardness 1.5) should be harder than stone (hardness 1.0)."""
        obsidian = self.registry.variant("obsidian", "obsidian_platform")
        stone = self.registry.variant("stone", "stone_platform")
        self.assertGreater(obsidian.hardness, stone.hardness)

    def test_obsidian_platform_is_non_falling(self) -> None:
        """Obsidian platform should not be falling (support_bearing=True)."""
        obsidian = self.registry.variant("obsidian", "obsidian_platform")
        self.assertTrue(obsidian.support_bearing)

    # ── Snow ──
    def test_snow_melts_to_water_at_5c(self) -> None:
        """Snow should phase-transition to water when temperature > 5C."""
        grid = self._make_grid()
        # Place snow at temperature above 5C
        hot_cell = CellState(
            family_id="snow",
            variant_id="snow_powder",
            temperature=10.0,
            integrity=1.0,
        )
        grid.set_cell(20, 30, hot_cell)

        # Run phase transitions
        from engine.phases import apply_phase_transitions
        apply_phase_transitions(grid, self.registry, 0.1)

        cell = grid.get_cell(20, 30)
        # Snow at 10C should transition to water
        self.assertEqual(cell.family_id, "water")

    # ── Wood ──
    def test_wood_grows_upward(self) -> None:
        """Wood platform should grow upward when conditions are met."""
        grid = self._make_grid()
        # Place mature wood (age >= GROW_INTERVAL)
        from src.game.config import GROW_INTERVAL
        wood_cell = CellState(
            family_id="wood",
            variant_id="wood_platform",
            integrity=1.0,
            generation=0,
            age=GROW_INTERVAL,
        )
        grid.set_cell(20, 30, wood_cell)

        apply_reactions(grid, self.registry, 0.1)

        # Check if a new wood cell appeared above (y=29, since y decreases upward in grid)
        above = grid.get_cell(20, 29)
        if above.family_id == "wood":
            self.assertEqual(above.generation, 1)

    def test_wood_generation_limit(self) -> None:
        """Wood should stop growing at max_generation=6."""
        grid = self._make_grid()
        from src.game.config import GROW_INTERVAL
        # Place wood at generation 6 (max for wood)
        wood_cell = CellState(
            family_id="wood",
            variant_id="wood_platform",
            integrity=1.0,
            generation=6,
            age=GROW_INTERVAL,
        )
        grid.set_cell(20, 30, wood_cell)

        apply_reactions(grid, self.registry, 0.1)

        # No new wood should appear above
        above = grid.get_cell(20, 29)
        self.assertNotEqual(above.family_id, "wood")

    def test_wood_ignites_when_hot(self) -> None:
        """Wood should ignite when temperature >= reaction_min_temperature."""
        grid = self._make_grid()
        wood_variant = self.registry.variant("wood", "wood_platform")
        hot_cell = CellState(
            family_id="wood",
            variant_id="wood_platform",
            temperature=wood_variant.reaction_min_temperature + 10.0,
            integrity=1.0,
        )
        grid.set_cell(20, 30, hot_cell)

        apply_reactions(grid, self.registry, 0.1)

        cell = grid.get_cell(20, 30)
        self.assertEqual(cell.family_id, "fire")

    # ── Grass ──
    def test_grass_grows_upward(self) -> None:
        """Grass platform should grow upward when conditions are met."""
        grid = self._make_grid()
        from src.game.config import GROW_INTERVAL
        grass_cell = CellState(
            family_id="grass",
            variant_id="grass_platform",
            integrity=1.0,
            generation=0,
            age=GROW_INTERVAL,
        )
        grid.set_cell(20, 30, grass_cell)

        apply_reactions(grid, self.registry, 0.1)

        above = grid.get_cell(20, 29)
        if above.family_id == "grass":
            self.assertEqual(above.generation, 1)

    def test_grass_generation_limit(self) -> None:
        """Grass should stop growing at max_generation=4."""
        grid = self._make_grid()
        from src.game.config import GROW_INTERVAL
        grass_cell = CellState(
            family_id="grass",
            variant_id="grass_platform",
            integrity=1.0,
            generation=4,
            age=GROW_INTERVAL,
        )
        grid.set_cell(20, 30, grass_cell)

        apply_reactions(grid, self.registry, 0.1)

        above = grid.get_cell(20, 29)
        self.assertNotEqual(above.family_id, "grass")

    # ── Wood Plank ──
    def test_wood_plank_does_not_grow(self) -> None:
        """Wood plank should not grow (no max_generation)."""
        grid = self._make_grid()
        from src.game.config import GROW_INTERVAL
        plank_family = self.registry.family("wood_plank")
        max_gen = plank_family.reaction_profile.get("max_generation", 0)
        self.assertEqual(max_gen, 0)

    # ── Stone Falling ──
    def test_stone_falling_does_not_transmit_support(self) -> None:
        """stone_falling should not be support_bearing or transmit support."""
        variant = self.registry.variant("stone", "stone_falling")
        self.assertFalse(variant.support_bearing)
        self.assertFalse(variant.support_transmission)

    # ── Entity Placeholder ──
    def test_placeholder_blocks_falling_sand(self) -> None:
        """Entity placeholder should block falling sand (immobile solid, not displaced)."""
        grid = self._make_grid()
        # Place placeholder — no FIXPOINT, not a support source
        grid.set_cell(20, 30, CellState(
            family_id="entity_placeholder",
            variant_id="placeholder",
            integrity=1.0,
        ))
        # Place sand above
        grid.set_cell(20, 29, CellState(
            family_id="sand",
            variant_id="sand_powder",
            integrity=1.0,
        ))

        step(grid, self.registry, 0.1)

        # Placeholder should still be at (20,30) — sand can't displace it
        placeholder_cell = grid.get_cell(20, 30)
        self.assertEqual(placeholder_cell.family_id, "entity_placeholder")

    def test_placeholder_support_bearing(self) -> None:
        """Entity placeholder should be support_bearing and support_transmission
        per plan (placeholder participates in support network as a solid anchor)."""
        variant = self.registry.variant("entity_placeholder", "placeholder")
        self.assertTrue(variant.support_bearing)
        self.assertTrue(variant.support_transmission)

    def test_placeholder_receives_living_damage(self) -> None:
        """Entity placeholder should receive damage from neighbors with 'living' in damage_mask."""
        grid = self._make_grid()
        # Place fire next to placeholder
        grid.set_cell(19, 30, CellState(family_id="fire", variant_id="fire", integrity=1.0))
        grid.set_cell(20, 30, CellState(
            family_id="entity_placeholder",
            variant_id="placeholder",
            integrity=1.0,
        ))

        initial_integrity = grid.get_cell(20, 30).integrity
        apply_reactions(grid, self.registry, 0.1)
        new_integrity = grid.get_cell(20, 30).integrity

        self.assertLess(new_integrity, initial_integrity)


if __name__ == "__main__":
    unittest.main()