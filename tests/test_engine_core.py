from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine.grid import create_grid
from engine.materials import build_material_registry
from engine.phases import apply_phase_transitions
from engine.render import build_rgba_frame
from engine.scenarios import populate_demo_scene
from engine.sim import inject_cells, step
from engine.types import CellFlag, CellState


class EngineCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_material_registry()

    def test_slm_imports_still_work_after_package_rename(self) -> None:
        import slm.data_generation  # noqa: F401
        import slm.io_utils  # noqa: F401
        import slm.model_socket_schema  # noqa: F401

    def test_powder_falls_downward(self) -> None:
        grid = create_grid(3, 4)
        inject_cells(grid, [(1, 0)], "sand", "sand_powder")
        for _ in range(3):
            step(grid, self.registry, 1.0)
        self.assertEqual(grid.get_cell(1, 3).variant_id, "sand_powder")

    def test_support_loss_turns_platform_into_same_family_powder(self) -> None:
        grid = create_grid(4, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT, "support_value": 1.0})
        inject_cells(grid, [(1, 0), (2, 0), (3, 0)], "stone", "stone_platform")
        for _ in range(8):
            step(grid, self.registry, 1.0)
        self.assertGreater(grid.get_cell(3, 0).support_value, 0.0)

        grid.set_cell(1, 0, CellState())
        for _ in range(50):
            step(grid, self.registry, 1.0)
        self.assertEqual(grid.get_cell(3, 0).variant_id, "stone_powder")

    def test_water_phase_transitions_cover_ice_and_steam(self) -> None:
        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": 120.0})
        apply_phase_transitions(grid, self.registry, 1.0)
        self.assertEqual(grid.get_cell(0, 0).variant_id, "steam")

        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": -5.0})
        apply_phase_transitions(grid, self.registry, 1.0)
        self.assertEqual(grid.get_cell(0, 0).variant_id, "ice")

    def test_sand_turns_into_molten_glass_when_overheated(self) -> None:
        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "sand", "sand_powder", {"temperature": 1300.0})
        apply_phase_transitions(grid, self.registry, 1.0)
        cell = grid.get_cell(0, 0)
        self.assertEqual(cell.family_id, "glass")
        self.assertEqual(cell.variant_id, "molten_glass")

    def test_fire_uses_age_and_burns_out(self) -> None:
        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire")
        for _ in range(8):
            step(grid, self.registry, 1.0)
        self.assertTrue(grid.get_cell(0, 0).is_empty)

    def test_render_frame_matches_grid_shape(self) -> None:
        grid = create_grid(4, 3)
        inject_cells(grid, [(1, 1)], "stone", "stone_platform")
        frame = build_rgba_frame(grid, self.registry)
        self.assertEqual(len(frame), 4 * 3 * 4)

    def test_demo_scene_contains_fixpoints_and_liquids(self) -> None:
        grid = create_grid(40, 24)
        populate_demo_scene(grid)
        fixpoints = 0
        liquids = 0
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get_cell(x, y)
                if cell.flags & CellFlag.FIXPOINT:
                    fixpoints += 1
                if cell.variant_id in {"water", "tar_liquid", "acid_liquid", "poison_liquid"}:
                    liquids += 1
        self.assertGreaterEqual(fixpoints, 2)
        self.assertGreater(liquids, 0)


if __name__ == "__main__":
    unittest.main()
