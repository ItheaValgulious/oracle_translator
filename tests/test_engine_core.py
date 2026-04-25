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

import moderngl
import pyglet

from engine.gpu_backend import GpuSimulator, GpuMaterialTables, pack_grid_state, unpack_grid_state
from engine.grid import create_grid
from engine.materials import build_material_registry
from engine.motion import apply_motion
from engine.phases import apply_phase_transitions
from engine.reactions import apply_reactions
from engine.render import DebugViewMode, build_rgba_frame
from engine.scenarios import populate_demo_scene
from engine.sim import inject_cells, step
from engine.support import SUPPORT_SOURCE_VALUE
from engine.thermal import apply_thermal
from engine.types import CellFlag, CellState
from scripts.run_engine_demo import parse_args


_GPU_TEST_WINDOWS: list[pyglet.window.Window] = []


def create_compute_context() -> moderngl.Context:
    try:
        return moderngl.create_standalone_context(require=430)
    except Exception as standalone_error:  # noqa: BLE001
        try:
            window = pyglet.window.Window(width=32, height=32, visible=False)
            window.switch_to()
            context = moderngl.create_context(require=430)
            _GPU_TEST_WINDOWS.append(window)
            return context
        except Exception as window_error:  # noqa: BLE001
            if "window" in locals():
                window.close()
            raise RuntimeError(
                "Failed to create an OpenGL 4.3 compute context via standalone or hidden-window paths. "
                f"standalone={standalone_error!r}; window={window_error!r}"
            ) from window_error


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

    def test_downward_gravity_does_not_inject_reverse_impulse(self) -> None:
        grid = create_grid(5, 5)
        inject_cells(grid, [(2, 1)], "sand", "sand_powder")
        step(grid, self.registry, 1 / 60.0)
        cell = grid.get_cell(2, 2)
        self.assertEqual(cell.variant_id, "sand_powder")
        self.assertGreaterEqual(cell.blocked_y, 0.0)

    def test_diagonal_motion_consumes_remaining_axis_intent(self) -> None:
        grid = create_grid(7, 7)
        inject_cells(grid, [(3, 3)], "sand", "sand_powder", {"vel_x": 1.0, "vel_y": 2.0})
        step(grid, self.registry, 1 / 60.0)
        step(grid, self.registry, 1 / 60.0)
        self.assertEqual(grid.get_cell(4, 5).variant_id, "sand_powder")

    def test_liquid_spreads_sideways_on_floor(self) -> None:
        grid = create_grid(9, 6)
        for x in range(9):
            inject_cells(grid, [(x, 5)], "stone", "stone_platform")
        inject_cells(grid, [(4, 1), (4, 2), (4, 3)], "water", "water")
        for _ in range(20):
            step(grid, self.registry, 1 / 60.0)
        xs = {x for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y).variant_id == "water"}
        self.assertGreater(len(xs), 1)

    def test_disabling_liquid_brownian_removes_horizontal_random_velocity(self) -> None:
        enabled_grid = create_grid(5, 5)
        enabled_grid.liquid_brownian_enabled = True
        inject_cells(enabled_grid, [(2, 1)], "water", "water")
        step(enabled_grid, self.registry, 1 / 60.0)
        enabled_water = next(cell for cell in enabled_grid.cells if cell.variant_id == "water")

        disabled_grid = create_grid(5, 5)
        disabled_grid.liquid_brownian_enabled = False
        inject_cells(disabled_grid, [(2, 1)], "water", "water")
        step(disabled_grid, self.registry, 1 / 60.0)
        disabled_water = next(cell for cell in disabled_grid.cells if cell.variant_id == "water")

        self.assertAlmostEqual(disabled_water.vel_x, 0.0, places=6)
        self.assertNotAlmostEqual(enabled_water.vel_x, 0.0, places=6)

    def test_disabling_blocked_impulse_clears_persisted_intent(self) -> None:
        enabled_grid = create_grid(5, 5)
        enabled_grid.liquid_brownian_enabled = False
        enabled_grid.blocked_impulse_enabled = True
        surround = [(x, y) for y in range(1, 4) for x in range(1, 4) if (x, y) != (2, 2)]
        inject_cells(enabled_grid, surround, "stone", "stone_platform")
        inject_cells(enabled_grid, [(2, 2)], "water", "water", {"blocked_x": 1.25, "blocked_y": -0.75})
        step(enabled_grid, self.registry, 1 / 60.0)
        enabled_water = enabled_grid.get_cell(2, 2)

        disabled_grid = create_grid(5, 5)
        disabled_grid.liquid_brownian_enabled = False
        disabled_grid.blocked_impulse_enabled = False
        inject_cells(disabled_grid, surround, "stone", "stone_platform")
        inject_cells(disabled_grid, [(2, 2)], "water", "water", {"blocked_x": 1.25, "blocked_y": -0.75})
        step(disabled_grid, self.registry, 1 / 60.0)
        disabled_water = disabled_grid.get_cell(2, 2)

        self.assertGreater(abs(enabled_water.blocked_x) + abs(enabled_water.blocked_y), 0.1)
        self.assertAlmostEqual(disabled_water.blocked_x, 0.0, places=6)
        self.assertAlmostEqual(disabled_water.blocked_y, 0.0, places=6)

    def test_gas_brownian_motion_changes_horizontal_position(self) -> None:
        grid = create_grid(32, 24)
        inject_cells(
            grid,
            {"x": 16, "y": 12, "radius": 64},
            "empty",
            "empty",
            {"temperature": 60.0},
            registry=self.registry,
        )
        inject_cells(grid, [(16, 12)], "poison", "poison_gas")
        visited_x = set()
        for _ in range(12):
            step(grid, self.registry, 1 / 60.0)
            poison_positions = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y).variant_id == "poison_gas"]
            self.assertEqual(len(poison_positions), 1)
            visited_x.add(poison_positions[0][0])
        self.assertGreater(len(visited_x), 1)

    def test_heavier_liquid_can_sink_through_lighter_liquid(self) -> None:
        grid = create_grid(3, 6)
        for y in range(6):
            inject_cells(grid, [(0, y), (2, y)], "stone", "stone_platform")
        for x in range(3):
            inject_cells(grid, [(x, 5)], "stone", "stone_platform")
        inject_cells(grid, [(1, 3)], "tar", "tar_liquid")
        inject_cells(grid, [(1, 4)], "water", "water")
        step(grid, self.registry, 1 / 60.0)
        self.assertEqual(grid.get_cell(1, 3).variant_id, "water")
        self.assertEqual(grid.get_cell(1, 4).variant_id, "tar_liquid")

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

    def test_temperature_view_uses_alternate_palette(self) -> None:
        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"temperature": 20.0})
        inject_cells(grid, [(1, 0)], "stone", "stone_platform", {"temperature": 1000.0})
        material_frame = build_rgba_frame(grid, self.registry)
        temperature_frame = build_rgba_frame(grid, self.registry, view_mode=DebugViewMode.TEMPERATURE)
        self.assertNotEqual(material_frame, temperature_frame)

    def test_temperature_does_not_relax_back_to_material_base_temperature(self) -> None:
        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": 80.0})
        apply_thermal(grid, self.registry, 1.0)
        self.assertEqual(grid.get_cell(0, 0).temperature, 80.0)

    def test_platforms_conduct_heat_to_each_other(self) -> None:
        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"temperature": 200.0})
        inject_cells(grid, [(1, 0)], "stone", "stone_platform", {"temperature": 20.0})
        apply_thermal(grid, self.registry, 1 / 60.0)
        self.assertLess(grid.get_cell(0, 0).temperature, 200.0)
        self.assertGreater(grid.get_cell(1, 0).temperature, 20.0)

    def test_hot_air_cools_by_conducting_to_cool_neighbors(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0), (2, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 600.0}, registry=self.registry)
        apply_thermal(grid, self.registry, 1 / 60.0)
        self.assertLess(grid.get_cell(1, 0).temperature, 560.0)
        self.assertGreater(grid.get_cell(0, 0).temperature, 40.0)
        self.assertGreater(grid.get_cell(2, 0).temperature, 40.0)

    def test_hot_empty_air_advects_upward(self) -> None:
        grid = create_grid(1, 5)
        inject_cells(grid, [(0, 3)], "empty", "empty", {"temperature": 200.0}, registry=self.registry)
        for _ in range(3):
            step(grid, self.registry, 1 / 60.0)
        hottest_y = max(range(grid.height), key=lambda y: grid.get_cell(0, y).temperature)
        self.assertLess(hottest_y, 3)

    def test_temperature_moves_with_moving_cell(self) -> None:
        grid = create_grid(3, 3)
        inject_cells(grid, [(1, 0)], "sand", "sand_powder", {"temperature": 80.0})
        apply_motion(grid, self.registry, 1 / 60.0)
        self.assertTrue(grid.get_cell(1, 0).is_empty)
        self.assertEqual(grid.get_cell(1, 1).variant_id, "sand_powder")
        self.assertEqual(grid.get_cell(1, 1).temperature, 80.0)

    def test_empty_air_temperature_is_preserved_when_cell_moves_into_it(self) -> None:
        grid = create_grid(2, 1)
        grid.liquid_brownian_enabled = False
        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": 20.0, "vel_x": 1.0})
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 90.0}, registry=self.registry)
        apply_motion(grid, self.registry, 1 / 60.0)
        self.assertTrue(grid.get_cell(0, 0).is_empty)
        self.assertEqual(grid.get_cell(0, 0).temperature, 90.0)
        self.assertEqual(grid.get_cell(1, 0).variant_id, "water")
        self.assertEqual(grid.get_cell(1, 0).temperature, 20.0)

    def test_fire_leaves_hot_air_when_it_moves(self) -> None:
        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire", {"temperature": 600.0, "vel_x": 1.0})
        apply_motion(grid, self.registry, 1 / 60.0)
        self.assertTrue(grid.get_cell(0, 0).is_empty)
        self.assertGreater(grid.get_cell(0, 0).temperature, 500.0)
        self.assertEqual(grid.get_cell(1, 0).variant_id, "fire")
        self.assertGreater(grid.get_cell(1, 0).temperature, 500.0)

    def test_expired_fire_leaves_hot_air(self) -> None:
        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire", {"temperature": 600.0, "age": 5.9})
        apply_reactions(grid, self.registry, 1.0)
        self.assertTrue(grid.get_cell(0, 0).is_empty)
        self.assertGreater(grid.get_cell(0, 0).temperature, 500.0)

    def test_fire_reaction_energy_heats_fire_cell_before_diffusion(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0), (2, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(1, 0)], "fire", "fire", {"temperature": 100.0})
        apply_reactions(grid, self.registry, 1.0)
        self.assertGreater(grid.get_cell(1, 0).temperature, 100.0)
        self.assertEqual(grid.get_cell(0, 0).temperature, 20.0)
        self.assertEqual(grid.get_cell(2, 0).temperature, 20.0)
        apply_thermal(grid, self.registry, 1 / 60.0)
        self.assertGreater(grid.get_cell(0, 0).temperature, 20.0)
        self.assertGreater(grid.get_cell(2, 0).temperature, 20.0)

    def test_fire_trail_hot_air_cools_after_fire_moves_away(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(1, 0)], "fire", "fire", {"temperature": 600.0, "vel_x": 1.0})
        apply_motion(grid, self.registry, 1 / 60.0)
        before = grid.get_cell(1, 0).temperature
        apply_thermal(grid, self.registry, 1 / 60.0)
        self.assertLess(grid.get_cell(1, 0).temperature, before)

    def test_pressure_view_uses_pressure_palette(self) -> None:
        grid = create_grid(1, 4)
        inject_cells(grid, [(0, 1), (0, 2)], "water", "water")
        step(grid, self.registry, 1 / 60.0)
        pressure_frame = build_rgba_frame(grid, self.registry, view_mode=DebugViewMode.PRESSURE)
        top_pixel = pressure_frame[4:8]
        bottom_pixel = pressure_frame[8:12]
        self.assertNotEqual(top_pixel, bottom_pixel)

    def test_support_signal_propagates_across_platform_chain(self) -> None:
        grid = create_grid(8, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT, "support_value": 1.0})
        inject_cells(grid, [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)], "stone", "stone_platform")
        for _ in range(7):
            step(grid, self.registry, 1 / 60.0)
        self.assertGreater(grid.get_cell(7, 0).support_value, 0.5)

    def test_support_signal_moves_one_cell_per_step(self) -> None:
        grid = create_grid(4, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT})
        inject_cells(grid, [(1, 0), (2, 0), (3, 0)], "stone", "stone_platform")
        step(grid, self.registry, 1 / 60.0)
        self.assertEqual(grid.get_cell(1, 0).support_value, SUPPORT_SOURCE_VALUE)
        self.assertEqual(grid.get_cell(2, 0).support_value, 0.0)

    def test_support_signal_has_no_distance_decay_along_connected_platforms(self) -> None:
        width = 64
        grid = create_grid(width, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT, "support_value": 1.0})
        inject_cells(grid, [(x, 0) for x in range(1, width)], "stone", "stone_platform")
        for _ in range(width - 1):
            step(grid, self.registry, 1 / 60.0)
        self.assertEqual(grid.get_cell(width - 1, 0).support_value, SUPPORT_SOURCE_VALUE)

    def test_platform_integrity_decays_only_after_support_timeout(self) -> None:
        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT})
        inject_cells(grid, [(1, 0)], "stone", "stone_platform")
        step(grid, self.registry, 1 / 60.0)

        grid.set_cell(0, 0, CellState())
        step(grid, self.registry, 9.0)
        self.assertEqual(grid.get_cell(1, 0).integrity, 1.0)

        step(grid, self.registry, 2.0)
        self.assertLess(grid.get_cell(1, 0).integrity, 1.0)

    def test_heat_can_propagate_through_air_gap(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"temperature": 600.0})
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(2, 0)], "stone", "stone_platform", {"temperature": 20.0})
        for _ in range(60):
            step(grid, self.registry, 1 / 60.0)
        self.assertGreater(grid.get_cell(1, 0).temperature, 20.5)

    def test_fire_heats_adjacent_water_before_motion(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire")
        inject_cells(grid, [(1, 0)], "water", "water", {"temperature": 20.0})
        step(grid, self.registry, 1 / 60.0)
        water = next(cell for cell in grid.cells if cell.variant_id == "water")
        self.assertGreater(water.temperature, 20.0)

    def test_fire_heats_water_through_air_gap(self) -> None:
        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire")
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(2, 0)], "water", "water", {"temperature": 20.0})
        for _ in range(3):
            step(grid, self.registry, 1 / 60.0)
        water = next(cell for cell in grid.cells if cell.variant_id == "water")
        self.assertGreater(water.temperature, 20.0)

    def test_acid_is_consumed_when_it_successfully_corroded_target(self) -> None:
        grid = create_grid(3, 3)
        inject_cells(grid, [(1, 1)], "acid", "acid_liquid")
        inject_cells(grid, [(1, 2)], "stone", "stone_platform")
        step(grid, self.registry, 1 / 60.0)
        self.assertTrue(grid.get_cell(1, 1).is_empty)
        self.assertLess(grid.get_cell(1, 2).integrity, 1.0)

    def test_fire_and_water_motion_do_not_create_extra_mass(self) -> None:
        grid = create_grid(20, 12)
        for x in range(20):
            inject_cells(grid, [(x, 11)], "stone", "stone_platform")
        inject_cells(grid, {"x": 10, "y": 8, "radius": 2}, "fire", "fire", registry=self.registry)
        inject_cells(grid, {"x": 10, "y": 5, "radius": 1}, "water", "water", registry=self.registry)

        initial_fire = sum(1 for cell in grid.cells if cell.variant_id == "fire")
        initial_water_family = sum(1 for cell in grid.cells if cell.variant_id in {"water", "steam", "ice"})

        for _ in range(10):
            step(grid, self.registry, 1 / 60.0)

        final_fire = sum(1 for cell in grid.cells if cell.variant_id == "fire")
        final_water_family = sum(1 for cell in grid.cells if cell.variant_id in {"water", "steam", "ice"})
        self.assertLessEqual(final_fire, initial_fire)
        self.assertLessEqual(final_water_family, initial_water_family)

    def test_falling_powder_does_not_lift_water_column_high_into_air(self) -> None:
        grid = create_grid(7, 12)
        for x in range(7):
            inject_cells(grid, [(x, 11)], "stone", "stone_platform")
        for y in range(2, 10):
            inject_cells(grid, [(3, y)], "sand", "sand_powder")
        inject_cells(grid, [(3, 10)], "water", "water")

        for _ in range(18):
            step(grid, self.registry, 1 / 60.0)

        water_positions = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.get_cell(x, y).variant_id == "water"]
        self.assertEqual(len(water_positions), 1)
        self.assertGreaterEqual(water_positions[0][1], 9)

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

    def test_gpu_grid_pack_roundtrip_preserves_state(self) -> None:
        grid = create_grid(4, 3)
        inject_cells(
            grid,
            [(1, 1)],
            "stone",
            "stone_platform",
            {"flags": CellFlag.FIXPOINT, "support_value": 1.0, "temperature": 220.0},
        )
        inject_cells(grid, [(2, 2)], "water", "water", {"vel_x": 0.5, "blocked_y": 0.25})
        tables = GpuMaterialTables.from_registry(self.registry)
        packed = pack_grid_state(grid, tables)
        restored = unpack_grid_state(grid.width, grid.height, tables, *packed)

        stone = restored.get_cell(1, 1)
        water = restored.get_cell(2, 2)
        self.assertEqual(stone.variant_id, "stone_platform")
        self.assertTrue(stone.flags & CellFlag.FIXPOINT)
        self.assertAlmostEqual(stone.temperature, 220.0)
        self.assertEqual(water.variant_id, "water")
        self.assertAlmostEqual(water.vel_x, 0.5)
        self.assertAlmostEqual(water.blocked_y, 0.25)

    def test_gpu_backend_runs_phase_and_lifetime_rules(self) -> None:
        ctx = create_compute_context()

        water_grid = create_grid(1, 1)
        inject_cells(water_grid, [(0, 0)], "water", "water", {"temperature": 120.0})
        gpu = GpuSimulator(ctx, water_grid, self.registry)
        gpu.step(0.0)
        self.assertEqual(gpu.readback_grid().get_cell(0, 0).variant_id, "steam")

        fire_grid = create_grid(1, 1)
        inject_cells(fire_grid, [(0, 0)], "fire", "fire")
        gpu.load_grid(fire_grid)
        for _ in range(8):
            gpu.step(1.0)
        self.assertTrue(gpu.readback_grid().get_cell(0, 0).is_empty)

    def test_gpu_disabling_liquid_brownian_removes_horizontal_random_velocity(self) -> None:
        ctx = create_compute_context()

        enabled_grid = create_grid(5, 5)
        enabled_grid.liquid_brownian_enabled = True
        inject_cells(enabled_grid, [(2, 1)], "water", "water")
        enabled_gpu = GpuSimulator(ctx, enabled_grid, self.registry)
        enabled_gpu.step(1 / 60.0)
        enabled_water = next(cell for cell in enabled_gpu.readback_grid().cells if cell.variant_id == "water")

        disabled_grid = create_grid(5, 5)
        disabled_grid.liquid_brownian_enabled = False
        inject_cells(disabled_grid, [(2, 1)], "water", "water")
        disabled_gpu = GpuSimulator(ctx, disabled_grid, self.registry)
        disabled_gpu.step(1 / 60.0)
        disabled_water = next(cell for cell in disabled_gpu.readback_grid().cells if cell.variant_id == "water")

        self.assertAlmostEqual(disabled_water.vel_x, 0.0, places=6)
        self.assertNotAlmostEqual(enabled_water.vel_x, 0.0, places=6)

    def test_gpu_disabling_blocked_impulse_clears_persisted_intent(self) -> None:
        ctx = create_compute_context()

        surround = [(x, y) for y in range(1, 4) for x in range(1, 4) if (x, y) != (2, 2)]

        enabled_grid = create_grid(5, 5)
        enabled_grid.liquid_brownian_enabled = False
        enabled_grid.blocked_impulse_enabled = True
        inject_cells(enabled_grid, surround, "stone", "stone_platform")
        inject_cells(enabled_grid, [(2, 2)], "water", "water", {"blocked_x": 1.25, "blocked_y": -0.75})
        enabled_gpu = GpuSimulator(ctx, enabled_grid, self.registry)
        enabled_gpu.step(1 / 60.0)
        enabled_water = enabled_gpu.readback_grid().get_cell(2, 2)

        disabled_grid = create_grid(5, 5)
        disabled_grid.liquid_brownian_enabled = False
        disabled_grid.blocked_impulse_enabled = False
        inject_cells(disabled_grid, surround, "stone", "stone_platform")
        inject_cells(disabled_grid, [(2, 2)], "water", "water", {"blocked_x": 1.25, "blocked_y": -0.75})
        disabled_gpu = GpuSimulator(ctx, disabled_grid, self.registry)
        disabled_gpu.step(1 / 60.0)
        disabled_water = disabled_gpu.readback_grid().get_cell(2, 2)

        self.assertGreater(abs(enabled_water.blocked_x) + abs(enabled_water.blocked_y), 0.1)
        self.assertAlmostEqual(disabled_water.blocked_x, 0.0, places=6)
        self.assertAlmostEqual(disabled_water.blocked_y, 0.0, places=6)

    def test_gpu_pressure_view_uses_pressure_palette(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(1, 4)
        inject_cells(grid, [(0, 1), (0, 2)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        material_frame = gpu.render(DebugViewMode.MATERIAL).read()
        pressure_frame = gpu.render(DebugViewMode.PRESSURE).read()
        self.assertNotEqual(material_frame, pressure_frame)

    def test_gpu_temperature_does_not_relax_back_to_material_base_temperature(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": 80.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(0.0)
        self.assertEqual(gpu.readback_grid().get_cell(0, 0).temperature, 80.0)

    def test_gpu_platforms_conduct_heat_to_each_other(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"temperature": 200.0})
        inject_cells(grid, [(1, 0)], "stone", "stone_platform", {"temperature": 20.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertLess(readback.get_cell(0, 0).temperature, 200.0)
        self.assertGreater(readback.get_cell(1, 0).temperature, 20.0)

    def test_gpu_hot_air_cools_by_conducting_to_cool_neighbors(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0), (2, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 600.0}, registry=self.registry)
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(0.0)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertLess(readback.get_cell(1, 0).temperature, 560.0)
        self.assertGreater(readback.get_cell(0, 0).temperature, 40.0)
        self.assertGreater(readback.get_cell(2, 0).temperature, 40.0)

    def test_gpu_hot_empty_air_advects_upward(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(1, 5)
        inject_cells(grid, [(0, 3)], "empty", "empty", {"temperature": 200.0}, registry=self.registry)
        gpu = GpuSimulator(ctx, grid, self.registry)
        for _ in range(3):
            gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        hottest_y = max(range(readback.height), key=lambda y: readback.get_cell(0, y).temperature)
        self.assertLess(hottest_y, 3)

    def test_gpu_temperature_moves_with_moving_cell(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 3)
        inject_cells(grid, [(1, 0)], "sand", "sand_powder", {"temperature": 80.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertTrue(readback.get_cell(1, 0).is_empty)
        self.assertEqual(readback.get_cell(1, 1).variant_id, "sand_powder")
        self.assertGreater(readback.get_cell(1, 1).temperature, 70.0)

    def test_gpu_empty_air_temperature_is_preserved_when_cell_moves_into_it(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(2, 1)
        grid.liquid_brownian_enabled = False
        inject_cells(grid, [(0, 0)], "water", "water", {"temperature": 20.0, "vel_x": 1.0})
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 90.0}, registry=self.registry)
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(0.0)
        readback = gpu.readback_grid()
        self.assertTrue(readback.get_cell(0, 0).is_empty)
        self.assertEqual(readback.get_cell(0, 0).temperature, 90.0)
        self.assertEqual(readback.get_cell(1, 0).variant_id, "water")
        self.assertEqual(readback.get_cell(1, 0).temperature, 20.0)

    def test_gpu_fire_leaves_hot_air_when_it_moves(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire", {"temperature": 600.0, "vel_x": 1.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(0.0)
        readback = gpu.readback_grid()
        self.assertTrue(readback.get_cell(0, 0).is_empty)
        self.assertGreater(readback.get_cell(0, 0).temperature, 500.0)
        self.assertEqual(readback.get_cell(1, 0).variant_id, "fire")
        self.assertGreater(readback.get_cell(1, 0).temperature, 500.0)

    def test_gpu_expired_fire_leaves_hot_air(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire", {"temperature": 600.0, "age": 5.9})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1.0)
        readback = gpu.readback_grid()
        self.assertTrue(readback.get_cell(0, 0).is_empty)
        self.assertGreater(readback.get_cell(0, 0).temperature, 500.0)

    def test_gpu_fire_reaction_energy_heats_fire_cell_before_diffusion(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(1, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire", {"temperature": 100.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1.0)
        readback = gpu.readback_grid()
        self.assertGreater(readback.get_cell(0, 0).temperature, 100.0)

    def test_gpu_fire_trail_hot_air_cools_after_fire_moves_away(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 1)
        inject_cells(grid, [(1, 0)], "fire", "fire", {"temperature": 600.0, "vel_x": 1.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(0.0)
        before = gpu.readback_grid().get_cell(1, 0).temperature
        gpu.step(1 / 60.0)
        self.assertLess(gpu.readback_grid().get_cell(1, 0).temperature, before)

    def test_gpu_fire_heats_adjacent_water_before_motion(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire")
        inject_cells(grid, [(1, 0)], "water", "water", {"temperature": 20.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        water = next(cell for cell in gpu.readback_grid().cells if cell.variant_id == "water")
        self.assertGreater(water.temperature, 20.0)

    def test_gpu_fire_heats_water_through_air_gap(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 1)
        inject_cells(grid, [(0, 0)], "fire", "fire")
        inject_cells(grid, [(1, 0)], "empty", "empty", {"temperature": 20.0}, registry=self.registry)
        inject_cells(grid, [(2, 0)], "water", "water", {"temperature": 20.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        for _ in range(3):
            gpu.step(1 / 60.0)
        water = next(cell for cell in gpu.readback_grid().cells if cell.variant_id == "water")
        self.assertGreater(water.temperature, 20.0)

    def test_gpu_motion_consumes_remaining_axis_intent(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(7, 7)
        inject_cells(grid, [(3, 3)], "sand", "sand_powder", {"vel_x": 1.0, "vel_y": 2.0})
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        gpu.step(1 / 60.0)
        self.assertEqual(gpu.readback_grid().get_cell(4, 5).variant_id, "sand_powder")

    def test_gpu_support_signal_moves_one_cell_per_step(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(4, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT})
        inject_cells(grid, [(1, 0), (2, 0), (3, 0)], "stone", "stone_platform")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertEqual(readback.get_cell(1, 0).support_value, SUPPORT_SOURCE_VALUE)
        self.assertEqual(readback.get_cell(2, 0).support_value, 0.0)

    def test_gpu_platform_integrity_decays_only_after_support_timeout(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(2, 1)
        inject_cells(grid, [(0, 0)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT})
        inject_cells(grid, [(1, 0)], "stone", "stone_platform")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        gpu.paint_circle(0, 0, 0, None, None)
        gpu.step(9.0)
        self.assertEqual(gpu.readback_grid().get_cell(1, 0).integrity, 1.0)

        gpu.step(2.0)
        self.assertLess(gpu.readback_grid().get_cell(1, 0).integrity, 1.0)

    def test_gpu_acid_is_consumed_when_it_corroded_target(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 3)
        inject_cells(grid, [(1, 1)], "acid", "acid_liquid")
        inject_cells(grid, [(1, 2)], "stone", "stone_platform")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertTrue(readback.get_cell(1, 1).is_empty)
        self.assertLess(readback.get_cell(1, 2).integrity, 1.0)

    def test_gpu_fire_and_water_motion_do_not_create_extra_mass(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(20, 12)
        for x in range(20):
            inject_cells(grid, [(x, 11)], "stone", "stone_platform")
        inject_cells(grid, {"x": 10, "y": 8, "radius": 2}, "fire", "fire", registry=self.registry)
        inject_cells(grid, {"x": 10, "y": 5, "radius": 1}, "water", "water", registry=self.registry)
        gpu = GpuSimulator(ctx, grid, self.registry)

        initial = gpu.readback_grid()
        initial_fire = sum(1 for cell in initial.cells if cell.variant_id == "fire")
        initial_water_family = sum(1 for cell in initial.cells if cell.variant_id in {"water", "steam", "ice"})

        for _ in range(10):
            gpu.step(1 / 60.0)

        final = gpu.readback_grid()
        final_fire = sum(1 for cell in final.cells if cell.variant_id == "fire")
        final_water_family = sum(1 for cell in final.cells if cell.variant_id in {"water", "steam", "ice"})
        self.assertLessEqual(final_fire, initial_fire)
        self.assertLessEqual(final_water_family, initial_water_family)

    def test_gpu_liquid_fall_preserves_water_cell_count(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(12, 10)
        for x in range(12):
            inject_cells(grid, [(x, 9)], "stone", "stone_platform")
        inject_cells(grid, {"x": 6, "y": 2, "radius": 2}, "water", "water", registry=self.registry)
        gpu = GpuSimulator(ctx, grid, self.registry)

        initial_water = sum(1 for cell in gpu.readback_grid().cells if cell.variant_id == "water")
        for _ in range(6):
            gpu.step(1 / 60.0)
        final_water = sum(1 for cell in gpu.readback_grid().cells if cell.variant_id == "water")
        self.assertEqual(final_water, initial_water)

    def test_gpu_liquid_moves_sideways_immediately_when_floor_blocked(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(5, 4)
        for x in range(5):
            inject_cells(grid, [(x, 3)], "stone", "stone_platform")
        inject_cells(grid, [(2, 2)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        water_positions = [(x, y) for y in range(readback.height) for x in range(readback.width) if readback.get_cell(x, y).variant_id == "water"]
        self.assertEqual(len(water_positions), 1)
        self.assertNotEqual(water_positions[0], (2, 2))

    def test_gpu_gas_brownian_motion_changes_horizontal_position(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(32, 24)
        inject_cells(
            grid,
            {"x": 16, "y": 12, "radius": 64},
            "empty",
            "empty",
            {"temperature": 60.0},
            registry=self.registry,
        )
        inject_cells(grid, [(16, 12)], "poison", "poison_gas")
        gpu = GpuSimulator(ctx, grid, self.registry)

        visited_x = set()
        for _ in range(12):
            gpu.step(1 / 60.0)
            readback = gpu.readback_grid()
            poison_positions = [(x, y) for y in range(readback.height) for x in range(readback.width) if readback.get_cell(x, y).variant_id == "poison_gas"]
            self.assertEqual(len(poison_positions), 1)
            visited_x.add(poison_positions[0][0])

        self.assertGreater(len(visited_x), 1)

    def test_gpu_large_liquid_reservoir_spreads_across_floor(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(64, 40)
        for x in range(64):
            inject_cells(grid, [(x, 39)], "stone", "stone_platform")
        for x in range(16):
            for y in range(12, 39):
                inject_cells(grid, [(x, y)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)

        for _ in range(60):
            gpu.step(1 / 60.0)

        readback = gpu.readback_grid()
        xs = {x for y in range(readback.height) for x in range(readback.width) if readback.get_cell(x, y).variant_id == "water"}
        self.assertGreaterEqual(len(xs), 48)

    def test_gpu_large_liquid_pile_lowers_its_peak_noticeably(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(80, 80)
        for x in range(80):
            inject_cells(grid, [(x, 79)], "stone", "stone_platform")
        center = 40
        for x in range(20, 61):
            dx = abs(x - center)
            top = 18 + dx // 2
            for y in range(top, 79):
                inject_cells(grid, [(x, y)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)

        for _ in range(60):
            gpu.step(1 / 60.0)

        readback = gpu.readback_grid()
        heights = []
        for x in range(readback.width):
            ys = [y for y in range(readback.height) if readback.get_cell(x, y).variant_id == "water"]
            heights.append((readback.height - 1 - min(ys)) if ys else 0)
        self.assertLessEqual(max(heights), 45)

    def test_gpu_heavier_liquid_swaps_downward_with_lighter_liquid(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 6)
        for y in range(6):
            inject_cells(grid, [(0, y), (2, y)], "stone", "stone_platform")
        for x in range(3):
            inject_cells(grid, [(x, 5)], "stone", "stone_platform")
        inject_cells(grid, [(1, 3)], "tar", "tar_liquid")
        inject_cells(grid, [(1, 4)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertEqual(readback.get_cell(1, 3).variant_id, "water")
        self.assertEqual(readback.get_cell(1, 4).variant_id, "tar_liquid")

    def test_gpu_liquid_swaps_downward_with_lighter_gas(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(3, 6)
        for y in range(6):
            inject_cells(grid, [(0, y), (2, y)], "stone", "stone_platform")
        for x in range(3):
            inject_cells(grid, [(x, 5)], "stone", "stone_platform")
        inject_cells(grid, [(1, 3)], "water", "water")
        inject_cells(grid, [(1, 4)], "poison", "poison_gas")
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.step(1 / 60.0)
        readback = gpu.readback_grid()
        self.assertEqual(readback.get_cell(1, 3).variant_id, "poison_gas")
        self.assertEqual(readback.get_cell(1, 4).variant_id, "water")

    def test_gpu_falling_powder_does_not_lift_water_column_high_into_air(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(7, 12)
        for x in range(7):
            inject_cells(grid, [(x, 11)], "stone", "stone_platform")
        for y in range(2, 10):
            inject_cells(grid, [(3, y)], "sand", "sand_powder")
        inject_cells(grid, [(3, 10)], "water", "water")
        gpu = GpuSimulator(ctx, grid, self.registry)

        for _ in range(18):
            gpu.step(1 / 60.0)

        readback = gpu.readback_grid()
        water_positions = [(x, y) for y in range(readback.height) for x in range(readback.width) if readback.get_cell(x, y).variant_id == "water"]
        self.assertEqual(len(water_positions), 1)
        self.assertGreaterEqual(water_positions[0][1], 9)

    def test_gpu_paint_circle_accepts_float_like_coordinates(self) -> None:
        ctx = create_compute_context()

        grid = create_grid(4, 4)
        gpu = GpuSimulator(ctx, grid, self.registry)
        gpu.paint_circle(1.5, 2.5, 1.0, "stone", "stone_platform")
        painted = gpu.readback_grid()
        self.assertEqual(painted.get_cell(1, 2).variant_id, "stone_platform")

    def test_demo_cli_parses_grid_and_window_options(self) -> None:
        args = parse_args(
            [
                "--grid-width",
                "200",
                "--grid-height",
                "120",
                "--window-width",
                "1440",
                "--window-height",
                "900",
                "--cell-scale",
                "4",
                "--substeps",
                "3",
                "--no-liquid-brownian",
                "--no-blocked-impulse",
                "--no-vsync",
            ]
        )
        self.assertEqual(args.grid_width, 200)
        self.assertEqual(args.grid_height, 120)
        self.assertEqual(args.window_width, 1440)
        self.assertEqual(args.window_height, 900)
        self.assertEqual(args.cell_scale, 4)
        self.assertEqual(args.substeps, 3)
        self.assertTrue(args.no_liquid_brownian)
        self.assertTrue(args.no_blocked_impulse)
        self.assertTrue(args.no_vsync)


if __name__ == "__main__":
    unittest.main()
