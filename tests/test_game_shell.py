"""Tests for Phase 1 game shell: hero, entity_manager, spell_system, animation, screens."""

from __future__ import annotations

import sys
import tempfile
import unittest
import time
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.materials import build_material_registry
from engine.types import CellFlag, CellState
from engine.render import DebugViewMode
from engine.world import ActiveWorldWindow, GpuChunkCache, WorldChunkStore, WorldRect
from engine.gpu_backend import GpuMaterialTables, pack_cells_state, unpack_grid_state

from game import config as cfg
from game.animation import AnimationManager, FrameData, HERO_ANIMATIONS
from game.app import GameApp
from game.enemy import EnemyA
from game.entity_manager import Entity, EntityManager
from game.hero import Hero, GridFeedback
from game.spell_system import (
    SPELL_CATALOG,
    expand_model_socket,
    execute_magic_socket,
    inject_stream_tick,
    MagicSocket,
    ActiveStream,
    SLM_AVAILABLE,
    slm_to_model_socket,
)
from game.terrain import TerrainGenerator


def _world_to_local(world: ActiveWorldWindow, wx: int, wy: int) -> tuple[int, int]:
    return (wx - world.active_origin_x, wy - world.active_origin_y)


def _local_to_world(world: ActiveWorldWindow, lx: int, ly: int) -> tuple[int, int]:
    return (lx + world.active_origin_x, ly + world.active_origin_y)


class HeroTests(unittest.TestCase):
    """Tests for Hero entity."""

    def test_hero_dimensions(self) -> None:
        hero = Hero()
        self.assertEqual(hero.width, cfg.HERO_WIDTH)
        self.assertEqual(hero.height, cfg.HERO_HEIGHT)

    def test_hero_initial_state(self) -> None:
        hero = Hero()
        self.assertEqual(hero.hp, cfg.HERO_MAX_HP)
        self.assertEqual(hero.mp, cfg.HERO_MAX_MP)
        self.assertEqual(hero.state, "idle")

    def test_hero_walk_left(self) -> None:
        hero = Hero()
        hero.input_left = True
        hero.update(0.1)
        self.assertLess(hero.vel_x, 0)
        self.assertFalse(hero.facing_right)
        self.assertEqual(hero.state, "walk")

    def test_hero_walk_right(self) -> None:
        hero = Hero()
        hero.input_right = True
        hero.update(0.1)
        self.assertGreater(hero.vel_x, 0)
        self.assertTrue(hero.facing_right)
        self.assertEqual(hero.state, "walk")

    def test_hero_gravity(self) -> None:
        hero = Hero()
        hero.on_ground = False
        hero.update(0.1)
        self.assertGreater(hero.vel_y, 0)  # positive = downward (y increases toward ground)

    def test_hero_jump(self) -> None:
        hero = Hero()
        hero.on_ground = True
        hero.input_jump = True
        hero.update(0.1)
        self.assertLess(hero.vel_y, 0)  # negative = upward (y decreases)
        self.assertEqual(hero.state, "jump")
        self.assertFalse(hero.on_ground)

    def test_hero_mp_regen(self) -> None:
        hero = Hero()
        hero.mp = 30.0
        hero.update(0.1)
        self.assertGreater(hero.mp, 30.0)

    def test_hero_consume_mp_success(self) -> None:
        hero = Hero()
        self.assertTrue(hero.consume_mp(10))
        self.assertEqual(hero.mp, cfg.HERO_MAX_MP - 10)

    def test_hero_consume_mp_failure(self) -> None:
        hero = Hero()
        hero.mp = 5.0
        self.assertFalse(hero.consume_mp(10))
        self.assertEqual(hero.mp, 5.0)

    def test_hero_chant_state_held(self) -> None:
        """Hero enters chant state when SPACE is held (input_chant_held=True)."""
        hero = Hero()
        hero.on_ground = True
        hero.input_chant_held = True
        hero.update(0.1)
        self.assertEqual(hero.state, "chant")
        hero.update(0.1)
        self.assertEqual(hero.state, "chant")

    def test_hero_chant_to_cast_on_release(self) -> None:
        """Hero transitions from chant to cast when SPACE is released."""
        hero = Hero()
        hero.on_ground = True
        hero.input_chant_held = True
        hero.update(0.1)
        self.assertEqual(hero.state, "chant")
        hero.input_chant_held = False
        hero.update(0.1)
        self.assertEqual(hero.state, "cast")
        hero.update(0.4)
        self.assertEqual(hero.state, "idle")

    def test_hero_chant_freezes_horizontal_movement(self) -> None:
        """Hero cannot move horizontally while chanting."""
        hero = Hero()
        hero.on_ground = True
        hero.input_chant_held = True
        hero.input_right = True
        hero.update(0.1)
        self.assertEqual(hero.state, "chant")
        self.assertEqual(hero.vel_x, 0.0)

    def test_hero_state_machine_has_all_states(self) -> None:
        valid_states = ("idle", "walk", "jump", "chant", "cast")
        for state in valid_states:
            self.assertIn(state, HERO_ANIMATIONS)

    def test_hero_blocked_left_stops_movement(self) -> None:
        hero = Hero()
        hero.on_ground = True
        hero.input_left = True
        feedback = GridFeedback(world_width=100.0, blocked_left=True, blocked_below=True)
        hero.update(0.1, grid_feedback=feedback)
        self.assertEqual(hero.vel_x, 0.0)

    def test_hero_blocked_right_stops_movement(self) -> None:
        hero = Hero()
        hero.on_ground = True
        hero.input_right = True
        feedback = GridFeedback(world_width=100.0, blocked_right=True, blocked_below=True)
        hero.update(0.1, grid_feedback=feedback)
        self.assertEqual(hero.vel_x, 0.0)

    def test_hero_climbs_one_cell_step_without_losing_move_intent(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        hero.on_ground = True
        hero.input_right = True
        feedback = GridFeedback(
            world_width=100.0,
            blocked_below=True,
            blocked_right=True,
            blocked_right_ahead=True,
        )
        old_y = hero.y
        hero.update(0.0, grid_feedback=feedback)
        self.assertEqual(hero.y, old_y - 1.0)
        self.assertGreater(hero.vel_x, 0.0)

    def test_hero_destuck_does_not_apply_while_moving_upward_airborne(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        hero.on_ground = False
        hero.vel_y = -5.0
        feedback = GridFeedback(world_width=100.0, embedded=True, blocked_up=False)
        old_y = hero.y
        hero.update(0.0, grid_feedback=feedback)
        self.assertEqual(hero.y, old_y)

    def test_hero_can_move_horizontally_while_airborne(self) -> None:
        hero = Hero()
        hero.state = "jump"
        hero.on_ground = False
        hero.input_right = True
        feedback = GridFeedback(world_width=100.0, blocked_below=False)
        hero.update(0.1, grid_feedback=feedback)
        self.assertGreater(hero.vel_x, 0.0)

    def test_hero_blocked_up_stops_jump_motion(self) -> None:
        hero = Hero()
        hero.state = "jump"
        hero.on_ground = False
        hero.vel_y = -10.0
        feedback = GridFeedback(world_width=100.0, blocked_up=True)
        hero.update(0.1, grid_feedback=feedback)
        self.assertEqual(hero.vel_y, 0.0)

    def test_hero_placeholder_cells(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        cells = hero.get_placeholder_cells()
        self.assertTrue(len(cells) > 0)
        for x, y in cells:
            self.assertGreaterEqual(x, int(hero.left))
            self.assertLessEqual(x, int(hero.right))
            self.assertGreaterEqual(y, int(hero.bottom))
            self.assertLessEqual(y, int(hero.top))


class EnemyTests(unittest.TestCase):
    def test_enemy_a_reverses_direction_when_wall_blocks_patrol(self) -> None:
        enemy = EnemyA.create("enemy_a_test", 40.0, 20.0)
        enemy.patrol_direction = 1.0
        enemy.state = "patrol"
        enemy.update(
            0.1,
            grid_feedback=GridFeedback(world_width=100.0, blocked_below=True, blocked_right=True),
            hero_x=500.0,
            hero_y=20.0,
        )
        self.assertLess(enemy.vel_x, 0.0)

    def test_enemy_a_patrol_cycle_advances_when_timer_expires(self) -> None:
        enemy = EnemyA.create("enemy_a_cycle", 40.0, 20.0)
        original_cycle = enemy.patrol_cycle
        enemy.patrol_time_remaining = 0.0
        enemy.update(
            0.1,
            grid_feedback=GridFeedback(world_width=100.0, blocked_below=True),
            hero_x=500.0,
            hero_y=20.0,
        )
        self.assertGreater(enemy.patrol_cycle, original_cycle)
        self.assertGreater(enemy.patrol_time_remaining, 0.0)
        self.assertEqual(enemy.state, "patrol")

    def test_enemy_a_destuck_lifts_body_up(self) -> None:
        enemy = EnemyA.create("enemy_a_stuck", 40.0, 20.0)
        original_y = enemy.y
        enemy.update(
            0.0,
            grid_feedback=GridFeedback(world_width=100.0, blocked_below=True, embedded=True),
            hero_x=500.0,
            hero_y=20.0,
        )
        self.assertLess(enemy.y, original_y)


class EntityManagerTests(unittest.TestCase):
    """Tests for GPU entity mask and GPU collision query runtime."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()
        try:
            import moderngl
            cls.ctx = moderngl.create_standalone_context()
        except Exception:
            cls.ctx = None

    def _make_world(self, w: int = 40, h: int = 40) -> ActiveWorldWindow:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(w, h, chunk_size=4)
        world = ActiveWorldWindow(store, self.registry, viewport_width=w, viewport_height=h, ctx=self.ctx)
        return world

    def _make_manager_with_hero(self, x: float = 20.0, y: float = 10.0) -> tuple[Hero, EntityManager, Entity]:
        hero = Hero(x=x, y=y)
        mgr = EntityManager(hero=hero)
        entity = Entity(
            entity_id="hero",
            x=hero.x,
            y=hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        )
        mgr.register_entity(entity)
        return hero, mgr, entity

    def _write_and_upload_cell(self, world: ActiveWorldWindow, wx: int, wy: int, cell: CellState) -> None:
        lx, ly = _world_to_local(world, wx, wy)
        tables = GpuMaterialTables.from_registry(self.registry)
        packed = pack_cells_state([cell], tables)
        world.gpu_simulator.write_region_bytes(lx, ly, 1, 1, packed.state_int, packed.state_vec, packed.state_misc)

    def _read_gpu_feedback(self, mgr: EntityManager, world: ActiveWorldWindow) -> GridFeedback:
        mgr.schedule_feedback(world)
        world.step(0.0)
        feedback = mgr.poll_all_feedback(world)
        hero_feedback = feedback.get("hero")
        self.assertIsNotNone(hero_feedback)
        return hero_feedback

    def test_tick_does_not_modify_gpu_cells_under_hero(self) -> None:
        _hero, mgr, entity = self._make_manager_with_hero()
        world = self._make_world()
        wx0 = int(entity.left)
        wx1 = int(entity.right) + 1
        wy0 = int(entity.bottom)
        wy1 = int(entity.top) + 1
        before: list[tuple[int, int, CellState]] = []
        for wy in range(wy0, wy1):
            for wx in range(wx0, wx1):
                lx, ly = _world_to_local(world, wx, wy)
                before.append((lx, ly, world.gpu_simulator.readback_cells_region(lx, ly, 1, 1)[0][0]))
        mgr.tick(world, 0.1)
        for lx, ly, expected in before:
            cell = world.gpu_simulator.readback_cells_region(lx, ly, 1, 1)[0][0]
            self.assertEqual(cell.family_id, expected.family_id)
            self.assertEqual(cell.variant_id, expected.variant_id)
            self.assertEqual(cell.generation, expected.generation)

    def test_grid_feedback_detects_ground(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        ground_wy = int(hero.top) + 1
        for wx in range(int(hero.left), int(hero.right) + 1):
            self._write_and_upload_cell(world, wx, ground_wy, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_below)

    def test_grid_feedback_detects_single_edge_ground_cell(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        self._write_and_upload_cell(world, int(hero.left), int(hero.top) + 1, CellState(
            family_id="stone", variant_id="stone_platform", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_below)

    def test_query_points_align_step_and_embedded_rows_to_body(self) -> None:
        _hero, mgr, entity = self._make_manager_with_hero()
        clip = (
            int(entity.left),
            int(entity.bottom),
            int(entity.right) + 1,
            int(entity.top) + 1,
        )
        lx0, _ly0, lx1, ly1 = clip
        points = mgr._build_query_points(entity, object(), clip)  # type: ignore[arg-type]

        right_step = [point for point in points if point[2] == 7]
        right_clearance = [point for point in points if point[2] == 10]
        embedded_points = [point for point in points if point[2] == 8]

        self.assertEqual(right_step, [(lx1, ly1 - 1, 7)])
        self.assertEqual(right_clearance, [(lx1, ly1 - 2, 10)])
        self.assertGreater(len(embedded_points), 1)
        self.assertGreater(len({point[0] for point in embedded_points}), 1)
        self.assertIn(ly1 - 1, {point[1] for point in embedded_points})
        self.assertIn(ly1 - 2, {point[1] for point in embedded_points})

    def test_grid_feedback_detects_horizontal_walls(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            self._write_and_upload_cell(world, int(hero.left) - 1, wy, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_left)

    def test_grid_feedback_detects_ceiling(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        for wx in range(int(hero.left), int(hero.right) + 1):
            self._write_and_upload_cell(world, wx, int(hero.bottom) - 1, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_up)

    def test_grid_feedback_detects_full_height_side_wall(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            self._write_and_upload_cell(world, int(hero.right) + 1, wy, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_right)

    def test_grid_feedback_detects_liquid(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="water", variant_id="water"))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.in_liquid)

    def test_grid_feedback_damage_from_fire_temperature(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="fire", variant_id="fire", temperature=600.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertGreater(feedback.damage, 0.2)

    def test_grid_feedback_damage_from_magic_acid_contact(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="magic_acid", variant_id="magic_acid_liquid", integrity=1.0))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.in_liquid)
        self.assertGreater(feedback.damage, 0.01)

    def test_grid_feedback_damage_from_fast_solid_impact(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="stone",
                    variant_id="stone_powder",
                    integrity=1.0,
                    vel_x=90.0,
                ))
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertGreater(feedback.damage, 0.05)

    def test_read_feedback_and_update_applies_gpu_damage_to_hero(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="fire", variant_id="fire", temperature=600.0))
        feedback = self._read_gpu_feedback(mgr, world)
        hp_before = hero.hp
        mgr.read_feedback_and_update(world, 0.0, feedback={"hero": feedback})
        self.assertLess(hero.hp, hp_before)

    def test_gpu_entity_mask_blocks_sand_from_falling_into_hero(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        sand_wx = int(hero.x)
        sand_wy = int(hero.bottom) - 1
        self._write_and_upload_cell(world, sand_wx, sand_wy, CellState(
            family_id="sand", variant_id="sand_powder", integrity=1.0))
        world.step(0.1)
        cell_x, cell_y = _world_to_local(world, sand_wx, int(hero.bottom))
        hero_cell = world.gpu_simulator.readback_cells_region(cell_x, cell_y, 1, 1)[0][0]
        above_cell = world.gpu_simulator.readback_cells_region(cell_x, cell_y - 1, 1, 1)[0][0]
        self.assertTrue(hero_cell.is_empty)
        self.assertEqual(above_cell.family_id, "sand")


class GpuChunkCacheTests(unittest.TestCase):
    """Tests for GPU packed chunk cache and paging persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()
        try:
            import moderngl
            cls.ctx = moderngl.create_standalone_context()
        except Exception:
            cls.ctx = None

    def test_missing_chunk_generates_once_and_writes_disk(self) -> None:
        calls: list[tuple[int, int]] = []

        def generate(store: WorldChunkStore, chunk_x: int, chunk_y: int, chunk_size: int, seed: int) -> None:
            calls.append((chunk_x, chunk_y))
            store.set_cell(chunk_x * chunk_size, chunk_y * chunk_size, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))

        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 16, chunk_size=4, seed=7, chunk_generator=generate)
            if self.ctx is None:
                self.skipTest("GPU context not available")
            world = ActiveWorldWindow(
                store,
                self.registry,
                viewport_width=4,
                viewport_height=4,
                halo_cells=0,
                ctx=self.ctx,
                initial_camera_x=0,
                initial_camera_y=0,
                chunk_save_dir=tmp,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            self.assertIn((0, 0), calls)
            world.chunk_cache.flush_dirty()
            self.assertTrue(any(Path(tmp).rglob("0_0.ogchunk")))

            calls.clear()
            store2 = WorldChunkStore(16, 16, chunk_size=4, seed=7, chunk_generator=generate)
            ActiveWorldWindow(
                store2,
                self.registry,
                viewport_width=4,
                viewport_height=4,
                halo_cells=0,
                ctx=self.ctx,
                initial_camera_x=0,
                initial_camera_y=0,
                chunk_save_dir=tmp,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            self.assertEqual(calls, [])
            self.assertIsNotNone(world)

    def test_prefetch_worker_generates_missing_chunk_to_disk(self) -> None:
        calls: list[tuple[int, int]] = []

        def generate(store: WorldChunkStore, chunk_x: int, chunk_y: int, chunk_size: int, seed: int) -> None:
            calls.append((chunk_x, chunk_y))
            store.set_cell(chunk_x * chunk_size, chunk_y * chunk_size, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))

        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 16, chunk_size=4, seed=7, chunk_generator=generate)
            cache = GpuChunkCache(store, self.registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
            packed = cache._load_for_worker(0, 0)
            self.assertIsNotNone(packed)
            self.assertEqual(calls, [(0, 0)])
            self.assertTrue(any(Path(tmp).rglob("0_0.ogchunk")))

    def test_prime_rect_on_disk_generates_missing_chunk_files(self) -> None:
        calls: list[tuple[int, int]] = []

        def generate(store: WorldChunkStore, chunk_x: int, chunk_y: int, chunk_size: int, seed: int) -> None:
            calls.append((chunk_x, chunk_y))
            store.set_cell(chunk_x * chunk_size, chunk_y * chunk_size, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))

        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 16, chunk_size=4, seed=7, chunk_generator=generate)
            cache = GpuChunkCache(store, self.registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
            cache.prime_rect_on_disk(WorldRect(0, 0, 4, 4))
            for _ in range(100):
                if any(Path(tmp).rglob("0_0.ogchunk")):
                    break
                time.sleep(0.02)
            self.assertTrue(any(Path(tmp).rglob("0_0.ogchunk")))

    def test_paged_out_gpu_cell_restores_from_binary_chunk(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 8, chunk_size=4, seed=11)
            world = ActiveWorldWindow(
                store,
                self.registry,
                viewport_width=4,
                viewport_height=4,
                halo_cells=0,
                page_shift_cells=4,
                safety_margin_cells=0,
                idle_flush_service_interval_seconds=0.0,
                ctx=self.ctx,
                chunk_save_dir=tmp,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            tables = GpuMaterialTables.from_registry(self.registry)
            packed = pack_cells_state([CellState(family_id="stone", variant_id="stone_platform", integrity=1.0)], tables)
            world.gpu_simulator.write_region_bytes(1, 1, 1, 1, packed.state_int, packed.state_vec, packed.state_misc)

            world.pan_camera(4, 0)
            while world.pending_writeback_count:
                world.mark_camera_activity(False, dt=1.0)
                world.service_background_io()
            world.pan_camera(-4, 0)

            cell = world.gpu_simulator.readback_cells_region(1, 1, 1, 1)[0][0]
            self.assertEqual(cell.family_id, "stone")
            self.assertEqual(cell.variant_id, "stone_platform")

    def test_world_close_flushes_active_gpu_cells_to_disk(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 8, chunk_size=4, seed=17)
            world = ActiveWorldWindow(
                store,
                self.registry,
                viewport_width=4,
                viewport_height=4,
                halo_cells=0,
                ctx=self.ctx,
                chunk_save_dir=tmp,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            tables = GpuMaterialTables.from_registry(self.registry)
            packed = pack_cells_state([CellState(family_id="stone", variant_id="stone_platform", integrity=1.0)], tables)
            world.gpu_simulator.write_region_bytes(1, 1, 1, 1, packed.state_int, packed.state_vec, packed.state_misc)
            world.close()

            store2 = WorldChunkStore(16, 8, chunk_size=4, seed=17)
            world2 = ActiveWorldWindow(
                store2,
                self.registry,
                viewport_width=4,
                viewport_height=4,
                halo_cells=0,
                ctx=self.ctx,
                chunk_save_dir=tmp,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            try:
                cell = world2.gpu_simulator.readback_cells_region(1, 1, 1, 1)[0][0]
                self.assertEqual(cell.family_id, "stone")
                self.assertEqual(cell.variant_id, "stone_platform")
            finally:
                world2.close()

    def test_set_camera_far_jump_shifts_active_window_once(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(256, 32, chunk_size=8)
        world = ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=16,
            viewport_height=16,
            halo_cells=4,
            page_shift_cells=4,
            safety_margin_cells=4,
            ctx=self.ctx,
            initial_camera_x=0,
            initial_camera_y=0,
            chunk_cache_prefetch_x=0,
            chunk_cache_prefetch_y=0,
        )
        world.set_camera(200, 0)
        self.assertEqual(world.paging_stats.shift_count, 1)
        self.assertLessEqual(world.active_origin_x, world.camera_x - world.safety_margin_cells)
        self.assertGreaterEqual(
            world.active_origin_x + world.active_width,
            world.camera_x + world.viewport_width + world.safety_margin_cells,
        )

    def test_prefetch_camera_region_does_not_move_camera(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(256, 32, chunk_size=8)
        world = ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=16,
            viewport_height=16,
            halo_cells=4,
            page_shift_cells=4,
            safety_margin_cells=4,
            ctx=self.ctx,
            initial_camera_x=0,
            initial_camera_y=0,
            chunk_cache_prefetch_x=0,
            chunk_cache_prefetch_y=0,
        )
        world.prefetch_camera_region(200, 0, submit_chunks=2)
        self.assertEqual((world.camera_x, world.camera_y), (0, 0))

    def test_prepare_camera_region_sync_does_not_move_camera(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(256, 32, chunk_size=8)
        world = ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=16,
            viewport_height=16,
            halo_cells=4,
            page_shift_cells=4,
            safety_margin_cells=4,
            ctx=self.ctx,
            initial_camera_x=0,
            initial_camera_y=0,
            chunk_cache_prefetch_x=0,
            chunk_cache_prefetch_y=0,
        )
        world.prepare_camera_region_sync(200, 0)
        self.assertEqual((world.camera_x, world.camera_y), (0, 0))

    def test_move_hero_to_primes_far_camera_jump(self) -> None:
        app = object.__new__(GameApp)
        app.hero = Hero()
        app.hero.x = 10.0
        app.hero.y = 10.0
        app.hero.vel_x = 1.0
        app.hero.vel_y = 2.0
        app.hero.on_ground = True
        app.entity_manager = EntityManager(hero=app.hero)
        app.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=app.hero.x,
            y=app.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))

        class DummyWorld:
            viewport_width = 16
            viewport_height = 16
            active_width = 48
            active_height = 48
            camera_x = 0
            camera_y = 0

            def __init__(self) -> None:
                self.prepared: list[tuple[int, int]] = []
                self.set_calls: list[tuple[int, int]] = []

            def prepare_camera_region_sync(self, camera_x: int, camera_y: int) -> None:
                self.prepared.append((camera_x, camera_y))

            def set_camera(self, camera_x: int, camera_y: int) -> None:
                self.set_calls.append((camera_x, camera_y))
                self.camera_x = camera_x
                self.camera_y = camera_y

        app.world = DummyWorld()
        GameApp.move_hero_to(app, 200.0, 120.0)
        self.assertEqual(app.hero.vel_x, 0.0)
        self.assertEqual(app.hero.vel_y, 0.0)
        self.assertFalse(app.hero.on_ground)
        self.assertEqual(app.world.prepared, [(192, 112)])
        self.assertEqual(app.world.set_calls, [(192, 112)])

    def test_move_hero_to_skips_prime_for_near_camera_move(self) -> None:
        app = object.__new__(GameApp)
        app.hero = Hero()
        app.entity_manager = EntityManager(hero=app.hero)
        app.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=app.hero.x,
            y=app.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))

        class DummyWorld:
            viewport_width = 16
            viewport_height = 16
            active_width = 96
            active_height = 96
            camera_x = 40
            camera_y = 40

            def __init__(self) -> None:
                self.prepared: list[tuple[int, int]] = []
                self.set_calls: list[tuple[int, int]] = []

            def prepare_camera_region_sync(self, camera_x: int, camera_y: int) -> None:
                self.prepared.append((camera_x, camera_y))

            def set_camera(self, camera_x: int, camera_y: int) -> None:
                self.set_calls.append((camera_x, camera_y))

        app.world = DummyWorld()
        GameApp.move_hero_to(app, 60.0, 60.0)
        self.assertEqual(app.world.prepared, [])
        self.assertEqual(app.world.set_calls, [(52, 52)])

    def test_terrain_gpu_chunk_generation_matches_point_query(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        chunk_size = 64
        chamber = gen._underground_chambers()[0]
        chunk_x = chamber["cx"] // chunk_size
        chunk_y = chamber["cy"] // chunk_size
        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(
                cfg.WORLD_WIDTH,
                cfg.WORLD_HEIGHT,
                chunk_size=chunk_size,
                seed=42,
                chunk_generator=gen.generate_chunk,
            )
            cache = GpuChunkCache(store, self.registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
            packed = cache._generate_chunk(chunk_x, chunk_y)
            grid = unpack_grid_state(
                chunk_size,
                chunk_size,
                cache.tables,
                bytes(packed.state_int),
                bytes(packed.state_vec),
                bytes(packed.state_misc),
            )
            world_x0 = chunk_x * chunk_size
            world_y0 = chunk_y * chunk_size
            for local_y in range(chunk_size):
                for local_x in range(chunk_size):
                    expected = gen.cell_at(world_x0 + local_x, world_y0 + local_y)
                    actual = grid.cells[local_y * chunk_size + local_x]
                    if expected is None:
                        self.assertTrue(actual.is_empty, f"expected empty at {(world_x0 + local_x, world_y0 + local_y)}")
                        continue
                    self.assertEqual(
                        (actual.family_id, actual.variant_id, actual.flags),
                        (expected.family_id, expected.variant_id, expected.flags),
                    )


class ExplosionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()
        try:
            import moderngl
            cls.ctx = moderngl.create_standalone_context()
        except Exception:
            cls.ctx = None

    def _make_world(self, size: int = 128) -> ActiveWorldWindow:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(size, size, chunk_size=8)
        return ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=size,
            viewport_height=size,
            halo_cells=0,
            ctx=self.ctx,
            chunk_cache_prefetch_x=0,
            chunk_cache_prefetch_y=0,
        )

    def _max_pressure_in_band(
        self,
        world: ActiveWorldWindow,
        *,
        center_x: int,
        center_y: int,
        inner_radius: int,
        outer_radius: int,
    ) -> float:
        grid = world.readback_gpu_snapshot()
        values: list[float] = []
        inner_sq = inner_radius * inner_radius
        outer_sq = outer_radius * outer_radius
        for local_y in range(grid.height):
            world_y = world.active_origin_y + local_y
            dy = world_y - center_y
            if abs(dy) > outer_radius:
                continue
            for local_x in range(grid.width):
                world_x = world.active_origin_x + local_x
                dx = world_x - center_x
                dist_sq = dx * dx + dy * dy
                if inner_sq <= dist_sq <= outer_sq:
                    values.append(grid.pressure[grid.index(local_x, local_y)])
        return max(values) if values else 0.0

    def test_gpu_snapshot_reads_pressure_after_step(self) -> None:
        self.skipTest("Replaced by scripts/http_debug_smoke.py runtime GPU pressure verification")

    def test_trigger_explosion_pushes_pressure_into_outer_annulus(self) -> None:
        self.skipTest("Replaced by scripts/http_debug_smoke.py runtime explosion verification")


class DebugViewTests(unittest.TestCase):
    def test_f4_toggles_temperature_view(self) -> None:
        from pyglet.window import key

        app = object.__new__(GameApp)
        app._keys_pressed = set()
        app.current_screen = "game"
        app.entity_manager = EntityManager(hero=Hero())
        app.view_mode = DebugViewMode.MATERIAL
        app.screens = {}
        GameApp.on_key_press(app, key.F4, 0)
        self.assertEqual(app.view_mode, DebugViewMode.TEMPERATURE)
        GameApp.on_key_press(app, key.F4, 0)
        self.assertEqual(app.view_mode, DebugViewMode.MATERIAL)

    def test_f5_toggles_pressure_view(self) -> None:
        from pyglet.window import key

        app = object.__new__(GameApp)
        app._keys_pressed = set()
        app.current_screen = "game"
        app.entity_manager = EntityManager(hero=Hero())
        app.view_mode = DebugViewMode.MATERIAL
        app.screens = {}
        GameApp.on_key_press(app, key.F5, 0)
        self.assertEqual(app.view_mode, DebugViewMode.PRESSURE)
        GameApp.on_key_press(app, key.F5, 0)
        self.assertEqual(app.view_mode, DebugViewMode.MATERIAL)


class SpellSystemTests(unittest.TestCase):
    """Tests for spell expansion and execution."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()
        try:
            import moderngl
            cls.ctx = moderngl.create_standalone_context()
        except Exception:
            cls.ctx = None

    def test_spell_catalog_has_8_spells(self) -> None:
        self.assertEqual(len(SPELL_CATALOG), 8)

    def test_expand_fireball(self) -> None:
        socket = SPELL_CATALOG[0]  # Fireball
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        self.assertEqual(magic.subject_family, "fire")
        self.assertEqual(magic.subject_variant, "fire")
        self.assertEqual(magic.powerness, 0.5)
        self.assertEqual(magic.reaction_template, "burn")
        self.assertGreater(magic.brush_radius, 0)

    def test_expand_water_wall_is_burst(self) -> None:
        socket = SPELL_CATALOG[1]  # Water Wall (appear/burst)
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        self.assertEqual(magic.subject_family, "water")
        self.assertEqual(magic.release_profile, "burst")

    def test_expand_fireball_is_stream(self) -> None:
        socket = SPELL_CATALOG[0]  # Fireball (spray/stream)
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        self.assertEqual(magic.release_profile, "stream")

    def test_expand_gives_direction_vector(self) -> None:
        socket = SPELL_CATALOG[0]
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        self.assertGreater(magic.direction[0], 0)

    def test_powerness_scales_brush_radius(self) -> None:
        socket_low = {"material": "fire", "reaction": "burn", "release": "spray",
                      "motion": "flow", "motion_direction": "forward",
                      "origin": "self", "target": "none", "politeness": 0.2}
        socket_high = {"material": "fire", "reaction": "burn", "release": "spray",
                       "motion": "flow", "motion_direction": "forward",
                       "origin": "self", "target": "none", "politeness": 1.0}
        magic_low = expand_model_socket(socket_low)
        magic_high = expand_model_socket(socket_high)
        self.assertGreater(magic_high.brush_radius, magic_low.brush_radius)

    def test_execute_burst_injects_cells(self) -> None:
        """Burst-type spells inject cells immediately via paint_world."""
        self.skipTest("GPU paint verification uses explicit GPU readback tests")

    def test_execute_stream_returns_active_stream(self) -> None:
        """Stream-type spells return an ActiveStream for multi-frame injection."""
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=40, viewport_height=40, ctx=self.ctx)
        socket = SPELL_CATALOG[0]  # Fireball (stream)
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        result = execute_magic_socket(magic, world, self.registry)
        self.assertIsInstance(result, ActiveStream)
        self.assertGreater(result.remaining_ticks, 0)

    def test_stream_inject_injects_cells(self) -> None:
        """inject_stream_tick should inject cells each tick."""
        self.skipTest("GPU paint verification uses explicit GPU readback tests")

    def test_execute_attaches_reaction_overrides(self) -> None:
        """execute_magic_socket should attach spell_* reaction overrides to injected cells."""
        self.skipTest("GPU paint verification uses explicit GPU readback tests")

    def test_no_light_or_lightning_in_expansion_table(self) -> None:
        from game.spell_system import SUBJECT_EXPANSION_TABLE
        self.assertNotIn("light", SUBJECT_EXPANSION_TABLE)
        self.assertNotIn("lightning", SUBJECT_EXPANSION_TABLE)

    def test_steam_in_expansion_table(self) -> None:
        from game.spell_system import SUBJECT_EXPANSION_TABLE
        self.assertIn("steam", SUBJECT_EXPANSION_TABLE)
        steam = SUBJECT_EXPANSION_TABLE["steam"]
        self.assertEqual(steam["family_id"], "water")
        self.assertEqual(steam["variant_id"], "steam")

    def test_slm_adapter_returns_none_without_checkpoint(self) -> None:
        result = slm_to_model_socket("火球术")
        self.assertIsNone(result)


class TerrainGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()

    def _assert_chunk_matches_point_query(
        self,
        gen: TerrainGenerator,
        *,
        chunk_x: int,
        chunk_y: int,
        chunk_size: int = 64,
    ) -> None:
        store = WorldChunkStore(
            cfg.WORLD_WIDTH,
            cfg.WORLD_HEIGHT,
            chunk_size=chunk_size,
            seed=42,
            chunk_generator=gen.generate_chunk,
        )
        store._ensure_chunk_generated(chunk_x, chunk_y)
        x0 = chunk_x * chunk_size
        y0 = chunk_y * chunk_size
        for wx in range(x0, x0 + chunk_size):
            for wy in range(y0, y0 + chunk_size):
                expected = gen.cell_at(wx, wy)
                actual = store.get_cell(wx, wy)
                if expected is None:
                    self.assertTrue(actual.is_empty, f"expected empty at {(wx, wy)}")
                    continue
                self.assertEqual(
                    (actual.family_id, actual.variant_id, actual.flags),
                    (expected.family_id, expected.variant_id, expected.flags),
                    f"mismatch at {(wx, wy)}",
                )

    def test_chunk_storage_root_uses_save_name_env(self) -> None:
        with patch.dict("os.environ", {cfg.SAVE_NAME_ENV_VAR: "slot_alpha"}, clear=False):
            self.assertEqual(cfg.chunk_storage_root(), str(Path("storage") / "slot_alpha"))

    def test_find_spawn_point_near_alpine_center_is_clear(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        spawn = gen.find_spawn_point_near(
            2 * cfg.BIOME_WIDTH + cfg.BIOME_WIDTH // 2,
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=cfg.BIOME_WIDTH // 3,
        )
        self.assertIsNotNone(spawn)
        spawn_x, spawn_y = spawn or (0.0, 0.0)
        floor_cell = gen.cell_at(int(spawn_x), int(spawn_y + cfg.HERO_HEIGHT + 1))
        self.assertTrue(floor_cell is not None and not floor_cell.is_empty)
        for sample_y in range(int(spawn_y), int(spawn_y + cfg.HERO_HEIGHT)):
            cell = gen.cell_at(int(spawn_x), sample_y)
            self.assertTrue(cell is None or cell.is_empty)

    def test_find_spawn_point_near_underground_center_is_clear(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        spawn = gen.find_spawn_point_near(
            3 * cfg.BIOME_WIDTH + cfg.BIOME_WIDTH // 2,
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=cfg.BIOME_WIDTH // 3,
        )
        self.assertIsNotNone(spawn)
        spawn_x, spawn_y = spawn or (0.0, 0.0)
        floor_cell = gen.cell_at(int(spawn_x), int(spawn_y + cfg.HERO_HEIGHT + 1))
        self.assertTrue(floor_cell is not None and not floor_cell.is_empty)
        for sample_y in range(int(spawn_y), int(spawn_y + cfg.HERO_HEIGHT)):
            cell = gen.cell_at(int(spawn_x), sample_y)
            self.assertTrue(cell is None or cell.is_empty)

    def test_plains_surface_columns_are_mixed_not_all_grass(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        seen = set()
        for wx in range(64, min(cfg.BIOME_WIDTH, 4096), 16):
            cell = gen._surface_cell_for_column(wx)
            seen.add((cell.family_id, cell.variant_id))
        self.assertIn(("grass", "grass_platform"), seen)
        self.assertIn(("sand", "sand_powder"), seen)
        self.assertIn(("stone", "stone_platform"), seen)

    def test_plains_tree_features_have_varied_heights_and_canopies(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        trees = []
        for wx in range(1, 6000):
            feature = gen._plains_tree_feature(wx)
            if feature is not None:
                trees.append(feature)
            if len(trees) >= 5:
                break
        self.assertGreaterEqual(len(trees), 3)
        self.assertGreater(len({tree.trunk_height for tree in trees}), 1)
        self.assertGreater(len({(tree.canopy_radius_x, tree.canopy_radius_y) for tree in trees}), 1)
        sample = trees[0]
        self.assertTrue(
            gen._tree_leaf_present(
                sample,
                sample.center_x + max(1, sample.canopy_radius_x // 2),
                int(round(sample.canopy_center_y)),
            )
        )

    def test_plains_pond_floor_uses_irregular_noise(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        pond = None
        for wx in range(1, 12000):
            feature = gen._plains_pond_feature(wx)
            if feature is not None and feature.center_x == wx:
                pond = feature
                break
        self.assertIsNotNone(pond)
        assert pond is not None
        dx = max(4, pond.half_width // 3)
        center_floor = gen._pond_floor_y(pond.center_x, pond.ground_y, pond)
        left_floor = gen._pond_floor_y(pond.center_x - dx, pond.ground_y, pond)
        right_floor = gen._pond_floor_y(pond.center_x + dx, pond.ground_y, pond)
        self.assertGreater(pond.half_width, cfg.PLAINS_POND_WIDTH // 2)
        self.assertGreater(center_floor, left_floor)
        self.assertGreater(center_floor, right_floor)
        self.assertGreater(abs(left_floor - right_floor), 0.25)

    def test_alpine_generated_island_chunk_matches_point_query(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        island = gen._alpine_islands()[0]
        chunk_size = 64
        self._assert_chunk_matches_point_query(
            gen,
            chunk_x=island["cx"] // chunk_size,
            chunk_y=island["cy"] // chunk_size,
            chunk_size=chunk_size,
        )

    def test_alpine_generated_bridge_chunk_matches_point_query(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        islands = gen._alpine_islands()
        bridge_pair: tuple[dict, dict] | None = None
        for i, island in enumerate(islands):
            for j in island.get("connections", []):
                if j > i:
                    bridge_pair = (island, islands[j])
                    break
            if bridge_pair is not None:
                break
        self.assertIsNotNone(bridge_pair)
        island_a, island_b = bridge_pair or ({}, {})
        bx0 = min(island_a["cx"], island_b["cx"])
        bx1 = max(island_a["cx"], island_b["cx"])
        mid_x = (bx0 + bx1) // 2
        span = max(1, bx1 - bx0)
        t = (mid_x - bx0) / span
        mid_y = int(island_a["cy"] + (island_b["cy"] - island_a["cy"]) * t)
        chunk_size = 64
        self._assert_chunk_matches_point_query(
            gen,
            chunk_x=mid_x // chunk_size,
            chunk_y=mid_y // chunk_size,
            chunk_size=chunk_size,
        )

    def test_underground_generated_chamber_chunk_matches_point_query(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        chamber = gen._underground_chambers()[0]
        chunk_size = 64
        self._assert_chunk_matches_point_query(
            gen,
            chunk_x=chamber["cx"] // chunk_size,
            chunk_y=chamber["cy"] // chunk_size,
            chunk_size=chunk_size,
        )

    def test_underground_generated_corridor_chunk_matches_point_query(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        chambers = gen._underground_chambers()
        corridor_pair: tuple[dict, dict] | None = None
        for i, chamber in enumerate(chambers):
            for j in chamber.get("connections", []):
                if j > i:
                    corridor_pair = (chamber, chambers[j])
                    break
            if corridor_pair is not None:
                break
        self.assertIsNotNone(corridor_pair)
        chamber_a, chamber_b = corridor_pair or ({}, {})
        bx0 = min(chamber_a["cx"], chamber_b["cx"])
        bx1 = max(chamber_a["cx"], chamber_b["cx"])
        mid_x = (bx0 + bx1) // 2
        span = max(1, bx1 - bx0)
        t = (mid_x - bx0) / span
        mid_y = int(chamber_a["cy"] + (chamber_b["cy"] - chamber_a["cy"]) * t)
        chunk_size = 64
        self._assert_chunk_matches_point_query(
            gen,
            chunk_x=mid_x // chunk_size,
            chunk_y=mid_y // chunk_size,
            chunk_size=chunk_size,
        )


class AnimationTests(unittest.TestCase):
    """Tests for animation system."""

    def test_hero_animations_has_all_states(self) -> None:
        expected_states = ("idle", "walk", "jump", "chant", "cast")
        for state in expected_states:
            self.assertIn(state, HERO_ANIMATIONS)
            frames = HERO_ANIMATIONS[state]
            self.assertEqual(len(frames), 4)

    def test_animation_manager_transitions(self) -> None:
        mgr = AnimationManager()
        frame = mgr.update(0.1, "walk")
        self.assertIsNotNone(frame)
        self.assertEqual(mgr.state, "walk")

    def test_animation_manager_frame_data(self) -> None:
        frame = HERO_ANIMATIONS["idle"][0]
        self.assertIsInstance(frame, FrameData)
        self.assertIsNotNone(frame.color)
        self.assertIsNotNone(frame.vertices)
        self.assertEqual(len(frame.vertices), 4)


if __name__ == "__main__":
    unittest.main()
