"""Tests for Phase 1 game shell: hero, entity_manager, spell_system, animation, screens."""

from __future__ import annotations

import math
import queue
import sys
import tempfile
import threading
import unittest
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.materials import build_material_registry
import engine.gpu_backend as gpu_backend
from engine.snapshot_mailbox import SnapshotMailboxRegistry
from engine.types import CellFlag, CellState
from engine.render import DebugViewMode
from engine.world import ActiveWorldWindow, GpuChunkCache, WorldChunkStore, WorldRect
from engine.gpu_backend import GpuMaterialTables, _delete_gl_sync, _gl_sync_signaled, pack_cells_state, unpack_grid_state
from src.engine.gpu_owner import GpuFramePayload

from game import config as cfg
from game.animation import AnimationManager, FrameData, HERO_ANIMATIONS
from game.app import GameApp, _fit_cell_scale_to_screen, run_game
from game.enemy import EnemyA
from game.projectile import Arrow
from game.entity_manager import Entity, EntityManager
from game.hero import Hero, GridFeedback, LocalCellSnapshot, feedback_from_snapshot
from game.screens import GameScreen
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

    def test_gpu_sync_helpers_tolerate_invalid_sync(self) -> None:
        with patch.object(gpu_backend, "_gl_client_wait_sync", side_effect=RuntimeError("bad sync")):
            self.assertTrue(_gl_sync_signaled(object()))
        with patch.object(gpu_backend, "_gl_delete_sync", side_effect=RuntimeError("bad sync")):
            _delete_gl_sync(object())

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

    def test_snapshot_feedback_only_marks_step_ahead_when_clearance_is_open(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        left = int(hero.left) - 2
        bottom = int(hero.bottom) - 2
        width = int(hero.right) - left + 4
        height = int(hero.top) - bottom + 4
        rows = [[0 for _ in range(width)] for _ in range(height)]
        right_wall_x = int(hero.right) + 1
        top_y = int(hero.top)
        clearance_y = max(int(hero.bottom), top_y - 1)
        stone_index = 1
        rows[top_y - bottom][right_wall_x - left] = stone_index
        rows[clearance_y - bottom][right_wall_x - left] = stone_index
        snapshot = LocalCellSnapshot(
            entity_id="hero",
            origin_x=left,
            origin_y=bottom,
            width=width,
            height=height,
            variant_indices=tuple(tuple(row) for row in rows),
            velocities=tuple(tuple((0.0, 0.0) for _ in range(width)) for _ in range(height)),
            temperatures=tuple(tuple(20.0 for _ in range(width)) for _ in range(height)),
            variant_families=("empty", "wood"),
            empty_variant_index=0,
            tick_id=1,
        )

        feedback = feedback_from_snapshot(
            snapshot=snapshot,
            center_x=hero.x,
            bottom_y=hero.y,
            width=hero.width,
            height=hero.height,
            world_width=cfg.WORLD_WIDTH,
        )

        self.assertTrue(feedback.blocked_right)
        self.assertFalse(feedback.blocked_right_ahead)

    def test_hero_does_not_climb_full_height_tree_trunk(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        hero.on_ground = True
        hero.input_right = True
        feedback = GridFeedback(
            world_width=100.0,
            blocked_below=True,
            blocked_right=True,
            blocked_right_ahead=False,
        )

        hero.update(0.0, grid_feedback=feedback)

        self.assertEqual(hero.y, 10.0)
        self.assertEqual(hero.vel_x, 0.0)

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

    def test_enemy_a_does_not_climb_same_step_every_frame_while_inching_forward(self) -> None:
        enemy = EnemyA.create("enemy_a_step_lock", 40.0, 20.0)
        enemy.patrol_direction = 1.0
        enemy.state = "patrol"
        original_y = enemy.y
        feedback = GridFeedback(
            world_width=100.0,
            blocked_below=True,
            blocked_right_ahead=True,
        )
        for _ in range(10):
            enemy.update(
                1.0 / 60.0,
                grid_feedback=feedback,
                hero_x=500.0,
                hero_y=20.0,
            )
        self.assertEqual(enemy.y, original_y - 1.0)
        self.assertGreater(enemy.x, 40.0)

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
        feedback: dict[str, GridFeedback] = {}
        for _ in range(8):
            world.step(0.0)
            feedback = mgr.poll_all_feedback(world)
            if not mgr._pending_feedback_tokens:
                break
        if mgr._pending_feedback_tokens:
            token = mgr._pending_feedback_tokens.popleft()
            forced = world.gpu_simulator.poll_batched_entity_feedback(token, force_ready=True)
            if forced is not None:
                feedback.update(forced)
                mgr._last_feedback = feedback
        hero_feedback = feedback.get("hero")
        self.assertIsNotNone(hero_feedback)
        return hero_feedback

    def test_read_feedback_uses_mirror_gpu_step_for_snapshot_age(self) -> None:
        _hero, mgr, _entity = self._make_manager_with_hero()
        snapshot = LocalCellSnapshot(
            entity_id="hero",
            origin_x=0,
            origin_y=0,
            width=1,
            height=1,
            variant_indices=((0,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((20.0,),),
            variant_families=("empty",),
            empty_variant_index=0,
            tick_id=5,
        )
        mgr.set_latest_snapshot("hero", snapshot)
        world = SimpleNamespace(world_width=cfg.WORLD_WIDTH, gpu_step_index=9)

        mgr.read_feedback_and_update(world, 0.0)

        self.assertEqual(snapshot.age_frames, 4)

    def test_read_feedback_and_update_uses_snapshot_without_grid_feedback_adapter(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        hero.on_ground = False
        hero.vel_y = 5.0
        world = SimpleNamespace(world_width=cfg.WORLD_WIDTH, gpu_step_index=9)
        left = int(hero.x - hero.width / 2.0)
        right = int(hero.x + hero.width / 2.0)
        bottom = int(hero.y) - 2
        top = int(hero.y + hero.height) + 4
        width = right - left + 1
        height = top - bottom + 1
        foot_y = int(hero.y + hero.height) + 1
        rows: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]
        for world_x in range(left, right + 1):
            rows[foot_y - bottom][world_x - left] = 1
        mgr._snapshot_registry.submit(
            SimpleNamespace(
                entity_id="hero",
                origin_x=left,
                origin_y=bottom,
                width=width,
                height=height,
                tick_id=8,
                packed=LocalCellSnapshot(
                    entity_id="hero",
                    origin_x=left,
                    origin_y=bottom,
                    width=width,
                    height=height,
                    variant_indices=tuple(tuple(row) for row in rows),
                    velocities=tuple(tuple((0.0, 0.0) for _ in range(width)) for _ in range(height)),
                    temperatures=tuple(tuple(20.0 for _ in range(width)) for _ in range(height)),
                    variant_families=("empty", "stone"),
                    empty_variant_index=0,
                    tick_id=8,
                ),
                submitted_at=0.0,
                age_frames=0,
            )
        )

        mgr.read_feedback_and_update(world, 0.0)

        self.assertTrue(hero.on_ground)
        self.assertEqual(hero.vel_y, 0.0)

    def test_latest_snapshot_prefers_newer_mailbox_snapshot_over_cached_snapshot(self) -> None:
        _hero, mgr, _entity = self._make_manager_with_hero()
        cached = LocalCellSnapshot(
            entity_id="hero",
            origin_x=0,
            origin_y=0,
            width=1,
            height=1,
            variant_indices=((0,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((20.0,),),
            variant_families=("empty",),
            empty_variant_index=0,
            tick_id=3,
        )
        newer = LocalCellSnapshot(
            entity_id="hero",
            origin_x=1,
            origin_y=2,
            width=1,
            height=1,
            variant_indices=((1,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((30.0,),),
            variant_families=("empty", "stone"),
            empty_variant_index=0,
            tick_id=5,
        )
        mgr.set_latest_snapshot("hero", cached)
        mgr._snapshot_registry.submit(
            SimpleNamespace(
                entity_id="hero",
                origin_x=newer.origin_x,
                origin_y=newer.origin_y,
                width=newer.width,
                height=newer.height,
                tick_id=newer.tick_id,
                packed=newer,
                submitted_at=0.0,
                age_frames=0,
            )
        )

        latest = mgr.latest_snapshot_for("hero", current_tick=8)

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.origin_x, newer.origin_x)
        self.assertEqual(latest.origin_y, newer.origin_y)
        self.assertEqual(latest.tick_id, newer.tick_id)
        self.assertEqual(latest.variant_indices, newer.variant_indices)
        self.assertEqual(latest.age_frames, 3)
        self.assertEqual(mgr._latest_snapshots["hero"].tick_id, newer.tick_id)

    def test_latest_snapshot_reuses_cached_snapshot_when_mailbox_has_no_fresh_value(self) -> None:
        _hero, mgr, _entity = self._make_manager_with_hero()
        cached = LocalCellSnapshot(
            entity_id="hero",
            origin_x=0,
            origin_y=0,
            width=1,
            height=1,
            variant_indices=((0,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((20.0,),),
            variant_families=("empty",),
            empty_variant_index=0,
            tick_id=4,
        )
        mgr.set_latest_snapshot("hero", cached)

        latest = mgr.latest_snapshot_for("hero", current_tick=9)

        self.assertIs(latest, cached)
        self.assertEqual(latest.age_frames, 5)

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

    def test_world_step_does_not_replace_terrain_with_entity_placeholders(self) -> None:
        _hero, mgr, entity = self._make_manager_with_hero()
        world = self._make_world()
        wx0 = int(entity.left)
        wx1 = int(entity.right) + 1
        wy0 = int(entity.bottom)
        wy1 = int(entity.top) + 1
        for wy in range(wy0, wy1):
            for wx in range(wx0, wx1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="stone",
                    variant_id="stone_platform",
                    integrity=1.0,
                ))

        mgr.tick(world, 0.0)
        world.step(0.0)

        for wy in range(wy0, wy1):
            for wx in range(wx0, wx1):
                lx, ly = _world_to_local(world, wx, wy)
                cell = world.gpu_simulator.readback_cells_region(lx, ly, 1, 1)[0][0]
                self.assertEqual(cell.family_id, "stone")
                self.assertEqual(cell.variant_id, "stone_platform")

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

    def test_grid_feedback_ignores_own_placeholder_cells(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        mgr.tick(world, 0.0)
        ground_wy = int(hero.top) + 1
        for wx in range(int(hero.left), int(hero.right) + 1):
            self._write_and_upload_cell(world, wx, ground_wy, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))
        world.step(0.0)
        feedback = self._read_gpu_feedback(mgr, world)
        self.assertTrue(feedback.blocked_below)
        self.assertFalse(feedback.blocked_left)
        self.assertFalse(feedback.blocked_right)
        self.assertFalse(feedback.embedded)

    def test_async_snapshot_request_polls_staged_payload(self) -> None:
        world = self._make_world()
        wx = world.active_origin_x + 5
        wy = world.active_origin_y + 6
        self._write_and_upload_cell(world, wx, wy, CellState(
            family_id="stone", variant_id="stone_platform", integrity=1.0))

        token = world.request_snapshot_cells_region_world(
            entity_id="hero",
            world_x=wx,
            world_y=wy,
            width=1,
            height=1,
        )
        snapshot = world.poll_snapshot_cells_region_world(token, force_ready=True)

        self.assertIsNotNone(snapshot)
        self.assertEqual((snapshot.world_x, snapshot.world_y), (wx, wy))
        self.assertEqual(snapshot.variant_families[snapshot.variant_indices[0][0]], "stone")

    def test_snapshot_request_can_use_step_gate_without_gl_sync(self) -> None:
        if self.ctx is None:
            self.skipTest("GPU context not available")
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=40,
            viewport_height=40,
            ctx=self.ctx,
            enable_gl_sync=False,
        )
        wx = world.active_origin_x + 5
        wy = world.active_origin_y + 6
        self._write_and_upload_cell(world, wx, wy, CellState(
            family_id="stone", variant_id="stone_platform", integrity=1.0))

        with patch.object(gpu_backend, "_gl_fence_sync", side_effect=AssertionError("glFenceSync should not be used")):
            token = world.request_snapshot_cells_region_world(
                entity_id="hero",
                world_x=wx,
                world_y=wy,
                width=1,
                height=1,
            )
        self.assertIsNone(token.sync)
        self.assertIsNone(world.poll_snapshot_cells_region_world(token, force_ready=False))

        world.step(0.0)
        snapshot = world.poll_snapshot_cells_region_world(token, force_ready=False)

        self.assertIsNotNone(snapshot)
        self.assertEqual((snapshot.world_x, snapshot.world_y), (wx, wy))
        self.assertEqual(snapshot.variant_families[snapshot.variant_indices[0][0]], "stone")

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

    def test_poll_all_feedback_never_forces_pending_feedback_ready(self) -> None:
        class DummyGpu:
            def __init__(self) -> None:
                self.force_ready_values: list[bool] = []

            def poll_batched_entity_feedback(self, token, *, force_ready: bool = False):
                self.force_ready_values.append(force_ready)
                return None

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 32
            active_height = 32
            world_width = cfg.WORLD_WIDTH

            def __init__(self) -> None:
                self.gpu_simulator = DummyGpu()

        _hero, mgr, _entity = self._make_manager_with_hero()
        world = DummyWorld()
        mgr._pending_feedback_tokens.append(object())  # type: ignore[arg-type]

        feedback = mgr.poll_all_feedback(world)  # type: ignore[arg-type]

        self.assertIn("hero", feedback)
        self.assertEqual(world.gpu_simulator.force_ready_values, [False])

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

    def test_read_feedback_and_update_applies_snapshot_damage_to_hero(self) -> None:
        hero, mgr, _entity = self._make_manager_with_hero()
        world = self._make_world()
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            for wx in range(int(hero.left), int(hero.right) + 1):
                self._write_and_upload_cell(world, wx, wy, CellState(
                    family_id="fire", variant_id="fire", temperature=600.0))
        token = world.request_snapshot_cells_region_world(
            entity_id="hero",
            world_x=int(hero.left) - 1,
            world_y=int(hero.bottom) - 1,
            width=int(hero.width) + 4,
            height=int(hero.height) + 4,
        )
        snapshot = world.poll_snapshot_cells_region_world(token, force_ready=True)
        self.assertIsNotNone(snapshot)
        mgr.set_latest_snapshot(
            "hero",
            LocalCellSnapshot(
                entity_id=snapshot.entity_id,
                origin_x=snapshot.world_x,
                origin_y=snapshot.world_y,
                width=snapshot.width,
                height=snapshot.height,
                variant_indices=snapshot.variant_indices,
                velocities=snapshot.velocities,
                temperatures=snapshot.temperatures,
                variant_families=snapshot.variant_families,
                empty_variant_index=snapshot.empty_variant_index,
                tick_id=snapshot.issued_step,
            ),
        )
        hp_before = hero.hp
        mgr.read_feedback_and_update(world, 0.0, feedback={"hero": GridFeedback(world_width=world.world_width)})
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
            try:
                cache.prime_rect_on_disk(WorldRect(0, 0, 4, 4))
                for _ in range(100):
                    if any(Path(tmp).rglob("0_0.ogchunk")):
                        break
                    time.sleep(0.02)
                self.assertTrue(any(Path(tmp).rglob("0_0.ogchunk")))
            finally:
                cache.shutdown()

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
            try:
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
            finally:
                world.close()

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

    def test_move_hero_to_prefetches_far_camera_jump_without_blocking_prepare(self) -> None:
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

        class DummyOwner:
            is_running = True

            def __init__(self) -> None:
                self.commands = []

            def fire_and_forget(self, command) -> None:
                self.commands.append(command)

            def submit(self, command):
                self.commands.append(command)
                return SimpleNamespace(result_if_ready=lambda: None)

        app.world = SimpleNamespace()
        app._gpu_owner_thread = DummyOwner()
        app._latest_gpu_world_status = {
            "camera": (0, 0),
            "active_origin": (0, 0),
            "active_size": (48, 48),
            "viewport_size": (16, 16),
            "world_size": (cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT),
            "gpu": {"step_index": 0},
        }
        GameApp.move_hero_to(app, 200.0, 120.0)
        self.assertEqual(app.hero.vel_x, 0.0)
        self.assertEqual(app.hero.vel_y, 0.0)
        self.assertFalse(app.hero.on_ground)
        teleports = [command.payload for command in app._gpu_owner_thread.commands if command.op.name == "CAMERA_TELEPORT"]
        self.assertEqual(len(teleports), 1)
        self.assertEqual((teleports[0].target_x, teleports[0].target_y, teleports[0].submit_chunks), (192, 112, 8))

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

        class DummyOwner:
            is_running = True

            def __init__(self) -> None:
                self.commands = []

            def fire_and_forget(self, command) -> None:
                self.commands.append(command)

            def submit(self, command):
                self.commands.append(command)
                return SimpleNamespace(result_if_ready=lambda: None)

        app.world = SimpleNamespace()
        app._gpu_owner_thread = DummyOwner()
        app._latest_gpu_world_status = {
            "camera": (40, 40),
            "active_origin": (0, 0),
            "active_size": (96, 96),
            "viewport_size": (16, 16),
            "world_size": (cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT),
            "gpu": {"step_index": 0},
        }
        GameApp.move_hero_to(app, 60.0, 60.0)
        teleports = [command.payload for command in app._gpu_owner_thread.commands if command.op.name == "CAMERA_TELEPORT"]
        self.assertEqual(len(teleports), 1)
        self.assertEqual((teleports[0].target_x, teleports[0].target_y, teleports[0].submit_chunks), (52, 52, 8))

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


class GameAppRuntimeTests(unittest.TestCase):
    def _make_app(self) -> GameApp:
        app = object.__new__(GameApp)
        app.hero = Hero()
        app.hero.x = 10.0
        app.hero.y = 10.0
        app._snapshot_registry = SnapshotMailboxRegistry()
        app.entity_manager = EntityManager(hero=app.hero, _snapshot_registry=app._snapshot_registry)
        app.enemies = {}
        app.projectiles = []
        app.active_streams = []
        app.active_pressure_bursts = []
        app._gpu_owner_thread = None
        app._pending_gpu_frame_handle = None
        app._latest_gpu_frame_payload = None
        app._pending_gpu_present_handle = None
        app._pending_gpu_snapshot_service_handle = None
        app._latest_gpu_world_status = None
        app.view_mode = DebugViewMode.MATERIAL
        app._sim_fps = 0.0
        app._sim_fps_count = 0
        app._sim_fps_started_at = 0.0
        app._sim_step_timestamps = deque(maxlen=240)
        app._gpu_tick_timestamps = deque(maxlen=240)
        app._perf_last_ms = {}
        app._perf_samples = {}
        app._chant_started = False
        app.stt = type("DummyStt", (), {"available": False})()
        return app

    class _CompletedHandle:
        def __init__(self, value=None, *, ok: bool = True) -> None:
            self._result = SimpleNamespace(ok=ok, value=value, error="" if ok else "failed")

        def done(self) -> bool:
            return True

        def wait(self, timeout: float = 0.0):
            return self._result

    class _RecordingOwner:
        is_running = True

        def __init__(self, snapshot=None, render_payload=None, status=None) -> None:
            self.commands = []
            self.snapshot = snapshot
            self.render_payload = render_payload
            self.status = status

        def fire_and_forget(self, command) -> None:
            self.commands.append(command)

        def submit(self, command):
            self.commands.append(command)
            if command.op.name == "SERVICE_SNAPSHOTS":
                requests = tuple(getattr(command.payload, "requests", ()) or ())
                snapshot = self.snapshot
                if snapshot is not None and requests:
                    request = requests[0]
                    app_registry = getattr(self, "snapshot_registry", None)
                    if app_registry is not None:
                        app_registry.submit(
                            SimpleNamespace(
                                entity_id=request.entity_id,
                                origin_x=getattr(snapshot, "world_x", request.world_x),
                                origin_y=getattr(snapshot, "world_y", request.world_y),
                                width=getattr(snapshot, "width", request.width),
                                height=getattr(snapshot, "height", request.height),
                                tick_id=getattr(snapshot, "issued_step", 0),
                                packed=snapshot,
                                submitted_at=request.submitted_at,
                                age_frames=0,
                            )
                        )
                return GameAppRuntimeTests._CompletedHandle(
                    {
                        "submitted": len(requests),
                        "completed": 1 if snapshot is not None and requests else 0,
                        "pending": 0,
                        "ready_delay_ms_max": 0.0,
                        "ready_delay_ms_mean": 0.0,
                    }
                )
            if command.op.name == "RENDER_FRAME":
                return GameAppRuntimeTests._CompletedHandle(self.render_payload)
            if command.op.name == "WORLD_STATUS":
                return GameAppRuntimeTests._CompletedHandle(self.status)
            return GameAppRuntimeTests._CompletedHandle(None)

    class _OwnerRouteWorld:
        active_origin_x = 0
        active_origin_y = 0
        active_width = 128
        active_height = 128
        viewport_width = 128
        viewport_height = 128
        world_width = cfg.WORLD_WIDTH
        camera_x = 0
        camera_y = 0
        gpu_simulator = None

        def __init__(self) -> None:
            self.chunk_cache = self

        def step(self, dt: float) -> None:
            raise AssertionError("owner route must not call world.step on the main thread")

        def schedule_prefetch_for_rect(self, *args, **kwargs) -> None:
            raise AssertionError("owner route must not schedule prefetch on the main thread")

        def pan_camera(self, *args, **kwargs) -> None:
            raise AssertionError("owner route must not pan the camera on the main thread")

        def set_camera(self, *args, **kwargs) -> None:
            raise AssertionError("owner route must not set the camera on the main thread")

        def mark_camera_activity(self, *args, **kwargs) -> None:
            raise AssertionError("owner route must not mark camera activity on the main thread")

        def prefetch_camera_region(self, *args, **kwargs) -> None:
            raise AssertionError("owner route must not prefetch camera regions on the main thread")

        def service_background_io(self) -> None:
            raise AssertionError("owner route must not service world IO on the main thread")

        def request_snapshot_cells_region_world(self, **kwargs):
            raise AssertionError("owner route must not submit snapshots through world directly")

        def poll_snapshot_cells_region_world(self, *args, **kwargs):
            raise AssertionError("owner route must not poll snapshots through world directly")

    def _make_owner_route_app(self, *, snapshot=None) -> tuple[GameApp, _RecordingOwner]:
        app = self._make_app()
        app.world = self._OwnerRouteWorld()
        status = {
            "camera": (0, 0),
            "active_origin": (0, 0),
            "active_size": (128, 128),
            "viewport_size": (128, 128),
            "world_size": (cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT),
            "gpu": {"step_index": 0},
        }
        owner = self._RecordingOwner(snapshot=snapshot, status=status)
        owner.snapshot_registry = app._snapshot_registry
        app._gpu_owner_thread = owner
        app.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=app.hero.x,
            y=app.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))
        return app, owner

    def test_fit_cell_scale_to_screen_clamps_oversized_default(self) -> None:
        self.assertEqual(
            _fit_cell_scale_to_screen(4, screen_width=1920, screen_height=1080),
            2,
        )

    def test_fit_cell_scale_to_screen_preserves_requested_scale_when_it_fits(self) -> None:
        self.assertEqual(
            _fit_cell_scale_to_screen(4, screen_width=3840, screen_height=2160),
            4,
        )

    def test_run_game_uses_120hz_redraw_interval(self) -> None:
        with patch("game.app.GameApp") as mock_game_app, patch("game.app.pyglet.app.run") as mock_run:
            run_game(seed=7, cell_scale=3)

        mock_game_app.assert_called_once_with(seed=7, cell_scale=3)
        mock_game_app.return_value.set_minimum_size.assert_called_once_with(400, 300)
        mock_run.assert_called_once_with(interval=1.0 / 120.0)

    def test_owner_created_world_is_required(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(GameApp._owner_created_world_requested())
        with patch.dict("os.environ", {"ORACLE_TRANSLATOR_OWNER_CREATED_WORLD": "0"}, clear=True):
            with self.assertRaises(RuntimeError):
                GameApp._owner_created_world_requested()
        with patch.dict("os.environ", {"ORACLE_TRANSLATOR_OWNER_CREATED_WORLD": "true"}, clear=True):
            self.assertTrue(GameApp._owner_created_world_requested())

    def test_owner_display_renderer_avoids_legacy_moderngl_factory(self) -> None:
        app = object.__new__(GameApp)
        app.ctx = None
        with patch.object(GameApp, "_owner_created_world_requested", return_value=True):
            with patch.object(GameApp, "_create_legacy_renderer", side_effect=AssertionError("legacy renderer should not be created")):
                renderer = GameApp._create_display_renderer(app, 640, 380)
        self.assertEqual(type(renderer).__name__, "OwnerFrameRenderer")
        self.assertIsNone(app.ctx)

    def test_gpu_frame_payload_snapshot_is_nonblocking_and_caches_owner_status(self) -> None:
        app = self._make_app()
        payload = GpuFramePayload(
            width=2,
            height=2,
            rgba=b"\x01" * 16,
            uv_rect=(0.0, 0.0, 1.0, 1.0),
            status={
                "camera": (3, 4),
                "active_origin": (0, 0),
                "active_size": (2, 2),
                "viewport_size": (2, 2),
                "world_size": (8, 8),
            },
            view_mode=DebugViewMode.MATERIAL,
        )
        app.world = self._OwnerRouteWorld()
        owner = self._RecordingOwner(render_payload=payload)
        app._gpu_owner_thread = owner

        first = GameApp.gpu_frame_payload_snapshot(app)

        self.assertIsNone(first)
        self.assertEqual([command.op.name for command in owner.commands], ["RENDER_FRAME"])
        self.assertTrue(owner.commands[0].payload.readback_rgba)

        second = GameApp.gpu_frame_payload_snapshot(app)

        self.assertIs(second, payload)
        self.assertIs(app._latest_gpu_frame_payload, payload)
        self.assertEqual(app._latest_gpu_world_status["camera"], (3, 4))
        self.assertEqual(
            [command.op.name for command in owner.commands],
            ["RENDER_FRAME", "RENDER_FRAME"],
        )

    def test_read_framebuffer_rgb_uses_owner_payload_without_gl_read(self) -> None:
        app = self._make_app()
        app._width = 4
        app._height = 4
        app._gpu_owner_thread = self._RecordingOwner()
        app._latest_gpu_frame_payload = GpuFramePayload(
            width=2,
            height=2,
            rgba=bytes((
                10, 20, 30, 255,
                40, 50, 60, 255,
                70, 80, 90, 255,
                100, 110, 120, 255,
            )),
            uv_rect=(0.0, 0.0, 1.0, 1.0),
            status={},
            view_mode=DebugViewMode.MATERIAL,
        )
        app.ctx = SimpleNamespace(
            screen=SimpleNamespace(read=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("owner mode must not read GL framebuffer")))
        )

        rgb = GameApp.read_framebuffer_rgb(app)

        self.assertEqual(len(rgb), 4 * 4 * 3)
        self.assertEqual(rgb[0:3], bytes((10, 20, 30)))
        self.assertEqual(rgb[6:9], bytes((40, 50, 60)))
        self.assertEqual(rgb[-3:], bytes((100, 110, 120)))
        frame_w, frame_h, frame_rgb = GameApp.owner_frame_view_rgb_snapshot(app)
        self.assertEqual((frame_w, frame_h), (2, 2))
        self.assertEqual(
            frame_rgb,
            bytes((
                10, 20, 30,
                40, 50, 60,
                70, 80, 90,
                100, 110, 120,
            )),
        )

    def test_owner_frame_view_rgb_snapshot_reads_presented_owner_frame(self) -> None:
        app = self._make_app()
        app.current_screen = "game"
        app._width = 4
        app._height = 3
        app.world = self._OwnerRouteWorld()
        owner = self._RecordingOwner(
            render_payload={
                "width": 4,
                "height": 3,
                "screen_rgb": bytes(range(4 * 3 * 3)),
                "status": {"camera": (7, 8)},
                "overlay_line_count": 2,
                "enemy_overlay_count": 0,
                "projectile_overlay_count": 0,
            }
        )
        app._gpu_owner_thread = owner

        width, height, rgb = GameApp.owner_frame_view_rgb_snapshot(app)

        self.assertEqual((width, height), (4, 3))
        self.assertEqual(rgb, bytes(range(4 * 3 * 3)))
        self.assertEqual(app._latest_gpu_world_status["camera"], (7, 8))
        self.assertEqual(app._last_owner_present_overlay_line_count, 2)
        self.assertEqual([command.op.name for command in owner.commands], ["RENDER_FRAME"])
        payload = owner.commands[0].payload
        self.assertTrue(payload.present)
        self.assertTrue(payload.readback_present_rgb)
        self.assertFalse(payload.readback_rgba)

    def test_submit_gpu_present_frame_sends_overlay_and_actor_counts(self) -> None:
        app = self._make_app()
        app.current_screen = "game"
        app._width = 640
        app._height = 384
        app._last_dt = 1.0 / 60.0
        app._debug_overlay_enabled = True
        app.world = self._OwnerRouteWorld()
        enemy = EnemyA.create("enemy_a", 24.0, 18.0, "plains")
        app.enemies[enemy.entity_id] = enemy
        app.projectiles = [Arrow(x=16.0, y=20.0, vel_x=8.0, vel_y=0.0)]
        app._latest_gpu_world_status = {
            "gpu": {"step_index": 9},
            "chunk": {
                "cached": 3,
                "clean_resident": 2,
                "dirty_resident": 1,
                "queued_read": 4,
                "queued_write": 5,
                "queued_generate": 6,
                "worker_fallback": 7,
            },
            "paging": {
                "shift_ms": 1.5,
                "last_shift_disk_loads": 2,
                "last_shift_disk_load_ms": 3.5,
                "last_shift_generates": 4,
                "last_shift_generate_ms": 5.5,
                "last_shift_saves": 6,
                "last_shift_save_ms": 7.5,
                "gpu_writeback_queue_depth": 8,
            },
        }
        app._snapshot_registry.submit(
            SimpleNamespace(
                entity_id="hero",
                origin_x=0,
                origin_y=0,
                width=1,
                height=1,
                tick_id=7,
                packed=LocalCellSnapshot(
                    entity_id="hero",
                    origin_x=0,
                    origin_y=0,
                    width=1,
                    height=1,
                    variant_indices=((0,),),
                    velocities=(((0.0, 0.0),),),
                    temperatures=((20.0,),),
                    variant_families=("empty",),
                    empty_variant_index=0,
                    tick_id=7,
                ),
                submitted_at=0.0,
                age_frames=0,
            )
        )
        owner = self._RecordingOwner(
            render_payload={
                "status": {"camera": (1, 2)},
                "overlay_line_count": 6,
                "enemy_overlay_count": 1,
                "projectile_overlay_count": 1,
            }
        )
        app._gpu_owner_thread = owner

        GameApp._submit_gpu_present_frame(app)

        self.assertIsNotNone(app._pending_gpu_present_handle)
        self.assertEqual([command.op.name for command in owner.commands], ["RENDER_FRAME"])
        payload = owner.commands[0].payload
        self.assertTrue(payload.present)
        self.assertFalse(payload.readback_rgba)
        self.assertEqual(payload.window_size, (640, 384))
        self.assertEqual(len(payload.enemies), 1)
        self.assertEqual(len(payload.projectiles), 1)
        self.assertGreaterEqual(len(payload.overlay_lines), 1)
        self.assertTrue(any("Snapshot age" in line for line in payload.overlay_lines))
        self.assertTrue(any("wb 8" in line for line in payload.overlay_lines))
        self.assertEqual(app._last_owner_present_enemy_count, 1)
        self.assertEqual(app._last_owner_present_projectile_count, 1)
        self.assertEqual(app._last_owner_present_overlay_line_count, 1 + len(payload.overlay_lines))

        GameApp._submit_gpu_present_frame(app)

        self.assertIsNotNone(app._pending_gpu_present_handle)
        self.assertEqual(app._latest_gpu_world_status["camera"], (1, 2))
        self.assertEqual(app._last_owner_present_overlay_line_count, 7)
        self.assertEqual(app._last_owner_present_enemy_count, 1)
        self.assertEqual(app._last_owner_present_projectile_count, 1)
        self.assertEqual([command.op.name for command in owner.commands], ["RENDER_FRAME", "RENDER_FRAME"])

    def test_submit_gpu_present_frame_includes_console_lines_when_console_open(self) -> None:
        app = self._make_app()
        app.current_screen = "game"
        app._width = 640
        app._height = 384
        app._last_dt = 1.0 / 60.0
        app.world = self._OwnerRouteWorld()
        app._game_screen = GameScreen(app)
        app.screens = {"game": app._game_screen}
        app._game_screen.show_console = True
        owner = self._RecordingOwner(render_payload={"status": {"camera": (0, 0)}})
        app._gpu_owner_thread = owner

        GameApp._submit_gpu_present_frame(app)

        payload = owner.commands[0].payload
        self.assertTrue(any("[Console]" in line for line in payload.console_lines))
        self.assertTrue(any("Spells:" in line for line in payload.console_lines))

    def test_process_enemy_ai_skips_offscreen_enemy_updates(self) -> None:
        app = self._make_app()

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 32
            active_height = 32
            world_width = cfg.WORLD_WIDTH

            def paint_world(self, *args, **kwargs) -> None:
                raise AssertionError("offscreen enemy should not touch world")

        app.world = DummyWorld()
        enemy = EnemyA.create("enemy_a", 200.0, 200.0, "plains")
        enemy.state = "attack"
        enemy.arrow_cooldown = 0.0
        app.enemies[enemy.entity_id] = enemy
        app.entity_manager.register_enemy(enemy)
        before = (enemy.x, enemy.y, enemy.arrow_cooldown)

        GameApp._process_enemy_ai(app, 1.0 / 60.0)

        self.assertEqual(app.projectiles, [])
        self.assertEqual((enemy.x, enemy.y, enemy.arrow_cooldown), before)

    def test_process_enemy_ai_uses_empty_snapshot_defaults_when_no_snapshot_exists(self) -> None:
        app = self._make_app()

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 128
            active_height = 128
            world_width = cfg.WORLD_WIDTH

            def paint_world(self, *args, **kwargs) -> None:
                raise AssertionError("enemy without feedback should not attack")

        app.world = DummyWorld()
        app.hero.x = 500.0
        app.hero.y = 30.0
        enemy = EnemyA.create("enemy_a", 30.0, 30.0, "plains")
        enemy.vel_y = 5.0
        app.enemies[enemy.entity_id] = enemy
        app.entity_manager.register_enemy(enemy)

        GameApp._process_enemy_ai(app, 0.5)

        self.assertGreater(enemy.y, 30.0)
        self.assertEqual(app.projectiles, [])

    def test_process_enemy_ai_prefers_latest_snapshot_over_grid_feedback(self) -> None:
        app = self._make_app()

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 128
            active_height = 128
            world_width = cfg.WORLD_WIDTH
            gpu_simulator = SimpleNamespace(step_index=9)

            def paint_world(self, *args, **kwargs) -> None:
                raise AssertionError("enemy should not attack in this test")

        app.world = DummyWorld()
        app.hero.x = 500.0
        app.hero.y = 30.0
        enemy = EnemyA.create("enemy_a", 30.0, 30.0, "plains")
        app.enemies[enemy.entity_id] = enemy
        app.entity_manager.register_enemy(enemy)
        app.entity_manager._last_feedback[enemy.entity_id] = GridFeedback(
            world_width=cfg.WORLD_WIDTH,
            blocked_below=False,
        )

        left = int(enemy.x - enemy.width / 2.0)
        right = int(enemy.x + enemy.width / 2.0)
        bottom = int(enemy.y) - 2
        top = int(enemy.y + enemy.height) + 4
        width = right - left + 1
        height = top - bottom + 1
        foot_y = int(enemy.y + enemy.height) + 1
        rows: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]
        for world_x in range(left, right + 1):
            rows[foot_y - bottom][world_x - left] = 1
        app.entity_manager.set_latest_snapshot(
            enemy.entity_id,
            LocalCellSnapshot(
                entity_id=enemy.entity_id,
                origin_x=left,
                origin_y=bottom,
                width=width,
                height=height,
                variant_indices=tuple(tuple(row) for row in rows),
                velocities=tuple(tuple((0.0, 0.0) for _ in range(width)) for _ in range(height)),
                temperatures=tuple(tuple(20.0 for _ in range(width)) for _ in range(height)),
                variant_families=("empty", "stone"),
                empty_variant_index=0,
                tick_id=8,
            ),
        )

        GameApp._process_enemy_ai(app, 0.5)

        self.assertTrue(enemy.on_ground)
        self.assertEqual(enemy.vel_y, 0.0)

    def test_enemy_b_snapshot_collision_path_uses_snapshot_without_grid_feedback(self) -> None:
        app = self._make_app()

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 128
            active_height = 128
            world_width = cfg.WORLD_WIDTH
            gpu_simulator = SimpleNamespace(step_index=9)

        app.world = DummyWorld()
        app.hero.x = 500.0
        app.hero.y = 30.0
        from game.enemy import EnemyB

        enemy = EnemyB.create("enemy_b", 30.0, 30.0, "plains")
        enemy.state = "charge"
        app.enemies[enemy.entity_id] = enemy
        app.entity_manager.register_enemy(enemy)
        before = (enemy.x, enemy.y)

        left = int(enemy.x - enemy.width / 2.0) - 2
        right = int(enemy.x + enemy.width / 2.0) + 2
        bottom = int(enemy.y) - 2
        top = int(enemy.y + enemy.height) + 2
        width = right - left + 1
        height = top - bottom + 1
        rows: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]
        block_x = int(enemy.x) - left
        block_y = int(enemy.y) - bottom
        rows[block_y][block_x] = 1
        app._snapshot_registry.submit(
            SimpleNamespace(
                entity_id=enemy.entity_id,
                origin_x=left,
                origin_y=bottom,
                width=width,
                height=height,
                tick_id=8,
                packed=LocalCellSnapshot(
                    entity_id=enemy.entity_id,
                    origin_x=left,
                    origin_y=bottom,
                    width=width,
                    height=height,
                    variant_indices=tuple(tuple(row) for row in rows),
                    velocities=tuple(tuple((0.0, 0.0) for _ in range(width)) for _ in range(height)),
                    temperatures=tuple(tuple(20.0 for _ in range(width)) for _ in range(height)),
                    variant_families=("empty", "stone"),
                    empty_variant_index=0,
                    tick_id=8,
                ),
                submitted_at=0.0,
                age_frames=0,
            )
        )

        GameApp._process_enemy_ai(app, 1.0 / 60.0)

        self.assertNotEqual((enemy.x, enemy.y), before)

    def test_update_game_ticks_gpu_with_enemy_position_after_enemy_ai(self) -> None:
        app, owner = self._make_owner_route_app()
        app.hero.x = 500.0
        app.hero.y = 30.0
        enemy = EnemyA.create("enemy_a", 30.0, 30.0, "plains")
        enemy.patrol_direction = 1.0
        enemy.state = "patrol"
        app.enemies[enemy.entity_id] = enemy
        app.entity_manager.register_enemy(enemy)
        app.entity_manager._last_feedback[enemy.entity_id] = GridFeedback(world_width=cfg.WORLD_WIDTH, blocked_below=True)
        app.entity_manager.read_feedback_and_update = lambda world, dt, feedback=None: None  # type: ignore[method-assign]
        app.entity_manager.debug_collision = False
        app.entity_manager.poll_all_feedback = lambda world: (_ for _ in ()).throw(AssertionError("feedback polling is debug-only"))  # type: ignore[method-assign]
        app.entity_manager.schedule_feedback = lambda world: (_ for _ in ()).throw(AssertionError("feedback scheduling is debug-only"))  # type: ignore[method-assign]

        app._service_pressure_bursts = lambda: None
        app._refresh_local_entity_snapshots = lambda: None
        app._set_perf_ms = lambda name, value: None
        app._record_sim_fps = lambda: None
        app._process_projectiles = lambda dt: None
        app._cleanup_dead_enemies = lambda: None

        GameApp.update_game(app, 1.0 / 60.0)

        self.assertGreater(app.entity_manager._entities[enemy.entity_id].x, 30.0)
        state_payloads = [
            command.payload
            for command in owner.commands
            if command.op.name == "SYNC_ENTITY_STATE" and command.payload.states
        ]
        self.assertTrue(state_payloads)
        uploaded_enemy = next(
            state
            for state in state_payloads[-1].states
            if state[0] == enemy.entity_id
        )
        self.assertEqual(uploaded_enemy[1], int(app.entity_manager._entities[enemy.entity_id].x))
        self.assertIn("STEP_WORLD", [command.op.name for command in owner.commands])

    def test_update_game_delegates_world_step_to_gpu_boundary(self) -> None:
        app, _owner = self._make_owner_route_app()
        app.entity_manager.read_feedback_and_update = lambda world, dt, feedback=None: None  # type: ignore[method-assign]
        app.entity_manager.debug_collision = False
        app.entity_manager.poll_all_feedback = lambda world: (_ for _ in ()).throw(AssertionError("feedback polling is debug-only"))  # type: ignore[method-assign]
        app.entity_manager.schedule_feedback = lambda world: (_ for _ in ()).throw(AssertionError("feedback scheduling is debug-only"))  # type: ignore[method-assign]
        app._process_enemy_ai = lambda dt: None
        app._service_pressure_bursts = lambda: None
        app._refresh_local_entity_snapshots = lambda: None
        app._process_projectiles = lambda dt: None
        app._cleanup_dead_enemies = lambda: None
        app._record_sim_fps = lambda: None
        submitted: list[float] = []
        app._submit_gpu_world_tick = lambda dt: submitted.append(dt)  # type: ignore[method-assign]

        GameApp.update_game(app, 1.0 / 60.0)

        self.assertEqual(submitted, [1.0 / 60.0])

    def test_owner_world_skips_legacy_feedback_even_when_debug_collision_enabled(self) -> None:
        app, owner = self._make_owner_route_app()
        app.entity_manager.read_feedback_and_update = lambda world, dt, feedback=None: None  # type: ignore[method-assign]
        app.entity_manager.debug_collision = True
        app.entity_manager.poll_all_feedback = lambda world: (_ for _ in ()).throw(AssertionError("owner route must not poll legacy feedback"))  # type: ignore[method-assign]
        app.entity_manager.schedule_feedback = lambda world: (_ for _ in ()).throw(AssertionError("owner route must not schedule legacy feedback"))  # type: ignore[method-assign]
        app._process_enemy_ai = lambda dt: None
        app._service_pressure_bursts = lambda: None
        app._refresh_local_entity_snapshots = lambda: None
        app._process_projectiles = lambda dt: None
        app._cleanup_dead_enemies = lambda: None
        app._record_sim_fps = lambda: None

        GameApp.update_game(app, 1.0 / 60.0)

        self.assertIn("STEP_WORLD", [command.op.name for command in owner.commands])

    def test_trigger_explosion_queues_world_mutations_until_flush(self) -> None:
        app, owner = self._make_owner_route_app()

        GameApp._trigger_explosion(app, 10, 20)

        self.assertGreater(app._world_mutation_sink().pending_count, 0)

        flushed = GameApp._flush_gpu_world_mutations(app)

        self.assertGreater(flushed, 0)
        self.assertEqual(app._world_mutation_sink().pending_count, 0)
        mutation_payloads = [
            command.payload
            for command in owner.commands
            if command.op.name == "APPLY_WORLD_MUTATIONS"
        ]
        self.assertTrue(mutation_payloads)
        mutation_types = {type(command).__name__ for command in mutation_payloads[-1]}
        self.assertIn("PaintWorldCommand", mutation_types)
        self.assertIn("InjectPressureWorldCommand", mutation_types)
        self.assertIn("InjectPressureRingWorldCommand", mutation_types)

    def test_sim_fps_uses_rolling_step_history(self) -> None:
        app = self._make_app()
        app._sim_fps = 0.0
        app._sim_fps_count = 0
        app._sim_fps_started_at = 100.0
        app._sim_step_timestamps = deque(maxlen=240)

        with patch("game.app.perf_counter", side_effect=(100.0, 100.0 + 1.0 / 60.0, 100.0 + 2.0 / 60.0, 100.0 + 3.0 / 60.0)):
            GameApp._record_sim_fps(app)
            GameApp._record_sim_fps(app)
            GameApp._record_sim_fps(app)
            GameApp._record_sim_fps(app)

        self.assertGreaterEqual(app.sim_fps, 59.9)
        self.assertLessEqual(app.sim_fps, 60.1)

    def test_gpu_tick_rate_uses_rolling_step_history(self) -> None:
        app = self._make_app()
        app._gpu_tick_timestamps = deque(maxlen=240)

        with patch("game.app.perf_counter", side_effect=(100.0, 100.0 + 1.0 / 60.0, 100.0 + 2.0 / 60.0, 100.0 + 3.0 / 60.0)):
            GameApp._record_gpu_tick(app)
            GameApp._record_gpu_tick(app)
            GameApp._record_gpu_tick(app)
            GameApp._record_gpu_tick(app)

        self.assertGreaterEqual(app.gpu_tick_rate, 59.9)
        self.assertLessEqual(app.gpu_tick_rate, 60.1)

    def test_refresh_local_snapshots_skips_offscreen_projectiles(self) -> None:
        app, owner = self._make_owner_route_app()
        app.projectiles = [Arrow(x=200.0, y=200.0, vel_x=0.0, vel_y=0.0)]
        app._snapshot_refresh_counter = 3

        GameApp._refresh_local_entity_snapshots(app)

        self.assertEqual(app.entity_manager._latest_snapshots, {})
        service_commands = [command for command in owner.commands if command.op.name == "SERVICE_SNAPSHOTS"]
        self.assertEqual(len(service_commands), 1)
        payload = service_commands[0].payload
        self.assertEqual([request.entity_id for request in payload.requests], ["hero"])

    def test_queue_local_snapshot_only_records_request_until_gpu_boundary(self) -> None:
        app = self._make_app()

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 64
            active_height = 64

            def request_snapshot_cells_region_world(self, **_kwargs):
                raise AssertionError("queue stage must not submit GPU snapshot requests")

        app.world = DummyWorld()

        GameApp._queue_local_snapshot(app, "hero", 1, 2, 4, 6)

        queued = list(app._snapshot_request_queue())
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].entity_id, "hero")
        self.assertEqual((queued[0].world_x, queued[0].world_y, queued[0].width, queued[0].height), (1, 2, 3, 4))

    def test_refresh_local_snapshots_uses_owner_snapshot_service(self) -> None:
        snapshot = SimpleNamespace(
            entity_id="hero",
            world_x=4,
            world_y=5,
            width=1,
            height=1,
            variant_indices=((0,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((20.0,),),
            variant_families=("empty",),
            empty_variant_index=0,
            issued_step=7,
        )
        app, owner = self._make_owner_route_app(snapshot=snapshot)
        app._snapshot_refresh_counter = 3
        GameApp._refresh_local_entity_snapshots(app)

        self.assertEqual([command.op.name for command in owner.commands], ["SERVICE_SNAPSHOTS"])
        latest = app.entity_manager.latest_snapshot_for("hero")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.tick_id, 7)

    def test_gpu_owner_routes_world_tick_and_mutations_without_direct_world_calls(self) -> None:
        app, owner = self._make_owner_route_app()

        GameApp._queue_world_pressure(app, 1, 2, 3, 4.0)
        GameApp._submit_gpu_world_tick(app, 1.0 / 60.0)

        op_names = [command.op.name for command in owner.commands]
        self.assertIn("APPLY_WORLD_MUTATIONS", op_names)
        self.assertIn("SYNC_ENTITY_STATE", op_names)
        self.assertIn("STEP_WORLD", op_names)
        step_commands = [command for command in owner.commands if command.op.name == "STEP_WORLD"]
        self.assertEqual(step_commands[-1].payload["dt"], 1.0 / 60.0)

    def test_gpu_owner_routes_camera_prefetch_entity_and_io_commands(self) -> None:
        app, owner = self._make_owner_route_app()

        GameApp._submit_gpu_chunk_prefetch(
            app,
            WorldRect(8, 12, 16, 20),
            margin_x=1,
            margin_y=2,
            prioritize=True,
        )
        GameApp._submit_gpu_entity_shape_registration(app)
        GameApp._submit_gpu_entity_state_update(app)
        GameApp._submit_gpu_entity_mask_update(app)
        GameApp._submit_gpu_camera_follow(app, 30, 40, dt=0.25)
        GameApp._submit_gpu_camera_teleport(app, 400, 500)
        GameApp._submit_gpu_background_io_service(app)

        op_names = [command.op.name for command in owner.commands]
        self.assertEqual(op_names[0], "SCHEDULE_PREFETCH")
        self.assertIn("CAMERA_FOLLOW", op_names)
        self.assertIn("CAMERA_TELEPORT", op_names)
        self.assertIn("SERVICE_BACKGROUND_IO", op_names)
        sync_payloads = [
            command.payload
            for command in owner.commands
            if command.op.name == "SYNC_ENTITY_STATE"
        ]
        self.assertTrue(any(payload.shapes for payload in sync_payloads))
        self.assertTrue(any(payload.states for payload in sync_payloads))
        self.assertTrue(any(payload.mask_rects for payload in sync_payloads))
        follow = next(command.payload for command in owner.commands if command.op.name == "CAMERA_FOLLOW")
        self.assertEqual((follow.target_x, follow.target_y, follow.dt), (30, 40, 0.25))
        teleport = next(command.payload for command in owner.commands if command.op.name == "CAMERA_TELEPORT")
        self.assertEqual((teleport.target_x, teleport.target_y, teleport.submit_chunks), (400, 500, 8))

    def test_submit_gpu_camera_follow_submits_owner_command(self) -> None:
        app, owner = self._make_owner_route_app()

        GameApp._submit_gpu_camera_follow(app, 30, 40, dt=0.25)

        self.assertIn("CAMERA_FOLLOW", [command.op.name for command in owner.commands])

    def test_experiment_camera_override_controls_follow_target(self) -> None:
        app, owner = self._make_owner_route_app()
        app.current_screen = "game"
        app._last_dt = 1.0 / 60.0
        app._width = cfg.VIEWPORT_WIDTH * cfg.CELL_SCALE
        app._height = cfg.VIEWPORT_HEIGHT * cfg.CELL_SCALE
        app._keys_pressed = set()
        app._sim_accumulator = 0.0
        app._owner_input_events = queue.SimpleQueue()
        app._pending_gpu_camera_follow_handle = None
        app._pending_gpu_background_io_handle = None
        app._pending_gpu_present_handle = None
        app._experiment_camera_override = (321, 654)
        app._experiment_freeze_gameplay = True
        app._game_screen = SimpleNamespace(update=lambda dt: None)
        app.screens = {"game": app._game_screen}

        GameApp._tick(app, 1.0 / 120.0)

        follow = next(command.payload for command in owner.commands if command.op.name == "CAMERA_FOLLOW")
        self.assertEqual((follow.target_x, follow.target_y), (321, 654))

    def test_update_game_experiment_freeze_skips_actor_sim_and_still_submits_world_tick(self) -> None:
        app, owner = self._make_owner_route_app()
        app._experiment_freeze_gameplay = True
        app.hero.state = "idle"
        app.projectiles = [Arrow(x=12.0, y=13.0, vel_x=1.0, vel_y=2.0)]
        app.enemies = {"enemy_a": EnemyA.create("enemy_a", 16.0, 20.0, "plains")}
        app._sim_fps_count = 0
        app._sim_fps_started_at = 0.0

        with patch.object(GameApp, "_process_enemy_ai", side_effect=AssertionError("enemy AI should be frozen")), \
             patch.object(GameApp, "_process_projectiles", side_effect=AssertionError("projectiles should be frozen")), \
             patch.object(GameApp, "_cleanup_dead_enemies", side_effect=AssertionError("cleanup should be frozen")):
            GameApp.update_game(app, 1.0 / 60.0)

        self.assertIn("STEP_WORLD", [command.op.name for command in owner.commands])
        self.assertGreaterEqual(app.sim_fps, 0.0)

    def test_gpu_owner_routes_snapshot_service_without_direct_world_calls(self) -> None:
        snapshot = SimpleNamespace(
            entity_id="hero",
            world_x=4,
            world_y=5,
            width=1,
            height=1,
            variant_indices=((0,),),
            velocities=(((0.0, 0.0),),),
            temperatures=((20.0,),),
            variant_families=("empty",),
            empty_variant_index=0,
            issued_step=7,
        )
        app, owner = self._make_owner_route_app(snapshot=snapshot)

        GameApp._queue_local_snapshot(app, "hero", 4, 5, 5, 6)
        submitted = GameApp._submit_gpu_snapshot_requests(app)

        self.assertEqual(submitted, 1)
        self.assertEqual([command.op.name for command in owner.commands], ["SERVICE_SNAPSHOTS"])
        latest = app.entity_manager.latest_snapshot_for("hero")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.tick_id, 7)
        self.assertIsNotNone(app._pending_gpu_snapshot_service_handle)

    def test_started_gpu_owner_runs_gameapp_world_commands_off_main_thread(self) -> None:
        app = self._make_app()

        class DummyGpu:
            step_index = 0

            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def upload_entity_states(self, states, *, origin_x: int, origin_y: int) -> None:
                self.calls.append(("states", threading.get_ident(), tuple(states), origin_x, origin_y))

            def update_entity_mask(self, rects) -> None:
                self.calls.append(("mask", threading.get_ident(), tuple(rects)))

            def register_entity_shape(self, entity_id: str, width: int, height: int) -> None:
                self.calls.append(("shape", threading.get_ident(), entity_id, width, height))

        class DummyWorld:
            active_origin_x = 0
            active_origin_y = 0
            active_width = 128
            active_height = 128
            world_width = cfg.WORLD_WIDTH
            camera_x = 0
            camera_y = 0

            def __init__(self) -> None:
                self.gpu_simulator = DummyGpu()
                self.calls: list[tuple] = []

            def step(self, dt: float) -> None:
                self.calls.append(("step", threading.get_ident(), dt))

            def close(self) -> None:
                self.calls.append(("close", threading.get_ident()))

        world = DummyWorld()
        app.world = world
        app.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=app.hero.x,
            y=app.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))

        self.assertTrue(GameApp._start_gpu_owner_thread(app))
        owner = app._gpu_owner_thread
        try:
            main_tid = threading.get_ident()
            GameApp._submit_gpu_entity_shape_registration(app)
            GameApp._submit_gpu_world_tick(app, 1.0 / 60.0)
            owner.run_on_owner(lambda: None, timeout=2.0)

            owner_tid = owner.owner_thread_id
            self.assertIsNotNone(owner_tid)
            self.assertNotEqual(owner_tid, main_tid)
            self.assertIn(("step", owner_tid, 1.0 / 60.0), world.calls)
            self.assertTrue(world.gpu_simulator.calls)
            self.assertTrue(all(call[1] == owner_tid for call in world.gpu_simulator.calls))

            GameApp._close_game_world(app)
            self.assertIn(("close", owner_tid), world.calls)
            self.assertIsNone(app.world)
            self.assertIsNone(app._gpu_owner_thread)
        finally:
            if getattr(app, "_gpu_owner_thread", None) is not None:
                app._gpu_owner_thread.shutdown()


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

    def _assert_spawn_is_clear(
        self,
        gen: TerrainGenerator,
        *,
        spawn_x: float,
        spawn_y: float,
        entity_width: float,
        entity_height: float,
    ) -> None:
        half_width = max(1, int(math.ceil(entity_width / 2.0)))
        height = max(1, int(math.ceil(entity_height)))
        floor_y = int(math.floor(spawn_y + entity_height + 1))
        for sample_x in range(int(spawn_x) - half_width, int(spawn_x) + half_width + 1):
            floor = gen.cell_at(sample_x, floor_y)
            self.assertTrue(floor is not None and not floor.is_empty, f"expected solid floor at {(sample_x, floor_y)}")
            for sample_y in range(int(math.floor(spawn_y)), int(math.ceil(spawn_y + height))):
                cell = gen.cell_at(sample_x, sample_y)
                self.assertTrue(cell is None or cell.is_empty, f"expected empty spawn cell at {(sample_x, sample_y)}")

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

    def test_surface_enemy_a_spawn_points_are_clear_of_terrain_and_foliage(self) -> None:
        gen = TerrainGenerator(42, self.registry)
        for point in gen.spawn_points():
            if point["type"] != "A" or point["biome"] == "underground":
                continue
            self._assert_spawn_is_clear(
                gen,
                spawn_x=float(point["x"]),
                spawn_y=float(point["y"]),
                entity_width=cfg.ENEMY_A_WIDTH,
                entity_height=cfg.ENEMY_A_HEIGHT,
            )

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
