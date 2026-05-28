"""Tests for Phase 1 game shell: hero, entity_manager, spell_system, animation, screens."""

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
from engine.sim import inject_cells, step
from engine.types import CellFlag, CellState
from engine.world import ActiveWorldWindow, WorldChunkStore

from game import config as cfg
from game.animation import AnimationManager, FrameData, HERO_ANIMATIONS
from game.entity_manager import Entity, EntityManager, PLACEHOLDER_FAMILY
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

    def test_hero_placeholder_cells(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        cells = hero.get_placeholder_cells()
        self.assertTrue(len(cells) > 0)
        for x, y in cells:
            self.assertGreaterEqual(x, int(hero.left))
            self.assertLessEqual(x, int(hero.right))
            self.assertGreaterEqual(y, int(hero.bottom))
            self.assertLessEqual(y, int(hero.top))


class EntityManagerTests(unittest.TestCase):
    """Tests for Entity-Grid Hybrid tick cycle.

    All entity_manager operations target active_grid (local coords).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()

    def _make_world(self, w: int = 40, h: int = 40) -> ActiveWorldWindow:
        store = WorldChunkStore(w, h, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=w, viewport_height=h, ctx=None)
        return world

    def test_write_placeholder_creates_cells(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()
        mgr.write_placeholder(world)

        for lx, ly in mgr._entity_local_cells(entity, world):
            cell = world.active_grid.get_cell(lx, ly)
            self.assertEqual(cell.family_id, PLACEHOLDER_FAMILY)

    def test_clear_placeholder_restores_originals(self) -> None:
        """clear_placeholder should restore saved original cells, not just empty."""
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        # Pre-fill active_grid with stone at hero's local positions
        local_cells = mgr._entity_local_cells(entity, world)
        for lx, ly in local_cells:
            world.active_grid.set_cell(lx, ly, CellState(
                family_id="stone", variant_id="stone_platform", integrity=1.0))

        mgr.write_placeholder(world)
        for lx, ly in local_cells:
            cell = world.active_grid.get_cell(lx, ly)
            self.assertEqual(cell.family_id, PLACEHOLDER_FAMILY)

        mgr.clear_placeholder(world)
        for lx, ly in local_cells:
            cell = world.active_grid.get_cell(lx, ly)
            self.assertEqual(cell.family_id, "stone")

    def test_tick_order_is_correct(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        mgr.write_placeholder(world)
        mgr.tick(world, 0.1)

        local_cells = mgr._entity_local_cells(entity, world)
        for lx, ly in local_cells:
            cell = world.active_grid.get_cell(lx, ly)
            self.assertEqual(cell.family_id, PLACEHOLDER_FAMILY)

        world.step(0.1)
        mgr.post_step(world, 0.1)

        # post_step no longer clears placeholders — they persist until the
        # next tick() clears them. This is intentional: clearing here would
        # restore destroyed terrain before the hero reads feedback, preventing
        # the hero from falling through burned ground.
        for lx, ly in local_cells:
            cell = world.active_grid.get_cell(lx, ly)
            self.assertEqual(cell.family_id, PLACEHOLDER_FAMILY)

    def test_grid_feedback_detects_ground(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        # Place stone below hero in active_grid (local coords)
        for wx in range(int(hero.left), int(hero.right) + 1):
            wy = int(hero.bottom) - 1
            if 0 <= wx < world.active_width and 0 <= wy < world.active_height:
                lx, ly = _world_to_local(world, wx, wy)
                world.active_grid.set_cell(lx, ly, CellState(
                    family_id="stone", variant_id="stone_platform", integrity=1.0))

        mgr.write_placeholder(world)
        feedback = mgr.read_entity_feedback(entity, world)
        self.assertTrue(feedback.blocked_below)

    def test_grid_feedback_detects_horizontal_walls(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        # Place stone left of hero in active_grid
        for wy in range(int(hero.bottom), int(hero.top) + 1):
            wx = int(hero.left) - 1
            if 0 <= wx < world.active_width and 0 <= wy < world.active_height:
                lx, ly = _world_to_local(world, wx, wy)
                world.active_grid.set_cell(lx, ly, CellState(
                    family_id="stone", variant_id="stone_platform", integrity=1.0))

        mgr.write_placeholder(world)
        feedback = mgr.read_entity_feedback(entity, world)
        self.assertTrue(feedback.blocked_left)

    def test_grid_feedback_detects_liquid(self) -> None:
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        # Pre-fill hero area with water so write_placeholder saves them as originals
        local_cells = mgr._entity_local_cells(entity, world)
        for lx, ly in local_cells:
            world.active_grid.set_cell(lx, ly, CellState(
                family_id="water", variant_id="water"))

        mgr.write_placeholder(world)
        feedback = mgr.read_entity_feedback(entity, world)
        self.assertTrue(feedback.in_liquid)

    def test_grid_feedback_detects_damage_via_integrity(self) -> None:
        """Damage is read from placeholder integrity loss (reaction system pathway).

        Fire adjacent to placeholder → reaction system reduces placeholder
        integrity → entity_manager detects integrity loss → damage > 0.
        """
        hero = Hero(x=20.0, y=10.0)
        mgr = EntityManager(hero=hero)
        entity = Entity(entity_id="hero", x=hero.x, y=hero.y,
                        width=cfg.HERO_WIDTH, height=cfg.HERO_HEIGHT)
        mgr.register_entity(entity)
        world = self._make_world()

        mgr.write_placeholder(world)
        # Place fire adjacent to placeholder in active_grid (local coords)
        wx = int(hero.left) - 1
        wy = int(hero.bottom)
        lx, ly = _world_to_local(world, wx, wy)
        world.active_grid.set_cell(lx, ly, CellState(
            family_id="fire", variant_id="fire", integrity=1.0))

        # Run simulation so reactions.py reduces placeholder integrity
        world.step(0.1)

        feedback = mgr.read_entity_feedback(entity, world)
        self.assertGreater(feedback.damage, 0.0)


class SpellSystemTests(unittest.TestCase):
    """Tests for spell expansion and execution."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_material_registry()

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
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=40, viewport_height=40, ctx=None)
        # Use Water Wall (burst) for immediate injection
        socket = SPELL_CATALOG[1]  # Water Wall
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        result = execute_magic_socket(magic, world, self.registry)
        self.assertIsNone(result)

        water_cells = [(x, y) for y in range(world.active_grid.height)
                       for x in range(world.active_grid.width)
                       if world.active_grid.get_cell(x, y).family_id == "water"]
        self.assertGreater(len(water_cells), 0)

    def test_execute_stream_returns_active_stream(self) -> None:
        """Stream-type spells return an ActiveStream for multi-frame injection."""
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=40, viewport_height=40, ctx=None)
        socket = SPELL_CATALOG[0]  # Fireball (stream)
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        result = execute_magic_socket(magic, world, self.registry)
        self.assertIsInstance(result, ActiveStream)
        self.assertGreater(result.remaining_ticks, 0)

    def test_stream_inject_injects_cells(self) -> None:
        """inject_stream_tick should inject cells each tick."""
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=40, viewport_height=40, ctx=None)
        socket = SPELL_CATALOG[0]  # Fireball (stream)
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        stream = execute_magic_socket(magic, world, self.registry)
        self.assertIsInstance(stream, ActiveStream)

        inject_stream_tick(stream, world, self.registry,
                           hero_x=20.0, hero_y=10.0, facing_right=True)
        fire_cells = [(x, y) for y in range(world.active_grid.height)
                      for x in range(world.active_grid.width)
                      if world.active_grid.get_cell(x, y).family_id == "fire"]
        self.assertGreater(len(fire_cells), 0)
        self.assertGreater(stream.ticks_total, stream.remaining_ticks)

    def test_execute_attaches_reaction_overrides(self) -> None:
        """execute_magic_socket should attach spell_* reaction overrides to injected cells."""
        store = WorldChunkStore(40, 40, chunk_size=4)
        world = ActiveWorldWindow(
            store, self.registry, viewport_width=40, viewport_height=40, ctx=None)
        # Ice Shield: freeze reaction (burst)
        socket = SPELL_CATALOG[2]  # Ice Shield
        magic = expand_model_socket(socket, hero_x=20.0, hero_y=10.0, facing_right=True)
        result = execute_magic_socket(magic, world, self.registry)
        self.assertIsNone(result)

        ice_cells = []
        for y in range(world.active_grid.height):
            for x in range(world.active_grid.width):
                cell = world.active_grid.get_cell(x, y)
                if cell.family_id == "water" and cell.variant_id == "ice":
                    ice_cells.append((x, y, cell))

        self.assertGreater(len(ice_cells), 0)
        for x, y, cell in ice_cells:
            self.assertEqual(cell.spell_convert_mode, "self")
            self.assertIn("water", cell.spell_damage_mask)
            self.assertIn("terrain", cell.spell_damage_mask)

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