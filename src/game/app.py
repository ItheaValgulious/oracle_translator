"""Game App: state machine (TITLE/GAME/OPTIONS/CONSOLE)."""

from __future__ import annotations

import logging
import time as _time
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import moderngl
import pyglet

log = logging.getLogger(__name__)
from pyglet.window import key

from src.engine.materials import build_material_registry
from src.engine.render import DebugViewMode
from src.engine.world import ActiveWorldWindow, WorldChunkStore
from src.game import config as cfg
from src.game.enemy import EnemyA, EnemyB, EnemyC
from src.game.entity_manager import Entity, EntityManager
from src.game.hero import Hero, GridFeedback, LocalCellSnapshot
from src.game.projectile import Arrow, Fireball, ProjectileBase
from src.game.renderer import GameRenderer
from src.game.screens import GameScreen, OptionsScreen, TitleScreen
from src.game.spell_system import (
    SPELL_CATALOG, ActiveStream, expand_model_socket,
    execute_magic_socket, inject_stream_tick,
)
from src.game.stt import SpeechToText
from src.game.terrain import TerrainGenerator


@dataclass
class PressureBurst:
    x: int
    y: int
    radius: int
    pressure: float
    ticks_remaining: int


class GameApp(pyglet.window.Window):
    """Main game application with screen state machine."""

    def __init__(self, seed: int = 42, cell_scale: int = cfg.CELL_SCALE) -> None:
        self._cell_scale = cell_scale
        window_width = cfg.VIEWPORT_WIDTH * cell_scale
        window_height = cfg.VIEWPORT_HEIGHT * cell_scale
        super().__init__(
            width=window_width,
            height=window_height,
            caption="Oracle Translator",
            resizable=False,
        )
        self.seed = seed
        self.registry = build_material_registry()
        self.world: ActiveWorldWindow | None = None
        self.terrain_gen: TerrainGenerator | None = None
        self.hero = Hero()
        self.entity_manager = EntityManager(hero=self.hero)
        self._latest_world_snapshots: dict[str, object] = {}
        self.view_mode = DebugViewMode.MATERIAL
        self._camera_target_x = 0
        self._camera_target_y = 0
        self.active_streams: list[ActiveStream] = []
        self.active_pressure_bursts: list[PressureBurst] = []
        self.enemies: dict[str, EnemyA | EnemyB | EnemyC] = {}
        self.projectiles: list[ProjectileBase] = []

        # STT (speech-to-text)
        self.stt = SpeechToText()
        self._chant_started = False

        # ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.BLEND)

        self.renderer = GameRenderer(self.ctx, window_width, window_height)

        # Screens
        self.screens: dict[str, pyglet.window.Window] = {}
        self.current_screen: str | None = None
        self._title_screen = TitleScreen(self)
        self._options_screen = OptionsScreen(self)
        self._game_screen = GameScreen(self)
        self.screens = {
            "title": self._title_screen,
            "options": self._options_screen,
            "game": self._game_screen,
        }
        self.change_screen("title")

        # Input
        self._keys_pressed: set[int] = set()
        self._last_dt = 1.0 / 60.0
        self._sim_accumulator = 0.0
        self._sim_fps = 0.0
        self._sim_fps_count = 0
        self._sim_fps_started_at = perf_counter()
        self._debug_overlay_enabled = False
        self._perf_samples: dict[str, deque[float]] = {}
        self._perf_last_ms: dict[str, float] = {}

        # Schedule tick
        pyglet.clock.schedule_interval(self._tick, 1.0 / 60.0)

        # Debug HTTP server
        from src.game.debug_server import DebugServer
        self._debug = DebugServer(self, port=9123)
        self._debug.start()

    def _refresh_local_entity_snapshots(self) -> None:
        if self.world is None:
            return
        for projectile in self.projectiles:
            entity_id = f"projectile_{id(projectile)}"
            lead_x = int(projectile.x + projectile.vel_x * 0.15)
            lead_y = int(projectile.y + projectile.vel_y * 0.15)
            min_x = min(int(projectile.x), lead_x) - 2
            max_x = max(int(projectile.x), lead_x) + 3
            min_y = min(int(projectile.y), lead_y) - 2
            max_y = max(int(projectile.y), lead_y) + 3
            snapshot = self.world.snapshot_cells_region_world(
                entity_id=entity_id,
                world_x=min_x,
                world_y=min_y,
                width=max(1, max_x - min_x),
                height=max(1, max_y - min_y),
            )
            self.entity_manager.set_latest_snapshot(
                entity_id,
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
                    age_frames=0,
                ),
            )

    def _prune_projectile_snapshots(self) -> None:
        active_ids = {f"projectile_{id(projectile)}" for projectile in self.projectiles if projectile.is_alive}
        stale_ids = [
            entity_id
            for entity_id in self.entity_manager._latest_snapshots
            if entity_id.startswith("projectile_") and entity_id not in active_ids
        ]
        for entity_id in stale_ids:
            self.entity_manager._latest_snapshots.pop(entity_id, None)
            self.entity_manager._last_feedback.pop(entity_id, None)

    def change_screen(self, name: str) -> None:
        self.current_screen = name

    def resize_window(self, new_cell_scale: int) -> None:
        """Resize the window for a new cell scale. Called from OptionsScreen."""
        self._cell_scale = new_cell_scale
        cfg.CELL_SCALE = new_cell_scale
        window_width = cfg.VIEWPORT_WIDTH * new_cell_scale
        window_height = cfg.VIEWPORT_HEIGHT * new_cell_scale
        self.set_size(window_width, window_height)
        self.renderer = GameRenderer(self.ctx, window_width, window_height)

    def _init_game_world(self) -> None:
        """Initialize a fresh game world."""
        t0 = perf_counter()
        log.info("[app] _init_game_world: seed=%d world=%dx%d viewport=%dx%d chunk_size=%d",
                 self.seed, cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                 cfg.VIEWPORT_WIDTH, cfg.VIEWPORT_HEIGHT, cfg.CHUNK_SIZE)
        terrain_gen = TerrainGenerator(self.seed, self.registry)
        self.terrain_gen = terrain_gen
        store = WorldChunkStore(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                                chunk_size=cfg.CHUNK_SIZE, seed=self.seed,
                                chunk_generator=terrain_gen.generate_chunk)
        hero_x = float(cfg.VIEWPORT_WIDTH // 2)
        spawn = terrain_gen.find_spawn_point_near(
            int(hero_x),
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=max(128, cfg.VIEWPORT_WIDTH // 2),
        )
        if spawn is None:
            ground_y = terrain_gen.ground_height_at(int(hero_x))
            hero_y = ground_y - 1 - cfg.HERO_HEIGHT
        else:
            hero_x, hero_y = spawn
            ground_y = hero_y + 1 + cfg.HERO_HEIGHT
        log.info("[app] hero spawn: x=%.1f y=%.1f ground_y=%.1f", hero_x, hero_y, ground_y)
        self.hero.reset(hero_x, hero_y)
        # Initialize camera at hero position to avoid loading empty world center
        cam_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        cam_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        self.world = ActiveWorldWindow(
            store,
            self.registry,
            viewport_width=cfg.VIEWPORT_WIDTH,
            viewport_height=cfg.VIEWPORT_HEIGHT,
            ctx=self.ctx,
            initial_camera_x=cam_x,
            initial_camera_y=cam_y,
            chunk_save_dir=cfg.chunk_storage_root(),
            chunk_cache_prefetch_x=cfg.CHUNK_CACHE_PREFETCH_X,
            chunk_cache_prefetch_y=cfg.CHUNK_CACHE_PREFETCH_Y,
        )
        log.info("[app] world initialized in %.1fms, camera=(%d,%d) hero=(%.1f,%.1f)",
                 (perf_counter() - t0) * 1000,
                 self.world.camera_x, self.world.camera_y,
                 self.hero.x, self.hero.y)
        self._sim_fps = 0.0
        self._sim_fps_count = 0
        self._sim_fps_started_at = perf_counter()
        self.active_streams.clear()
        self.active_pressure_bursts.clear()
        self.enemies.clear()
        self.projectiles.clear()
        # Register hero entity for GPU query / GPU occupancy mask / GPU placeholder.
        self.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=self.hero.x,
            y=self.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))
        # Spawn enemies from terrain seed distribution
        enemy_idx = 0
        for sp in terrain_gen.spawn_points():
            eid = f"enemy_{sp['type']}_{enemy_idx}"
            enemy_idx += 1
            if sp["type"] == "A":
                enemy = EnemyA.create(eid, sp["x"], sp["y"], sp["biome"])
            elif sp["type"] == "B":
                enemy = EnemyB.create(eid, sp["x"], sp["y"], sp["biome"])
            elif sp["type"] == "C":
                enemy = EnemyC.create(eid, sp["x"], sp["y"], sp["biome"])
            else:
                continue
            self.enemies[eid] = enemy
            self.entity_manager.register_enemy(enemy)
        if self.world is not None:
            self.entity_manager.update_gpu_entity_mask(self.world)
            self.entity_manager.register_entity_shapes(self.world)
            self.entity_manager.schedule_feedback(self.world)
            self._refresh_local_entity_snapshots()

    def move_hero_to(self, x: float, y: float) -> None:
        self.hero.x = x
        self.hero.y = y
        self.hero.vel_x = 0.0
        self.hero.vel_y = 0.0
        self.hero.on_ground = False
        hero_entity = self.entity_manager._entities.get("hero")
        if hero_entity is not None:
            hero_entity.x = x
            hero_entity.y = y
        if self.world is not None:
            target_x = int(x) - self.world.viewport_width // 2
            target_y = int(y) - self.world.viewport_height // 2
            if (
                abs(target_x - self.world.camera_x) >= self.world.active_width
                or abs(target_y - self.world.camera_y) >= self.world.active_height
            ):
                self.world.prepare_camera_region_sync(target_x, target_y)
            self.world.set_camera(target_x, target_y)

    def teleport_to_surface_x(self, world_x: float) -> bool:
        if self.terrain_gen is None:
            return False
        spawn = self.terrain_gen.find_spawn_point_near(
            int(world_x),
            entity_width=cfg.HERO_WIDTH,
            entity_height=cfg.HERO_HEIGHT,
            search_radius=max(256, cfg.BIOME_WIDTH // 5),
        )
        if spawn is None:
            return False
        self.move_hero_to(*spawn)
        return True

    def cast_spell_by_index(self, idx: int) -> None:
        """Cast a spell from the catalog by index."""
        if not SPELL_CATALOG or idx < 0 or idx >= len(SPELL_CATALOG):
            return
        spell = SPELL_CATALOG[idx]
        if self.world is None:
            return
        if not self.hero.consume_mp(spell["mp"]):
            return
        self.hero.selected_spell_idx = idx
        magic = expand_model_socket(spell, self.hero.x, self.hero.y, self.hero.facing_right)
        result = execute_magic_socket(magic, self.world, self.registry)
        if result is not None and isinstance(result, ActiveStream):
            self.active_streams.append(result)

    def _handle_hero_input(self, dt: float) -> None:
        """Read keyboard input and update hero."""
        self.hero.input_left = key.A in self._keys_pressed or key.LEFT in self._keys_pressed
        self.hero.input_right = key.D in self._keys_pressed or key.RIGHT in self._keys_pressed
        self.hero.input_jump = key.W in self._keys_pressed or key.UP in self._keys_pressed
        self.hero.input_chant_held = key.SPACE in self._keys_pressed

    def _mock_slm_to_spell_index(self, text: str) -> int:
        """Mock SLM: return the currently selected spell index.

        In the real pipeline this would be:
            text -> SLM inference -> Model Socket -> expand -> execute
        For now, just return the selected spell from the catalog.
        """
        return self.hero.selected_spell_idx

    def update_game(self, dt: float) -> None:
        """Update game logic for one tick."""
        if self.world is None:
            return

        # Input is handled in _tick() before calling this method

        # Detect chant entry: start STT recording
        if self.hero.state == "chant" and not self._chant_started:
            self._chant_started = True
            if self.stt.available:
                self.stt.start()

        # Detect chant→cast transition: hero just entered cast state
        was_chanting = self.hero.state == "chant"
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[update] PRE-tick: hero y=%.2f vel_y=%.3f on_ground=%s state=%s",
                     self.hero.y, self.hero.vel_y, self.hero.on_ground, self.hero.state)
        t0 = _time.perf_counter()
        # Snapshot-driven collision is populated asynchronously; keep the old
        # feedback object only as a compatibility shim during migration.
        all_feedback = self.entity_manager.poll_all_feedback(self.world)
        t_poll = _time.perf_counter()
        # Apply feedback before scheduling the next query so corrected positions are sampled.
        self.entity_manager.read_feedback_and_update(self.world, dt, feedback=all_feedback)
        t_update = _time.perf_counter()
        # Upload entity states/mask BEFORE step so placeholder occupancy matches
        # the entities that participate in this simulation tick.
        self.entity_manager.tick(self.world, dt)
        self._service_pressure_bursts()
        t_tick = _time.perf_counter()
        self.world.step(dt)
        t_step_submit = _time.perf_counter()
        # Submit feedback against the post-step world. Scheduling this before the
        # next frame's poll avoids readback stalls caused by querying a pre-step snapshot.
        self.entity_manager.schedule_feedback(self.world)
        t_schedule = _time.perf_counter()
        self._refresh_local_entity_snapshots()
        t_snapshot = _time.perf_counter()
        # Reset GL state after GPU compute so pyglet text/shaders can bind cleanly
        self.ctx.clear()
        t_step = _time.perf_counter()
        self._set_perf_ms("update_game_poll", (t_poll - t0) * 1000.0)
        self._set_perf_ms("update_game_apply", (t_update - t_poll) * 1000.0)
        self._set_perf_ms("update_game_tick", (t_tick - t_update) * 1000.0)
        self._set_perf_ms("update_game_world_step_submit", (t_step_submit - t_tick) * 1000.0)
        self._set_perf_ms("update_game_schedule_feedback", (t_schedule - t_step_submit) * 1000.0)
        self._set_perf_ms("update_game_snapshot", (t_snapshot - t_schedule) * 1000.0)
        self._set_perf_ms("update_game_ctx_clear", (t_step - t_snapshot) * 1000.0)
        self._set_perf_ms("update_game_total", (t_step - t0) * 1000.0)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[perf] poll=%.1f update=%.1f tick=%.1f step+schedule=%.1f total=%.1f ms",
                     (t_poll-t0)*1000, (t_update-t_poll)*1000, (t_tick-t_update)*1000,
                     (t_step-t_tick)*1000,
                     (_time.perf_counter()-t0)*1000)

        # If hero transitioned from chant to cast, execute the spell
        if was_chanting and self.hero.state == "cast":
            # Stop STT and get transcribed text
            stt_text = ""
            if self._chant_started and self.stt.available:
                stt_text = self.stt.stop()
            self._chant_started = False

            # Mock SLM: transcribed text -> spell index
            spell_idx = self._mock_slm_to_spell_index(stt_text)
            self.cast_spell_by_index(spell_idx)

        # Inject active stream ticks
        remaining: list[ActiveStream] = []
        for stream in self.active_streams:
            if stream.remaining_ticks > 0:
                inject_stream_tick(
                    stream, self.world, self.registry,
                    self.hero.x, self.hero.y, self.hero.facing_right,
                )
                remaining.append(stream)
                # Spell-to-enemy damage
                self._damage_enemies_with_spell(stream)
        self.active_streams = remaining

        # ── Enemy physics: movement, gravity, collision ──
        self._process_enemy_ai(dt)

        # ── Projectile update + GPU collision ──
        self._process_projectiles(dt)

        # ── Clean up dead enemies ──
        self._cleanup_dead_enemies()

        self._record_sim_fps()

    def _queue_pressure_burst(self, world_x: int, world_y: int, radius: int, pressure: float) -> None:
        self.active_pressure_bursts.append(
            PressureBurst(
                x=int(world_x),
                y=int(world_y),
                radius=int(radius),
                pressure=float(pressure),
                ticks_remaining=max(1, int(cfg.BLAST_PRESSURE_BURST_TICKS)),
            )
        )

    def _service_pressure_bursts(self) -> None:
        if self.world is None or not self.active_pressure_bursts:
            return
        remaining: list[PressureBurst] = []
        for burst in self.active_pressure_bursts:
            age = max(0, cfg.BLAST_PRESSURE_BURST_TICKS - burst.ticks_remaining)
            scale = cfg.BLAST_PRESSURE_DECAY ** age
            pulse = burst.pressure * scale
            if pulse <= 1.0:
                continue
            core_radius = max(2, burst.radius - age)
            core_pulse = pulse * float(cfg.BLAST_PRESSURE_CORE_SCALE)
            if core_pulse > 1.0:
                self.world.inject_pressure_world(burst.x, burst.y, core_radius, core_pulse)

            shell_mid_radius = (
                float(cfg.BLAST_RING_DISTANCE)
                + float(burst.radius)
                + float(age) * float(cfg.BLAST_PRESSURE_SHELL_SPEED)
            )
            shell_half_width = max(1.0, float(cfg.BLAST_PRESSURE_SHELL_WIDTH) * 0.5)
            shell_inner = max(1, int(round(shell_mid_radius - shell_half_width)))
            shell_outer = max(shell_inner + 1, int(round(shell_mid_radius + shell_half_width)))
            self.world.inject_pressure_ring_world(
                burst.x,
                burst.y,
                shell_inner,
                shell_outer,
                pulse * float(cfg.BLAST_PRESSURE_SHELL_SCALE),
            )

            trail_outer = max(1, shell_inner)
            trail_inner = max(0, trail_outer - max(1, int(cfg.BLAST_PRESSURE_SHELL_WIDTH)))
            trail_pulse = pulse * float(cfg.BLAST_SECONDARY_PRESSURE_SCALE)
            if trail_pulse > 1.0:
                self.world.inject_pressure_ring_world(
                    burst.x,
                    burst.y,
                    trail_inner,
                    trail_outer,
                    trail_pulse,
                )
            burst.ticks_remaining -= 1
            if burst.ticks_remaining > 0:
                remaining.append(burst)
        self.active_pressure_bursts = remaining

    def _trigger_explosion(self, world_x: int, world_y: int, *, pressure: float | None = None, radius: int | None = None) -> None:
        if self.world is None:
            return
        center_x = int(world_x)
        center_y = int(world_y)
        pressure_value = float(cfg.BLAST_PRESSURE if pressure is None else pressure)
        pressure_radius = int(cfg.BLAST_PRESSURE_RADIUS if radius is None else radius)
        self.world.paint_world(center_x, center_y, int(cfg.BLAST_FIRE_RADIUS), "fire", "fire", overrides={"vel_y": -2.0})
        self.world.paint_world(center_x, center_y, int(cfg.BLAST_GAS_RADIUS), "water", "steam", overrides={"temperature": 220.0})
        self.world.inject_pressure_world(center_x, center_y, pressure_radius, pressure_value)
        shell_half_width = max(1.0, float(cfg.BLAST_PRESSURE_SHELL_WIDTH) * 0.5)
        shell_inner = max(1, int(round(float(pressure_radius) - shell_half_width)))
        shell_outer = max(shell_inner + 1, int(round(float(pressure_radius) + shell_half_width)))
        self.world.inject_pressure_ring_world(
            center_x,
            center_y,
            shell_inner,
            shell_outer,
            pressure_value * float(cfg.BLAST_PRESSURE_SHELL_SCALE),
        )
        self._queue_pressure_burst(center_x, center_y, pressure_radius, pressure_value)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            vx = dx * cfg.BLAST_GAS_VELOCITY
            vy = dy * cfg.BLAST_GAS_VELOCITY
            px = center_x + int(dx * cfg.BLAST_RING_DISTANCE)
            py = center_y + int(dy * cfg.BLAST_RING_DISTANCE)
            self.world.paint_world(px, py, int(cfg.BLAST_GAS_RADIUS), "water", "steam", overrides={"vel_x": vx, "vel_y": vy, "temperature": 180.0})
            self.world.paint_world(px, py, 1, "stone", "stone_powder", overrides={"vel_x": dx * cfg.BLAST_DEBRIS_VELOCITY, "vel_y": dy * cfg.BLAST_DEBRIS_VELOCITY})

    def _process_enemy_ai(self, dt: float) -> None:
        """Process enemy AI: physics, movement, arrows, boss attacks."""
        if self.world is None:
            return
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                continue
            # Physics/movement update with collision feedback
            enemy_fb = self.entity_manager._last_feedback.get(eid) if isinstance(self.entity_manager._last_feedback, dict) else None
            enemy.update(dt, grid_feedback=enemy_fb or GridFeedback(world_width=self.world.world_width),
                         hero_x=self.hero.x, hero_y=self.hero.y)
            # Sync entity position
            entity = self.entity_manager._entities.get(eid)
            if entity is not None:
                entity.x = enemy.x
                entity.y = enemy.y
            # Attack actions
            if isinstance(enemy, EnemyA) and enemy.should_fire_arrow():
                arrow = Arrow.create(enemy.x, enemy.y, self.hero.x, self.hero.y, enemy.facing_right)
                self.projectiles.append(arrow)
                enemy.fire_arrow()
            elif isinstance(enemy, EnemyC):
                if enemy.should_spray_oil:
                    dx = cfg.BOSS_OIL_SPRAY_WIDTH if enemy.facing_right else -cfg.BOSS_OIL_SPRAY_WIDTH
                    spray_x = int(enemy.x + dx)
                    spray_y = int(enemy.y + enemy.height * 0.5)
                    if 0 <= spray_x < cfg.WORLD_WIDTH and 0 <= spray_y < cfg.WORLD_HEIGHT:
                        self.world.paint_world(spray_x, spray_y, int(cfg.BOSS_OIL_SPRAY_WIDTH // 2), "tar", "tar_liquid")
                elif enemy.should_fire_fireball:
                    fb = Fireball.create(enemy.x, enemy.y, self.hero.x, self.hero.y)
                    self.projectiles.append(fb)
                elif enemy.should_collapse:
                    cx = int(enemy.x)
                    cy = int(enemy.y) - int(cfg.BOSS_COLLAPSE_HEIGHT)
                    if 0 <= cx < cfg.WORLD_WIDTH and 0 <= cy < cfg.WORLD_HEIGHT:
                        self.world.paint_world(cx, cy, int(cfg.BOSS_COLLAPSE_WIDTH // 2), None, None)

    def _projectile_hero_overlap(self, projectile: ProjectileBase) -> bool:
        hero_left = self.hero.x - cfg.HERO_WIDTH / 2.0
        hero_right = self.hero.x + cfg.HERO_WIDTH / 2.0
        hero_bottom = self.hero.y
        hero_top = self.hero.y + cfg.HERO_HEIGHT
        return (
            projectile.x > hero_left
            and projectile.x < hero_right
            and projectile.y > hero_bottom
            and projectile.y < hero_top
        )

    def _projectile_snapshot_terrain_hit(
        self,
        projectile: ProjectileBase,
        previous_x: float,
        previous_y: float,
    ) -> tuple[int, int] | None:
        snapshot = self.entity_manager.latest_snapshot_for(f"projectile_{id(projectile)}")
        if snapshot is None:
            return None
        dx = projectile.x - previous_x
        dy = projectile.y - previous_y
        steps = max(1, int(max(abs(dx), abs(dy)) * 2.0))
        for step_index in range(steps + 1):
            t = step_index / steps
            sample_x = previous_x + dx * t
            sample_y = previous_y + dy * t
            world_x = int(round(sample_x))
            world_y = int(round(sample_y))
            if not (0 <= world_x < cfg.WORLD_WIDTH and 0 <= world_y < cfg.WORLD_HEIGHT):
                return (world_x, world_y)
            variant_index = snapshot.variant_index_at_world(world_x, world_y)
            if variant_index is not None and variant_index != snapshot.empty_variant_index:
                return (world_x, world_y)
        return None

    def _process_projectiles(self, dt: float) -> None:
        """Move projectiles, check GPU collision, handle hits."""
        if self.world is None:
            return
        previous_positions: list[tuple[float, float]] = []
        for p in self.projectiles:
            previous_positions.append((p.x, p.y))
            p.update(dt)

        for i, p in enumerate(self.projectiles):
            if not p.is_alive:
                continue
            previous_x, previous_y = previous_positions[i]
            terrain_hit = self._projectile_snapshot_terrain_hit(p, previous_x, previous_y)
            if terrain_hit is not None:
                hit_x, hit_y = terrain_hit
                if isinstance(p, Arrow):
                    self.world.paint_world(
                        int(hit_x), int(hit_y), 2, "stone", "stone_powder",
                        overrides={"vel_x": p.vel_x * 0.3, "vel_y": -2.0},
                    )
                elif isinstance(p, Fireball):
                    self._trigger_explosion(int(hit_x), int(hit_y))
                p.is_alive = False
                continue

            if self._projectile_hero_overlap(p):
                if isinstance(p, Arrow):
                    self.world.paint_world(
                        int(p.x),
                        int(p.y),
                        1,
                        "iron",
                        "iron_grit",
                        overrides={"vel_x": p.vel_x, "vel_y": p.vel_y},
                    )
                elif isinstance(p, Fireball):
                    self._trigger_explosion(int(p.x), int(p.y))
                p.is_alive = False

        self.projectiles = [p for p in self.projectiles if p.is_alive]
        self._prune_projectile_snapshots()

    def _cleanup_dead_enemies(self) -> None:
        """Remove dead enemies, handle EnemyB explosion."""
        if self.world is None:
            return
        dead_ids: list[str] = []
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                dead_ids.append(eid)
                if isinstance(enemy, EnemyB) and enemy.should_explode:
                    # EnemyB explodes on death
                    self._trigger_explosion(int(enemy.x), int(enemy.y))
        for eid in dead_ids:
            self.enemies.pop(eid, None)
            self.entity_manager.unregister_enemy(eid)

    def _damage_enemies_with_spell(self, stream: ActiveStream) -> None:
        """Damage enemies within the spell's brush radius."""
        magic = stream.magic
        # Spell origin: same logic as inject_stream_tick
        if magic.force_strength > 0.0 or magic.carrier_velocity > 0.0:
            sx = self.hero.x + (1.0 if self.hero.facing_right else -1.0) * cfg.HERO_WIDTH
            sy = self.hero.y + cfg.HERO_HEIGHT * 0.5
        else:
            sx, sy = magic.origin
        radius = float(magic.brush_radius)
        damage = cfg.SPELL_BASE_DAMAGE * (0.5 + magic.powerness * 0.5)
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                continue
            dist = ((enemy.x - sx) ** 2 + (enemy.y - sy) ** 2) ** 0.5
            if dist <= radius + enemy.width * 0.5:
                enemy.take_damage(damage)

    def _record_sim_fps(self) -> None:
        """Track actual completed simulation ticks for the F3 overlay."""
        now = perf_counter()
        self._sim_fps_count += 1
        elapsed = now - self._sim_fps_started_at
        if elapsed >= 1.0:
            self._sim_fps = self._sim_fps_count / elapsed
            self._sim_fps_count = 0
            self._sim_fps_started_at = now

    @property
    def sim_fps(self) -> float:
        elapsed = perf_counter() - self._sim_fps_started_at
        if self._sim_fps_count > 0 and elapsed > 0.0:
            return self._sim_fps_count / elapsed
        return self._sim_fps

    def _set_perf_ms(self, key: str, value_ms: float, *, history: int = 120) -> None:
        if not hasattr(self, "_perf_last_ms"):
            self._perf_last_ms = {}
        if not hasattr(self, "_perf_samples"):
            self._perf_samples = {}
        self._perf_last_ms[key] = float(value_ms)
        samples = self._perf_samples.get(key)
        if samples is None:
            samples = deque(maxlen=history)
            self._perf_samples[key] = samples
        samples.append(float(value_ms))

    def perf_last_ms(self, key: str) -> float:
        if not hasattr(self, "_perf_last_ms"):
            return 0.0
        return float(self._perf_last_ms.get(key, 0.0))

    def perf_avg_ms(self, key: str) -> float:
        if not hasattr(self, "_perf_samples"):
            return 0.0
        samples = self._perf_samples.get(key)
        if not samples:
            return 0.0
        return float(sum(samples) / len(samples))

    def debug_perf_snapshot(self) -> dict[str, float]:
        keys = (
            "main_tick_total",
            "main_tick_input",
            "main_tick_sim",
            "main_tick_camera",
            "main_tick_background_io",
            "update_game_total",
            "update_game_poll",
            "update_game_apply",
            "update_game_tick",
            "update_game_world_step_submit",
            "update_game_schedule_feedback",
            "update_game_snapshot",
            "update_game_ctx_clear",
        )
        snapshot: dict[str, float] = {}
        for key in keys:
            snapshot[f"{key}_last_ms"] = round(self.perf_last_ms(key), 3)
            snapshot[f"{key}_avg_ms"] = round(self.perf_avg_ms(key), 3)
        return snapshot

    @property
    def debug_overlay_enabled(self) -> bool:
        return bool(getattr(self, "_debug_overlay_enabled", False))

    _MAX_DT = 1.0 / 15.0  # cap large pauses without forcing a long catch-up loop
    _SIM_DT = 1.0 / 50.0  # GPU simulation fixed step (50Hz)
    _MAX_STEPS_PER_FRAME = 2

    def _tick(self, dt: float) -> None:
        """Main tick loop. Simulation runs at 50Hz; rendering remains 60Hz/vsync."""
        tick_started_at = perf_counter()
        dt = min(dt, self._MAX_DT)
        self._last_dt = dt
        if self.current_screen != "game":
            return
        if self.world is None:
            self._game_screen.update(dt)
            return
        # Always handle input every frame for responsiveness
        input_started_at = perf_counter()
        self._handle_hero_input(dt)
        input_finished_at = perf_counter()
        self._sim_accumulator += dt
        steps = 0
        sim_started_at = perf_counter()
        while self._sim_accumulator >= self._SIM_DT and steps < self._MAX_STEPS_PER_FRAME:
            self._sim_accumulator -= self._SIM_DT
            self._game_screen.update(self._SIM_DT)
            steps += 1
        if steps >= self._MAX_STEPS_PER_FRAME and self._sim_accumulator >= self._SIM_DT:
            self._sim_accumulator = 0.0
        sim_finished_at = perf_counter()
        # Camera follows hero every render tick, independent of sim cadence.
        target_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        target_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        camera_started_at = perf_counter()
        self.world.pan_camera(target_x - self.world.camera_x, target_y - self.world.camera_y)
        self.world.mark_camera_activity(False, dt=dt)
        camera_finished_at = perf_counter()
        bg_io_started_at = perf_counter()
        self.world.service_background_io()
        tick_finished_at = perf_counter()
        self._set_perf_ms("main_tick_input", (input_finished_at - input_started_at) * 1000.0)
        self._set_perf_ms("main_tick_sim", (sim_finished_at - sim_started_at) * 1000.0)
        self._set_perf_ms("main_tick_camera", (camera_finished_at - camera_started_at) * 1000.0)
        self._set_perf_ms("main_tick_background_io", (tick_finished_at - bg_io_started_at) * 1000.0)
        self._set_perf_ms("main_tick_total", (tick_finished_at - tick_started_at) * 1000.0)

    def on_draw(self) -> None:
        screen = self.screens.get(self.current_screen)
        if screen is not None:
            screen.on_draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.add(symbol)
        log.debug("[app] key_press: symbol=%d screen=%s", symbol, self.current_screen)

        # Global: ENTER on title -> init world + switch to game
        if self.current_screen == "title" and symbol == key.ENTER:
            log.info("[app] ENTER pressed on title, initializing game world...")
            try:
                self._init_game_world()
                self.change_screen("game")
                log.info("[app] switched to game screen")
            except Exception:
                import traceback
                log.error("[app] _init_game_world failed:\n%s", traceback.format_exc())
            return

        # F3: toggle perf/debug overlay
        if symbol == key.F3:
            self._debug_overlay_enabled = not self._debug_overlay_enabled
            return
        if symbol == key.F6:
            self.entity_manager.debug_collision = not self.entity_manager.debug_collision
            if not self.entity_manager.debug_collision:
                self.entity_manager.last_debug = None
            return
        if symbol == key.F4:
            self.view_mode = (
                DebugViewMode.MATERIAL
                if self.view_mode == DebugViewMode.TEMPERATURE
                else DebugViewMode.TEMPERATURE
            )
            return
        if symbol == key.F5:
            self.view_mode = (
                DebugViewMode.MATERIAL
                if self.view_mode == DebugViewMode.PRESSURE
                else DebugViewMode.PRESSURE
            )
            return

        # Spell hotkeys 1-8 set selected spell (don't cast immediately)
        if self.current_screen == "game" and key._1 <= symbol <= key._8:
            self.hero.selected_spell_idx = symbol - key._1

        screen = self.screens.get(self.current_screen)
        if screen is not None:
            screen.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.discard(symbol)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        screen = self.screens.get(self.current_screen)
        if screen is not None:
            screen.on_mouse_press(x, y, button, modifiers)

    def on_close(self) -> None:
        self._debug.stop()
        if self.world is not None:
            self.world.close()
        super().on_close()


def run_game(*, seed: int = 42, cell_scale: int = cfg.CELL_SCALE) -> None:
    app = GameApp(seed=seed, cell_scale=cell_scale)
    app.set_minimum_size(400, 300)
    pyglet.app.run()
