"""In-window profiler: runs game for ~5 seconds, prints per-stage timing, then exits."""
from __future__ import annotations
import sys, os, logging, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
log = logging.getLogger("profiler")

import moderngl
import pyglet
from src.engine.materials import build_material_registry
from src.engine.world import ActiveWorldWindow, WorldChunkStore
from src.game import config as cfg
from src.game.enemy import EnemyA, EnemyB, EnemyC
from src.game.entity_manager import Entity, EntityManager
from src.game.hero import Hero, GridFeedback
from src.game.projectile import ProjectileBase
from src.game.terrain import TerrainGenerator

class ProfileApp(pyglet.window.Window):
    def __init__(self):
        super().__init__(
            width=cfg.VIEWPORT_WIDTH * cfg.CELL_SCALE,
            height=cfg.VIEWPORT_HEIGHT * cfg.CELL_SCALE,
            caption="Profiler",
            resizable=False,
        )
        self.ctx = moderngl.create_context()
        self.registry = build_material_registry()
        self.terrain_gen = TerrainGenerator(42, self.registry)
        self.hero = Hero()
        self.entity_manager = EntityManager(hero=self.hero)
        self.enemies: dict = {}
        self.projectiles: list[ProjectileBase] = []

        # Timing stats
        self.stage_times: dict[str, collections.deque] = {}
        self.total_ticks = 0
        self.max_ticks = 300
        self.frame_times: collections.deque = collections.deque(maxlen=60)

        self._init_world()

    def _init_world(self):
        t0 = time.perf_counter()
        store = WorldChunkStore(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                                chunk_size=cfg.CHUNK_SIZE, seed=42,
                                chunk_generator=self.terrain_gen.generate_chunk)
        hero_x = float(cfg.VIEWPORT_WIDTH // 2)
        ground_y = self.terrain_gen.ground_height_at(int(hero_x))
        hero_y = ground_y - cfg.HERO_HEIGHT
        self.hero.reset(hero_x, hero_y)
        cam_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        cam_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        self.world = ActiveWorldWindow(
            store, self.registry,
            viewport_width=cfg.VIEWPORT_WIDTH,
            viewport_height=cfg.VIEWPORT_HEIGHT,
            ctx=self.ctx,
            initial_camera_x=cam_x,
            initial_camera_y=cam_y,
        )
        t1 = time.perf_counter()
        log.info(f"World init: {(t1-t0)*1000:.1f}ms")

        # Register hero entity
        self.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=self.hero.x,
            y=self.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))

        # Spawn enemies
        enemy_idx = 0
        for sp in self.terrain_gen.spawn_points():
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

        self.entity_manager.update_gpu_entity_mask(self.world)
        self.entity_manager.register_entity_shapes(self.world)
        self.entity_manager.schedule_feedback(self.world)

        # Check enemy visibility
        cam_x = self.world.camera_x
        cam_y = self.world.camera_y
        cam_right = cam_x + cfg.VIEWPORT_WIDTH
        cam_top = cam_y + cfg.VIEWPORT_HEIGHT
        visible = [e for e in self.enemies.values()
                   if e.is_alive and e.left >= cam_x and e.right <= cam_right
                   and e.bottom >= cam_y and e.top <= cam_top]
        log.info(f"Enemies total: {len(self.enemies)}, visible in viewport: {len(visible)}")

    def _record(self, stage: str, ms: float) -> None:
        if stage not in self.stage_times:
            self.stage_times[stage] = collections.deque(maxlen=300)
        self.stage_times[stage].append(ms)

    def _tick(self, dt: float) -> None:
        dt = min(dt, 1.0 / 15.0)
        t_frame_start = time.perf_counter()

        # poll_all_feedback BEFORE step (1-frame delayed, no GPU stall)
        t0 = time.perf_counter()
        all_feedback = self.entity_manager.poll_all_feedback(self.world)
        self._record("feedback_poll", (time.perf_counter() - t0) * 1000)

        # Apply feedback before scheduling the next query so corrected positions are sampled.
        t0 = time.perf_counter()
        self.entity_manager.read_feedback_and_update(self.world, dt, feedback=all_feedback)
        self._record("feedback_update", (time.perf_counter() - t0) * 1000)

        # entity_manager.tick + schedule_feedback BEFORE step
        t0 = time.perf_counter()
        self.entity_manager.tick(self.world, dt)
        self.entity_manager.schedule_feedback(self.world)
        self._record("entity_tick", (time.perf_counter() - t0) * 1000)

        # world.step (GPU sim, increments step_index)
        t0 = time.perf_counter()
        self.world.step(dt)
        self._record("gpu_step", (time.perf_counter() - t0) * 1000)

        # enemy AI
        t0 = time.perf_counter()
        for eid, enemy in list(self.enemies.items()):
            if not enemy.is_alive:
                continue
            enemy_fb = all_feedback.get(eid) if isinstance(all_feedback, dict) else None
            enemy.update(dt, grid_feedback=enemy_fb or GridFeedback(world_width=self.world.world_width),
                        hero_x=self.hero.x, hero_y=self.hero.y)
            entity = self.entity_manager._entities.get(eid)
            if entity is not None:
                entity.x = enemy.x
                entity.y = enemy.y
        self._record("enemy_ai", (time.perf_counter() - t0) * 1000)

        # camera follow
        t0 = time.perf_counter()
        target_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        target_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        self.world.pan_camera(target_x - self.world.camera_x, target_y - self.world.camera_y)
        self.world.mark_camera_activity(False, dt=dt)
        self.world.service_background_io()
        self._record("camera_io", (time.perf_counter() - t0) * 1000)

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        self.frame_times.append(frame_ms)
        self.total_ticks += 1

        if self.total_ticks >= self.max_ticks:
            self._print_stats()
            pyglet.app.exit()

    def on_draw(self) -> None:
        self.ctx.clear(0.04, 0.05, 0.07, 1.0)

    def _print_stats(self) -> None:
        log.info("=== TIMING STATS ===")
        alive = sum(1 for e in self.enemies.values() if e.is_alive)
        log.info(f"Alive enemies: {alive}/{len(self.enemies)}")
        log.info(f"Total ticks: {self.total_ticks}")
        if self.frame_times:
            avg_frame = sum(self.frame_times) / len(self.frame_times)
            log.info(f"Frame time: avg={avg_frame:.2f}ms max={max(self.frame_times):.2f}ms "
                     f"FPS={1000/avg_frame:.1f}")
        for stage, times in sorted(self.stage_times.items()):
            if times:
                avg = sum(times) / len(times)
                log.info(f"  {stage}: avg={avg:.2f}ms max={max(times):.2f}ms "
                         f"p50={sorted(times)[len(times)//2]:.2f}ms")

if __name__ == "__main__":
    app = ProfileApp()
    pyglet.clock.schedule_interval(app._tick, 1.0 / 60.0)
    pyglet.app.run()
