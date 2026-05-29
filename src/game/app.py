"""Game App: state machine (TITLE/GAME/OPTIONS/CONSOLE)."""

from __future__ import annotations

import logging
import time as _time
from time import perf_counter

import moderngl
import pyglet

log = logging.getLogger(__name__)
from pyglet.window import key

from src.engine.demo_app import DEFAULT_TICK_RATE_HZ
from src.engine.materials import build_material_registry
from src.engine.render import DebugViewMode
from src.engine.world import ActiveWorldWindow, WorldChunkStore
from src.game import config as cfg
from src.game.entity_manager import Entity, EntityManager, PLACEHOLDER_FAMILY
from src.game.hero import Hero
from src.game.renderer import GameRenderer
from src.game.screens import GameScreen, OptionsScreen, TitleScreen
from src.game.spell_system import (
    SPELL_CATALOG, ActiveStream, expand_model_socket,
    execute_magic_socket, inject_stream_tick,
)
from src.game.stt import SpeechToText
from src.game.terrain import TerrainGenerator


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
        self.hero = Hero()
        self.entity_manager = EntityManager(hero=self.hero)
        self.view_mode = DebugViewMode.MATERIAL
        self._camera_target_x = 0
        self._camera_target_y = 0
        self.active_streams: list[ActiveStream] = []

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
        self._last_dt = 1.0 / DEFAULT_TICK_RATE_HZ
        self._sim_accumulator = 0.0

        # Schedule tick
        pyglet.clock.schedule_interval(self._tick, 1.0 / DEFAULT_TICK_RATE_HZ)

        # Debug HTTP server
        from src.game.debug_server import start_debug_server
        self._debug_server = start_debug_server(self, port=9123)

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
        store = WorldChunkStore(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                                chunk_size=cfg.CHUNK_SIZE, seed=self.seed,
                                chunk_generator=terrain_gen.generate_chunk)
        hero_x = float(cfg.VIEWPORT_WIDTH // 2)
        ground_y = terrain_gen.ground_height_at(int(hero_x))
        hero_y = ground_y - cfg.HERO_HEIGHT
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
        )
        log.info("[app] world initialized in %.1fms, camera=(%d,%d) hero=(%.1f,%.1f)",
                 (perf_counter() - t0) * 1000,
                 self.world.camera_x, self.world.camera_y,
                 self.hero.x, self.hero.y)
        self.active_streams.clear()
        # Register hero entity for placeholder system
        self.entity_manager.register_entity(Entity(
            entity_id="hero",
            x=self.hero.x,
            y=self.hero.y,
            width=cfg.HERO_WIDTH,
            height=cfg.HERO_HEIGHT,
        ))
        if self.world is not None:
            self.entity_manager.write_placeholder(self.world)

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
        log.debug("[update] PRE-tick: hero y=%.2f vel_y=%.3f on_ground=%s state=%s",
                 self.hero.y, self.hero.vel_y, self.hero.on_ground, self.hero.state)
        t0 = _time.perf_counter()
        self.entity_manager.tick(self.world, dt)
        t1 = _time.perf_counter()
        self.world.step(dt)
        t2 = _time.perf_counter()
        # Sync GPU→CPU for hero surroundings so collision sees dynamic terrain changes
        margin = 8
        hx = int(self.hero.x)
        hy = int(self.hero.y + self.hero.height / 2)
        ax = self.world.active_origin_x
        ay = self.world.active_origin_y
        gx = max(0, hx - ax - margin)
        gy = max(0, hy - ay - margin)
        gw = min(self.world.active_width - gx, int(self.hero.width) + margin * 2 + 4)
        gh = min(self.world.active_height - gy, int(self.hero.height) + margin * 2 + 4)
        self.world.sync_cpu_region_from_gpu(gx, gy, gw, gh)
        t3 = _time.perf_counter()
        self.entity_manager.read_feedback_and_update(self.world, dt)
        t4 = _time.perf_counter()
        self.entity_manager.clear_placeholder(self.world)
        t5 = _time.perf_counter()
        log.info("[perf] placeholder=%.1f gpu_sim=%.1f readback=%.1f feedback=%.1f clear=%.1f total=%.1f ms",
                 (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000, (t5-t0)*1000)

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
        self.active_streams = remaining

    _MAX_DT = 1.0 / 20.0  # cap to 50ms to prevent huge physics jumps after long init
    _SIM_DT = 1.0 / 20.0  # simulation fixed step (20Hz)

    def _tick(self, dt: float) -> None:
        """Main tick loop. Simulation runs at fixed _SIM_DT rate."""
        dt = min(dt, self._MAX_DT)
        self._last_dt = dt
        if self.current_screen != "game":
            return
        if self.world is None:
            self._game_screen.update(dt)
            return
        # Always handle input every frame for responsiveness
        self._handle_hero_input(dt)
        # Run at most one sim step per frame to avoid spiral-of-death
        self._sim_accumulator += dt
        stepped = False
        if self._sim_accumulator >= self._SIM_DT:
            step_dt = min(self._sim_accumulator, self._MAX_DT)
            self._sim_accumulator = 0.0
            self._game_screen.update(step_dt)
            stepped = True
        # Camera follows hero every frame (smooth even between sim steps)
        if stepped and self.world is not None:
            target_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
            target_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
            self.world.pan_camera(target_x - self.world.camera_x, target_y - self.world.camera_y)

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

        # F3: toggle debug collision overlay
        if symbol == key.F3:
            self.entity_manager.debug_collision = not self.entity_manager.debug_collision
            if not self.entity_manager.debug_collision:
                self.entity_manager.last_debug = None
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


def run_game(*, seed: int = 42, cell_scale: int = cfg.CELL_SCALE) -> None:
    app = GameApp(seed=seed, cell_scale=cell_scale)
    app.set_minimum_size(400, 300)
    pyglet.app.run()