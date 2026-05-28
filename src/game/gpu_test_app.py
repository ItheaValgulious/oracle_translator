"""GPU backend test with hero + full material scene.

Launch: python src/game/gpu_test_app.py

Controls:
  WASD        Move hero
  Space       Chant → Cast spell chain
  1-8         Cast spell from catalog
  C           Open/close spell menu
  T           Cycle view mode (material/temperature/pressure)
  V           Cycle view mode
  Mouse       Paint material (left=draw, right=erase)
  Bracket     Brush size [  ]
  P           Pause/resume
  ESC         Quit
"""

from __future__ import annotations

from array import array
from time import perf_counter

import moderngl
import pyglet
from pyglet.window import key, mouse

from src.engine.gpu_backend import ComputeBackendUnavailable
from src.engine.materials import build_material_registry
from src.engine.render import DebugViewMode
from src.engine.sim import inject_cells
from src.engine.types import CellFlag, CellState
from src.engine.world import DEFAULT_HALO_CELLS, DEFAULT_PAGE_SHIFT_CELLS, ActiveWorldWindow, WorldChunkStore
from src.game import config as cfg
from src.game.entity_manager import EntityManager, PLACEHOLDER_FAMILY
from src.game.hero import Hero, GridFeedback
from src.game.spell_system import SPELL_CATALOG, expand_model_socket, execute_magic_socket


# ── Paint tools ──
class ToolSpec:
    def __init__(self, label: str, family_id: str | None, variant_id: str | None,
                 overrides: dict | None = None) -> None:
        self.label = label
        self.family_id = family_id
        self.variant_id = variant_id
        self.overrides = overrides or {}


TOOLS = {
    key._1: ToolSpec("Stone Platform", "stone", "stone_platform"),
    key._2: ToolSpec("Stone Fixpoint", "stone", "stone_platform", {"flags": CellFlag.FIXPOINT}),
    key._3: ToolSpec("Stone Falling", "stone", "stone_falling"),
    key._4: ToolSpec("Sand", "sand", "sand_powder"),
    key._5: ToolSpec("Water", "water", "water"),
    key._6: ToolSpec("Fire", "fire", "fire"),
    key._7: ToolSpec("Oil", "oil", "oil_liquid"),
    key._8: ToolSpec("Magic Acid", "magic_acid", "magic_acid_liquid"),
    key._9: ToolSpec("Tar", "tar", "tar_liquid"),
    key._0: ToolSpec("Erase", None, None),
}


VIEW_MODE_ORDER = (DebugViewMode.MATERIAL, DebugViewMode.TEMPERATURE, DebugViewMode.PRESSURE)

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D frame_tex;
uniform vec2 view_uv_origin;
uniform vec2 view_uv_scale;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(frame_tex, view_uv_origin + v_uv * view_uv_scale);
}
"""


def populate_test_scene(store: WorldChunkStore, registry) -> None:
    """Create a rich scene with all material types, terrain, and hazards."""
    w = store.width
    h = store.height
    ground_y = h - 5  # top of stone floor
    surface_y = ground_y - 1  # grass surface

    # ── Floor: thick stone layer ──
    for y in range(ground_y, h):
        for x in range(w):
            store.set_cell(x, y, CellState(family_id="stone", variant_id="stone_platform",
                                           flags=CellFlag.FIXPOINT if y == h - 1 else CellFlag.NONE))

    # ── Ground surface: grass on top of stone ──
    for x in range(0, w):
        store.set_cell(x, surface_y, CellState(family_id="grass", variant_id="grass_platform",
                                              flags=CellFlag.FIXPOINT, integrity=1.0))

    # ── Walls: stone/obsidian pillars ──
    wall_x_left = 5
    wall_x_right = w - 6
    for y in range(surface_y - 30, ground_y):
        store.set_cell(wall_x_left, y, CellState(family_id="stone", variant_id="stone_platform",
                                                   flags=CellFlag.FIXPOINT))
        store.set_cell(wall_x_right, y, CellState(family_id="obsidian", variant_id="obsidian_platform",
                                                    flags=CellFlag.FIXPOINT))

    # ── Bridge: wood planks spanning between walls ──
    bridge_y = surface_y - 12
    for x in range(wall_x_left + 1, wall_x_right):
        store.set_cell(x, bridge_y, CellState(family_id="wood_plank", variant_id="wood_plank",
                                               flags=CellFlag.FIXPOINT, integrity=1.0))

    # ── Water pool (dug into stone floor, 2 rows deep) ──
    pool_left = 20
    pool_right = 40
    for x in range(pool_left, pool_right):
        store.set_cell(x, ground_y, CellState(family_id="water", variant_id="water"))
        store.set_cell(x, ground_y + 1, CellState(family_id="water", variant_id="water"))

    # ── Oil layer floating on water ──
    for x in range(pool_left + 3, pool_right - 3):
        store.set_cell(x, ground_y, CellState(family_id="oil", variant_id="oil_liquid"))

    # ── Tar pit (dug into stone floor) ──
    tar_left = 50
    tar_right = 60
    for x in range(tar_left, tar_right):
        store.set_cell(x, ground_y, CellState(family_id="tar", variant_id="tar_liquid"))
        store.set_cell(x, ground_y + 1, CellState(family_id="tar", variant_id="tar_liquid"))

    # ── Acid puddle ──
    for x in range(65, 72):
        store.set_cell(x, ground_y, CellState(family_id="acid", variant_id="acid_liquid"))

    # ── Magic acid puddle (persistent corrosive) ──
    for x in range(75, 80):
        store.set_cell(x, ground_y, CellState(family_id="magic_acid", variant_id="magic_acid_liquid"))

    # ── Poison patch ──
    for x in range(100, 108):
        store.set_cell(x, ground_y, CellState(family_id="poison", variant_id="poison_liquid"))

    # ── Ice shelf ──
    ice_left = 110
    ice_right = 125
    for x in range(ice_left, ice_right):
        store.set_cell(x, surface_y, CellState(family_id="water", variant_id="ice",
                                             flags=CellFlag.FIXPOINT, integrity=1.0))

    # ── Snow on ice ──
    for x in range(ice_left + 2, ice_right - 2):
        store.set_cell(x, surface_y - 1, CellState(family_id="snow", variant_id="snow_powder",
                                                  temperature=-5.0))

    # ── Sand pile ──
    sand_center = 135
    for dy in range(-4, 0):
        radius = 4 - dy
        for dx in range(-radius, radius + 1):
            sx = sand_center + dx
            sy = surface_y + dy
            if store.in_bounds(sx, sy):
                store.set_cell(sx, sy, CellState(family_id="sand", variant_id="sand_powder"))

    # ── Wood tree (grow-capable) ──
    tree_x = 45
    trunk_height = 8
    for y in range(surface_y - trunk_height, surface_y):
        store.set_cell(tree_x, y, CellState(family_id="wood", variant_id="wood_platform",
                                             flags=CellFlag.FIXPOINT, integrity=1.0, age=5.0))
    # Canopy
    canopy_y = surface_y - trunk_height - 2
    for dx in range(-2, 3):
        for dy in range(-2, 1):
            cx, cy = tree_x + dx, canopy_y + dy
            if store.in_bounds(cx, cy):
                store.set_cell(cx, cy, CellState(family_id="grass", variant_id="grass_platform",
                                                  flags=CellFlag.FIXPOINT, integrity=1.0))

    # ── Glass platform ──
    glass_y = surface_y - 20
    for x in range(80, 95):
        store.set_cell(x, glass_y, CellState(family_id="glass", variant_id="glass_platform",
                                              flags=CellFlag.FIXPOINT, integrity=1.0))

    # ── Iron pillar ──
    iron_x = 90
    for y in range(glass_y + 1, ground_y):
        store.set_cell(iron_x, y, CellState(family_id="iron", variant_id="iron_platform",
                                              flags=CellFlag.FIXPOINT))

    # ── Magma vent ──
    for x in range(145, 150):
        store.set_cell(x, ground_y, CellState(family_id="stone", variant_id="magma"))

    # ── Falling stone column (unsupported, will collapse) ──
    col_x = 155
    for y in range(surface_y - 15, surface_y):
        store.set_cell(col_x, y, CellState(family_id="stone", variant_id="stone_falling",
                                             integrity=1.0))


class GpuTestWindow(pyglet.window.Window):
    def __init__(self, *, cell_scale: int = 8) -> None:
        vp_w = cfg.VIEWPORT_WIDTH
        vp_h = cfg.VIEWPORT_HEIGHT
        window_w = vp_w * cell_scale
        window_h = vp_h * cell_scale
        super().__init__(width=window_w, height=window_h,
                         caption="GPU Test — Hero + Materials", resizable=True)
        self.cell_scale = cell_scale
        self.registry = build_material_registry()
        self.view_mode = DebugViewMode.MATERIAL
        self.paused = False
        self.brush_radius = 2
        self.current_tool_key = key._1
        self._keys_pressed: set[int] = set()

        # ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.BLEND)

        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        quad = self.ctx.buffer(data=array("f", [
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ]).tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(quad, "2f 2f", "in_pos", "in_uv")])

        # Build world with GPU backend
        store = WorldChunkStore(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                                chunk_size=cfg.CHUNK_SIZE, seed=42)
        populate_test_scene(store, self.registry)
        store.recompute_anchored_support(self.registry)

        try:
            self.world = ActiveWorldWindow(
                store, self.registry,
                viewport_width=vp_w, viewport_height=vp_h,
                halo_cells=DEFAULT_HALO_CELLS,
                page_shift_cells=DEFAULT_PAGE_SHIFT_CELLS,
                ctx=self.ctx,
            )
            self.backend_label = "GPU Compute"
        except (ComputeBackendUnavailable, Exception):
            self.world = ActiveWorldWindow(
                store, self.registry,
                viewport_width=vp_w, viewport_height=vp_h,
                halo_cells=DEFAULT_HALO_CELLS,
                page_shift_cells=DEFAULT_PAGE_SHIFT_CELLS,
                ctx=None,
            )
            self.backend_label = "CPU Fallback"

        # Hero + entity manager
        self.hero = Hero()
        ground_y = cfg.WORLD_HEIGHT - 5 - cfg.HERO_HEIGHT
        self.hero.reset(80.0, float(ground_y))
        self.entity_mgr = EntityManager(hero=self.hero)

        # Write initial placeholder
        self.entity_mgr.write_placeholder(self.world)

        # Texture
        self.texture: moderngl.Texture | None = None
        self._bind_texture()
        self._update_uv()

        # HUD label
        self.overlay = pyglet.text.Label(
            "", x=12, y=self.height - 12, anchor_x="left", anchor_y="top",
            multiline=True, width=self.width - 24, color=(240, 240, 240, 255),
            font_size=10,
        )

        # Spell menu
        self.show_console = False
        self.console_selected = 0

        pyglet.clock.schedule_interval(self._tick, 1.0 / 60.0)

    def _bind_texture(self) -> None:
        if self.world.gpu_simulator is not None:
            self.texture = self.world.gpu_simulator.frame_texture
        else:
            tex = self.ctx.texture((self.world.active_width, self.world.active_height), 4,
                                   self.world.render(self.view_mode))
            self.texture = tex
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.texture.use(location=0)
        self.program["frame_tex"].value = 0

    def _update_uv(self) -> None:
        o_x, o_y, s_x, s_y = self.world.visible_uv_rect()
        self.program["view_uv_origin"].value = (o_x, o_y)
        self.program["view_uv_scale"].value = (s_x, s_y)

    def _upload_frame(self) -> None:
        if self.world.gpu_simulator is not None:
            self.texture = self.world.render(self.view_mode)
        else:
            self.texture.write(self.world.render(self.view_mode))
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._update_uv()

    def _screen_to_world(self, sx: int, sy: int) -> tuple[int, int]:
        return self.world.screen_to_world(sx, sy,
                                           screen_width=max(1, self.width),
                                           screen_height=max(1, self.height))

    def _paint(self, world_x: int, world_y: int, erase: bool = False) -> None:
        tool = TOOLS.get(self.current_tool_key)
        if tool is None:
            return
        self.world.paint_world(
            world_x, world_y, self.brush_radius,
            None if erase or tool.family_id is None else tool.family_id,
            None if erase or tool.variant_id is None else tool.variant_id,
            overrides=dict(tool.overrides),
        )

    def _tick(self, dt: float) -> None:
        if self.paused:
            return

        # ── Hero input ──
        self.hero.input_left = key.A in self._keys_pressed or key.LEFT in self._keys_pressed
        self.hero.input_right = key.D in self._keys_pressed or key.RIGHT in self._keys_pressed
        self.hero.input_jump = key.W in self._keys_pressed or key.UP in self._keys_pressed
        self.hero.input_chant = key.SPACE in self._keys_pressed

        # ── Entity-Grid Hybrid cycle ──
        self.entity_mgr.tick(self.world, dt)
        self.world.step(dt)
        self.entity_mgr.post_step(self.world, dt)

        # ── Camera follows hero ──
        target_x = int(self.hero.x) - cfg.VIEWPORT_WIDTH // 2
        target_y = int(self.hero.y) - cfg.VIEWPORT_HEIGHT // 2
        dx = target_x - self.world.camera_x
        dy = target_y - self.world.camera_y
        if dx != 0 or dy != 0:
            self.world.pan_camera(dx, dy)

        self.world.service_background_io()
        self._upload_frame()
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        tool = TOOLS.get(self.current_tool_key, ToolSpec("?", None, None))
        hero_state = self.hero.state
        backend = self.backend_label
        gpu_note = f" | GPU pending={self.world.pending_writeback_count}" if self.world.gpu_simulator else ""
        status = "PAUSED" if self.paused else "Running"
        self.overlay.text = (
            f"{status} | {backend}{gpu_note} | View: {self.view_mode.value}\n"
            f"Hero: pos=({self.hero.x:.1f},{self.hero.y:.1f}) state={hero_state} "
            f"HP={self.hero.hp:.0f}/{cfg.HERO_MAX_HP:.0f} MP={self.hero.mp:.0f}/{cfg.HERO_MAX_MP:.0f}\n"
            f"Camera: ({self.world.camera_x},{self.world.camera_y}) "
            f"Paging: shifts={self.world.paging_stats.shift_count} "
            f"avg={self.world.shift_time_average_ms():.1f}ms\n"
            f"Tool: {tool.label} | Brush: {self.brush_radius}\n"
            f"WASD move | Space chant | 1-8 spell | C console | T view | P pause | ESC quit"
        )

    def on_draw(self) -> None:
        self.ctx.clear(0.04, 0.05, 0.07, 1.0)

        # Grid
        if self.texture is not None:
            self.texture.use(location=0)
            self.vao.render(moderngl.TRIANGLE_STRIP)

        # Hero overlay (rectangle at hero screen position)
        cam_x = self.world.camera_x
        cam_y = self.world.camera_y
        sx = (self.hero.x - cam_x) * self.cell_scale
        sy = (self.hero.y - cam_y) * self.cell_scale
        w = self.hero.width * self.cell_scale
        h = self.hero.height * self.cell_scale

        # Color by state
        state_colors = {
            "idle": (200, 60, 40), "walk": (220, 80, 50),
            "jump": (240, 100, 60), "chant": (230, 180, 80),
            "cast": (240, 200, 100),
        }
        color = state_colors.get(self.hero.state, (200, 60, 40))
        if not self.hero.is_alive:
            color = (80, 80, 80)

        # Flip x if facing left
        draw_x = sx if self.hero.facing_right else sx + w

        hero_rect = pyglet.shapes.Rectangle(draw_x, sy, w, h, color=color)
        hero_rect.opacity = 200
        hero_rect.draw()

        # HP/MP bars
        bar_x, bar_y = 10, 10
        bar_w = 150
        bar_h = 12

        # HP
        hp_ratio = max(0.0, min(1.0, self.hero.hp / cfg.HERO_MAX_HP))
        pyglet.shapes.Rectangle(bar_x, bar_y + bar_h + 2, bar_w, bar_h,
                                color=(60, 0, 0)).draw()
        pyglet.shapes.Rectangle(bar_x, bar_y + bar_h + 2, int(bar_w * hp_ratio), bar_h,
                                color=(220, 30, 30)).draw()

        # MP
        mp_ratio = max(0.0, min(1.0, self.hero.mp / cfg.HERO_MAX_MP))
        pyglet.shapes.Rectangle(bar_x, bar_y, bar_w, bar_h,
                                color=(0, 0, 60)).draw()
        pyglet.shapes.Rectangle(bar_x, bar_y, int(bar_w * mp_ratio), bar_h,
                                color=(30, 30, 220)).draw()

        # Labels
        pyglet.text.Label(
            f"HP:{self.hero.hp:.0f} MP:{self.hero.mp:.0f}",
            x=bar_x + 4, y=bar_y + 2, font_size=9, color=(255, 255, 255, 255),
        ).draw()

        # Spell menu overlay
        if self.show_console:
            overlay_w = self.width
            overlay_h = self.height
            pyglet.shapes.Rectangle(0, 0, overlay_w, overlay_h,
                                    color=(0, 0, 0)).draw()
            pyglet.text.Label("-- Spell Menu --", font_size=18,
                              x=overlay_w // 2, y=overlay_h - 40,
                              anchor_x="center", color=(240, 220, 180, 255)).draw()
            for i, spell in enumerate(SPELL_CATALOG):
                color = (255, 255, 0) if i == self.console_selected else (200, 200, 200)
                prefix = "> " if i == self.console_selected else "  "
                pyglet.text.Label(
                    f"{prefix}{i + 1}. {spell['name']} (MP:{spell['mp']})",
                    font_size=12, x=overlay_w // 2 - 100, y=overlay_h - 70 - i * 20,
                    color=color,
                ).draw()

        self.overlay.draw()

    def cast_spell_by_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(SPELL_CATALOG):
            return
        spell = SPELL_CATALOG[idx]
        if not self.hero.consume_mp(spell["mp"]):
            return
        magic = expand_model_socket(spell, self.hero.x, self.hero.y, self.hero.facing_right)
        execute_magic_socket(magic, self.world, self.registry)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.add(symbol)

        if symbol == key.ESCAPE:
            self.close()
            return

        if symbol == key.P:
            self.paused = not self.paused
            return

        if symbol == key.T:
            idx = VIEW_MODE_ORDER.index(self.view_mode)
            self.view_mode = VIEW_MODE_ORDER[(idx + 1) % len(VIEW_MODE_ORDER)]
            self._upload_frame()
            return

        if symbol == key.C:
            self.show_console = not self.show_console
            return

        if symbol == key.BRACKETLEFT:
            self.brush_radius = max(0, self.brush_radius - 1)
            return
        if symbol == key.BRACKETRIGHT:
            self.brush_radius = min(8, self.brush_radius + 1)
            return

        # Console navigation
        if self.show_console:
            if symbol == key.UP:
                self.console_selected = max(0, self.console_selected - 1)
                return
            if symbol == key.DOWN:
                self.console_selected = min(len(SPELL_CATALOG) - 1, self.console_selected + 1)
                return
            if symbol in (key.ENTER, key.SPACE):
                self.cast_spell_by_index(self.console_selected)
                self.show_console = False
                return
            return  # block other keys while console open

        # Spell hotkeys
        if key._1 <= symbol <= key._8:
            self.cast_spell_by_index(symbol - key._1)
            return

        # Tool selection (key 9, 0)
        if symbol in TOOLS:
            self.current_tool_key = symbol

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._keys_pressed.discard(symbol)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        world_x, world_y = self._screen_to_world(x, y)
        self._paint(world_x, world_y, erase=button == mouse.RIGHT)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        world_x, world_y = self._screen_to_world(x, y)
        erase = bool(buttons & mouse.RIGHT)
        draw = bool(buttons & mouse.LEFT)
        if draw or erase:
            self._paint(world_x, world_y, erase=erase)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.ctx.viewport = (0, 0, width, height)
        self.overlay.y = height - 12
        self.overlay.width = max(120, width - 24)


def run_gpu_test() -> None:
    window = GpuTestWindow(cell_scale=cfg.CELL_SCALE)
    window.set_minimum_size(400, 300)
    pyglet.app.run()


if __name__ == "__main__":
    run_gpu_test()