"""Game renderer: grid + entity overlay + UI."""

from __future__ import annotations

import logging
from array import array
from time import perf_counter

log = logging.getLogger(__name__)

import moderngl
import pyglet

from src.engine.render import DebugViewMode
from src.game import config as cfg
from src.game.animation import AnimationManager, SimpleSpriteBatch
from src.game.entity_manager import DebugCollisionInfo
from src.game.health_bar import HealthBar
from src.game.hero import Hero


class GameRenderer:
    """Renders the world grid, hero overlay, and UI."""

    def __init__(self, ctx: moderngl.Context, window_width: int, window_height: int) -> None:
        self.ctx = ctx
        self.window_width = window_width
        self.window_height = window_height
        self.health_bar = HealthBar(x=10, y=10)
        self.anim_mgr = AnimationManager()
        self.sprite_batch = SimpleSpriteBatch()

        # FPS tracking
        self._render_fps_history: list[float] = []
        self._last_render_at: float | None = None
        self._last_render_dt = 0.0
        self._render_fps = 0.0
        self._fps_label_updated_at = 0.0
        self._fps_label_update_interval = 0.25
        self._fps_label = pyglet.text.Label(
            "", font_size=10, x=10, y=window_height - 15,
            color=(255, 255, 0, 200),
        )

        # Shader for grid rendering (full-screen quad)
        vertex_shader = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            v_uv = in_uv;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        fragment_shader = """
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
        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        quad = self.ctx.buffer(
            data=array(
                "f",
                [-1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                 -1.0, 1.0, 0.0, 1.0,
                 1.0, 1.0, 1.0, 1.0],
            ).tobytes()
        )
        self.vao = self.ctx.vertex_array(self.program, [(quad, "2f 2f", "in_pos", "in_uv")])
        self._texture: moderngl.Texture | None = None
        self._hero_rect = pyglet.shapes.Rectangle(0, 0, 1, 1, color=(200, 60, 40))
        self._debug_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec4 in_color;
            out vec4 v_color;
            void main() {
                v_color = in_color;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = v_color;
            }
            """,
        )
        self._debug_vbo: moderngl.Buffer | None = None
        self._debug_vao: moderngl.VertexArray | None = None
        self._debug_capacity_bytes = 0

    def render_frame(self, world, view_mode: DebugViewMode) -> moderngl.Texture:
        """Render the world grid to a texture."""
        if world.gpu_simulator is not None:
            tex = world.render(view_mode)
        else:
            rgba = world.render(view_mode)
            if self._texture is not None and self._texture.size == (world.active_width, world.active_height):
                self._texture.write(rgba)
                tex = self._texture
            else:
                tex = self.ctx.texture((world.active_width, world.active_height), 4, rgba)
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                tex.repeat_x = False
                tex.repeat_y = False
        self._texture = tex
        return tex

    def draw_grid(self, world, view_mode: DebugViewMode) -> None:
        """Draw the world grid as a fullscreen quad."""
        tex = self.render_frame(world, view_mode)
        origin_x, origin_y, scale_x, scale_y = world.visible_uv_rect()
        self.program["view_uv_origin"].value = (origin_x, origin_y)
        self.program["view_uv_scale"].value = (scale_x, scale_y)
        tex.use(location=0)
        self.program["frame_tex"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def draw_hero(self, hero: Hero, camera_x: int, camera_y: int, dt: float) -> None:
        """Draw the hero rectangle matching the debug overlay coordinate system."""
        cs = cfg.CELL_SCALE
        # Same coords as debug overlay: world cell (wx, wy) → screen (sx, sy)
        # hero.top = y + height (head), highest world y = lowest screen y
        # Use float coords for smooth sub-pixel movement
        sx = (hero.left - camera_x) * cs
        sy = self.window_height - (hero.top - camera_y) * cs - cs

        w = hero.width * cs
        h = hero.height * cs
        color = (200, 60, 40)

        state = hero.state
        if state not in ("idle", "walk", "jump", "chant", "cast"):
            state = "idle"
        frame = self.anim_mgr.update(dt, state)
        if frame is not None:
            color = frame.color[:3]

        r = self._hero_rect
        r.x = sx
        r.y = sy
        r.width = w
        r.height = h
        r.color = color
        # Flip: for a solid-color rect this has no visual effect, but
        # store the flag so sprite rendering can use texture UV flipping.
        # Position stays the same — no sx shift.
        self._hero_flipped = not hero.facing_right
        r.draw()

    def _append_debug_rect(
        self,
        vertices: array,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        opacity: int,
    ) -> None:
        if x1 <= 0.0 or y1 <= 0.0 or x0 >= self.window_width or y0 >= self.window_height:
            return
        x0 = max(0.0, min(float(self.window_width), x0))
        y0 = max(0.0, min(float(self.window_height), y0))
        x1 = max(0.0, min(float(self.window_width), x1))
        y1 = max(0.0, min(float(self.window_height), y1))
        if x1 <= x0 or y1 <= y0:
            return

        nx0 = x0 / self.window_width * 2.0 - 1.0
        ny0 = y0 / self.window_height * 2.0 - 1.0
        nx1 = x1 / self.window_width * 2.0 - 1.0
        ny1 = y1 / self.window_height * 2.0 - 1.0
        r = color[0] / 255.0
        g = color[1] / 255.0
        b = color[2] / 255.0
        a = opacity / 255.0
        vertices.extend((
            nx0, ny0, r, g, b, a,
            nx1, ny0, r, g, b, a,
            nx1, ny1, r, g, b, a,
            nx0, ny0, r, g, b, a,
            nx1, ny1, r, g, b, a,
            nx0, ny1, r, g, b, a,
        ))

    def _append_debug_cell(
        self,
        vertices: array,
        camera_x: int,
        camera_y: int,
        wx: int,
        wy: int,
        color: tuple[int, int, int],
        opacity: int,
    ) -> None:
        cs = cfg.CELL_SCALE
        x0 = (wx - camera_x) * cs
        y0 = self.window_height - (wy - camera_y) * cs - cs
        self._append_debug_rect(vertices, x0, y0, x0 + cs, y0 + cs, color, opacity)

    def _append_debug_world_rect(
        self,
        vertices: array,
        camera_x: int,
        camera_y: int,
        wx0: int,
        wy0: int,
        wx1: int,
        wy1: int,
        color: tuple[int, int, int],
        opacity: int,
    ) -> None:
        cs = cfg.CELL_SCALE
        x0 = (wx0 - camera_x) * cs
        y0 = self.window_height - (wy1 - camera_y) * cs
        x1 = (wx1 - camera_x) * cs
        y1 = self.window_height - (wy0 - camera_y) * cs
        self._append_debug_rect(vertices, x0, y0, x1, y1, color, opacity)

    def _draw_debug_vertices(self, vertices: array) -> None:
        if not vertices:
            return
        data = vertices.tobytes()
        if self._debug_vbo is None or len(data) > self._debug_capacity_bytes:
            self._debug_capacity_bytes = max(4096, len(data) * 2)
            self._debug_vbo = self.ctx.buffer(reserve=self._debug_capacity_bytes)
            self._debug_vao = self.ctx.vertex_array(
                self._debug_program,
                [(self._debug_vbo, "2f 4f", "in_pos", "in_color")],
            )
        self._debug_vbo.write(data)
        self.ctx.enable(moderngl.BLEND)
        if self._debug_vao is not None:
            self._debug_vao.render(moderngl.TRIANGLES, vertices=len(vertices) // 6)

    def draw_debug_collision(self, camera_x: int, camera_y: int, debug: DebugCollisionInfo) -> None:
        """Draw debug overlay showing collision-checked cells."""
        vertices = array("f")
        # Ground cells: yellow
        for wx, wy in debug.ground_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (255, 255, 0), 120)
        # Ceiling cells: purple
        for wx, wy in debug.ceiling_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (180, 80, 255), 120)
        # Left wall cells: red
        for wx, wy in debug.left_wall_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (255, 0, 0), 120)
        # Right wall cells: blue
        for wx, wy in debug.right_wall_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (0, 100, 255), 120)
        # Hero AABB: green translucent fill.
        for wx0, wy0, wx1, wy1 in debug.hero_rects:
            self._append_debug_world_rect(
                vertices,
                camera_x,
                camera_y,
                wx0,
                wy0,
                wx1,
                wy1,
                (0, 255, 0),
                60,
            )
        # Compatibility for older DebugCollisionInfo producers.
        for wx, wy in debug.hero_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (0, 255, 0), 60)
        self._draw_debug_vertices(vertices)

    def draw_ui(self, hero: Hero) -> None:
        """Draw HP/MP bars and any other UI."""
        self.health_bar.draw(hero.hp, cfg.HERO_MAX_HP, hero.mp, cfg.HERO_MAX_MP)

    def _record_render_fps(self) -> None:
        now = perf_counter()
        if self._last_render_at is not None:
            dt = now - self._last_render_at
            if dt > 0:
                self._last_render_dt = dt
                self._render_fps_history.append(1.0 / dt)
                if len(self._render_fps_history) > 60:
                    self._render_fps_history.pop(0)
                self._render_fps = sum(self._render_fps_history) / len(self._render_fps_history)
        self._last_render_at = now

    def draw_fps(self, sim_fps: float) -> None:
        """Draw render/simulation FPS counters (call when F3 debug mode is active)."""
        now = perf_counter()
        if now - self._fps_label_updated_at >= self._fps_label_update_interval:
            if self._render_fps_history:
                text = (
                    f"Render FPS: {self._render_fps:.0f}  "
                    f"Sim FPS: {sim_fps:.0f}  "
                    f"render dt: {self._last_render_dt*1000:.1f}ms"
                )
            else:
                text = f"Render FPS: --  Sim FPS: {sim_fps:.0f}"
            if text != self._fps_label.text:
                self._fps_label.text = text
            self._fps_label_updated_at = now
        self._fps_label.draw()

    def draw(self, world, hero: Hero, camera_x: int, camera_y: int, view_mode: DebugViewMode, dt: float = 1.0 / 60.0) -> None:
        """Full render pass."""
        self._record_render_fps()
        self.draw_grid(world, view_mode)
        self.draw_hero(hero, camera_x, camera_y, dt)
        self.draw_ui(hero)
