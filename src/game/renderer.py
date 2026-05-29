"""Game renderer: grid + entity overlay + UI."""

from __future__ import annotations

import logging
from array import array

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
        self._fps_history: list[float] = []
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

    def draw_debug_collision(self, camera_x: int, camera_y: int, debug: DebugCollisionInfo) -> None:
        """Draw debug overlay showing collision-checked cells."""
        cs = cfg.CELL_SCALE
        h = self.window_height
        # Ground cells: yellow
        for wx, wy in debug.ground_cells:
            sy = h - (wy - camera_y) * cs - cs
            r = pyglet.shapes.Rectangle(
                (wx - camera_x) * cs, sy, cs, cs,
                color=(255, 255, 0),
            )
            r.opacity = 120
            r.draw()
        # Left wall cells: red
        for wx, wy in debug.left_wall_cells:
            sy = h - (wy - camera_y) * cs - cs
            r = pyglet.shapes.Rectangle(
                (wx - camera_x) * cs, sy, cs, cs,
                color=(255, 0, 0),
            )
            r.opacity = 120
            r.draw()
        # Right wall cells: blue
        for wx, wy in debug.right_wall_cells:
            sy = h - (wy - camera_y) * cs - cs
            r = pyglet.shapes.Rectangle(
                (wx - camera_x) * cs, sy, cs, cs,
                color=(0, 100, 255),
            )
            r.opacity = 120
            r.draw()
        # Hero AABB cells: green outline (draw border lines)
        for wx, wy in debug.hero_cells:
            x0 = (wx - camera_x) * cs
            y0 = h - (wy - camera_y) * cs - cs
            border = pyglet.shapes.Rectangle(x0, y0, cs, cs, color=(0, 255, 0))
            border.opacity = 60
            border.draw()

    def draw_ui(self, hero: Hero) -> None:
        """Draw HP/MP bars and any other UI."""
        self.health_bar.draw(hero.hp, cfg.HERO_MAX_HP, hero.mp, cfg.HERO_MAX_MP)

    def draw_fps(self, dt: float) -> None:
        """Draw FPS counter (call when F3 debug mode is active)."""
        if dt > 0:
            self._fps_history.append(1.0 / dt)
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)
        if self._fps_history:
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            self._fps_label.text = f"FPS: {avg_fps:.0f}  dt: {dt*1000:.1f}ms"
        self._fps_label.draw()

    def draw(self, world, hero: Hero, camera_x: int, camera_y: int, view_mode: DebugViewMode, dt: float = 1.0 / 60.0) -> None:
        """Full render pass."""
        self.draw_grid(world, view_mode)
        self.draw_hero(hero, camera_x, camera_y, dt)
        self.draw_ui(hero)
