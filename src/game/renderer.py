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
        """Draw the hero using animation FrameData vertices."""
        screen_x = (hero.x - camera_x) * cfg.CELL_SCALE
        screen_y = (hero.y - camera_y) * cfg.CELL_SCALE

        # Determine animation state
        state = hero.state
        valid_states = ("idle", "walk", "jump", "chant", "cast")
        if state not in valid_states:
            state = "idle"
        frame = self.anim_mgr.update(dt, state)
        if frame is None:
            w = hero.width * cfg.CELL_SCALE
            h = hero.height * cfg.CELL_SCALE
            color = (200, 60, 40)
        else:
            color = frame.color[:3]
            verts = frame.vertices
            min_vx = min(v[0] for v in verts)
            max_vx = max(v[0] for v in verts)
            min_vy = min(v[1] for v in verts)
            max_vy = max(v[1] for v in verts)
            screen_x += min_vx
            screen_y += min_vy
            w = max_vx - min_vx
            h = max_vy - min_vy

        if not hero.facing_right and w > 0:
            screen_x += w

        r = self._hero_rect
        r.x = screen_x
        r.y = screen_y
        r.width = w
        r.height = h
        r.color = color
        r.draw()

    def draw_ui(self, hero: Hero) -> None:
        """Draw HP/MP bars and any other UI."""
        self.health_bar.draw(hero.hp, cfg.HERO_MAX_HP, hero.mp, cfg.HERO_MAX_MP)

    def draw(self, world, hero: Hero, camera_x: int, camera_y: int, view_mode: DebugViewMode, dt: float = 1.0 / 60.0) -> None:
        """Full render pass."""
        self.draw_grid(world, view_mode)
        self.draw_hero(hero, camera_x, camera_y, dt)
        self.draw_ui(hero)
