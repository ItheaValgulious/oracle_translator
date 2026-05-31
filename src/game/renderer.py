"""Game renderer: grid + entity overlay + UI.

All overlay drawing (hero, enemies, projectiles, UI, text) uses ModernGL directly.
pyglet.text.Label / pyglet.shapes are NEVER used because GPU compute shaders
invalidate their internal GL program IDs, causing GL_INVALID_VALUE crashes.
Text is rendered via a minimal 5x7 bitmap font baked into the overlay.
"""

from __future__ import annotations

import logging
import math
from array import array
from time import perf_counter

log = logging.getLogger(__name__)

import moderngl

from src.engine.render import DebugViewMode
from src.game import config as cfg
from src.game.animation import AnimationManager, SimpleSpriteBatch
from src.game.enemy import EnemyA, EnemyB, EnemyC, EnemyBase
from src.game.entity_manager import DebugCollisionInfo
from src.game.hero import Hero
from src.game.projectile import Arrow, Fireball, ProjectileBase

# ── Bitmap font: 4 wide x 6 tall, each row is 4 bits ────────────
# Encodes ASCII 32 (space) through 126 (~). 95 characters.
_FONT_WIDTH = 5
_FONT_HEIGHT = 7
_FONT_DATA: list[int] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # 32 ' '
    0x00, 0x00, 0x5F, 0x00, 0x00, 0x00,  # 33 '!'
    0x00, 0x07, 0x00, 0x07, 0x00, 0x00,  # 34 '"'
    0x14, 0x7F, 0x14, 0x7F, 0x14, 0x00,  # 35 '#'
    0x24, 0x2A, 0x7F, 0x2A, 0x12, 0x00,  # 36 '$'
    0x23, 0x13, 0x08, 0x64, 0x62, 0x00,  # 37 '%'
    0x36, 0x49, 0x55, 0x22, 0x50, 0x00,  # 38 '&'
    0x00, 0x05, 0x03, 0x00, 0x00, 0x00,  # 39 '''
    0x00, 0x1C, 0x22, 0x41, 0x00, 0x00,  # 40 '('
    0x00, 0x41, 0x22, 0x1C, 0x00, 0x00,  # 41 ')'
    0x14, 0x08, 0x3E, 0x08, 0x14, 0x00,  # 42 '*'
    0x08, 0x08, 0x3E, 0x08, 0x08, 0x00,  # 43 '+'
    0x00, 0x50, 0x30, 0x00, 0x00, 0x00,  # 44 ','
    0x08, 0x08, 0x08, 0x08, 0x08, 0x00,  # 45 '-'
    0x00, 0x60, 0x60, 0x00, 0x00, 0x00,  # 46 '.'
    0x20, 0x10, 0x08, 0x04, 0x02, 0x00,  # 47 '/'
    0x3E, 0x51, 0x49, 0x45, 0x3E, 0x00,  # 48 '0'
    0x00, 0x42, 0x7F, 0x40, 0x00, 0x00,  # 49 '1'
    0x42, 0x61, 0x51, 0x49, 0x46, 0x00,  # 50 '2'
    0x21, 0x41, 0x45, 0x4B, 0x31, 0x00,  # 51 '3'
    0x18, 0x14, 0x12, 0x7F, 0x10, 0x00,  # 52 '4'
    0x27, 0x45, 0x45, 0x45, 0x39, 0x00,  # 53 '5'
    0x3C, 0x4A, 0x49, 0x49, 0x30, 0x00,  # 54 '6'
    0x01, 0x71, 0x09, 0x05, 0x03, 0x00,  # 55 '7'
    0x36, 0x49, 0x49, 0x49, 0x36, 0x00,  # 56 '8'
    0x06, 0x49, 0x49, 0x29, 0x1E, 0x00,  # 57 '9'
    0x00, 0x36, 0x36, 0x00, 0x00, 0x00,  # 58 ':'
    0x00, 0x56, 0x36, 0x00, 0x00, 0x00,  # 59 ';'
    0x08, 0x14, 0x22, 0x41, 0x00, 0x00,  # 60 '<'
    0x14, 0x14, 0x14, 0x14, 0x14, 0x00,  # 61 '='
    0x00, 0x41, 0x22, 0x14, 0x08, 0x00,  # 62 '>'
    0x02, 0x01, 0x51, 0x09, 0x06, 0x00,  # 63 '?'
    0x32, 0x49, 0x79, 0x41, 0x3E, 0x00,  # 64 '@'
    0x7E, 0x11, 0x11, 0x11, 0x7E, 0x00,  # 65 'A'
    0x7F, 0x49, 0x49, 0x49, 0x36, 0x00,  # 66 'B'
    0x3E, 0x41, 0x41, 0x41, 0x22, 0x00,  # 67 'C'
    0x7F, 0x41, 0x41, 0x22, 0x1C, 0x00,  # 68 'D'
    0x7F, 0x49, 0x49, 0x49, 0x41, 0x00,  # 69 'E'
    0x7F, 0x09, 0x09, 0x09, 0x01, 0x00,  # 70 'F'
    0x3E, 0x41, 0x49, 0x49, 0x7A, 0x00,  # 71 'G'
    0x7F, 0x08, 0x08, 0x08, 0x7F, 0x00,  # 72 'H'
    0x00, 0x41, 0x7F, 0x41, 0x00, 0x00,  # 73 'I'
    0x20, 0x40, 0x41, 0x3F, 0x01, 0x00,  # 74 'J'
    0x7F, 0x08, 0x14, 0x22, 0x41, 0x00,  # 75 'K'
    0x7F, 0x40, 0x40, 0x40, 0x40, 0x00,  # 76 'L'
    0x7F, 0x02, 0x0C, 0x02, 0x7F, 0x00,  # 77 'M'
    0x7F, 0x04, 0x08, 0x10, 0x7F, 0x00,  # 78 'N'
    0x3E, 0x41, 0x41, 0x41, 0x3E, 0x00,  # 79 'O'
    0x7F, 0x09, 0x09, 0x09, 0x06, 0x00,  # 80 'P'
    0x3E, 0x41, 0x51, 0x21, 0x5E, 0x00,  # 81 'Q'
    0x7F, 0x09, 0x19, 0x29, 0x46, 0x00,  # 82 'R'
    0x46, 0x49, 0x49, 0x49, 0x31, 0x00,  # 83 'S'
    0x01, 0x01, 0x7F, 0x01, 0x01, 0x00,  # 84 'T'
    0x3F, 0x40, 0x40, 0x40, 0x3F, 0x00,  # 85 'U'
    0x1F, 0x20, 0x40, 0x20, 0x1F, 0x00,  # 86 'V'
    0x3F, 0x40, 0x38, 0x40, 0x3F, 0x00,  # 87 'W'
    0x63, 0x14, 0x08, 0x14, 0x63, 0x00,  # 88 'X'
    0x07, 0x08, 0x70, 0x08, 0x07, 0x00,  # 89 'Y'
    0x61, 0x51, 0x49, 0x45, 0x43, 0x00,  # 90 'Z'
    0x00, 0x7F, 0x41, 0x41, 0x00, 0x00,  # 91 '['
    0x02, 0x04, 0x08, 0x10, 0x20, 0x00,  # 92 '\'
    0x00, 0x41, 0x41, 0x7F, 0x00, 0x00,  # 93 ']'
    0x04, 0x02, 0x01, 0x02, 0x04, 0x00,  # 94 '^'
    0x40, 0x40, 0x40, 0x40, 0x40, 0x00,  # 95 '_'
    0x00, 0x01, 0x02, 0x04, 0x00, 0x00,  # 96 '`'
    0x20, 0x54, 0x54, 0x54, 0x78, 0x00,  # 97 'a'
    0x7F, 0x48, 0x44, 0x44, 0x38, 0x00,  # 98 'b'
    0x38, 0x44, 0x44, 0x44, 0x20, 0x00,  # 99 'c'
    0x38, 0x44, 0x44, 0x48, 0x7F, 0x00,  # 100 'd'
    0x38, 0x54, 0x54, 0x54, 0x18, 0x00,  # 101 'e'
    0x08, 0x7E, 0x09, 0x01, 0x02, 0x00,  # 102 'f'
    0x0C, 0x52, 0x52, 0x52, 0x3E, 0x00,  # 103 'g'
    0x7F, 0x08, 0x04, 0x04, 0x78, 0x00,  # 104 'h'
    0x00, 0x44, 0x7D, 0x40, 0x00, 0x00,  # 105 'i'
    0x20, 0x40, 0x44, 0x3D, 0x00, 0x00,  # 106 'j'
    0x7F, 0x10, 0x28, 0x44, 0x00, 0x00,  # 107 'k'
    0x00, 0x41, 0x7F, 0x40, 0x00, 0x00,  # 108 'l'
    0x7C, 0x04, 0x18, 0x04, 0x78, 0x00,  # 109 'm'
    0x7C, 0x08, 0x04, 0x04, 0x78, 0x00,  # 110 'n'
    0x38, 0x44, 0x44, 0x44, 0x38, 0x00,  # 111 'o'
    0x7C, 0x14, 0x14, 0x14, 0x08, 0x00,  # 112 'p'
    0x08, 0x14, 0x14, 0x18, 0x7C, 0x00,  # 113 'q'
    0x7C, 0x08, 0x04, 0x04, 0x08, 0x00,  # 114 'r'
    0x48, 0x54, 0x54, 0x54, 0x20, 0x00,  # 115 's'
    0x04, 0x3F, 0x44, 0x40, 0x20, 0x00,  # 116 't'
    0x3C, 0x40, 0x40, 0x20, 0x7C, 0x00,  # 117 'u'
    0x1C, 0x20, 0x40, 0x20, 0x1C, 0x00,  # 118 'v'
    0x3C, 0x40, 0x30, 0x40, 0x3C, 0x00,  # 119 'w'
    0x44, 0x28, 0x10, 0x28, 0x44, 0x00,  # 120 'x'
    0x0C, 0x50, 0x50, 0x50, 0x3C, 0x00,  # 121 'y'
    0x44, 0x64, 0x54, 0x4C, 0x44, 0x00,  # 122 'z'
    0x00, 0x08, 0x36, 0x41, 0x00, 0x00,  # 123 '{'
    0x00, 0x00, 0x7F, 0x00, 0x00, 0x00,  # 124 '|'
    0x00, 0x41, 0x36, 0x08, 0x00, 0x00,  # 125 '}'
    0x10, 0x08, 0x08, 0x10, 0x08, 0x00,  # 126 '~'
]


class GameRenderer:
    """Renders the world grid, hero overlay, and UI."""

    def __init__(self, ctx: moderngl.Context, window_width: int, window_height: int) -> None:
        self.ctx = ctx
        self.window_width = window_width
        self.window_height = window_height
        self.anim_mgr = AnimationManager()
        self.sprite_batch = SimpleSpriteBatch()

        # FPS tracking
        self._render_fps_history: list[float] = []
        self._last_render_at: float | None = None
        self._last_render_dt = 0.0
        self._render_fps = 0.0
        self._fps_label_updated_at = 0.0
        self._fps_label_update_interval = 0.25
        self._fps_text = ""

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

        # Overlay shader (shared by hero, enemies, projectiles, UI, text)
        self._overlay_program = self.ctx.program(
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
        self._overlay_vbo: moderngl.Buffer | None = None
        self._overlay_vao: moderngl.VertexArray | None = None
        self._overlay_capacity_bytes = 0

    # ── Overlay helpers ────────────────────────────────────────────

    def _append_rect(
        self,
        vertices: array,
        x0: float, y0: float,
        x1: float, y1: float,
        color: tuple[int, int, int],
        opacity: int = 255,
    ) -> None:
        """Append a screen-space rectangle as two NDC triangles."""
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

    def _append_line(
        self,
        vertices: array,
        x0: float, y0: float,
        x1: float, y1: float,
        color: tuple[int, int, int],
        thickness: float = 2.0,
        opacity: int = 255,
    ) -> None:
        """Append a line as a thin rectangle."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 0.01:
            return
        nx = -dy / length * thickness * 0.5
        ny = dx / length * thickness * 0.5
        self._append_rect(
            vertices,
            min(x0 + nx, x1 + nx, x1 - nx, x0 - nx),
            min(y0 + ny, y1 + ny, y1 - ny, y0 - ny),
            max(x0 + nx, x1 + nx, x1 - nx, x0 - nx),
            max(y0 + ny, y1 + ny, y1 - ny, y0 - ny),
            color, opacity,
        )

    def _append_circle(
        self,
        vertices: array,
        cx: float, cy: float,
        radius: float,
        color: tuple[int, int, int],
        segments: int = 16,
        opacity: int = 255,
    ) -> None:
        """Append a filled circle as a triangle fan (decomposed to triangles)."""
        r = color[0] / 255.0
        g = color[1] / 255.0
        b = color[2] / 255.0
        a = opacity / 255.0
        cx_ndc = cx / self.window_width * 2.0 - 1.0
        cy_ndc = cy / self.window_height * 2.0 - 1.0
        rx = radius / self.window_width * 2.0
        ry = radius / self.window_height * 2.0
        prev_x = cx_ndc + rx
        prev_y = cy_ndc
        for i in range(1, segments + 1):
            angle = 2.0 * math.pi * i / segments
            cur_x = cx_ndc + rx * math.cos(angle)
            cur_y = cy_ndc + ry * math.sin(angle)
            vertices.extend((cx_ndc, cy_ndc, r, g, b, a))
            vertices.extend((prev_x, prev_y, r, g, b, a))
            vertices.extend((cur_x, cur_y, r, g, b, a))
            prev_x = cur_x
            prev_y = cur_y

    def _append_text(
        self,
        vertices: array,
        x: float, y: float,
        text: str,
        color: tuple[int, int, int],
        pixel_size: float = 2.0,
        opacity: int = 255,
    ) -> float:
        """Append bitmap-font text as tiny overlay rectangles.

        Returns the x position after the last character (for chaining).
        Font data is 5x7 column-major, 6 bytes per char (5 data + 1 padding).
        """
        r = color[0] / 255.0
        g = color[1] / 255.0
        b = color[2] / 255.0
        a = opacity / 255.0
        ww = float(self.window_width)
        wh = float(self.window_height)
        cx = x
        char_spacing = (_FONT_WIDTH + 1) * pixel_size
        _BYTES_PER_CHAR = 6  # 5 data columns + 1 padding
        for ch in text:
            code = ord(ch)
            if code < 32 or code > 126:
                cx += char_spacing
                continue
            idx = (code - 32) * _BYTES_PER_CHAR
            for col in range(_FONT_WIDTH):
                bits = _FONT_DATA[idx + col]
                for row in range(_FONT_HEIGHT):
                    if bits & (1 << row):
                        px = cx + col * pixel_size
                        # row 0 = top of glyph → highest y in bottom-up coords
                        py = y + (_FONT_HEIGHT - 1 - row) * pixel_size
                        nx0 = px / ww * 2.0 - 1.0
                        ny0 = py / wh * 2.0 - 1.0
                        nx1 = (px + pixel_size) / ww * 2.0 - 1.0
                        ny1 = (py + pixel_size) / wh * 2.0 - 1.0
                        vertices.extend((
                            nx0, ny0, r, g, b, a,
                            nx1, ny0, r, g, b, a,
                            nx1, ny1, r, g, b, a,
                            nx0, ny0, r, g, b, a,
                            nx1, ny1, r, g, b, a,
                            nx0, ny1, r, g, b, a,
                        ))
            cx += char_spacing
        return cx

    def _flush_overlay(self, vertices: array, mode: int = moderngl.TRIANGLES) -> None:
        """Upload and render accumulated overlay vertices."""
        if not vertices:
            return
        data = vertices.tobytes()
        count = len(vertices) // 6
        if self._overlay_vbo is None or len(data) > self._overlay_capacity_bytes:
            self._overlay_capacity_bytes = max(4096, len(data) * 2)
            self._overlay_vbo = self.ctx.buffer(reserve=self._overlay_capacity_bytes)
            self._overlay_vao = self.ctx.vertex_array(
                self._overlay_program,
                [(self._overlay_vbo, "2f 4f", "in_pos", "in_color")],
            )
        self._overlay_vbo.write(data)
        self.ctx.enable(moderngl.BLEND)
        if self._overlay_vao is not None:
            self._overlay_vao.render(mode, vertices=count)

    # ── Grid ───────────────────────────────────────────────────────

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

    # ── Entity overlays ────────────────────────────────────────────

    def draw_hero(self, hero: Hero, camera_x: int, camera_y: int, dt: float) -> None:
        """Draw the hero rectangle."""
        cs = cfg.CELL_SCALE
        sx = (hero.left - camera_x) * cs
        sy = self.window_height - (hero.top - camera_y) * cs - cs
        w = hero.width * cs
        h = hero.height * cs
        color: tuple[int, int, int] = (200, 60, 40)

        state = hero.state
        if state not in ("idle", "walk", "jump", "chant", "cast"):
            state = "idle"
        frame = self.anim_mgr.update(dt, state)
        if frame is not None:
            color = frame.color[:3]

        self._hero_flipped = not hero.facing_right
        verts = array("f")
        self._append_rect(verts, sx, sy, sx + w, sy + h, color)
        self._flush_overlay(verts)

    def draw_enemies(self, enemies: dict, camera_x: int, camera_y: int) -> None:
        """Draw all enemy rectangles."""
        cs = cfg.CELL_SCALE
        verts = array("f")
        for eid, enemy in enemies.items():
            if not enemy.is_alive:
                continue
            sx = (enemy.left - camera_x) * cs
            sy = self.window_height - (enemy.top - camera_y) * cs - cs
            w = enemy.width * cs
            h = enemy.height * cs
            color: tuple[int, int, int] = (50, 100, 220)
            if isinstance(enemy, EnemyB):
                color = (160, 40, 200)
            elif isinstance(enemy, EnemyC):
                color = (220, 30, 30)
            if enemy.damage_flash_timer > 0.0:
                color = (255, 255, 255)
            self._append_rect(verts, sx, sy, sx + w, sy + h, color)
            # Boss HP bar
            if isinstance(enemy, EnemyC):
                bar_w = w * 0.8
                bar_h = cs * 2
                bar_x = sx + w * 0.1
                bar_y = sy - bar_h - 2
                hp_ratio = enemy.hp / enemy.max_hp
                self._append_rect(verts, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, (60, 0, 0))
                self._append_rect(verts, bar_x, bar_y, bar_x + bar_w * hp_ratio, bar_y + bar_h, (200, 30, 30))
        self._flush_overlay(verts)

    def draw_projectiles(self, projectiles: list, camera_x: int, camera_y: int) -> None:
        """Draw all active projectiles."""
        cs = cfg.CELL_SCALE
        verts = array("f")
        for p in projectiles:
            if not p.is_alive:
                continue
            sx = (p.x - camera_x) * cs
            sy = self.window_height - (p.y - camera_y) * cs - cs
            if isinstance(p, Arrow):
                # Fixed 2x2 cell arrow, oriented by velocity direction
                arrow_cs = cs * 2
                self._append_rect(
                    verts,
                    sx - arrow_cs * 0.5, sy - arrow_cs * 0.5,
                    sx + arrow_cs * 0.5, sy + arrow_cs * 0.5,
                    (180, 120, 60), opacity=220,
                )
            elif isinstance(p, Fireball):
                self._append_circle(verts, sx, sy, cs * 2, (255, 140, 0))
        self._flush_overlay(verts)

    # ── Debug overlay ──────────────────────────────────────────────

    def _append_debug_cell(
        self,
        vertices: array,
        camera_x: int, camera_y: int,
        wx: int, wy: int,
        color: tuple[int, int, int],
        opacity: int,
    ) -> None:
        cs = cfg.CELL_SCALE
        x0 = (wx - camera_x) * cs
        y0 = self.window_height - (wy - camera_y) * cs - cs
        self._append_rect(vertices, x0, y0, x0 + cs, y0 + cs, color, opacity)

    def _append_debug_world_rect(
        self,
        vertices: array,
        camera_x: int, camera_y: int,
        wx0: int, wy0: int, wx1: int, wy1: int,
        color: tuple[int, int, int],
        opacity: int,
    ) -> None:
        cs = cfg.CELL_SCALE
        x0 = (wx0 - camera_x) * cs
        y0 = self.window_height - (wy1 - camera_y) * cs
        x1 = (wx1 - camera_x) * cs
        y1 = self.window_height - (wy0 - camera_y) * cs
        self._append_rect(vertices, x0, y0, x1, y1, color, opacity)

    def draw_debug_collision(self, camera_x: int, camera_y: int, debug: DebugCollisionInfo) -> None:
        """Draw debug overlay showing collision-checked cells."""
        vertices = array("f")
        for wx, wy in debug.ground_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (255, 255, 0), 120)
        for wx, wy in debug.ceiling_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (180, 80, 255), 120)
        for wx, wy in debug.left_wall_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (255, 0, 0), 120)
        for wx, wy in debug.right_wall_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (0, 100, 255), 120)
        for wx0, wy0, wx1, wy1 in debug.hero_rects:
            self._append_debug_world_rect(vertices, camera_x, camera_y, wx0, wy0, wx1, wy1, (0, 255, 0), 60)
        for wx, wy in debug.hero_cells:
            self._append_debug_cell(vertices, camera_x, camera_y, wx, wy, (0, 255, 0), 60)
        self._flush_overlay(vertices)

    # ── UI ─────────────────────────────────────────────────────────

    def draw_ui(self, hero: Hero) -> None:
        """Draw HP/MP bars + text via overlay (no pyglet labels)."""
        verts = array("f")
        bar_x, bar_y = 10.0, 10.0
        bar_w, bar_h = 150.0, 12.0
        # HP bar background + fill
        hp_ratio = max(0.0, min(1.0, hero.hp / cfg.HERO_MAX_HP))
        self._append_rect(verts, bar_x, bar_y + bar_h + 2, bar_x + bar_w, bar_y + bar_h + 2 + bar_h, (60, 0, 0), opacity=160)
        self._append_rect(verts, bar_x, bar_y + bar_h + 2, bar_x + bar_w * hp_ratio, bar_y + bar_h + 2 + bar_h, (220, 30, 30), opacity=160)
        # MP bar background + fill
        mp_ratio = max(0.0, min(1.0, hero.mp / cfg.HERO_MAX_MP))
        self._append_rect(verts, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, (0, 0, 60), opacity=160)
        self._append_rect(verts, bar_x, bar_y, bar_x + bar_w * mp_ratio, bar_y + bar_h, (30, 30, 220), opacity=160)
        # Text labels via bitmap font
        self._append_text(verts, 14.0, bar_y + bar_h + 4.0,
                          f"HP:{int(hero.hp)}/{int(cfg.HERO_MAX_HP)}", (255, 255, 255), pixel_size=3.0)
        self._append_text(verts, 14.0, bar_y + 2.0,
                          f"MP:{int(hero.mp)}/{int(cfg.HERO_MAX_MP)}", (255, 255, 255), pixel_size=3.0)
        self._flush_overlay(verts)

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
        """Draw FPS counter via bitmap font overlay."""
        now = perf_counter()
        if now - self._fps_label_updated_at >= self._fps_label_update_interval:
            if self._render_fps_history:
                self._fps_text = (
                    f"R:{self._render_fps:.0f} S:{sim_fps:.0f} "
                    f"dt:{self._last_render_dt*1000:.1f}ms"
                )
            else:
                self._fps_text = f"R:-- S:{sim_fps:.0f}"
            self._fps_label_updated_at = now
        if self._fps_text:
            verts = array("f")
            self._append_text(verts, 10.0, float(self.window_height) - 20.0,
                              self._fps_text, (255, 255, 0), pixel_size=3.0)
            self._flush_overlay(verts)

    def draw_perf_overlay(self, app) -> None:
        """Draw a compact multi-line perf overlay.

        Important: CPU timings here are submission / main-thread timings.
        In particular, world.step submit time is not the same as GPU completion time.
        """
        world = app.world
        if world is None:
            return
        self.draw_fps(app.sim_fps)
        chunk_stats = world.chunk_cache.snapshot_stats()
        perf = app.debug_perf_snapshot()
        lines = [
            (
                "Main "
                f"tick {perf['main_tick_total_last_ms']:.1f}/{perf['main_tick_total_avg_ms']:.1f} ms "
                f"sim {perf['main_tick_sim_last_ms']:.1f}/{perf['main_tick_sim_avg_ms']:.1f} "
                f"cam {perf['main_tick_camera_last_ms']:.1f}/{perf['main_tick_camera_avg_ms']:.1f} "
                f"io {perf['main_tick_background_io_last_ms']:.1f}/{perf['main_tick_background_io_avg_ms']:.1f}"
            ),
            (
                "Sim "
                f"upd {perf['update_game_total_last_ms']:.1f}/{perf['update_game_total_avg_ms']:.1f} ms "
                f"poll {perf['update_game_poll_last_ms']:.1f}/{perf['update_game_poll_avg_ms']:.1f} "
                f"sched {perf['update_game_schedule_feedback_last_ms']:.1f}/{perf['update_game_schedule_feedback_avg_ms']:.1f}"
            ),
            (
                "CPU submit "
                f"step {perf['update_game_world_step_submit_last_ms']:.1f}/{perf['update_game_world_step_submit_avg_ms']:.1f} ms "
                f"snap {perf['update_game_snapshot_last_ms']:.1f}/{perf['update_game_snapshot_avg_ms']:.1f} "
                f"clear {perf['update_game_ctx_clear_last_ms']:.1f}/{perf['update_game_ctx_clear_avg_ms']:.1f}"
            ),
            (
                "Paging "
                f"shift {world.shift_time_last_ms():.1f} ms "
                f"in {world.incoming_load_time_last_ms():.1f} "
                f"ov {world.overlap_copy_time_last_ms():.1f} "
                f"ev {world.stage_time_last_ms():.1f} "
                f"anc {world.anchor_build_time_last_ms():.1f}+{world.anchor_upload_time_last_ms():.1f}"
            ),
            (
                "Shift chunks "
                f"cache {world.paging_stats.last_shift_cache_hits} "
                f"empty {world.paging_stats.last_shift_empty_hits} "
                f"wait {world.paging_stats.last_shift_inflight_wait_hits} "
                f"load {world.paging_stats.last_shift_disk_loads}/{world.paging_stats.last_shift_disk_load_seconds * 1000.0:.1f} ms "
                f"gen {world.paging_stats.last_shift_generates}/{world.paging_stats.last_shift_generate_seconds * 1000.0:.1f} ms "
                f"save {world.paging_stats.last_shift_saves}/{world.paging_stats.last_shift_save_seconds * 1000.0:.1f} ms"
            ),
            (
                "Chunk totals "
                f"load {chunk_stats.disk_load_count}/{chunk_stats.disk_load_total_seconds * 1000.0:.1f} ms "
                f"gen {chunk_stats.generate_count}/{chunk_stats.generate_total_seconds * 1000.0:.1f} ms "
                f"save {chunk_stats.save_count}/{chunk_stats.save_total_seconds * 1000.0:.1f} ms "
                f"cached {chunk_stats.cached_chunks} inflight {chunk_stats.prefetch_inflight} queued {chunk_stats.prefetch_queued}"
            ),
            "F3 perf overlay, F6 collision samples, F4 temp view, F5 pressure view",
        ]
        bg = array("f")
        text = array("f")
        line_h = 18.0
        top = float(self.window_height) - 44.0
        max_chars = max(len(line) for line in lines)
        panel_w = min(float(self.window_width) - 12.0, 10.0 + max_chars * 12.0)
        panel_h = 10.0 + len(lines) * line_h
        self._append_rect(bg, 6.0, top - panel_h + 6.0, 6.0 + panel_w, top + 12.0, (0, 0, 0), opacity=180)
        for idx, line in enumerate(lines):
            y = top - idx * line_h
            self._append_text(text, 12.0, y, line, (255, 230, 120), pixel_size=2.0)
        self._flush_overlay(bg)
        self._flush_overlay(text)

    def draw(self, world, hero: Hero, camera_x: int, camera_y: int, view_mode: DebugViewMode, dt: float = 1.0 / 60.0, enemies: dict | None = None, projectiles: list | None = None) -> None:
        """Full render pass."""
        self._record_render_fps()
        self.draw_grid(world, view_mode)
        self.draw_hero(hero, camera_x, camera_y, dt)
        if enemies is not None:
            self.draw_enemies(enemies, camera_x, camera_y)
        if projectiles is not None:
            self.draw_projectiles(projectiles, camera_x, camera_y)
        self.draw_ui(hero)
