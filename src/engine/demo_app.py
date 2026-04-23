from __future__ import annotations

from array import array
from dataclasses import dataclass

import moderngl
import pyglet
from pyglet.window import key, mouse

from .grid import create_grid
from .materials import build_material_registry
from .render import build_rgba_frame
from .scenarios import populate_demo_scene
from .sim import inject_cells, step
from .types import CellFlag


@dataclass(frozen=True)
class ToolSpec:
    label: str
    family_id: str | None
    variant_id: str | None
    overrides: dict[str, object]


TOOLS: dict[int, ToolSpec] = {
    key._1: ToolSpec("Stone Platform", "stone", "stone_platform", {}),
    key._2: ToolSpec("Stone Fixpoint", "stone", "stone_platform", {"flags": CellFlag.FIXPOINT, "support_value": 1.0}),
    key._3: ToolSpec("Sand", "sand", "sand_powder", {}),
    key._4: ToolSpec("Water", "water", "water", {}),
    key._5: ToolSpec("Fire", "fire", "fire", {}),
    key._6: ToolSpec("Acid", "acid", "acid_liquid", {}),
    key._7: ToolSpec("Tar", "tar", "tar_liquid", {}),
    key._8: ToolSpec("Poison", "poison", "poison_liquid", {}),
    key._9: ToolSpec("Ice", "water", "ice", {}),
    key._0: ToolSpec("Erase", None, None, {}),
}


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
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(frame_tex, v_uv);
}
"""


class VoxelDemoWindow(pyglet.window.Window):
    def __init__(self, grid_width: int = 160, grid_height: int = 96, cell_scale: int = 8) -> None:
        super().__init__(
            width=grid_width * cell_scale,
            height=grid_height * cell_scale,
            caption="Voxel Engine Demo",
            resizable=False,
            vsync=True,
        )
        self.grid = create_grid(grid_width, grid_height)
        self.registry = build_material_registry()
        populate_demo_scene(self.grid, self.registry)
        self.cell_scale = cell_scale
        self.current_tool_key = key._1
        self.brush_radius = 2
        self.paused = False
        self.steps_per_tick = 1

        try:
            self.ctx = moderngl.create_context()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ModernGL could not create an OpenGL context. "
                "Run the demo on a desktop session with working OpenGL drivers."
            ) from exc
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.BLEND)

        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        quad = self.ctx.buffer(
            data=array(
                "f",
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ).tobytes()
        )
        self.vao = self.ctx.vertex_array(
            self.program,
            [(quad, "2f 2f", "in_pos", "in_uv")],
        )

        self.texture = self.ctx.texture((grid_width, grid_height), 4, build_rgba_frame(self.grid, self.registry))
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.texture.use(location=0)
        self.program["frame_tex"].value = 0

        self.overlay = pyglet.text.Label(
            "",
            x=12,
            y=self.height - 12,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=self.width - 24,
            color=(240, 240, 240, 255),
        )
        self._refresh_overlay()
        pyglet.clock.schedule_interval(self.tick, 1 / 60.0)

    def _screen_to_grid(self, sx: int, sy: int) -> tuple[int, int]:
        gx = max(0, min(self.grid.width - 1, sx // self.cell_scale))
        gy = max(0, min(self.grid.height - 1, self.grid.height - 1 - sy // self.cell_scale))
        return gx, gy

    def _refresh_overlay(self) -> None:
        tool = TOOLS[self.current_tool_key]
        status = "Paused" if self.paused else "Running"
        self.overlay.text = (
            f"{status} | Tool: {tool.label} | Brush: {self.brush_radius}\n"
            "1-9/0 switch tools, [ ] brush size, Space pause, N single-step, R reset scene, C clear"
        )

    def _upload_frame(self) -> None:
        self.texture.write(build_rgba_frame(self.grid, self.registry))

    def _paint(self, gx: int, gy: int, erase: bool = False) -> None:
        if erase or TOOLS[self.current_tool_key].family_id is None:
            inject_cells(
                self.grid,
                {"x": gx, "y": gy, "radius": self.brush_radius},
                "empty",
                "empty",
                registry=self.registry,
            )
            return
        tool = TOOLS[self.current_tool_key]
        inject_cells(
            self.grid,
            {"x": gx, "y": gy, "radius": self.brush_radius},
            tool.family_id,
            tool.variant_id,
            dict(tool.overrides),
            registry=self.registry,
        )

    def tick(self, dt: float) -> None:
        if not self.paused:
            for _ in range(self.steps_per_tick):
                step(self.grid, self.registry, dt)
        self._upload_frame()

    def on_draw(self) -> None:
        self.clear()
        self.ctx.clear(0.04, 0.05, 0.07, 1.0)
        self.texture.use(location=0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.overlay.draw()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        del modifiers
        gx, gy = self._screen_to_grid(x, y)
        self._paint(gx, gy, erase=button == mouse.RIGHT)
        self._upload_frame()

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        del dx, dy, modifiers
        gx, gy = self._screen_to_grid(x, y)
        erase = bool(buttons & mouse.RIGHT)
        draw = bool(buttons & mouse.LEFT)
        if draw or erase:
            self._paint(gx, gy, erase=erase)
            self._upload_frame()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        del modifiers
        if symbol in TOOLS:
            self.current_tool_key = symbol
        elif symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.N:
            step(self.grid, self.registry, 1 / 60.0)
            self._upload_frame()
        elif symbol == key.R:
            self.grid = create_grid(self.grid.width, self.grid.height)
            populate_demo_scene(self.grid, self.registry)
            self._upload_frame()
        elif symbol == key.C:
            self.grid = create_grid(self.grid.width, self.grid.height)
            self._upload_frame()
        elif symbol == key.BRACKETLEFT:
            self.brush_radius = max(0, self.brush_radius - 1)
        elif symbol == key.BRACKETRIGHT:
            self.brush_radius = min(8, self.brush_radius + 1)
        elif symbol == key.ESCAPE:
            self.close()
            return
        self._refresh_overlay()


def run_demo() -> None:
    window = VoxelDemoWindow()
    window.set_minimum_size(window.width, window.height)
    pyglet.app.run()
