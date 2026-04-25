from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import moderngl
import pyglet
from pyglet.window import key, mouse

from .gpu_backend import ComputeBackendUnavailable, GpuSimulator
from .grid import create_grid
from .materials import build_material_registry
from .render import DebugViewMode, build_rgba_frame
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
    key._2: ToolSpec("Stone Fixpoint", "stone", "stone_platform", {"flags": CellFlag.FIXPOINT}),
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


DEFAULT_TICK_RATE_HZ = 60.0
DEFAULT_SIMULATION_SUBSTEPS = 2
VIEW_MODE_ORDER = (
    DebugViewMode.MATERIAL,
    DebugViewMode.TEMPERATURE,
    DebugViewMode.PRESSURE,
)


class VoxelDemoWindow(pyglet.window.Window):
    def __init__(
        self,
        grid_width: int = 160,
        grid_height: int = 96,
        *,
        cell_scale: int = 8,
        window_width: int | None = None,
        window_height: int | None = None,
        simulation_substeps: int = DEFAULT_SIMULATION_SUBSTEPS,
        liquid_brownian_enabled: bool = True,
        blocked_impulse_enabled: bool = True,
        vsync: bool = True,
    ) -> None:
        initial_width = int(window_width if window_width is not None else grid_width * cell_scale)
        initial_height = int(window_height if window_height is not None else grid_height * cell_scale)
        super().__init__(
            width=initial_width,
            height=initial_height,
            caption="Voxel Engine Demo",
            resizable=True,
            vsync=vsync,
        )
        self.grid = create_grid(grid_width, grid_height)
        self.grid.liquid_brownian_enabled = bool(liquid_brownian_enabled)
        self.grid.blocked_impulse_enabled = bool(blocked_impulse_enabled)
        self.registry = build_material_registry()
        populate_demo_scene(self.grid, self.registry)
        self.gpu_simulator: GpuSimulator | None = None
        self.cell_scale = cell_scale
        self.target_tick_hz = DEFAULT_TICK_RATE_HZ
        self.last_tick_dt = 1.0 / self.target_tick_hz
        self.draw_fps = 0.0
        self._draw_counter = 0
        self._draw_sample_started_at = perf_counter()
        self.view_mode = DebugViewMode.MATERIAL
        self.current_tool_key = key._1
        self.brush_radius = 2
        self.paused = False
        self.steps_per_tick = max(1, int(simulation_substeps))
        self.liquid_brownian_enabled = bool(liquid_brownian_enabled)
        self.blocked_impulse_enabled = bool(blocked_impulse_enabled)
        self.backend_label = "CPU Reference"
        self.backend_detail = ""

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

        try:
            self.gpu_simulator = GpuSimulator(self.ctx, self.grid, self.registry)
            self.texture = self.gpu_simulator.frame_texture
            self.backend_label = "GPU Compute"
        except (ComputeBackendUnavailable, Exception) as exc:  # noqa: BLE001
            self.gpu_simulator = None
            self.texture = self.ctx.texture((grid_width, grid_height), 4, build_rgba_frame(self.grid, self.registry))
            self.backend_detail = f"{type(exc).__name__}: {exc}"
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
        pyglet.clock.schedule_interval(self.tick, 1.0 / self.target_tick_hz)

    def _screen_to_grid(self, sx: int, sy: int) -> tuple[int, int]:
        window_width = max(1, int(self.width))
        window_height = max(1, int(self.height))
        screen_x = float(sx)
        screen_y = float(sy)
        gx = max(0, min(self.grid.width - 1, int(screen_x * self.grid.width / window_width)))
        gy_from_bottom = int(screen_y * self.grid.height / window_height)
        gy = max(0, min(self.grid.height - 1, self.grid.height - 1 - gy_from_bottom))
        return gx, gy

    def _refresh_overlay(self) -> None:
        tool = TOOLS[self.current_tool_key]
        status = "Paused" if self.paused else "Running"
        refresh_hz = 0.0
        if self.last_tick_dt > 0.0 and isfinite(self.last_tick_dt):
            refresh_hz = 1.0 / self.last_tick_dt
        backend_note = f"\nFallback: {self.backend_detail}" if self.backend_detail else ""
        self.overlay.text = (
            f"{status} | Backend: {self.backend_label} | Refresh: {refresh_hz:.1f} Hz | Draw: {self.draw_fps:.1f} FPS\n"
            f"Grid: {self.grid.width}x{self.grid.height} | Window: {self.width}x{self.height} | View: {self.view_mode.value.title()} | Substeps: {self.steps_per_tick}\n"
            f"Liquid Brownian: {'On' if self.liquid_brownian_enabled else 'Off'}\n"
            f"Blocked Impulse: {'On' if self.blocked_impulse_enabled else 'Off'}\n"
            f"Tool: {tool.label} | Brush: {self.brush_radius}\n"
            "1-9/0 switch tools, [ ] brush size, -/= substeps, T cycle views, B toggle liquid Brownian, I toggle blocked impulse, Space pause, N single-step, R reset scene, C clear, drag border to resize"
            f"{backend_note}"
        )

    def _upload_frame(self) -> None:
        if self.gpu_simulator is not None:
            self.texture = self.gpu_simulator.render(self.view_mode)
            return
        self.texture.write(build_rgba_frame(self.grid, self.registry, view_mode=self.view_mode))

    def _paint(self, gx: int, gy: int, erase: bool = False) -> None:
        tool = TOOLS[self.current_tool_key]
        if self.gpu_simulator is not None:
            self.gpu_simulator.paint_circle(
                gx,
                gy,
                self.brush_radius,
                None if erase or tool.family_id is None else tool.family_id,
                None if erase or tool.family_id is None else tool.variant_id,
                overrides=dict(tool.overrides),
            )
            return
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
        self.last_tick_dt = dt
        if not self.paused:
            substep_dt = dt / self.steps_per_tick
            for _ in range(self.steps_per_tick):
                if self.gpu_simulator is not None:
                    self.gpu_simulator.step(substep_dt)
                else:
                    step(self.grid, self.registry, substep_dt)
        self._upload_frame()
        self._refresh_overlay()

    def on_draw(self) -> None:
        self._draw_counter += 1
        now = perf_counter()
        elapsed = now - self._draw_sample_started_at
        if elapsed >= 0.25:
            self.draw_fps = self._draw_counter / elapsed
            self._draw_counter = 0
            self._draw_sample_started_at = now
        self.clear()
        self.ctx.clear(0.04, 0.05, 0.07, 1.0)
        self.texture.use(location=0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.overlay.draw()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        if getattr(self, "ctx", None) is not None:
            self.ctx.viewport = (0, 0, width, height)
        if getattr(self, "overlay", None) is not None:
            self.overlay.y = height - 12
            self.overlay.width = max(120, width - 24)
            self._refresh_overlay()

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
            single_step_dt = (1 / 60.0) / self.steps_per_tick
            for _ in range(self.steps_per_tick):
                if self.gpu_simulator is not None:
                    self.gpu_simulator.step(single_step_dt)
                else:
                    step(self.grid, self.registry, single_step_dt)
            self._upload_frame()
        elif symbol == key.R:
            self.grid = create_grid(self.grid.width, self.grid.height)
            self.grid.liquid_brownian_enabled = self.liquid_brownian_enabled
            self.grid.blocked_impulse_enabled = self.blocked_impulse_enabled
            populate_demo_scene(self.grid, self.registry)
            if self.gpu_simulator is not None:
                self.gpu_simulator.load_grid(self.grid)
            self._upload_frame()
        elif symbol == key.C:
            self.grid = create_grid(self.grid.width, self.grid.height)
            self.grid.liquid_brownian_enabled = self.liquid_brownian_enabled
            self.grid.blocked_impulse_enabled = self.blocked_impulse_enabled
            if self.gpu_simulator is not None:
                self.gpu_simulator.load_grid(self.grid)
            self._upload_frame()
        elif symbol == key.T:
            current_index = VIEW_MODE_ORDER.index(self.view_mode)
            self.view_mode = VIEW_MODE_ORDER[(current_index + 1) % len(VIEW_MODE_ORDER)]
            self._upload_frame()
        elif symbol == key.B:
            self.liquid_brownian_enabled = not self.liquid_brownian_enabled
            self.grid.liquid_brownian_enabled = self.liquid_brownian_enabled
            if self.gpu_simulator is not None:
                self.gpu_simulator.set_liquid_brownian_enabled(self.liquid_brownian_enabled)
        elif symbol == key.I:
            self.blocked_impulse_enabled = not self.blocked_impulse_enabled
            self.grid.blocked_impulse_enabled = self.blocked_impulse_enabled
            if self.gpu_simulator is not None:
                self.gpu_simulator.set_blocked_impulse_enabled(self.blocked_impulse_enabled)
        elif symbol == key.MINUS:
            self.steps_per_tick = max(1, self.steps_per_tick - 1)
        elif symbol == key.EQUAL:
            self.steps_per_tick = min(16, self.steps_per_tick + 1)
        elif symbol == key.BRACKETLEFT:
            self.brush_radius = max(0, self.brush_radius - 1)
        elif symbol == key.BRACKETRIGHT:
            self.brush_radius = min(8, self.brush_radius + 1)
        elif symbol == key.ESCAPE:
            self.close()
            return
        self._refresh_overlay()


def run_demo(
    *,
    grid_width: int = 160,
    grid_height: int = 96,
    cell_scale: int = 8,
    window_width: int | None = None,
    window_height: int | None = None,
    simulation_substeps: int = DEFAULT_SIMULATION_SUBSTEPS,
    liquid_brownian_enabled: bool = True,
    blocked_impulse_enabled: bool = True,
    vsync: bool = True,
) -> None:
    window = VoxelDemoWindow(
        grid_width=grid_width,
        grid_height=grid_height,
        cell_scale=cell_scale,
        window_width=window_width,
        window_height=window_height,
        simulation_substeps=simulation_substeps,
        liquid_brownian_enabled=liquid_brownian_enabled,
        blocked_impulse_enabled=blocked_impulse_enabled,
        vsync=vsync,
    )
    window.set_minimum_size(200, 150)
    pyglet.app.run()
