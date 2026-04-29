from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import moderngl
import pyglet
from pyglet.window import key, mouse

from .gpu_backend import ComputeBackendUnavailable
from .materials import build_material_registry
from .render import DebugViewMode
from .scenarios import populate_demo_scene
from .types import CellFlag
from .world import (
    DEFAULT_HALO_CELLS,
    DEFAULT_PAGE_SHIFT_CELLS,
    ActiveWorldWindow,
    WorldChunkStore,
)


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
uniform vec2 view_uv_origin;
uniform vec2 view_uv_scale;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(frame_tex, view_uv_origin + v_uv * view_uv_scale);
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
        world_width: int | None = None,
        world_height: int | None = None,
        halo_cells: int = DEFAULT_HALO_CELLS,
        page_shift_cells: int = DEFAULT_PAGE_SHIFT_CELLS,
        cell_scale: int = 8,
        window_width: int | None = None,
        window_height: int | None = None,
        simulation_substeps: int = DEFAULT_SIMULATION_SUBSTEPS,
        liquid_brownian_enabled: bool = True,
        blocked_impulse_enabled: bool = True,
        directional_fallback_enabled: bool = True,
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

        self.registry = build_material_registry()
        self.viewport_grid_width = int(grid_width)
        self.viewport_grid_height = int(grid_height)
        self.world_grid_width = int(world_width if world_width is not None else max(grid_width * 2, grid_width + halo_cells * 4))
        self.world_grid_height = int(world_height if world_height is not None else max(grid_height * 2, grid_height + halo_cells * 4))
        self.halo_cells = int(halo_cells)
        self.page_shift_cells = int(page_shift_cells)
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
        self.directional_fallback_enabled = bool(directional_fallback_enabled)
        self.directional_fallback_angle_limit_degrees = 45.0
        self.backend_label = "CPU Reference"
        self.backend_detail = ""
        self.camera_pan_cells = max(8, self.page_shift_cells // 4)
        self.world: ActiveWorldWindow | None = None

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

        self.texture: moderngl.Texture | None = None
        self._rebuild_world(populate_scene=True)

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

    def _build_world_store(self, *, populate_scene: bool) -> WorldChunkStore:
        store = WorldChunkStore(self.world_grid_width, self.world_grid_height)
        if populate_scene:
            populate_demo_scene(store, self.registry)
        store.recompute_anchored_support(self.registry)
        return store

    def _rebuild_world(self, *, populate_scene: bool) -> None:
        store = self._build_world_store(populate_scene=populate_scene)
        try:
            self.world = ActiveWorldWindow(
                store,
                self.registry,
                viewport_width=self.viewport_grid_width,
                viewport_height=self.viewport_grid_height,
                halo_cells=self.halo_cells,
                page_shift_cells=self.page_shift_cells,
                ctx=self.ctx,
                liquid_brownian_enabled=self.liquid_brownian_enabled,
                blocked_impulse_enabled=self.blocked_impulse_enabled,
                directional_fallback_enabled=self.directional_fallback_enabled,
                directional_fallback_angle_limit_degrees=self.directional_fallback_angle_limit_degrees,
            )
            self.backend_label = "GPU Compute"
            self.backend_detail = ""
        except (ComputeBackendUnavailable, Exception) as exc:  # noqa: BLE001
            self.world = ActiveWorldWindow(
                store,
                self.registry,
                viewport_width=self.viewport_grid_width,
                viewport_height=self.viewport_grid_height,
                halo_cells=self.halo_cells,
                page_shift_cells=self.page_shift_cells,
                ctx=None,
                liquid_brownian_enabled=self.liquid_brownian_enabled,
                blocked_impulse_enabled=self.blocked_impulse_enabled,
                directional_fallback_enabled=self.directional_fallback_enabled,
                directional_fallback_angle_limit_degrees=self.directional_fallback_angle_limit_degrees,
            )
            self.backend_label = "CPU Reference"
            self.backend_detail = f"{type(exc).__name__}: {exc}"
        self._bind_world_texture()

    def _bind_world_texture(self) -> None:
        assert self.world is not None
        if self.world.gpu_simulator is not None:
            self.texture = self.world.gpu_simulator.frame_texture
        else:
            self.texture = self.ctx.texture(
                (self.world.active_width, self.world.active_height),
                4,
                self.world.render(self.view_mode),
            )
        assert self.texture is not None
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.texture.use(location=0)
        self.program["frame_tex"].value = 0
        self._update_view_uv_uniforms()

    def _update_view_uv_uniforms(self) -> None:
        assert self.world is not None
        origin_x, origin_y, scale_x, scale_y = self.world.visible_uv_rect()
        self.program["view_uv_origin"].value = (origin_x, origin_y)
        self.program["view_uv_scale"].value = (scale_x, scale_y)

    def _screen_to_world(self, sx: int, sy: int) -> tuple[int, int]:
        assert self.world is not None
        return self.world.screen_to_world(
            sx,
            sy,
            screen_width=max(1, int(self.width)),
            screen_height=max(1, int(self.height)),
        )

    def _refresh_overlay(self) -> None:
        assert self.world is not None
        tool = TOOLS[self.current_tool_key]
        status = "Paused" if self.paused else "Running"
        refresh_hz = 0.0
        if self.last_tick_dt > 0.0 and isfinite(self.last_tick_dt):
            refresh_hz = 1.0 / self.last_tick_dt
        backend_note = f"\nFallback: {self.backend_detail}" if self.backend_detail else ""
        self.overlay.text = (
            f"{status} | Backend: {self.backend_label} | Refresh: {refresh_hz:.1f} Hz | Draw: {self.draw_fps:.1f} FPS\n"
            f"World: {self.world.world_width}x{self.world.world_height} | Active: {self.world.active_width}x{self.world.active_height} | Viewport: {self.world.viewport_width}x{self.world.viewport_height}\n"
            f"Camera: ({self.world.camera_x}, {self.world.camera_y}) | Window Origin: ({self.world.active_origin_x}, {self.world.active_origin_y}) | Window: {self.width}x{self.height}\n"
            f"View: {self.view_mode.value.title()} | Substeps: {self.steps_per_tick} | Brush: {self.brush_radius} | Tool: {tool.label}\n"
            f"Liquid Brownian: {'On' if self.liquid_brownian_enabled else 'Off'}\n"
            f"Blocked Impulse: {'On' if self.blocked_impulse_enabled else 'Off'}\n"
            f"Directional Fallback: {'On' if self.directional_fallback_enabled else 'Off'} (<= {self.directional_fallback_angle_limit_degrees:.0f} deg)\n"
            "1-9/0 switch tools, [ ] brush size, -/= substeps, T cycle views, B toggle liquid Brownian, I toggle blocked impulse, F toggle directional fallback, WASD or arrows pan camera, Space pause, N single-step, R reset scene, C clear"
            f"{backend_note}"
        )

    def _upload_frame(self) -> None:
        assert self.world is not None
        if self.world.gpu_simulator is not None:
            self.texture = self.world.render(self.view_mode)
        else:
            assert self.texture is not None
            self.texture.write(self.world.render(self.view_mode))
        self._update_view_uv_uniforms()

    def _paint(self, world_x: int, world_y: int, erase: bool = False) -> None:
        assert self.world is not None
        tool = TOOLS[self.current_tool_key]
        self.world.paint_world(
            world_x,
            world_y,
            self.brush_radius,
            None if erase or tool.family_id is None else tool.family_id,
            None if erase or tool.family_id is None else tool.variant_id,
            overrides=dict(tool.overrides),
        )

    def _pan_camera(self, dx: int, dy: int) -> None:
        assert self.world is not None
        self.world.pan_camera(dx, dy)
        self._upload_frame()

    def tick(self, dt: float) -> None:
        assert self.world is not None
        self.last_tick_dt = dt
        if not self.paused:
            substep_dt = dt / self.steps_per_tick
            for _ in range(self.steps_per_tick):
                self.world.step(substep_dt)
        self.world.service_background_io()
        self._upload_frame()
        self._refresh_overlay()

    def on_draw(self) -> None:
        assert self.texture is not None
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
        world_x, world_y = self._screen_to_world(x, y)
        self._paint(world_x, world_y, erase=button == mouse.RIGHT)
        self._upload_frame()

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        del dx, dy, modifiers
        world_x, world_y = self._screen_to_world(x, y)
        erase = bool(buttons & mouse.RIGHT)
        draw = bool(buttons & mouse.LEFT)
        if draw or erase:
            self._paint(world_x, world_y, erase=erase)
            self._upload_frame()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        del modifiers
        if symbol in TOOLS:
            self.current_tool_key = symbol
        elif symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.N:
            assert self.world is not None
            single_step_dt = (1 / 60.0) / self.steps_per_tick
            for _ in range(self.steps_per_tick):
                self.world.step(single_step_dt)
            self._upload_frame()
        elif symbol == key.R:
            self._rebuild_world(populate_scene=True)
            self._upload_frame()
        elif symbol == key.C:
            self._rebuild_world(populate_scene=False)
            self._upload_frame()
        elif symbol == key.T:
            current_index = VIEW_MODE_ORDER.index(self.view_mode)
            self.view_mode = VIEW_MODE_ORDER[(current_index + 1) % len(VIEW_MODE_ORDER)]
            self._upload_frame()
        elif symbol == key.B:
            assert self.world is not None
            self.liquid_brownian_enabled = not self.liquid_brownian_enabled
            self.world.set_liquid_brownian_enabled(self.liquid_brownian_enabled)
        elif symbol == key.I:
            assert self.world is not None
            self.blocked_impulse_enabled = not self.blocked_impulse_enabled
            self.world.set_blocked_impulse_enabled(self.blocked_impulse_enabled)
        elif symbol == key.F:
            assert self.world is not None
            self.directional_fallback_enabled = not self.directional_fallback_enabled
            self.world.set_directional_fallback_enabled(self.directional_fallback_enabled)
        elif symbol == key.MINUS:
            self.steps_per_tick = max(1, self.steps_per_tick - 1)
        elif symbol == key.EQUAL:
            self.steps_per_tick = min(16, self.steps_per_tick + 1)
        elif symbol == key.BRACKETLEFT:
            self.brush_radius = max(0, self.brush_radius - 1)
        elif symbol == key.BRACKETRIGHT:
            self.brush_radius = min(8, self.brush_radius + 1)
        elif symbol in {key.A, key.LEFT}:
            self._pan_camera(-self.camera_pan_cells, 0)
        elif symbol in {key.D, key.RIGHT}:
            self._pan_camera(self.camera_pan_cells, 0)
        elif symbol in {key.W, key.UP}:
            self._pan_camera(0, -self.camera_pan_cells)
        elif symbol in {key.S, key.DOWN}:
            self._pan_camera(0, self.camera_pan_cells)
        elif symbol == key.ESCAPE:
            self.close()
            return
        self._refresh_overlay()


def run_demo(
    *,
    grid_width: int = 160,
    grid_height: int = 96,
    world_width: int | None = None,
    world_height: int | None = None,
    halo_cells: int = DEFAULT_HALO_CELLS,
    page_shift_cells: int = DEFAULT_PAGE_SHIFT_CELLS,
    cell_scale: int = 8,
    window_width: int | None = None,
    window_height: int | None = None,
    simulation_substeps: int = DEFAULT_SIMULATION_SUBSTEPS,
    liquid_brownian_enabled: bool = True,
    blocked_impulse_enabled: bool = True,
    directional_fallback_enabled: bool = True,
    vsync: bool = True,
) -> None:
    window = VoxelDemoWindow(
        grid_width=grid_width,
        grid_height=grid_height,
        world_width=world_width,
        world_height=world_height,
        halo_cells=halo_cells,
        page_shift_cells=page_shift_cells,
        cell_scale=cell_scale,
        window_width=window_width,
        window_height=window_height,
        simulation_substeps=simulation_substeps,
        liquid_brownian_enabled=liquid_brownian_enabled,
        blocked_impulse_enabled=blocked_impulse_enabled,
        directional_fallback_enabled=directional_fallback_enabled,
        vsync=vsync,
    )
    window.set_minimum_size(200, 150)
    pyglet.app.run()
