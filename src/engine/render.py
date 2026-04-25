from __future__ import annotations

from enum import Enum
from math import log1p

from .grid import Grid
from .support import SUPPORT_SOURCE_VALUE
from .types import CellFlag, MaterialRegistry


class DebugViewMode(str, Enum):
    MATERIAL = "material"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _lerp_channel(start: float, end: float, factor: float) -> int:
    return _clamp_channel(start + (end - start) * factor)


def _temperature_rgba(temperature: float) -> tuple[int, int, int, int]:
    stops = (
        (-40.0, (20, 28, 70)),
        (0.0, (50, 120, 220)),
        (20.0, (70, 180, 255)),
        (100.0, (90, 220, 140)),
        (300.0, (240, 220, 80)),
        (800.0, (255, 120, 40)),
        (1400.0, (255, 245, 235)),
    )
    if temperature <= stops[0][0]:
        red, green, blue = stops[0][1]
        return (red, green, blue, 255)
    if temperature >= stops[-1][0]:
        red, green, blue = stops[-1][1]
        return (red, green, blue, 255)

    for index in range(len(stops) - 1):
        left_temp, left_color = stops[index]
        right_temp, right_color = stops[index + 1]
        if left_temp <= temperature <= right_temp:
            factor = (temperature - left_temp) / max(0.001, right_temp - left_temp)
            return (
                _lerp_channel(left_color[0], right_color[0], factor),
                _lerp_channel(left_color[1], right_color[1], factor),
                _lerp_channel(left_color[2], right_color[2], factor),
                255,
            )

    return (255, 255, 255, 255)


def _pressure_rgba(pressure: float) -> tuple[int, int, int, int]:
    excess_pressure = max(0.0, pressure - 1.0)
    factor = min(1.0, log1p(excess_pressure) / log1p(128.0))
    stops = (
        (0.0, (8, 12, 28)),
        (0.15, (25, 70, 150)),
        (0.35, (45, 150, 235)),
        (0.6, (100, 220, 170)),
        (0.82, (245, 210, 70)),
        (1.0, (255, 245, 235)),
    )
    if factor <= stops[0][0]:
        red, green, blue = stops[0][1]
        return (red, green, blue, 255)

    for index in range(len(stops) - 1):
        left_factor, left_color = stops[index]
        right_factor, right_color = stops[index + 1]
        if left_factor <= factor <= right_factor:
            local_factor = (factor - left_factor) / max(0.001, right_factor - left_factor)
            return (
                _lerp_channel(left_color[0], right_color[0], local_factor),
                _lerp_channel(left_color[1], right_color[1], local_factor),
                _lerp_channel(left_color[2], right_color[2], local_factor),
                255,
            )

    red, green, blue = stops[-1][1]
    return (red, green, blue, 255)


def _material_rgba(grid: Grid, registry: MaterialRegistry, x: int, y: int) -> tuple[int, int, int, int]:
    cell = grid.get_cell(x, y)
    if cell.is_empty:
        return (10, 12, 16, 255)

    variant = registry.variant(cell.family_id, cell.variant_id)
    red, green, blue = variant.render_color

    if cell.family_id == "fire":
        pulse = min(1.0, cell.age / 6.0)
        red = _clamp_channel(red + 25 * (1.0 - pulse))
        green = _clamp_channel(green - 20 * pulse)
        blue = _clamp_channel(blue - 10 * pulse)

    if cell.flags & CellFlag.FIXPOINT:
        red = _clamp_channel(red + 35)
        green = _clamp_channel(green + 35)

    if variant.support_bearing:
        support_mix = min(1.0, max(0.0, cell.support_value / SUPPORT_SOURCE_VALUE))
        red = _clamp_channel(red * (0.85 + support_mix * 0.15))
        green = _clamp_channel(green * (0.85 + support_mix * 0.15))
        blue = _clamp_channel(blue * (0.85 + support_mix * 0.15))

    integrity_tint = max(0.2, min(1.0, cell.integrity))
    red = _clamp_channel(red * integrity_tint + 30 * (1.0 - integrity_tint))
    green = _clamp_channel(green * integrity_tint)
    blue = _clamp_channel(blue * integrity_tint)

    if cell.temperature > 120.0 and cell.family_id != "fire" and not cell.is_empty:
        heat = min(1.0, (cell.temperature - 120.0) / 900.0)
        red = _clamp_channel(red + 120 * heat)
        green = _clamp_channel(green + 30 * heat)

    return (red, green, blue, 255)


def _cell_rgba(
    grid: Grid,
    registry: MaterialRegistry,
    x: int,
    y: int,
    *,
    view_mode: DebugViewMode,
) -> tuple[int, int, int, int]:
    if view_mode == DebugViewMode.TEMPERATURE:
        return _temperature_rgba(grid.get_cell(x, y).temperature)
    if view_mode == DebugViewMode.PRESSURE:
        return _pressure_rgba(grid.pressure[grid.index(x, y)])
    return _material_rgba(grid, registry, x, y)


def build_rgba_frame(
    grid: Grid,
    registry: MaterialRegistry,
    *,
    view_mode: DebugViewMode = DebugViewMode.MATERIAL,
) -> bytes:
    """Return a tightly packed RGBA framebuffer for the current grid state."""
    frame = bytearray(grid.width * grid.height * 4)
    index = 0
    for y in range(grid.height - 1, -1, -1):
        for x in range(grid.width):
            red, green, blue, alpha = _cell_rgba(grid, registry, x, y, view_mode=view_mode)
            frame[index : index + 4] = bytes((red, green, blue, alpha))
            index += 4
    return bytes(frame)
