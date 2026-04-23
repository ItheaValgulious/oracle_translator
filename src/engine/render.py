from __future__ import annotations

from .grid import Grid
from .types import CellFlag, MaterialRegistry, SimKind


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _cell_rgba(grid: Grid, registry: MaterialRegistry, x: int, y: int) -> tuple[int, int, int, int]:
    cell = grid.get_cell(x, y)
    if cell.is_empty:
        return (10, 12, 16, 255)

    variant = registry.variant(cell.family_id, cell.variant_id)
    red, green, blue = variant.render_color

    if variant.sim_kind == SimKind.FIRE:
        pulse = min(1.0, cell.age / 6.0)
        red = _clamp_channel(red + 25 * (1.0 - pulse))
        green = _clamp_channel(green - 20 * pulse)
        blue = _clamp_channel(blue - 10 * pulse)

    if cell.flags & CellFlag.FIXPOINT:
        red = _clamp_channel(red + 35)
        green = _clamp_channel(green + 35)

    if variant.support_bearing:
        support_mix = min(1.0, max(0.0, cell.support_value))
        red = _clamp_channel(red * (0.85 + support_mix * 0.15))
        green = _clamp_channel(green * (0.85 + support_mix * 0.15))
        blue = _clamp_channel(blue * (0.85 + support_mix * 0.15))

    integrity_tint = max(0.2, min(1.0, cell.integrity))
    red = _clamp_channel(red * integrity_tint + 30 * (1.0 - integrity_tint))
    green = _clamp_channel(green * integrity_tint)
    blue = _clamp_channel(blue * integrity_tint)

    if cell.temperature > 120.0 and variant.sim_kind not in {SimKind.FIRE, SimKind.EMPTY}:
        heat = min(1.0, (cell.temperature - 120.0) / 900.0)
        red = _clamp_channel(red + 120 * heat)
        green = _clamp_channel(green + 30 * heat)

    return (red, green, blue, 255)


def build_rgba_frame(grid: Grid, registry: MaterialRegistry) -> bytes:
    """Return a tightly packed RGBA framebuffer for the current grid state."""
    frame = bytearray(grid.width * grid.height * 4)
    index = 0
    for y in range(grid.height - 1, -1, -1):
        for x in range(grid.width):
            red, green, blue, alpha = _cell_rgba(grid, registry, x, y)
            frame[index : index + 4] = bytes((red, green, blue, alpha))
            index += 4
    return bytes(frame)
