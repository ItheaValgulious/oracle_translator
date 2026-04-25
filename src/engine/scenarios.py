from __future__ import annotations

from .grid import Grid
from .materials import build_material_registry
from .sim import inject_cells
from .types import MaterialRegistry
from .types import CellFlag


def populate_demo_scene(grid: Grid, registry: MaterialRegistry | None = None) -> None:
    """Seed a small playground that shows support, liquids, heat, and collapse."""
    registry = registry or build_material_registry()
    width = grid.width
    height = grid.height

    floor_y = height - 3
    for x in range(width):
        inject_cells(grid, [(x, floor_y), (x, floor_y + 1), (x, floor_y + 2)], "stone", "stone_platform", registry=registry)

    left_anchor = max(2, width // 8)
    right_anchor = min(width - 3, width - width // 8)
    bridge_y = max(6, height // 2)

    inject_cells(grid, [(left_anchor, bridge_y)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT}, registry=registry)
    inject_cells(grid, [(right_anchor, bridge_y)], "stone", "stone_platform", {"flags": CellFlag.FIXPOINT}, registry=registry)
    for x in range(left_anchor + 1, right_anchor):
        inject_cells(grid, [(x, bridge_y)], "stone", "stone_platform", registry=registry)

    for y in range(bridge_y + 1, floor_y):
        inject_cells(grid, [(left_anchor, y), (right_anchor, y)], "stone", "stone_platform", registry=registry)

    water_pool_y = floor_y - 1
    for x in range(width // 5, width // 5 + width // 8):
        inject_cells(grid, [(x, water_pool_y)], "water", "water", registry=registry)

    tar_start = width // 2 + 4
    for x in range(tar_start, min(width - 2, tar_start + width // 10)):
        inject_cells(grid, [(x, floor_y - 1)], "tar", "tar_liquid", registry=registry)

    for x in range(max(2, width // 3), max(3, width // 3 + width // 12)):
        inject_cells(grid, [(x, bridge_y - 3)], "sand", "sand_powder", registry=registry)

    for x in range(right_anchor - 3, right_anchor + 1):
        inject_cells(grid, [(x, bridge_y - 1)], "water", "ice", registry=registry)

    inject_cells(grid, [(width // 2, floor_y - 1)], "poison", "poison_liquid", registry=registry)
    inject_cells(grid, [(width // 2 - 8, floor_y - 1)], "acid", "acid_liquid", registry=registry)
