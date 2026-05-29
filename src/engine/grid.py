from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .atmosphere import default_ambient_air_temperature_for_row
from .types import CellFlag, CellState, MaterialRegistry, SUPPORT_SOURCE_VALUE


@dataclass
class Grid:
    width: int
    height: int
    step_id: int = 0
    liquid_brownian_enabled: bool = True
    blocked_impulse_enabled: bool = True
    directional_fallback_enabled: bool = True
    directional_fallback_angle_limit_degrees: float = 45.0
    cells: list[CellState] = field(default_factory=list)
    scratch: list[CellState] = field(default_factory=list)
    external_support_anchors: list[bool] = field(default_factory=list)
    pressure: list[float] = field(default_factory=list)
    source_force_x: list[float] = field(default_factory=list)
    source_force_y: list[float] = field(default_factory=list)
    prev_source_force_x: list[float] = field(default_factory=list)
    prev_source_force_y: list[float] = field(default_factory=list)
    force_wave_x: list[float] = field(default_factory=list)
    force_wave_y: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = self.width * self.height
        if not self.cells:
            self.cells = [
                CellState(temperature=default_ambient_air_temperature_for_row(self.height, y))
                for y in range(self.height)
                for _ in range(self.width)
            ]
        if not self.scratch:
            self.scratch = [
                CellState(temperature=default_ambient_air_temperature_for_row(self.height, y))
                for y in range(self.height)
                for _ in range(self.width)
            ]
        if not self.pressure:
            self.pressure = [1.0 for _ in range(expected)]
        if not self.external_support_anchors:
            self.external_support_anchors = [False for _ in range(expected)]
        if not self.source_force_x:
            self.source_force_x = [0.0 for _ in range(expected)]
        if not self.source_force_y:
            self.source_force_y = [0.0 for _ in range(expected)]
        if not self.prev_source_force_x:
            self.prev_source_force_x = [0.0 for _ in range(expected)]
        if not self.prev_source_force_y:
            self.prev_source_force_y = [0.0 for _ in range(expected)]
        if not self.force_wave_x:
            self.force_wave_x = [0.0 for _ in range(expected)]
        if not self.force_wave_y:
            self.force_wave_y = [0.0 for _ in range(expected)]

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int, *, use_scratch: bool = False) -> CellState:
        cells = self.scratch if use_scratch else self.cells
        return cells[self.index(x, y)]

    def set_cell(self, x: int, y: int, cell: CellState, *, use_scratch: bool = False) -> None:
        cells = self.scratch if use_scratch else self.cells
        cells[self.index(x, y)] = cell

    def copy_cells_to_scratch(self) -> None:
        self.scratch = [cell.copy() for cell in self.cells]

    def clear_scratch(self) -> None:
        self.scratch = [
            CellState(temperature=default_ambient_air_temperature_for_row(self.height, y))
            for y in range(self.height)
            for _ in range(self.width)
        ]

    def clear_external_support_anchors(self) -> None:
        self.external_support_anchors = [False for _ in range(self.width * self.height)]

    def swap_buffers(self) -> None:
        self.cells, self.scratch = self.scratch, self.cells


def create_grid(width: int, height: int) -> Grid:
    return Grid(width=width, height=height)


def inject_cells(
    grid: Grid,
    brush_or_cells: dict[str, int] | Iterable[tuple[int, int]],
    family_id: str,
    variant_id: str,
    overrides: dict[str, object] | None = None,
    registry: MaterialRegistry | None = None,
) -> None:
    from .materials import build_material_registry
    registry = registry or build_material_registry()
    variant = registry.variant(family_id, variant_id)
    base = CellState(
        family_id=family_id,
        variant_id=variant_id,
        temperature=variant.base_temperature,
    )
    overrides = overrides or {}
    for key, value in overrides.items():
        setattr(base, key, value)

    targets: list[tuple[int, int]] = []
    if isinstance(brush_or_cells, dict):
        center_x = int(brush_or_cells["x"])
        center_y = int(brush_or_cells["y"])
        radius = int(brush_or_cells.get("radius", 0))
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                if grid.in_bounds(x, y) and (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    targets.append((x, y))
    else:
        targets.extend(brush_or_cells)

    for x, y in targets:
        if not grid.in_bounds(x, y):
            continue
        cell = base.copy()
        if family_id == "empty" and variant_id == "empty" and "temperature" not in overrides:
            cell.temperature = default_ambient_air_temperature_for_row(grid.height, y)
        if cell.flags & CellFlag.FIXPOINT:
            cell.support_value = SUPPORT_SOURCE_VALUE
        grid.set_cell(x, y, cell)
