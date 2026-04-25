from __future__ import annotations

from dataclasses import dataclass, field

from .types import CellState, empty_cell


@dataclass
class Grid:
    width: int
    height: int
    step_id: int = 0
    liquid_brownian_enabled: bool = True
    blocked_impulse_enabled: bool = True
    cells: list[CellState] = field(default_factory=list)
    scratch: list[CellState] = field(default_factory=list)
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
            self.cells = [empty_cell() for _ in range(expected)]
        if not self.scratch:
            self.scratch = [empty_cell() for _ in range(expected)]
        if not self.pressure:
            self.pressure = [1.0 for _ in range(expected)]
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
        self.scratch = [empty_cell() for _ in range(self.width * self.height)]

    def swap_buffers(self) -> None:
        self.cells, self.scratch = self.scratch, self.cells


def create_grid(width: int, height: int) -> Grid:
    return Grid(width=width, height=height)
