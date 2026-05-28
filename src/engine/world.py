from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Callable

log = logging.getLogger(__name__)

from .atmosphere import default_ambient_air_temperature_for_row
from .grid import Grid, create_grid
from .render import DebugViewMode, build_rgba_frame
from .sim import inject_cells, step
from .types import CellFlag, CellState, MaterialRegistry, empty_cell

if TYPE_CHECKING:
    import moderngl


DEFAULT_WORLD_CHUNK_SIZE = 320
DEFAULT_HALO_CELLS = 32
DEFAULT_PAGE_SHIFT_CELLS = 16
DEFAULT_SAFETY_MARGIN_CELLS = 16
DEFAULT_IDLE_FLUSH_COOLDOWN_SECONDS = 0.2
DEFAULT_IDLE_FLUSH_SERVICE_INTERVAL_SECONDS = 300.0
DEFAULT_PENDING_WRITEBACK_LIMIT = 256
DEFAULT_GPU_WRITEBACK_SLICE_CELL_BUDGET = 64

NEIGHBORS_8 = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


@dataclass(frozen=True)
class WorldRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def intersection(self, other: WorldRect) -> WorldRect | None:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return WorldRect(left, top, right - left, bottom - top)


@dataclass
class GridSlice:
    width: int
    height: int
    cells: list[CellState] = field(default_factory=list)
    anchored_support_mask: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = self.width * self.height
        if not self.cells:
            self.cells = [CellState() for _ in range(expected)]
        if not self.anchored_support_mask:
            self.anchored_support_mask = [False for _ in range(expected)]

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def get_cell(self, x: int, y: int) -> CellState:
        return self.cells[self.index(x, y)]

    def set_cell(self, x: int, y: int, cell: CellState) -> None:
        self.cells[self.index(x, y)] = cell


@dataclass
class _WorldChunk:
    cells: dict[int, CellState] = field(default_factory=dict)
    anchored_support_indices: set[int] = field(default_factory=set)


@dataclass
class _PendingGpuWriteback:
    rect: WorldRect
    staged_region: object
    snapshot_region: GridSlice | None = None
    major_axis_offset: int = 0


@dataclass
class PagingStats:
    shift_count: int = 0
    total_shift_seconds: float = 0.0
    last_shift_seconds: float = 0.0
    max_shift_seconds: float = 0.0
    last_evict_stage_seconds: float = 0.0
    last_overlap_copy_seconds: float = 0.0
    last_overlap_transient_copy_seconds: float = 0.0
    last_incoming_load_seconds: float = 0.0
    last_incoming_transient_clear_seconds: float = 0.0
    last_anchor_build_seconds: float = 0.0
    last_anchor_upload_seconds: float = 0.0
    max_evict_stage_seconds: float = 0.0
    max_overlap_copy_seconds: float = 0.0
    max_overlap_transient_copy_seconds: float = 0.0
    max_incoming_load_seconds: float = 0.0
    max_incoming_transient_clear_seconds: float = 0.0
    max_anchor_build_seconds: float = 0.0
    max_anchor_upload_seconds: float = 0.0


@dataclass(frozen=True)
class _AnchorRegionUpdate:
    local_x: int
    local_y: int
    width: int
    height: int
    values: list[bool]


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _default_world_cell(world_height: int, y: int) -> CellState:
    return CellState(temperature=default_ambient_air_temperature_for_row(world_height, y))


def _is_default_empty_cell(cell: CellState, world_height: int, y: int) -> bool:
    ambient = default_ambient_air_temperature_for_row(world_height, y)
    return (
        cell.family_id == "empty"
        and cell.variant_id == "empty"
        and cell.vel_x == 0.0
        and cell.vel_y == 0.0
        and cell.blocked_x == 0.0
        and cell.blocked_y == 0.0
        and cell.temperature == ambient
        and cell.support_value == 0.0
        and cell.integrity == 1.0
        and cell.generation == 0
        and cell.age == 0.0
        and cell.flags == CellFlag.NONE
    )


def _copy_runtime_flags(target: Grid, source: Grid) -> None:
    target.step_id = source.step_id
    target.liquid_brownian_enabled = source.liquid_brownian_enabled
    target.blocked_impulse_enabled = source.blocked_impulse_enabled
    target.directional_fallback_enabled = source.directional_fallback_enabled
    target.directional_fallback_angle_limit_degrees = source.directional_fallback_angle_limit_degrees


def _capture_grid_region(grid: Grid, x: int, y: int, width: int, height: int) -> GridSlice:
    cells = [
        grid.get_cell(x + local_x, y + local_y).copy()
        for local_y in range(height)
        for local_x in range(width)
    ]
    anchors = [
        bool(grid.external_support_anchors[grid.index(x + local_x, y + local_y)])
        for local_y in range(height)
        for local_x in range(width)
    ]
    return GridSlice(width=width, height=height, cells=cells, anchored_support_mask=anchors)


def _write_grid_region(grid: Grid, x: int, y: int, region: GridSlice) -> None:
    set_cell = grid.set_cell
    for local_y in range(region.height):
        row_offset = local_y * region.width
        gy = y + local_y
        for local_x in range(region.width):
            set_cell(x + local_x, gy, region.cells[row_offset + local_x])


def _slice_grid_region(region: GridSlice, x: int, y: int, width: int, height: int) -> GridSlice:
    cells = [
        region.get_cell(x + local_x, y + local_y).copy()
        for local_y in range(height)
        for local_x in range(width)
    ]
    anchors = [
        bool(region.anchored_support_mask[region.index(x + local_x, y + local_y)])
        for local_y in range(height)
        for local_x in range(width)
    ]
    return GridSlice(width=width, height=height, cells=cells, anchored_support_mask=anchors)


def _rect_difference(rect: WorldRect, overlap: WorldRect | None) -> list[WorldRect]:
    if overlap is None:
        return [] if rect.is_empty else [rect]

    parts: list[WorldRect] = []
    if overlap.x > rect.x:
        parts.append(WorldRect(rect.x, rect.y, overlap.x - rect.x, rect.height))
    if overlap.right < rect.right:
        parts.append(WorldRect(overlap.right, rect.y, rect.right - overlap.right, rect.height))
    if overlap.y > rect.y:
        parts.append(WorldRect(overlap.x, rect.y, overlap.width, overlap.y - rect.y))
    if overlap.bottom < rect.bottom:
        parts.append(WorldRect(overlap.x, overlap.bottom, overlap.width, rect.bottom - overlap.bottom))
    return [part for part in parts if not part.is_empty]


class WorldChunkStore:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        chunk_size: int = DEFAULT_WORLD_CHUNK_SIZE,
        seed: int = 0,
        chunk_generator: Callable[[WorldChunkStore, int, int, int, int], None] | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("World dimensions must be positive.")
        self.width = int(width)
        self.height = int(height)
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.chunk_generator = chunk_generator
        self._chunks: dict[tuple[int, int], _WorldChunk] = {}
        self._generated_chunks: set[tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _chunk_coord(self, x: int, y: int) -> tuple[int, int]:
        return (x // self.chunk_size, y // self.chunk_size)

    def _chunk_local_index(self, x: int, y: int) -> int:
        return (y % self.chunk_size) * self.chunk_size + (x % self.chunk_size)

    def _chunk(self, chunk_x: int, chunk_y: int, *, create: bool) -> _WorldChunk | None:
        key = (chunk_x, chunk_y)
        chunk = self._chunks.get(key)
        if chunk is None and create:
            chunk = _WorldChunk()
            self._chunks[key] = chunk
        return chunk

    def _ensure_chunk_generated(self, chunk_x: int, chunk_y: int) -> None:
        key = (chunk_x, chunk_y)
        if key in self._generated_chunks:
            return
        self._generated_chunks.add(key)
        if self.chunk_generator is not None:
            log.info("[world] generating chunk (%d,%d) for the first time", chunk_x, chunk_y)
            self.chunk_generator(self, chunk_x, chunk_y, self.chunk_size, self.seed)
            log.info("[world] chunk (%d,%d) generation complete, total_generated=%d",
                     chunk_x, chunk_y, len(self._generated_chunks))

    def _cell_ref(self, x: int, y: int) -> CellState | None:
        if not self.in_bounds(x, y):
            return None
        chunk = self._chunk(*self._chunk_coord(x, y), create=False)
        if chunk is None:
            return None
        return chunk.cells.get(self._chunk_local_index(x, y))

    def get_cell(self, x: int, y: int) -> CellState:
        if not self.in_bounds(x, y):
            raise IndexError("World coordinate is out of bounds.")
        chunk_x, chunk_y = self._chunk_coord(x, y)
        chunk = self._chunk(chunk_x, chunk_y, create=False)
        if chunk is None:
            self._ensure_chunk_generated(chunk_x, chunk_y)
            chunk = self._chunk(chunk_x, chunk_y, create=False)
        if chunk is None:
            return _default_world_cell(self.height, y)
        cell = chunk.cells.get(self._chunk_local_index(x, y))
        return cell.copy() if cell is not None else _default_world_cell(self.height, y)

    def set_cell(self, x: int, y: int, cell: CellState) -> None:
        if not self.in_bounds(x, y):
            return
        chunk_x, chunk_y = self._chunk_coord(x, y)
        chunk = self._chunk(chunk_x, chunk_y, create=True)
        assert chunk is not None
        local_index = self._chunk_local_index(x, y)
        if _is_default_empty_cell(cell, self.height, y):
            chunk.cells.pop(local_index, None)
            chunk.anchored_support_indices.discard(local_index)
            if not chunk.cells and not chunk.anchored_support_indices:
                self._chunks.pop((chunk_x, chunk_y), None)
            return
        chunk.cells[local_index] = cell.copy()

    def anchored_support_at(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        chunk = self._chunk(*self._chunk_coord(x, y), create=False)
        if chunk is None:
            return False
        return self._chunk_local_index(x, y) in chunk.anchored_support_indices

    def _chunk_rect_local_bounds(self, rect: WorldRect, chunk_x: int, chunk_y: int) -> tuple[int, int, int, int] | None:
        chunk_world_x = chunk_x * self.chunk_size
        chunk_world_y = chunk_y * self.chunk_size
        local_x0 = max(0, rect.x - chunk_world_x)
        local_y0 = max(0, rect.y - chunk_world_y)
        local_x1 = min(self.chunk_size, rect.right - chunk_world_x)
        local_y1 = min(self.chunk_size, rect.bottom - chunk_world_y)
        if local_x1 <= local_x0 or local_y1 <= local_y0:
            return None
        return (local_x0, local_y0, local_x1, local_y1)

    def rect_has_stored_cells(self, rect: WorldRect) -> bool:
        clipped = rect.intersection(WorldRect(0, 0, self.width, self.height))
        if clipped is None:
            return False
        chunk_min_x = clipped.x // self.chunk_size
        chunk_max_x = (clipped.right - 1) // self.chunk_size
        chunk_min_y = clipped.y // self.chunk_size
        chunk_max_y = (clipped.bottom - 1) // self.chunk_size
        # Ensure chunks are generated before checking for stored cells
        for chunk_y in range(chunk_min_y, chunk_max_y + 1):
            for chunk_x in range(chunk_min_x, chunk_max_x + 1):
                self._ensure_chunk_generated(chunk_x, chunk_y)
        for chunk_y in range(chunk_min_y, chunk_max_y + 1):
            for chunk_x in range(chunk_min_x, chunk_max_x + 1):
                chunk = self._chunk(chunk_x, chunk_y, create=False)
                if chunk is None or not chunk.cells:
                    continue
                local_bounds = self._chunk_rect_local_bounds(clipped, chunk_x, chunk_y)
                if local_bounds is None:
                    continue
                local_x0, local_y0, local_x1, local_y1 = local_bounds
                for local_index in chunk.cells:
                    cell_local_x = local_index % self.chunk_size
                    cell_local_y = local_index // self.chunk_size
                    if local_x0 <= cell_local_x < local_x1 and local_y0 <= cell_local_y < local_y1:
                        return True
        return False

    def read_rect(self, world_x: int, world_y: int, width: int, height: int) -> GridSlice:
        total = width * height
        cells: list[CellState | None] = [None] * total
        anchors = [False for _ in range(total)]
        rect = WorldRect(world_x, world_y, width, height)
        clipped = rect.intersection(WorldRect(0, 0, self.width, self.height))

        if clipped is None:
            return GridSlice(width=width, height=height, cells=cells, anchored_support_mask=anchors)

        chunk_min_x = clipped.x // self.chunk_size
        chunk_max_x = (clipped.right - 1) // self.chunk_size
        chunk_min_y = clipped.y // self.chunk_size
        chunk_max_y = (clipped.bottom - 1) // self.chunk_size
        for chunk_y in range(chunk_min_y, chunk_max_y + 1):
            for chunk_x in range(chunk_min_x, chunk_max_x + 1):
                self._ensure_chunk_generated(chunk_x, chunk_y)
        for chunk_y in range(chunk_min_y, chunk_max_y + 1):
            for chunk_x in range(chunk_min_x, chunk_max_x + 1):
                chunk = self._chunk(chunk_x, chunk_y, create=False)
                if chunk is None or (not chunk.cells and not chunk.anchored_support_indices):
                    continue
                local_bounds = self._chunk_rect_local_bounds(clipped, chunk_x, chunk_y)
                if local_bounds is None:
                    continue
                local_x0, local_y0, local_x1, local_y1 = local_bounds
                chunk_world_x = chunk_x * self.chunk_size
                chunk_world_y = chunk_y * self.chunk_size

                if chunk.cells:
                    for local_index, cell in chunk.cells.items():
                        cell_local_x = local_index % self.chunk_size
                        cell_local_y = local_index // self.chunk_size
                        if not (local_x0 <= cell_local_x < local_x1 and local_y0 <= cell_local_y < local_y1):
                            continue
                        rect_local_x = chunk_world_x + cell_local_x - world_x
                        rect_local_y = chunk_world_y + cell_local_y - world_y
                        cells[rect_local_y * width + rect_local_x] = cell.copy()

                if chunk.anchored_support_indices:
                    for local_index in chunk.anchored_support_indices:
                        cell_local_x = local_index % self.chunk_size
                        cell_local_y = local_index // self.chunk_size
                        if not (local_x0 <= cell_local_x < local_x1 and local_y0 <= cell_local_y < local_y1):
                            continue
                        rect_local_x = chunk_world_x + cell_local_x - world_x
                        rect_local_y = chunk_world_y + cell_local_y - world_y
                        anchors[rect_local_y * width + rect_local_x] = True

        # Fill in default cells for empty (None) entries
        for local_y in range(height):
            cell_y = world_y + local_y
            base = local_y * width
            if 0 <= cell_y < self.height:
                ambient = default_ambient_air_temperature_for_row(self.height, cell_y)
                default = CellState(temperature=ambient)
            else:
                default = CellState()
            for i in range(base, base + width):
                if cells[i] is None:
                    cells[i] = default

        return GridSlice(width=width, height=height, cells=cells, anchored_support_mask=anchors)

    def write_rect(self, world_x: int, world_y: int, region: GridSlice) -> None:
        rect = WorldRect(world_x, world_y, region.width, region.height)
        clipped = rect.intersection(WorldRect(0, 0, self.width, self.height))
        if clipped is None:
            return

        for target_y in range(clipped.y, clipped.bottom):
            region_y = target_y - world_y
            chunk_y = target_y // self.chunk_size
            chunk_local_y = target_y % self.chunk_size
            target_x = clipped.x
            region_x = clipped.x - world_x
            row_offset = region_y * region.width

            while target_x < clipped.right:
                chunk_x = target_x // self.chunk_size
                chunk_world_x = chunk_x * self.chunk_size
                segment_right = min(clipped.right, chunk_world_x + self.chunk_size)
                chunk = self._chunk(chunk_x, chunk_y, create=False)
                local_x_start = target_x - chunk_world_x
                base_local_index = chunk_local_y * self.chunk_size + local_x_start
                segment_width = segment_right - target_x

                for delta_x in range(segment_width):
                    index = row_offset + region_x + delta_x
                    local_index = base_local_index + delta_x
                    cell = region.cells[index]
                    if _is_default_empty_cell(cell, self.height, target_y):
                        if chunk is None:
                            continue
                        chunk.cells.pop(local_index, None)
                        chunk.anchored_support_indices.discard(local_index)
                        continue
                    if chunk is None:
                        chunk = self._chunk(chunk_x, chunk_y, create=True)
                        assert chunk is not None
                    chunk.cells[local_index] = cell.copy()
                    if region.anchored_support_mask[index]:
                        chunk.anchored_support_indices.add(local_index)
                    else:
                        chunk.anchored_support_indices.discard(local_index)

                if chunk is not None and not chunk.cells and not chunk.anchored_support_indices:
                    self._chunks.pop((chunk_x, chunk_y), None)

                target_x = segment_right
                region_x += segment_width

    def recompute_anchored_support(self, registry: MaterialRegistry) -> None:
        support_cells: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        visited: set[tuple[int, int]] = set()

        for (chunk_x, chunk_y), chunk in self._chunks.items():
            chunk.anchored_support_indices.clear()
            for local_index, cell in chunk.cells.items():
                local_x = local_index % self.chunk_size
                local_y = local_index // self.chunk_size
                world_x = chunk_x * self.chunk_size + local_x
                world_y = chunk_y * self.chunk_size + local_y
                variant = registry.variant(cell.family_id, cell.variant_id)
                if not variant.support_transmission:
                    continue
                coord = (world_x, world_y)
                support_cells.add(coord)
                if cell.flags & CellFlag.FIXPOINT:
                    queue.append(coord)
                    visited.add(coord)

        while queue:
            x, y = queue.popleft()
            for dx, dy in NEIGHBORS_8:
                neighbor = (x + dx, y + dy)
                if neighbor in visited or neighbor not in support_cells:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        for world_x, world_y in visited:
            chunk = self._chunk(*self._chunk_coord(world_x, world_y), create=False)
            if chunk is None:
                continue
            chunk.anchored_support_indices.add(self._chunk_local_index(world_x, world_y))

    def has_support_anchor_source(
        self,
        x: int,
        y: int,
        *,
        support_transmission_keys: set[tuple[str, str]],
    ) -> bool:
        cell = self._cell_ref(x, y)
        if cell is None:
            return False
        if (cell.family_id, cell.variant_id) not in support_transmission_keys:
            return False
        if cell.flags & CellFlag.FIXPOINT:
            return True
        chunk = self._chunk(*self._chunk_coord(x, y), create=False)
        if chunk is None:
            return False
        return self._chunk_local_index(x, y) in chunk.anchored_support_indices


class ActiveWorldWindow:
    def __init__(
        self,
        store: WorldChunkStore,
        registry: MaterialRegistry,
        *,
        viewport_width: int,
        viewport_height: int,
        halo_cells: int = DEFAULT_HALO_CELLS,
        page_shift_cells: int = DEFAULT_PAGE_SHIFT_CELLS,
        safety_margin_cells: int = DEFAULT_SAFETY_MARGIN_CELLS,
        idle_flush_cooldown_seconds: float = DEFAULT_IDLE_FLUSH_COOLDOWN_SECONDS,
        idle_flush_service_interval_seconds: float = DEFAULT_IDLE_FLUSH_SERVICE_INTERVAL_SECONDS,
        pending_writeback_limit: int = DEFAULT_PENDING_WRITEBACK_LIMIT,
        ctx: moderngl.Context | None = None,
        liquid_brownian_enabled: bool = True,
        blocked_impulse_enabled: bool = True,
        directional_fallback_enabled: bool = True,
        directional_fallback_angle_limit_degrees: float = 45.0,
        initial_camera_x: int | None = None,
        initial_camera_y: int | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.viewport_width = min(int(viewport_width), store.width)
        self.viewport_height = min(int(viewport_height), store.height)
        self._support_transmission_keys = {
            (family_id, variant_id)
            for family_id, family in registry.families.items()
            for variant_id, variant in family.variants.items()
            if variant.support_transmission
        }
        self.halo_cells = max(0, int(halo_cells))
        self.page_shift_cells = max(1, int(page_shift_cells))
        self.safety_margin_cells = max(0, int(safety_margin_cells))
        self.idle_flush_cooldown_seconds = max(0.0, float(idle_flush_cooldown_seconds))
        self.idle_flush_service_interval_seconds = max(0.0, float(idle_flush_service_interval_seconds))
        self.pending_writeback_limit = max(1, int(pending_writeback_limit))
        self.pending_writeback_slice_cell_budget = max(64, int(DEFAULT_GPU_WRITEBACK_SLICE_CELL_BUDGET))
        self.active_width = min(store.width, self.viewport_width + self.halo_cells * 2)
        self.active_height = min(store.height, self.viewport_height + self.halo_cells * 2)
        if initial_camera_x is not None:
            self.camera_x = _clamp(int(initial_camera_x), 0, max(0, store.width - self.viewport_width))
        else:
            self.camera_x = max(0, (store.width - self.viewport_width) // 2)
        if initial_camera_y is not None:
            self.camera_y = _clamp(int(initial_camera_y), 0, max(0, store.height - self.viewport_height))
        else:
            self.camera_y = max(0, (store.height - self.viewport_height) // 2)
        self.active_origin_x = _clamp(self.camera_x - self.halo_cells, 0, max(0, store.width - self.active_width))
        self.active_origin_y = _clamp(self.camera_y - self.halo_cells, 0, max(0, store.height - self.active_height))
        self.active_grid = create_grid(self.active_width, self.active_height)
        self.active_grid.liquid_brownian_enabled = bool(liquid_brownian_enabled)
        self.active_grid.blocked_impulse_enabled = bool(blocked_impulse_enabled)
        self.active_grid.directional_fallback_enabled = bool(directional_fallback_enabled)
        self.active_grid.directional_fallback_angle_limit_degrees = float(directional_fallback_angle_limit_degrees)
        self.gpu_simulator = None
        self._pending_gpu_writebacks: list[_PendingGpuWriteback] = []
        self._pending_flush_cooldown_steps = 0
        self._camera_idle_elapsed_seconds = self.idle_flush_cooldown_seconds
        self._last_background_io_flush_idle_seconds = 0.0
        self._camera_recently_moved = False
        self.paging_stats = PagingStats()
        log.info("[world] ActiveWorldWindow init: store=%dx%d viewport=%dx%d active=%dx%d halo=%d",
                 store.width, store.height, self.viewport_width, self.viewport_height,
                 self.active_width, self.active_height, self.halo_cells)
        log.info("[world] camera=(%d,%d) active_origin=(%d,%d)",
                 self.camera_x, self.camera_y, self.active_origin_x, self.active_origin_y)
        self._materialize_active_grid_from_store()
        if ctx is not None:
            from .gpu_backend import GpuSimulator

            self.gpu_simulator = GpuSimulator(ctx, self.active_grid, self.registry)
            self.gpu_simulator.set_external_support_anchors(self.active_grid.external_support_anchors)
            log.info("[world] GPU simulator created")
        else:
            log.info("[world] CPU-only mode (no ctx)")

    @property
    def world_width(self) -> int:
        return self.store.width

    @property
    def world_height(self) -> int:
        return self.store.height

    @property
    def viewport_rect(self) -> WorldRect:
        return WorldRect(self.camera_x, self.camera_y, self.viewport_width, self.viewport_height)

    @property
    def active_rect(self) -> WorldRect:
        return WorldRect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)

    @property
    def pending_writeback_count(self) -> int:
        return len(self._pending_gpu_writebacks)

    @property
    def pending_writeback_pressure_count(self) -> int:
        return max(0, len(self._pending_gpu_writebacks) - self.pending_writeback_limit)

    def _rematerialize_cpu_grid_for_shift(
        self,
        old_rect: WorldRect,
        new_rect: WorldRect,
        overlap: WorldRect | None,
        incoming_rects: list[WorldRect],
    ) -> None:
        """Incrementally rebuild CPU grid after a GPU window shift.

        Reuse the old grid's cell list to avoid creating 312K new CellState
        objects.  Shift the overlap region in-place, then fill incoming
        regions from the store.
        """
        t0 = perf_counter()
        grid = self.active_grid
        old_cells = grid.cells
        w = grid.width

        # Shift overlap region in-place within the same cell list.
        # Copy direction must avoid overwriting source cells.
        if overlap is not None:
            src_x = overlap.x - old_rect.x
            src_y = overlap.y - old_rect.y
            dst_x = overlap.x - new_rect.x
            dst_y = overlap.y - new_rect.y
            if dst_y <= src_y:
                for row in range(overlap.height):
                    src_off = (src_y + row) * w + src_x
                    dst_off = (dst_y + row) * w + dst_x
                    old_cells[dst_off:dst_off + overlap.width] = old_cells[src_off:src_off + overlap.width]
            else:
                for row in range(overlap.height - 1, -1, -1):
                    src_off = (src_y + row) * w + src_x
                    dst_off = (dst_y + row) * w + dst_x
                    old_cells[dst_off:dst_off + overlap.width] = old_cells[src_off:src_off + overlap.width]

        t_overlap = perf_counter()

        # Clear evicted regions and fill incoming from store
        evicted_rects = _rect_difference(old_rect, overlap if overlap else old_rect)
        _empty = empty_cell()
        for ev in evicted_rects:
            lx = ev.x - new_rect.x
            ly = ev.y - new_rect.y
            for row in range(ev.height):
                off = (ly + row) * w + lx
                old_cells[off:off + ev.width] = [_empty] * ev.width

        for incoming_rect in incoming_rects:
            incoming = self.store.read_rect(incoming_rect.x, incoming_rect.y, incoming_rect.width, incoming_rect.height)
            _write_grid_region(grid, incoming_rect.x - new_rect.x, incoming_rect.y - new_rect.y, incoming)

        t_incoming = perf_counter()

        grid.external_support_anchors = self._build_external_support_anchor_mask(
            WorldRect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        )
        t_anchors = perf_counter()
        log.info("[world] shift-materialized in %.1fms "
                 "(overlap=%.1f incoming=%.1f anchors=%.1f) incoming=%s",
                 (t_anchors - t0) * 1000,
                 (t_overlap - t0) * 1000,
                 (t_incoming - t_overlap) * 1000,
                 (t_anchors - t_incoming) * 1000,
                 [(r.width, r.height) for r in incoming_rects])

    def _materialize_active_grid_from_store(self) -> None:
        t0 = perf_counter()
        log.info("[world] materializing active grid: origin=(%d,%d) size=%dx%d",
                 self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        loaded = self.store.read_rect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        t1 = perf_counter()
        grid = create_grid(self.active_width, self.active_height)
        _copy_runtime_flags(grid, self.active_grid)
        _write_grid_region(grid, 0, 0, loaded)
        t2 = perf_counter()
        grid.external_support_anchors = self._build_external_support_anchor_mask(
            WorldRect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        )
        t3 = perf_counter()
        self.active_grid = grid
        non_empty = sum(1 for c in loaded.cells if c.family_id != "empty")
        elapsed = (perf_counter() - t0) * 1000
        log.info("[world] materialized in %.1fms (read=%.1fms write=%.1fms anchors=%.1fms), non_empty_cells=%d / %d total",
                 elapsed, (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, non_empty, len(loaded.cells))

    def _border_has_external_support_anchor(self, rect: WorldRect, local_x: int, local_y: int) -> bool:
        world_x = rect.x + local_x
        world_y = rect.y + local_y
        for dx, dy in NEIGHBORS_8:
            neighbor_x = world_x + dx
            neighbor_y = world_y + dy
            if rect.x <= neighbor_x < rect.right and rect.y <= neighbor_y < rect.bottom:
                continue
            if self.store.has_support_anchor_source(
                neighbor_x,
                neighbor_y,
                support_transmission_keys=self._support_transmission_keys,
            ):
                return True
        return False

    def _neighboring_anchor_positions(self, positions: set[int]) -> set[int]:
        anchored: set[int] = set()
        for position in positions:
            anchored.add(position - 1)
            anchored.add(position)
            anchored.add(position + 1)
        return anchored

    def _top_anchor_row(self, rect: WorldRect) -> list[bool]:
        if rect.y <= 0:
            return [False for _ in range(rect.width)]
        positions: set[int] = set()
        world_y = rect.y - 1
        for world_x in range(rect.x - 1, rect.right + 1):
            if self.store.has_support_anchor_source(
                world_x,
                world_y,
                support_transmission_keys=self._support_transmission_keys,
            ):
                positions.add(world_x)
        neighboring = self._neighboring_anchor_positions(positions)
        return [rect.x + local_x in neighboring for local_x in range(rect.width)]

    def _bottom_anchor_row(self, rect: WorldRect) -> list[bool]:
        if rect.bottom >= self.store.height:
            return [False for _ in range(rect.width)]
        positions: set[int] = set()
        world_y = rect.bottom
        for world_x in range(rect.x - 1, rect.right + 1):
            if self.store.has_support_anchor_source(
                world_x,
                world_y,
                support_transmission_keys=self._support_transmission_keys,
            ):
                positions.add(world_x)
        neighboring = self._neighboring_anchor_positions(positions)
        return [rect.x + local_x in neighboring for local_x in range(rect.width)]

    def _left_anchor_column(self, rect: WorldRect) -> list[bool]:
        if rect.x <= 0:
            return [False for _ in range(max(0, rect.height - 2))]
        positions: set[int] = set()
        world_x = rect.x - 1
        for world_y in range(rect.y, rect.bottom):
            if self.store.has_support_anchor_source(
                world_x,
                world_y,
                support_transmission_keys=self._support_transmission_keys,
            ):
                positions.add(world_y)
        if rect.y > 0:
            corner_y = rect.y - 1
            for world_x_candidate in (rect.x - 1, rect.x):
                if self.store.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_transmission_keys=self._support_transmission_keys,
                ):
                    positions.add(corner_y)
        if rect.bottom < self.store.height:
            corner_y = rect.bottom
            for world_x_candidate in (rect.x - 1, rect.x):
                if self.store.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_transmission_keys=self._support_transmission_keys,
                ):
                    positions.add(corner_y)
        neighboring = self._neighboring_anchor_positions(positions)
        return [rect.y + local_y in neighboring for local_y in range(1, rect.height - 1)]

    def _right_anchor_column(self, rect: WorldRect) -> list[bool]:
        if rect.right >= self.store.width:
            return [False for _ in range(max(0, rect.height - 2))]
        positions: set[int] = set()
        world_x = rect.right
        for world_y in range(rect.y, rect.bottom):
            if self.store.has_support_anchor_source(
                world_x,
                world_y,
                support_transmission_keys=self._support_transmission_keys,
            ):
                positions.add(world_y)
        if rect.y > 0:
            corner_y = rect.y - 1
            for world_x_candidate in (rect.right - 1, rect.right):
                if self.store.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_transmission_keys=self._support_transmission_keys,
                ):
                    positions.add(corner_y)
        if rect.bottom < self.store.height:
            corner_y = rect.bottom
            for world_x_candidate in (rect.right - 1, rect.right):
                if self.store.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_transmission_keys=self._support_transmission_keys,
                ):
                    positions.add(corner_y)
        neighboring = self._neighboring_anchor_positions(positions)
        return [rect.y + local_y in neighboring for local_y in range(1, rect.height - 1)]

    def _build_external_support_anchor_mask(self, rect: WorldRect) -> list[bool]:
        anchors = [False for _ in range(rect.width * rect.height)]
        if rect.width <= 0 or rect.height <= 0:
            return anchors
        top_y = 0
        bottom_y = rect.height - 1
        left_x = 0
        right_x = rect.width - 1
        for local_x in range(rect.width):
            anchors[top_y * rect.width + local_x] = self._border_has_external_support_anchor(rect, local_x, top_y)
            if bottom_y != top_y:
                anchors[bottom_y * rect.width + local_x] = self._border_has_external_support_anchor(rect, local_x, bottom_y)
        for local_y in range(1, bottom_y):
            anchors[local_y * rect.width + left_x] = self._border_has_external_support_anchor(rect, left_x, local_y)
            if right_x != left_x:
                anchors[local_y * rect.width + right_x] = self._border_has_external_support_anchor(rect, right_x, local_y)
        return anchors

    def _build_external_support_anchor_updates(self, rect: WorldRect) -> tuple[list[bool], list[_AnchorRegionUpdate]]:
        anchors = self.active_grid.external_support_anchors
        expected_size = rect.width * rect.height
        if len(anchors) != expected_size:
            anchors = [False for _ in range(expected_size)]
        updates: list[_AnchorRegionUpdate] = []
        if rect.width <= 0 or rect.height <= 0:
            return anchors, updates
        top_values = self._top_anchor_row(rect)
        previous_top_values = anchors[: rect.width]
        anchors[: rect.width] = top_values
        if top_values != previous_top_values:
            updates.append(_AnchorRegionUpdate(local_x=0, local_y=0, width=rect.width, height=1, values=top_values))
        if rect.height > 1:
            bottom_offset = (rect.height - 1) * rect.width
            bottom_values = self._bottom_anchor_row(rect)
            previous_bottom_values = anchors[bottom_offset : bottom_offset + rect.width]
            anchors[bottom_offset : bottom_offset + rect.width] = bottom_values
            if bottom_values != previous_bottom_values:
                updates.append(
                    _AnchorRegionUpdate(
                        local_x=0,
                        local_y=rect.height - 1,
                        width=rect.width,
                        height=1,
                        values=bottom_values,
                    )
                )
        if rect.height > 2:
            left_values = self._left_anchor_column(rect)
            previous_left_values = [anchors[local_y * rect.width] for local_y in range(1, rect.height - 1)]
            for local_y, value in enumerate(left_values, start=1):
                anchors[local_y * rect.width] = value
            if left_values != previous_left_values:
                updates.append(
                    _AnchorRegionUpdate(
                        local_x=0,
                        local_y=1,
                        width=1,
                        height=rect.height - 2,
                        values=left_values,
                    )
                )
            if rect.width > 1:
                right_values = self._right_anchor_column(rect)
                previous_right_values = [
                    anchors[local_y * rect.width + (rect.width - 1)]
                    for local_y in range(1, rect.height - 1)
                ]
                for local_y, value in enumerate(right_values, start=1):
                    anchors[local_y * rect.width + (rect.width - 1)] = value
                if right_values != previous_right_values:
                    updates.append(
                        _AnchorRegionUpdate(
                            local_x=rect.width - 1,
                            local_y=1,
                            width=1,
                            height=rect.height - 2,
                            values=right_values,
                        )
                    )
        return anchors, updates

    def _set_external_support_anchors(self) -> None:
        build_started_at = perf_counter()
        anchors, updates = self._build_external_support_anchor_updates(self.active_rect)
        build_elapsed = perf_counter() - build_started_at
        self.active_grid.external_support_anchors = anchors
        upload_started_at = perf_counter()
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_external_support_anchors(anchors)
        upload_elapsed = perf_counter() - upload_started_at
        self.paging_stats.last_anchor_build_seconds = build_elapsed
        self.paging_stats.max_anchor_build_seconds = max(self.paging_stats.max_anchor_build_seconds, build_elapsed)
        self.paging_stats.last_anchor_upload_seconds = upload_elapsed
        self.paging_stats.max_anchor_upload_seconds = max(self.paging_stats.max_anchor_upload_seconds, upload_elapsed)

    def _pending_overlap(self, rect: WorldRect) -> list[tuple[_PendingGpuWriteback, WorldRect]]:
        overlaps: list[tuple[_PendingGpuWriteback, WorldRect]] = []
        for pending in self._pending_gpu_writebacks:
            overlap = pending.rect.intersection(rect)
            if overlap is not None:
                overlaps.append((pending, overlap))
        return overlaps

    def _capture_frozen_support_snapshot(self, region: GridSlice) -> GridSlice:
        anchored_mask = [
            (cell.family_id, cell.variant_id) in self._support_transmission_keys
            and ((cell.flags & CellFlag.FIXPOINT) != 0 or cell.support_value > 0.0)
            for cell in region.cells
        ]
        region.anchored_support_mask = anchored_mask
        return region

    def _stage_evicted_region(self, rect: WorldRect) -> None:
        if self.gpu_simulator is None:
            return
        staged = self.gpu_simulator.stage_region(
            rect.x - self.active_origin_x,
            rect.y - self.active_origin_y,
            rect.width,
            rect.height,
        )
        self._pending_gpu_writebacks.append(_PendingGpuWriteback(rect=rect, staged_region=staged))

    def _flush_one_pending_gpu_writeback(self) -> bool:
        if self.gpu_simulator is None or not self._pending_gpu_writebacks:
            return False
        pending = self._pending_gpu_writebacks[0]
        flush_along_y = pending.rect.height >= pending.rect.width
        if flush_along_y:
            slice_width = pending.rect.width
            slice_height = min(
                pending.rect.height - pending.major_axis_offset,
                1,
            )
            src_x = 0
            src_y = pending.major_axis_offset
            world_x = pending.rect.x
            world_y = pending.rect.y + pending.major_axis_offset
            pending.major_axis_offset += slice_height
            flush_complete = pending.major_axis_offset >= pending.rect.height
        else:
            slice_width = min(
                pending.rect.width - pending.major_axis_offset,
                1,
            )
            slice_height = pending.rect.height
            src_x = pending.major_axis_offset
            src_y = 0
            world_x = pending.rect.x + pending.major_axis_offset
            world_y = pending.rect.y
            pending.major_axis_offset += slice_width
            flush_complete = pending.major_axis_offset >= pending.rect.width

        if pending.snapshot_region is None:
            region = self._capture_frozen_support_snapshot(
                self.gpu_simulator.read_staged_region(
                    pending.staged_region,
                    x=src_x,
                    y=src_y,
                    width=slice_width,
                    height=slice_height,
                )
            )
        else:
            region = _slice_grid_region(pending.snapshot_region, src_x, src_y, slice_width, slice_height)
        self.store.write_rect(world_x, world_y, region)
        if not flush_complete:
            return True
        self._pending_gpu_writebacks.pop(0)
        self.gpu_simulator.release_staged_region(pending.staged_region)
        return True

    def _load_incoming_rect_into_gpu_buffer(
        self,
        rect: WorldRect,
        *,
        target_buffer_index: int,
        target_origin_x: int,
        target_origin_y: int,
    ) -> None:
        assert self.gpu_simulator is not None
        remaining = [rect]
        consumed_pending: list[_PendingGpuWriteback] = []
        for pending, overlap in self._pending_overlap(rect):
            self.gpu_simulator.copy_from_staged_region(
                pending.staged_region,
                src_x=overlap.x - pending.rect.x,
                src_y=overlap.y - pending.rect.y,
                width=overlap.width,
                height=overlap.height,
                dst_x=overlap.x - target_origin_x,
                dst_y=overlap.y - target_origin_y,
                dst_buffer_index=target_buffer_index,
            )
            next_remaining: list[WorldRect] = []
            for candidate in remaining:
                next_remaining.extend(_rect_difference(candidate, candidate.intersection(overlap)))
            remaining = next_remaining
            if overlap == pending.rect:
                consumed_pending.append(pending)
        for missing in remaining:
            if self.store.rect_has_stored_cells(missing):
                self.gpu_simulator.write_store_rect(
                    self.store,
                    world_x=missing.x,
                    world_y=missing.y,
                    width=missing.width,
                    height=missing.height,
                    dst_x=missing.x - target_origin_x,
                    dst_y=missing.y - target_origin_y,
                    buffer_index=target_buffer_index,
                )
            else:
                self.gpu_simulator.fill_empty_region(
                    missing.x - target_origin_x,
                    missing.y - target_origin_y,
                    missing.width,
                    missing.height,
                    world_row_offset=missing.y,
                    world_height=self.store.height,
                    buffer_index=target_buffer_index,
                )
        for pending in consumed_pending:
            if pending in self._pending_gpu_writebacks:
                self._pending_gpu_writebacks.remove(pending)
                self.gpu_simulator.release_staged_region(pending.staged_region)

    def _capture_active_region(self, rect: WorldRect) -> GridSlice:
        local_x = rect.x - self.active_origin_x
        local_y = rect.y - self.active_origin_y
        if self.gpu_simulator is not None:
            return self.gpu_simulator.read_region(local_x, local_y, rect.width, rect.height)
        return _capture_grid_region(self.active_grid, local_x, local_y, rect.width, rect.height)

    def _shift_active_window(self, new_origin_x: int, new_origin_y: int) -> None:
        log.info("[world] shifting active window: (%d,%d) -> (%d,%d), camera=(%d,%d)",
                 self.active_origin_x, self.active_origin_y, new_origin_x, new_origin_y,
                 self.camera_x, self.camera_y)
        shift_started_at = perf_counter()
        self.paging_stats.last_evict_stage_seconds = 0.0
        self.paging_stats.last_overlap_copy_seconds = 0.0
        self.paging_stats.last_overlap_transient_copy_seconds = 0.0
        self.paging_stats.last_incoming_load_seconds = 0.0
        self.paging_stats.last_incoming_transient_clear_seconds = 0.0
        old_rect = self.active_rect
        new_rect = WorldRect(new_origin_x, new_origin_y, self.active_width, self.active_height)
        overlap = old_rect.intersection(new_rect)
        evicted_rects = _rect_difference(old_rect, overlap)
        incoming_rects = _rect_difference(new_rect, overlap)

        evict_started_at = perf_counter()
        if self.gpu_simulator is None:
            for evicted_rect in evicted_rects:
                region = self._capture_active_region(evicted_rect)
                self.store.write_rect(evicted_rect.x, evicted_rect.y, region)
            self.store.recompute_anchored_support(self.registry)
        else:
            for evicted_rect in evicted_rects:
                self._stage_evicted_region(evicted_rect)
        evict_elapsed = perf_counter() - evict_started_at
        self.paging_stats.last_evict_stage_seconds = evict_elapsed
        self.paging_stats.max_evict_stage_seconds = max(self.paging_stats.max_evict_stage_seconds, evict_elapsed)

        if self.gpu_simulator is None:
            next_grid = create_grid(self.active_width, self.active_height)
            _copy_runtime_flags(next_grid, self.active_grid)
            if overlap is not None:
                overlap_slice = _capture_grid_region(
                    self.active_grid,
                    overlap.x - old_rect.x,
                    overlap.y - old_rect.y,
                    overlap.width,
                    overlap.height,
                )
                _write_grid_region(next_grid, overlap.x - new_rect.x, overlap.y - new_rect.y, overlap_slice)
            for incoming_rect in incoming_rects:
                incoming = self.store.read_rect(incoming_rect.x, incoming_rect.y, incoming_rect.width, incoming_rect.height)
                _write_grid_region(next_grid, incoming_rect.x - new_rect.x, incoming_rect.y - new_rect.y, incoming)
            self.active_grid = next_grid
        else:
            target_buffer_index = 1 - self.gpu_simulator.front_index
            if overlap is not None:
                overlap_copy_started_at = perf_counter()
                self.gpu_simulator.copy_region(
                    overlap.x - old_rect.x,
                    overlap.y - old_rect.y,
                    overlap.width,
                    overlap.height,
                    overlap.x - new_rect.x,
                    overlap.y - new_rect.y,
                    dst_buffer_index=target_buffer_index,
                )
                overlap_copy_elapsed = perf_counter() - overlap_copy_started_at
                self.paging_stats.last_overlap_copy_seconds = overlap_copy_elapsed
                self.paging_stats.max_overlap_copy_seconds = max(
                    self.paging_stats.max_overlap_copy_seconds,
                    overlap_copy_elapsed,
                )
                overlap_transient_started_at = perf_counter()
                self.gpu_simulator.copy_transient_region(
                    overlap.x - old_rect.x,
                    overlap.y - old_rect.y,
                    overlap.width,
                    overlap.height,
                    overlap.x - new_rect.x,
                    overlap.y - new_rect.y,
                )
                overlap_transient_elapsed = perf_counter() - overlap_transient_started_at
                self.paging_stats.last_overlap_transient_copy_seconds = overlap_transient_elapsed
                self.paging_stats.max_overlap_transient_copy_seconds = max(
                    self.paging_stats.max_overlap_transient_copy_seconds,
                    overlap_transient_elapsed,
                )
            incoming_load_started_at = perf_counter()
            for incoming_rect in incoming_rects:
                self._load_incoming_rect_into_gpu_buffer(
                    incoming_rect,
                    target_buffer_index=target_buffer_index,
                    target_origin_x=new_rect.x,
                    target_origin_y=new_rect.y,
                )
            incoming_load_elapsed = perf_counter() - incoming_load_started_at
            self.paging_stats.last_incoming_load_seconds = incoming_load_elapsed
            self.paging_stats.max_incoming_load_seconds = max(
                self.paging_stats.max_incoming_load_seconds,
                incoming_load_elapsed,
            )
            transient_clear_started_at = perf_counter()
            for incoming_rect in incoming_rects:
                self.gpu_simulator.clear_region_transients(
                    incoming_rect.x - new_rect.x,
                    incoming_rect.y - new_rect.y,
                    incoming_rect.width,
                    incoming_rect.height,
                )
            transient_clear_elapsed = perf_counter() - transient_clear_started_at
            self.paging_stats.last_incoming_transient_clear_seconds = transient_clear_elapsed
            self.paging_stats.max_incoming_transient_clear_seconds = max(
                self.paging_stats.max_incoming_transient_clear_seconds,
                transient_clear_elapsed,
            )
            self.gpu_simulator.front_index = target_buffer_index

        self.active_origin_x = new_origin_x
        self.active_origin_y = new_origin_y
        # Re-materialize CPU grid from store so collision detection sees correct terrain
        if self.gpu_simulator is not None:
            self._rematerialize_cpu_grid_for_shift(old_rect, new_rect, overlap, incoming_rects)
        self._set_external_support_anchors()
        self._record_shift_stats(perf_counter() - shift_started_at)

    def _record_shift_stats(self, shift_seconds: float) -> None:
        self.paging_stats.shift_count += 1
        self.paging_stats.total_shift_seconds += shift_seconds
        self.paging_stats.last_shift_seconds = shift_seconds
        self.paging_stats.max_shift_seconds = max(self.paging_stats.max_shift_seconds, shift_seconds)

    def mark_camera_activity(self, moved: bool, *, dt: float | None = None) -> None:
        if moved:
            self._camera_recently_moved = True
            self._camera_idle_elapsed_seconds = 0.0
            self._last_background_io_flush_idle_seconds = 0.0
            return
        if dt is not None:
            self._camera_idle_elapsed_seconds += max(0.0, float(dt))
        if self._camera_idle_elapsed_seconds >= self.idle_flush_cooldown_seconds:
            self._camera_recently_moved = False

    def ensure_resident_for_camera(self) -> None:
        log.debug("[world] ensure_resident: camera=(%d,%d) active_origin=(%d,%d) active_size=%dx%d",
                  self.camera_x, self.camera_y, self.active_origin_x, self.active_origin_y,
                  self.active_width, self.active_height)
        new_origin_x = self.active_origin_x
        new_origin_y = self.active_origin_y
        max_origin_x = max(0, self.store.width - self.active_width)
        max_origin_y = max(0, self.store.height - self.active_height)

        while self.camera_x - new_origin_x < self.safety_margin_cells and new_origin_x > 0:
            new_origin_x = max(0, new_origin_x - self.page_shift_cells)
        while (new_origin_x + self.active_width) - (self.camera_x + self.viewport_width) < self.safety_margin_cells and new_origin_x < max_origin_x:
            new_origin_x = min(max_origin_x, new_origin_x + self.page_shift_cells)

        while self.camera_y - new_origin_y < self.safety_margin_cells and new_origin_y > 0:
            new_origin_y = max(0, new_origin_y - self.page_shift_cells)
        while (new_origin_y + self.active_height) - (self.camera_y + self.viewport_height) < self.safety_margin_cells and new_origin_y < max_origin_y:
            new_origin_y = min(max_origin_y, new_origin_y + self.page_shift_cells)

        if new_origin_x != self.active_origin_x or new_origin_y != self.active_origin_y:
            self._shift_active_window(new_origin_x, new_origin_y)

    def pan_camera(self, dx: int, dy: int) -> None:
        next_camera_x = _clamp(self.camera_x + int(dx), 0, max(0, self.store.width - self.viewport_width))
        next_camera_y = _clamp(self.camera_y + int(dy), 0, max(0, self.store.height - self.viewport_height))
        moved = next_camera_x != self.camera_x or next_camera_y != self.camera_y
        self.camera_x = next_camera_x
        self.camera_y = next_camera_y
        if moved:
            log.info("[world] pan_camera: new camera=(%d,%d) (dx=%d, dy=%d)",
                     self.camera_x, self.camera_y, dx, dy)
        self.mark_camera_activity(moved)
        self.ensure_resident_for_camera()

    def read_cell(self, world_x: int, world_y: int) -> CellState | None:
        """Read a cell at world coordinates. Returns None if outside active window."""
        if not (self.active_rect.x <= world_x < self.active_rect.right and
                self.active_rect.y <= world_y < self.active_rect.bottom):
            return None
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        return self.active_grid.get_cell(local_x, local_y)

    def screen_to_world(self, sx: int, sy: int, *, screen_width: int, screen_height: int) -> tuple[int, int]:
        viewport_x = max(0, min(self.viewport_width - 1, int(float(sx) * self.viewport_width / max(1, screen_width))))
        viewport_y_from_bottom = int(float(sy) * self.viewport_height / max(1, screen_height))
        viewport_y = max(0, min(self.viewport_height - 1, self.viewport_height - 1 - viewport_y_from_bottom))
        return (self.camera_x + viewport_x, self.camera_y + viewport_y)

    def paint_world(
        self,
        world_x: int,
        world_y: int,
        radius: int,
        family_id: str | None,
        variant_id: str | None,
        *,
        overrides: dict[str, object] | None = None,
    ) -> None:
        if not self.active_rect.x <= world_x < self.active_rect.right or not self.active_rect.y <= world_y < self.active_rect.bottom:
            return
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        if self.gpu_simulator is not None:
            self.gpu_simulator.paint_circle(local_x, local_y, radius, family_id, variant_id, overrides=overrides)
            return
        if family_id is None or variant_id is None:
            inject_cells(
                self.active_grid,
                {"x": local_x, "y": local_y, "radius": radius},
                "empty",
                "empty",
                registry=self.registry,
            )
        else:
            inject_cells(
                self.active_grid,
                {"x": local_x, "y": local_y, "radius": radius},
                family_id,
                variant_id,
                overrides,
                registry=self.registry,
            )

    def set_liquid_brownian_enabled(self, enabled: bool) -> None:
        self.active_grid.liquid_brownian_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_liquid_brownian_enabled(enabled)

    def set_blocked_impulse_enabled(self, enabled: bool) -> None:
        self.active_grid.blocked_impulse_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_blocked_impulse_enabled(enabled)

    def set_directional_fallback_enabled(self, enabled: bool) -> None:
        self.active_grid.directional_fallback_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_directional_fallback_enabled(enabled)

    def set_directional_fallback_angle_limit_degrees(self, angle_limit_degrees: float) -> None:
        self.active_grid.directional_fallback_angle_limit_degrees = float(angle_limit_degrees)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_directional_fallback_angle_limit_degrees(angle_limit_degrees)

    def service_background_io(self) -> None:
        if self.gpu_simulator is None:
            return
        if self._camera_recently_moved:
            return
        if self._camera_idle_elapsed_seconds - self._last_background_io_flush_idle_seconds < self.idle_flush_service_interval_seconds:
            return
        if self._pending_gpu_writebacks:
            if self._flush_one_pending_gpu_writeback():
                self._last_background_io_flush_idle_seconds = self._camera_idle_elapsed_seconds

    def step(self, dt: float) -> None:
        if self.gpu_simulator is not None:
            self.gpu_simulator.step(dt)
            return
        step(self.active_grid, self.registry, dt)

    def render(self, view_mode: DebugViewMode = DebugViewMode.MATERIAL):
        if self.gpu_simulator is not None:
            return self.gpu_simulator.render(view_mode)
        return build_rgba_frame(self.active_grid, self.registry, view_mode=view_mode)

    def visible_uv_rect(self) -> tuple[float, float, float, float]:
        local_x = self.camera_x - self.active_origin_x
        local_y = self.camera_y - self.active_origin_y
        uv_origin_y = 1.0 - (local_y + self.viewport_height) / max(1, self.active_height)
        return (
            local_x / max(1, self.active_width),
            max(0.0, uv_origin_y),
            self.viewport_width / max(1, self.active_width),
            self.viewport_height / max(1, self.active_height),
        )

    def shift_time_average_ms(self) -> float:
        if self.paging_stats.shift_count <= 0:
            return 0.0
        return self.paging_stats.total_shift_seconds * 1000.0 / self.paging_stats.shift_count

    def shift_time_last_ms(self) -> float:
        return self.paging_stats.last_shift_seconds * 1000.0

    def shift_time_max_ms(self) -> float:
        return self.paging_stats.max_shift_seconds * 1000.0

    def stage_time_last_ms(self) -> float:
        return self.paging_stats.last_evict_stage_seconds * 1000.0

    def overlap_copy_time_last_ms(self) -> float:
        return self.paging_stats.last_overlap_copy_seconds * 1000.0

    def overlap_transient_copy_time_last_ms(self) -> float:
        return self.paging_stats.last_overlap_transient_copy_seconds * 1000.0

    def incoming_load_time_last_ms(self) -> float:
        return self.paging_stats.last_incoming_load_seconds * 1000.0

    def incoming_transient_clear_time_last_ms(self) -> float:
        return self.paging_stats.last_incoming_transient_clear_seconds * 1000.0

    def anchor_build_time_last_ms(self) -> float:
        return self.paging_stats.last_anchor_build_seconds * 1000.0

    def anchor_upload_time_last_ms(self) -> float:
        return self.paging_stats.last_anchor_upload_seconds * 1000.0

    def readback_active_grid(self) -> Grid:
        if self.gpu_simulator is not None:
            grid = self.gpu_simulator.readback_grid()
            grid.external_support_anchors = list(self.active_grid.external_support_anchors)
            return grid
        return self.active_grid
