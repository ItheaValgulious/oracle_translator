from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .atmosphere import default_ambient_air_temperature_for_row
from .grid import Grid, create_grid
from .render import DebugViewMode, build_rgba_frame
from .sim import inject_cells, step
from .types import CellFlag, CellState, MaterialRegistry

if TYPE_CHECKING:
    import moderngl


DEFAULT_WORLD_CHUNK_SIZE = 320
DEFAULT_HALO_CELLS = 64
DEFAULT_PAGE_SHIFT_CELLS = 64
DEFAULT_SAFETY_MARGIN_CELLS = 32

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
    for local_y in range(region.height):
        for local_x in range(region.width):
            grid.set_cell(x + local_x, y + local_y, region.get_cell(local_x, local_y).copy())


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
    def __init__(self, width: int, height: int, *, chunk_size: int = DEFAULT_WORLD_CHUNK_SIZE) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("World dimensions must be positive.")
        self.width = int(width)
        self.height = int(height)
        self.chunk_size = int(chunk_size)
        self._chunks: dict[tuple[int, int], _WorldChunk] = {}

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
        chunk = self._chunk(*self._chunk_coord(x, y), create=False)
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

    def read_rect(self, world_x: int, world_y: int, width: int, height: int) -> GridSlice:
        cells: list[CellState] = []
        anchors: list[bool] = []
        for y in range(height):
            for x in range(width):
                cell_x = world_x + x
                cell_y = world_y + y
                if self.in_bounds(cell_x, cell_y):
                    cell_ref = self._cell_ref(cell_x, cell_y)
                    cells.append(cell_ref.copy() if cell_ref is not None else _default_world_cell(self.height, cell_y))
                    chunk = self._chunk(*self._chunk_coord(cell_x, cell_y), create=False)
                    local_index = self._chunk_local_index(cell_x, cell_y)
                    anchors.append(bool(chunk is not None and local_index in chunk.anchored_support_indices))
                else:
                    cells.append(CellState())
                    anchors.append(False)
        return GridSlice(width=width, height=height, cells=cells, anchored_support_mask=anchors)

    def write_rect(self, world_x: int, world_y: int, region: GridSlice) -> None:
        for y in range(region.height):
            for x in range(region.width):
                target_x = world_x + x
                target_y = world_y + y
                if not self.in_bounds(target_x, target_y):
                    continue
                cell = region.get_cell(x, y)
                chunk_x, chunk_y = self._chunk_coord(target_x, target_y)
                chunk = self._chunk(chunk_x, chunk_y, create=True)
                assert chunk is not None
                local_index = self._chunk_local_index(target_x, target_y)
                if _is_default_empty_cell(cell, self.height, target_y):
                    chunk.cells.pop(local_index, None)
                    chunk.anchored_support_indices.discard(local_index)
                    if not chunk.cells and not chunk.anchored_support_indices:
                        self._chunks.pop((chunk_x, chunk_y), None)
                    continue
                chunk.cells[local_index] = cell.copy()

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
        ctx: moderngl.Context | None = None,
        liquid_brownian_enabled: bool = True,
        blocked_impulse_enabled: bool = True,
        directional_fallback_enabled: bool = True,
        directional_fallback_angle_limit_degrees: float = 45.0,
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
        self.active_width = min(store.width, self.viewport_width + self.halo_cells * 2)
        self.active_height = min(store.height, self.viewport_height + self.halo_cells * 2)
        self.camera_x = max(0, (store.width - self.viewport_width) // 2)
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
        self._materialize_active_grid_from_store()
        if ctx is not None:
            from .gpu_backend import GpuSimulator

            self.gpu_simulator = GpuSimulator(ctx, self.active_grid, self.registry)
            self.gpu_simulator.set_external_support_anchors(self.active_grid.external_support_anchors)

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

    def _materialize_active_grid_from_store(self) -> None:
        loaded = self.store.read_rect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        grid = create_grid(self.active_width, self.active_height)
        _copy_runtime_flags(grid, self.active_grid)
        _write_grid_region(grid, 0, 0, loaded)
        grid.external_support_anchors = self._build_external_support_anchor_mask(
            WorldRect(self.active_origin_x, self.active_origin_y, self.active_width, self.active_height)
        )
        self.active_grid = grid

    def _build_external_support_anchor_mask(self, rect: WorldRect) -> list[bool]:
        anchors = [False for _ in range(rect.width * rect.height)]
        for local_y in range(rect.height):
            for local_x in range(rect.width):
                if local_x not in {0, rect.width - 1} and local_y not in {0, rect.height - 1}:
                    continue
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
                        anchors[local_y * rect.width + local_x] = True
                        break
        return anchors

    def _set_external_support_anchors(self) -> None:
        anchors = self._build_external_support_anchor_mask(self.active_rect)
        self.active_grid.external_support_anchors = anchors
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_external_support_anchors(anchors)

    def _pending_overlap(self, rect: WorldRect) -> list[tuple[_PendingGpuWriteback, WorldRect]]:
        overlaps: list[tuple[_PendingGpuWriteback, WorldRect]] = []
        for pending in self._pending_gpu_writebacks:
            overlap = pending.rect.intersection(rect)
            if overlap is not None:
                overlaps.append((pending, overlap))
        return overlaps

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
        pending = self._pending_gpu_writebacks.pop(0)
        region = self.gpu_simulator.read_staged_region(pending.staged_region)
        self.store.write_rect(pending.rect.x, pending.rect.y, region)
        self.store.recompute_anchored_support(self.registry)
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
            incoming = self.store.read_rect(missing.x, missing.y, missing.width, missing.height)
            self.gpu_simulator.write_region(
                missing.x - target_origin_x,
                missing.y - target_origin_y,
                incoming,
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
        old_rect = self.active_rect
        new_rect = WorldRect(new_origin_x, new_origin_y, self.active_width, self.active_height)
        overlap = old_rect.intersection(new_rect)
        evicted_rects = _rect_difference(old_rect, overlap)
        incoming_rects = _rect_difference(new_rect, overlap)

        if self.gpu_simulator is None:
            for evicted_rect in evicted_rects:
                region = self._capture_active_region(evicted_rect)
                self.store.write_rect(evicted_rect.x, evicted_rect.y, region)
            self.store.recompute_anchored_support(self.registry)
        else:
            for evicted_rect in evicted_rects:
                self._stage_evicted_region(evicted_rect)

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
                self.gpu_simulator.copy_region(
                    overlap.x - old_rect.x,
                    overlap.y - old_rect.y,
                    overlap.width,
                    overlap.height,
                    overlap.x - new_rect.x,
                    overlap.y - new_rect.y,
                    dst_buffer_index=target_buffer_index,
                )
                self.gpu_simulator.copy_transient_region(
                    overlap.x - old_rect.x,
                    overlap.y - old_rect.y,
                    overlap.width,
                    overlap.height,
                    overlap.x - new_rect.x,
                    overlap.y - new_rect.y,
                )
            for incoming_rect in incoming_rects:
                self._load_incoming_rect_into_gpu_buffer(
                    incoming_rect,
                    target_buffer_index=target_buffer_index,
                    target_origin_x=new_rect.x,
                    target_origin_y=new_rect.y,
                )
                self.gpu_simulator.clear_region_transients(
                    incoming_rect.x - new_rect.x,
                    incoming_rect.y - new_rect.y,
                    incoming_rect.width,
                    incoming_rect.height,
                )
            self.gpu_simulator.front_index = target_buffer_index
            self._pending_flush_cooldown_steps = max(self._pending_flush_cooldown_steps, 8)

        self.active_origin_x = new_origin_x
        self.active_origin_y = new_origin_y
        self._set_external_support_anchors()

    def ensure_resident_for_camera(self) -> None:
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
        self.camera_x = _clamp(self.camera_x + int(dx), 0, max(0, self.store.width - self.viewport_width))
        self.camera_y = _clamp(self.camera_y + int(dy), 0, max(0, self.store.height - self.viewport_height))
        self.ensure_resident_for_camera()

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
        if self._pending_flush_cooldown_steps > 0:
            self._pending_flush_cooldown_steps -= 1
            return
        if self._pending_gpu_writebacks:
            self._flush_one_pending_gpu_writeback()

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
        return (
            local_x / max(1, self.active_width),
            local_y / max(1, self.active_height),
            self.viewport_width / max(1, self.active_width),
            self.viewport_height / max(1, self.active_height),
        )

    def readback_active_grid(self) -> Grid:
        if self.gpu_simulator is not None:
            grid = self.gpu_simulator.readback_grid()
            grid.external_support_anchors = list(self.active_grid.external_support_anchors)
            return grid
        return self.active_grid
