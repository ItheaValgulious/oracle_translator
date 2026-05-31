from __future__ import annotations

import logging
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Callable

log = logging.getLogger(__name__)

from .atmosphere import default_ambient_air_temperature_for_row
from .gpu_backend import LocalCellsSnapshot, PackedStateRegion
from .grid import Grid
from .render import DebugViewMode
from .types import CellFlag, CellState, MaterialRegistry

if TYPE_CHECKING:
    import moderngl


DEFAULT_WORLD_CHUNK_SIZE = 320
DEFAULT_HALO_CELLS = 16
DEFAULT_PAGE_SHIFT_CELLS = 16
DEFAULT_SAFETY_MARGIN_CELLS = 16
DEFAULT_IDLE_FLUSH_COOLDOWN_SECONDS = 0.2
DEFAULT_IDLE_FLUSH_SERVICE_INTERVAL_SECONDS = 300.0
DEFAULT_PENDING_WRITEBACK_LIMIT = 256
DEFAULT_GPU_WRITEBACK_SLICE_CELL_BUDGET = 64
DEFAULT_CHUNK_IO_WORKERS = 1
DEFAULT_CHUNK_CACHE_PREFETCH_X = 3
DEFAULT_CHUNK_CACHE_PREFETCH_Y = 2
GPU_CHUNK_FILE_MAGIC = b"OGCHUNK1"
GPU_CHUNK_FILE_VERSION = 1
GPU_CHUNK_HEADER_FORMAT = "<8siiiiiiqq"
GPU_CHUNK_HEADER_SIZE = struct.calcsize(GPU_CHUNK_HEADER_FORMAT)
GPU_STATE_PIXEL_BYTES = 16

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
    snapshot_region: object | None = None
    major_axis_offset: int = 0


@dataclass
class PackedChunk:
    width: int
    height: int
    state_int: bytearray
    state_vec: bytearray
    state_misc: bytearray
    dirty: bool = False


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
    last_shift_cache_hits: int = 0
    last_shift_empty_hits: int = 0
    last_shift_inflight_wait_hits: int = 0
    last_shift_disk_loads: int = 0
    last_shift_generates: int = 0
    last_shift_saves: int = 0
    last_shift_disk_load_seconds: float = 0.0
    last_shift_generate_seconds: float = 0.0
    last_shift_save_seconds: float = 0.0


@dataclass
class ChunkIoStats:
    cache_hits: int = 0
    empty_hits: int = 0
    inflight_wait_hits: int = 0
    disk_load_count: int = 0
    disk_load_total_seconds: float = 0.0
    disk_load_last_seconds: float = 0.0
    disk_load_max_seconds: float = 0.0
    generate_count: int = 0
    generate_total_seconds: float = 0.0
    generate_last_seconds: float = 0.0
    generate_max_seconds: float = 0.0
    save_count: int = 0
    save_total_seconds: float = 0.0
    save_last_seconds: float = 0.0
    save_max_seconds: float = 0.0


@dataclass(frozen=True)
class ChunkCacheDebugSnapshot:
    cached_chunks: int
    empty_chunks: int
    prefetch_queued: int
    prefetch_inflight: int
    pinned_chunks: int
    cache_hits: int
    empty_hits: int
    inflight_wait_hits: int
    disk_load_count: int
    disk_load_total_seconds: float
    disk_load_last_seconds: float
    disk_load_max_seconds: float
    generate_count: int
    generate_total_seconds: float
    generate_last_seconds: float
    generate_max_seconds: float
    save_count: int
    save_total_seconds: float
    save_last_seconds: float
    save_max_seconds: float


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


def _chunks_for_rect(rect: WorldRect, chunk_size: int) -> list[tuple[int, int]]:
    if rect.is_empty:
        return []
    chunk_min_x = max(0, rect.x) // chunk_size
    chunk_max_x = max(0, rect.right - 1) // chunk_size
    chunk_min_y = max(0, rect.y) // chunk_size
    chunk_max_y = max(0, rect.bottom - 1) // chunk_size
    return [
        (chunk_x, chunk_y)
        for chunk_y in range(chunk_min_y, chunk_max_y + 1)
        for chunk_x in range(chunk_min_x, chunk_max_x + 1)
    ]


def _variant_fingerprint(registry: MaterialRegistry) -> int:
    value = 1469598103934665603
    for family_id, family in registry.families.items():
        for variant_id in family.variants.keys():
            for byte in f"{family_id}:{variant_id};".encode("utf-8"):
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    if value >= 1 << 63:
        value -= 1 << 64
    return value


def _blank_chunk(
    chunk_size: int,
    world_height: int,
    chunk_y: int,
    empty_variant_index: int,
) -> PackedChunk:
    cell_count = chunk_size * chunk_size
    state_int = array("i", [empty_variant_index, 0, int(CellFlag.NONE), 0]) * cell_count
    state_vec = array("f", [0.0]) * (cell_count * 4)
    state_misc = array("f")
    world_y0 = chunk_y * chunk_size
    for local_y in range(chunk_size):
        world_y = world_y0 + local_y
        ambient = default_ambient_air_temperature_for_row(world_height, world_y) if 0 <= world_y < world_height else 20.0
        state_misc.extend(array("f", [ambient, 0.0, 1.0, 0.0]) * chunk_size)
    return PackedChunk(
        width=chunk_size,
        height=chunk_size,
        state_int=bytearray(state_int.tobytes()),
        state_vec=bytearray(state_vec.tobytes()),
        state_misc=bytearray(state_misc.tobytes()),
    )


class _PackedChunkTerrainWriter:
    def __init__(
        self,
        *,
        world_width: int,
        world_height: int,
        chunk_x: int,
        chunk_y: int,
        chunk_size: int,
        seed: int,
        tables,
    ) -> None:
        self.width = int(world_width)
        self.height = int(world_height)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self._chunk_x = int(chunk_x)
        self._chunk_y = int(chunk_y)
        self._origin_x = self._chunk_x * self.chunk_size
        self._origin_y = self._chunk_y * self.chunk_size
        self._tables = tables
        blank = _blank_chunk(self.chunk_size, self.height, self._chunk_y, tables.empty_variant_index)
        self._chunk = blank
        self._state_int = array("i")
        self._state_int.frombytes(blank.state_int)
        self._state_vec = array("f")
        self._state_vec.frombytes(blank.state_vec)
        self._state_misc = array("f")
        self._state_misc.frombytes(blank.state_misc)
        self._encoded_cell_cache: dict[tuple[object, ...], tuple[int, int, int, float, float, float, float, float, float, float, float]] = {}
        self._nondefault_writes = 0

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _encode_cell(
        self,
        cell: CellState,
    ) -> tuple[int, int, int, float, float, float, float, float, float, float, float]:
        key = (
            cell.family_id,
            cell.variant_id,
            cell.generation,
            int(cell.flags),
            cell.vel_x,
            cell.vel_y,
            cell.blocked_x,
            cell.blocked_y,
            cell.temperature,
            cell.support_value,
            cell.integrity,
            cell.age,
        )
        encoded = self._encoded_cell_cache.get(key)
        if encoded is not None:
            return encoded
        encoded = (
            self._tables.variant_index_by_key[(cell.family_id, cell.variant_id)],
            cell.generation,
            int(cell.flags),
            cell.vel_x,
            cell.vel_y,
            cell.blocked_x,
            cell.blocked_y,
            cell.temperature,
            cell.support_value,
            cell.integrity,
            cell.age,
        )
        self._encoded_cell_cache[key] = encoded
        return encoded

    def _write_local_index(self, local_index: int, world_y: int, cell: CellState) -> None:
        int_offset = local_index * 4
        if _is_default_empty_cell(cell, self.height, world_y):
            self._state_int[int_offset] = self._tables.empty_variant_index
            self._state_int[int_offset + 1] = 0
            self._state_int[int_offset + 2] = int(CellFlag.NONE)
            self._state_int[int_offset + 3] = 0
            self._state_vec[int_offset] = 0.0
            self._state_vec[int_offset + 1] = 0.0
            self._state_vec[int_offset + 2] = 0.0
            self._state_vec[int_offset + 3] = 0.0
            ambient = default_ambient_air_temperature_for_row(self.height, world_y)
            self._state_misc[int_offset] = ambient
            self._state_misc[int_offset + 1] = 0.0
            self._state_misc[int_offset + 2] = 1.0
            self._state_misc[int_offset + 3] = 0.0
            return
        (
            variant_index,
            generation,
            flags,
            vel_x,
            vel_y,
            blocked_x,
            blocked_y,
            temperature,
            support_value,
            integrity,
            age,
        ) = self._encode_cell(cell)
        self._state_int[int_offset] = variant_index
        self._state_int[int_offset + 1] = generation
        self._state_int[int_offset + 2] = flags
        self._state_int[int_offset + 3] = 0
        self._state_vec[int_offset] = vel_x
        self._state_vec[int_offset + 1] = vel_y
        self._state_vec[int_offset + 2] = blocked_x
        self._state_vec[int_offset + 3] = blocked_y
        self._state_misc[int_offset] = temperature
        self._state_misc[int_offset + 1] = support_value
        self._state_misc[int_offset + 2] = integrity
        self._state_misc[int_offset + 3] = age
        self._nondefault_writes += 1

    def _terrain_write_span(self, world_x: int, world_y0: int, world_y1: int, cell: CellState) -> None:
        if not (self._origin_x <= world_x < self._origin_x + self.chunk_size):
            return
        clipped_y0 = max(int(world_y0), self._origin_y)
        clipped_y1 = min(int(world_y1), self._origin_y + self.chunk_size)
        if clipped_y0 >= clipped_y1:
            return
        local_x = world_x - self._origin_x
        for world_y in range(clipped_y0, clipped_y1):
            local_y = world_y - self._origin_y
            self._write_local_index(local_y * self.chunk_size + local_x, world_y, cell)

    def _terrain_write_cell(self, world_x: int, world_y: int, cell: CellState) -> None:
        self.set_cell(world_x, world_y, cell)

    def set_cell(self, x: int, y: int, cell: CellState) -> None:
        if not self.in_bounds(x, y):
            return
        if x // self.chunk_size != self._chunk_x or y // self.chunk_size != self._chunk_y:
            return
        local_x = x - self._origin_x
        local_y = y - self._origin_y
        self._write_local_index(local_y * self.chunk_size + local_x, y, cell)

    def finalize(self) -> PackedChunk:
        if self._nondefault_writes <= 0:
            return self._chunk
        self._chunk.state_int = bytearray(self._state_int.tobytes())
        self._chunk.state_vec = bytearray(self._state_vec.tobytes())
        self._chunk.state_misc = bytearray(self._state_misc.tobytes())
        self._chunk.dirty = True
        return self._chunk


def _copy_packed_rect(
    src,
    dst: PackedChunk,
    *,
    src_x: int,
    src_y: int,
    dst_x: int,
    dst_y: int,
    width: int,
    height: int,
) -> None:
    if width <= 0 or height <= 0:
        return
    src_width = src.width
    dst_width = dst.width
    for row in range(height):
        src_start = ((src_y + row) * src_width + src_x) * GPU_STATE_PIXEL_BYTES
        src_end = src_start + width * GPU_STATE_PIXEL_BYTES
        dst_start = ((dst_y + row) * dst_width + dst_x) * GPU_STATE_PIXEL_BYTES
        dst_end = dst_start + width * GPU_STATE_PIXEL_BYTES
        dst.state_int[dst_start:dst_end] = src.state_int[src_start:src_end]
        dst.state_vec[dst_start:dst_end] = src.state_vec[src_start:src_end]
        dst.state_misc[dst_start:dst_end] = src.state_misc[src_start:src_end]
    dst.dirty = True


def _extract_packed_rect(
    src: PackedChunk,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
):
    if width <= 0 or height <= 0:
        return PackedStateRegion(width=0, height=0, state_int=b"", state_vec=b"", state_misc=b"")
    state_int = bytearray()
    state_vec = bytearray()
    state_misc = bytearray()
    for row in range(height):
        start = ((y + row) * src.width + x) * GPU_STATE_PIXEL_BYTES
        end = start + width * GPU_STATE_PIXEL_BYTES
        state_int.extend(src.state_int[start:end])
        state_vec.extend(src.state_vec[start:end])
        state_misc.extend(src.state_misc[start:end])
    return PackedStateRegion(
        width=width,
        height=height,
        state_int=bytes(state_int),
        state_vec=bytes(state_vec),
        state_misc=bytes(state_misc),
    )


def _sanitize_packed_region_for_storage(
    packed_region,
    *,
    world_y0: int,
    world_height: int,
    placeholder_variant_index: int,
    empty_variant_index: int,
):
    if placeholder_variant_index < 0:
        return packed_region
    int_values = array("i")
    int_values.frombytes(bytes(packed_region.state_int))
    changed = False
    placeholder_pixels: list[int] = []
    for pixel_index in range(0, len(int_values), 4):
        if int_values[pixel_index] != placeholder_variant_index:
            continue
        int_values[pixel_index] = empty_variant_index
        int_values[pixel_index + 1] = 0
        int_values[pixel_index + 2] = int(CellFlag.NONE)
        int_values[pixel_index + 3] = 0
        placeholder_pixels.append(pixel_index)
        changed = True
    if not changed:
        return packed_region

    vec_values = array("f")
    vec_values.frombytes(bytes(packed_region.state_vec))
    misc_values = array("f")
    misc_values.frombytes(bytes(packed_region.state_misc))
    width = int(packed_region.width)
    for pixel_index in placeholder_pixels:
        cell_index = pixel_index // 4
        world_y = world_y0 + cell_index // width
        ambient = default_ambient_air_temperature_for_row(world_height, world_y)
        vec_values[pixel_index] = 0.0
        vec_values[pixel_index + 1] = 0.0
        vec_values[pixel_index + 2] = 0.0
        vec_values[pixel_index + 3] = 0.0
        misc_values[pixel_index] = ambient
        misc_values[pixel_index + 1] = 0.0
        misc_values[pixel_index + 2] = 1.0
        misc_values[pixel_index + 3] = 0.0

    from .gpu_backend import PackedStateRegion

    return PackedStateRegion(
        width=packed_region.width,
        height=packed_region.height,
        state_int=int_values.tobytes(),
        state_vec=vec_values.tobytes(),
        state_misc=misc_values.tobytes(),
    )


class GpuChunkCache:
    def __init__(
        self,
        store: WorldChunkStore,
        registry: MaterialRegistry,
        *,
        save_dir: str | Path,
        prefetch_x: int = DEFAULT_CHUNK_CACHE_PREFETCH_X,
        prefetch_y: int = DEFAULT_CHUNK_CACHE_PREFETCH_Y,
    ) -> None:
        from .gpu_backend import GpuMaterialTables

        self.store = store
        self.registry = registry
        self.tables = GpuMaterialTables.from_registry(registry)
        self.chunk_size = store.chunk_size
        self.prefetch_x = max(0, int(prefetch_x))
        self.prefetch_y = max(0, int(prefetch_y))
        self.save_dir = Path(save_dir) / f"seed_{store.seed}" / f"chunk_{self.chunk_size}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[tuple[int, int], PackedChunk] = {}
        self._prefetch_queue: deque[tuple[int, int]] = deque()
        self._prefetch_queued: set[tuple[int, int]] = set()
        self._inflight: dict[tuple[int, int], Future[PackedChunk | None]] = {}
        self._prime_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu-chunk-prime")
        self._prime_inflight: set[tuple[int, int]] = set()
        self._pinned_chunks: set[tuple[int, int]] = set()
        self._empty_chunks: set[tuple[int, int]] = set()
        self._blank_chunk_cache: dict[int, PackedChunk] = {}
        self._executor = ThreadPoolExecutor(max_workers=DEFAULT_CHUNK_IO_WORKERS, thread_name_prefix="gpu-chunk")
        self._lock = threading.Lock()
        self._material_fingerprint = _variant_fingerprint(registry)
        self._placeholder_variant_index = self.tables.variant_index_by_key.get(("entity_placeholder", "placeholder"), -1)
        self.stats = ChunkIoStats()

    def snapshot_stats(self) -> ChunkCacheDebugSnapshot:
        with self._lock:
            cached_chunks = len(self._chunks)
            empty_chunks = len(self._empty_chunks)
            prefetch_queued = len(self._prefetch_queue)
            prefetch_inflight = len(self._inflight)
            pinned_chunks = len(self._pinned_chunks)
        return ChunkCacheDebugSnapshot(
            cached_chunks=cached_chunks,
            empty_chunks=empty_chunks,
            prefetch_queued=prefetch_queued,
            prefetch_inflight=prefetch_inflight,
            pinned_chunks=pinned_chunks,
            cache_hits=self.stats.cache_hits,
            empty_hits=self.stats.empty_hits,
            inflight_wait_hits=self.stats.inflight_wait_hits,
            disk_load_count=self.stats.disk_load_count,
            disk_load_total_seconds=self.stats.disk_load_total_seconds,
            disk_load_last_seconds=self.stats.disk_load_last_seconds,
            disk_load_max_seconds=self.stats.disk_load_max_seconds,
            generate_count=self.stats.generate_count,
            generate_total_seconds=self.stats.generate_total_seconds,
            generate_last_seconds=self.stats.generate_last_seconds,
            generate_max_seconds=self.stats.generate_max_seconds,
            save_count=self.stats.save_count,
            save_total_seconds=self.stats.save_total_seconds,
            save_last_seconds=self.stats.save_last_seconds,
            save_max_seconds=self.stats.save_max_seconds,
        )

    def _chunk_path(self, chunk_x: int, chunk_y: int) -> Path:
        return self.save_dir / f"{chunk_x}_{chunk_y}.ogchunk"

    def _chunk_file_exists(self, chunk_x: int, chunk_y: int) -> bool:
        return self._chunk_path(chunk_x, chunk_y).exists()

    def _can_fill_empty_chunk(self, chunk_x: int, chunk_y: int) -> bool:
        key = (chunk_x, chunk_y)
        if key in self._empty_chunks:
            return True
        if key in self._chunks or self._chunk_file_exists(chunk_x, chunk_y):
            return False
        if self.store._chunk(chunk_x, chunk_y, create=False) is not None:  # noqa: SLF001
            return False
        return self.store.chunk_generator is None

    def _chunk_world_rect(self, chunk_x: int, chunk_y: int) -> WorldRect:
        return WorldRect(
            chunk_x * self.chunk_size,
            chunk_y * self.chunk_size,
            self.chunk_size,
            self.chunk_size,
        )

    def _blank_chunk_for(self, chunk_y: int) -> PackedChunk:
        chunk = self._blank_chunk_cache.get(chunk_y)
        if chunk is None:
            chunk = _blank_chunk(self.chunk_size, self.store.height, chunk_y, self.tables.empty_variant_index)
            self._blank_chunk_cache[chunk_y] = chunk
        return chunk

    def _load_from_disk(self, chunk_x: int, chunk_y: int) -> PackedChunk | None:
        started_at = perf_counter()
        path = self._chunk_path(chunk_x, chunk_y)
        if not path.exists():
            elapsed = perf_counter() - started_at
            self.stats.disk_load_last_seconds = elapsed
            self.stats.disk_load_max_seconds = max(self.stats.disk_load_max_seconds, elapsed)
            return None
        data = path.read_bytes()
        if len(data) < GPU_CHUNK_HEADER_SIZE:
            elapsed = perf_counter() - started_at
            self.stats.disk_load_last_seconds = elapsed
            self.stats.disk_load_max_seconds = max(self.stats.disk_load_max_seconds, elapsed)
            return None
        (
            magic,
            version,
            stored_seed,
            chunk_size,
            width,
            height,
            _stored_chunk_x,
            _stored_chunk_y,
            fingerprint,
        ) = struct.unpack(GPU_CHUNK_HEADER_FORMAT, data[:GPU_CHUNK_HEADER_SIZE])
        if (
            magic != GPU_CHUNK_FILE_MAGIC
            or version != GPU_CHUNK_FILE_VERSION
            or stored_seed != self.store.seed
            or chunk_size != self.chunk_size
            or width != self.chunk_size
            or height != self.chunk_size
            or fingerprint != self._material_fingerprint
        ):
            elapsed = perf_counter() - started_at
            self.stats.disk_load_last_seconds = elapsed
            self.stats.disk_load_max_seconds = max(self.stats.disk_load_max_seconds, elapsed)
            return None
        plane_size = width * height * GPU_STATE_PIXEL_BYTES
        expected = GPU_CHUNK_HEADER_SIZE + plane_size * 3
        if len(data) != expected:
            elapsed = perf_counter() - started_at
            self.stats.disk_load_last_seconds = elapsed
            self.stats.disk_load_max_seconds = max(self.stats.disk_load_max_seconds, elapsed)
            return None
        offset = GPU_CHUNK_HEADER_SIZE
        chunk = PackedChunk(
            width=width,
            height=height,
            state_int=bytearray(data[offset:offset + plane_size]),
            state_vec=bytearray(data[offset + plane_size:offset + plane_size * 2]),
            state_misc=bytearray(data[offset + plane_size * 2:offset + plane_size * 3]),
            dirty=False,
        )
        elapsed = perf_counter() - started_at
        self.stats.disk_load_count += 1
        self.stats.disk_load_total_seconds += elapsed
        self.stats.disk_load_last_seconds = elapsed
        self.stats.disk_load_max_seconds = max(self.stats.disk_load_max_seconds, elapsed)
        log.debug("[world] loaded GPU chunk (%d,%d) from disk", chunk_x, chunk_y)
        return chunk

    def _save_to_disk(self, chunk_x: int, chunk_y: int, chunk: PackedChunk) -> None:
        started_at = perf_counter()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self._chunk_path(chunk_x, chunk_y)
        header = struct.pack(
            GPU_CHUNK_HEADER_FORMAT,
            GPU_CHUNK_FILE_MAGIC,
            GPU_CHUNK_FILE_VERSION,
            int(self.store.seed),
            self.chunk_size,
            chunk.width,
            chunk.height,
            int(chunk_x),
            int(chunk_y),
            self._material_fingerprint,
        )
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(header + bytes(chunk.state_int) + bytes(chunk.state_vec) + bytes(chunk.state_misc))
        tmp_path.replace(path)
        chunk.dirty = False
        elapsed = perf_counter() - started_at
        self.stats.save_count += 1
        self.stats.save_total_seconds += elapsed
        self.stats.save_last_seconds = elapsed
        self.stats.save_max_seconds = max(self.stats.save_max_seconds, elapsed)
        log.debug("[world] saved GPU chunk (%d,%d) to disk", chunk_x, chunk_y)

    def _generate_chunk(self, chunk_x: int, chunk_y: int) -> PackedChunk:
        started_at = perf_counter()
        log.debug("[world] generating GPU chunk (%d,%d) for the first time", chunk_x, chunk_y)
        store_chunk = self.store._chunk(chunk_x, chunk_y, create=False)  # noqa: SLF001
        chunk_generator = self.store.chunk_generator
        generator_owner = getattr(chunk_generator, "__self__", None) if chunk_generator is not None else None
        if (
            store_chunk is None
            and chunk_generator is not None
            and getattr(generator_owner, "supports_packed_chunk_generation", False)
        ):
            writer = _PackedChunkTerrainWriter(
                world_width=self.store.width,
                world_height=self.store.height,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_size=self.chunk_size,
                seed=self.store.seed,
                tables=self.tables,
            )
            chunk_generator(writer, chunk_x, chunk_y, self.chunk_size, self.store.seed)
            chunk = writer.finalize()
            if writer._nondefault_writes <= 0:  # noqa: SLF001
                self._empty_chunks.add((chunk_x, chunk_y))
            elapsed = perf_counter() - started_at
            self.stats.generate_count += 1
            self.stats.generate_total_seconds += elapsed
            self.stats.generate_last_seconds = elapsed
            self.stats.generate_max_seconds = max(self.stats.generate_max_seconds, elapsed)
            return chunk

        chunk = _blank_chunk(self.chunk_size, self.store.height, chunk_y, self.tables.empty_variant_index)
        if store_chunk is None and chunk_generator is not None:
            temp_store = WorldChunkStore(
                self.store.width,
                self.store.height,
                chunk_size=self.chunk_size,
                seed=self.store.seed,
            )
            chunk_generator(temp_store, chunk_x, chunk_y, self.chunk_size, self.store.seed)
            store_chunk = temp_store._chunk(chunk_x, chunk_y, create=False)  # noqa: SLF001
        if store_chunk is None or not store_chunk.cells:
            self._empty_chunks.add((chunk_x, chunk_y))
            elapsed = perf_counter() - started_at
            self.stats.generate_count += 1
            self.stats.generate_total_seconds += elapsed
            self.stats.generate_last_seconds = elapsed
            self.stats.generate_max_seconds = max(self.stats.generate_max_seconds, elapsed)
            return self._blank_chunk_for(chunk_y)
        if store_chunk is not None and store_chunk.cells:
            state_int = array("i")
            state_int.frombytes(chunk.state_int)
            state_vec = array("f")
            state_vec.frombytes(chunk.state_vec)
            state_misc = array("f")
            state_misc.frombytes(chunk.state_misc)
            for local_index, cell in store_chunk.cells.items():
                int_offset = local_index * 4
                vec_offset = local_index * 4
                state_int[int_offset] = self.tables.variant_index_by_key[(cell.family_id, cell.variant_id)]
                state_int[int_offset + 1] = cell.generation
                state_int[int_offset + 2] = int(cell.flags)
                state_vec[vec_offset] = cell.vel_x
                state_vec[vec_offset + 1] = cell.vel_y
                state_vec[vec_offset + 2] = cell.blocked_x
                state_vec[vec_offset + 3] = cell.blocked_y
                state_misc[vec_offset] = cell.temperature
                state_misc[vec_offset + 1] = cell.support_value
                state_misc[vec_offset + 2] = cell.integrity
                state_misc[vec_offset + 3] = cell.age
            chunk.state_int = bytearray(state_int.tobytes())
            chunk.state_vec = bytearray(state_vec.tobytes())
            chunk.state_misc = bytearray(state_misc.tobytes())
        chunk.dirty = True
        elapsed = perf_counter() - started_at
        self.stats.generate_count += 1
        self.stats.generate_total_seconds += elapsed
        self.stats.generate_last_seconds = elapsed
        self.stats.generate_max_seconds = max(self.stats.generate_max_seconds, elapsed)
        return chunk

    def ensure_chunk_cached(self, chunk_x: int, chunk_y: int) -> PackedChunk:
        key = (chunk_x, chunk_y)
        if key in self._empty_chunks:
            self.stats.empty_hits += 1
            return self._blank_chunk_for(chunk_y)
        with self._lock:
            chunk = self._chunks.get(key)
            if chunk is not None:
                self.stats.cache_hits += 1
                return chunk
            future = self._inflight.get(key)
        if future is not None:
            self.stats.inflight_wait_hits += 1
            chunk = future.result()
            with self._lock:
                self._inflight.pop(key, None)
                if chunk is not None:
                    self._chunks[key] = chunk
                    return chunk
        chunk = self._load_from_disk(chunk_x, chunk_y)
        if chunk is None:
            chunk = self._generate_chunk(chunk_x, chunk_y)
        with self._lock:
            self._chunks[key] = chunk
        return chunk

    def prefetch_chunks_for_rect(self, rect: WorldRect) -> None:
        for chunk_x, chunk_y in self._prefetch_keys_for_rect(rect):
            self.ensure_chunk_cached(chunk_x, chunk_y)

    def _prefetch_keys_for_rect(
        self,
        rect: WorldRect,
        *,
        margin_x: int | None = None,
        margin_y: int | None = None,
    ) -> list[tuple[int, int]]:
        if rect.is_empty:
            return []
        pad_x = self.prefetch_x if margin_x is None else max(0, int(margin_x))
        pad_y = self.prefetch_y if margin_y is None else max(0, int(margin_y))
        chunk_min_x = max(0, rect.x) // self.chunk_size
        chunk_max_x = max(0, rect.right - 1) // self.chunk_size
        chunk_min_y = max(0, rect.y) // self.chunk_size
        chunk_max_y = max(0, rect.bottom - 1) // self.chunk_size
        max_chunk_x = max(0, (self.store.width - 1) // self.chunk_size)
        max_chunk_y = max(0, (self.store.height - 1) // self.chunk_size)
        keys: list[tuple[int, int]] = []
        for chunk_y in range(max(0, chunk_min_y - pad_y), min(max_chunk_y, chunk_max_y + pad_y) + 1):
            for chunk_x in range(max(0, chunk_min_x - pad_x), min(max_chunk_x, chunk_max_x + pad_x) + 1):
                keys.append((chunk_x, chunk_y))
        return keys

    def schedule_prefetch_for_rect(
        self,
        rect: WorldRect,
        *,
        margin_x: int | None = None,
        margin_y: int | None = None,
        prioritize: bool = False,
    ) -> None:
        for key in self._prefetch_keys_for_rect(rect, margin_x=margin_x, margin_y=margin_y):
            with self._lock:
                already_known = key in self._chunks or key in self._prefetch_queued or key in self._inflight
            if already_known:
                continue
            if prioritize:
                self._prefetch_queue.appendleft(key)
            else:
                self._prefetch_queue.append(key)
            self._prefetch_queued.add(key)

    def _load_for_worker(self, chunk_x: int, chunk_y: int) -> PackedChunk | None:
        chunk = self._load_from_disk(chunk_x, chunk_y)
        if chunk is not None:
            return chunk
        chunk = self._generate_chunk(chunk_x, chunk_y)
        if (chunk_x, chunk_y) in self._empty_chunks:
            return None
        if chunk.dirty:
            self._save_to_disk(chunk_x, chunk_y, chunk)
        return chunk

    def _prime_for_worker(self, chunk_x: int, chunk_y: int) -> None:
        chunk = self._load_from_disk(chunk_x, chunk_y)
        if chunk is not None:
            with self._lock:
                self._chunks[(chunk_x, chunk_y)] = chunk
            return
        chunk = self._generate_chunk(chunk_x, chunk_y)
        if (chunk_x, chunk_y) in self._empty_chunks:
            return
        if chunk.dirty:
            self._save_to_disk(chunk_x, chunk_y, chunk)
        with self._lock:
            self._chunks[(chunk_x, chunk_y)] = chunk

    def prime_rect_on_disk(self, rect: WorldRect) -> None:
        for key in self._prefetch_keys_for_rect(rect, margin_x=0, margin_y=0):
            self._pinned_chunks.add(key)
            with self._lock:
                already_known = (
                    key in self._chunks
                    or key in self._prefetch_queued
                    or key in self._inflight
                    or key in self._prime_inflight
                )
            if already_known:
                continue
            with self._lock:
                self._prime_inflight.add(key)
            future = self._prime_executor.submit(self._prime_for_worker, *key)

            def _clear(_future, *, key=key) -> None:
                with self._lock:
                    self._prime_inflight.discard(key)

            future.add_done_callback(_clear)

    def prime_rect_now(self, rect: WorldRect) -> None:
        for chunk_x, chunk_y in self._prefetch_keys_for_rect(rect, margin_x=0, margin_y=0):
            key = (chunk_x, chunk_y)
            self._pinned_chunks.add(key)
            chunk = self._chunks.get(key)
            if chunk is None:
                chunk = self._load_from_disk(chunk_x, chunk_y)
            if chunk is None:
                chunk = self._generate_chunk(chunk_x, chunk_y)
            if key in self._empty_chunks:
                continue
            if chunk.dirty:
                self._save_to_disk(chunk_x, chunk_y, chunk)
            with self._lock:
                self._chunks[key] = chunk

    def prepare_rect_now(self, rect: WorldRect) -> None:
        keys = self._prefetch_keys_for_rect(rect, margin_x=0, margin_y=0)
        if not keys:
            return
        for key in keys:
            self._pinned_chunks.add(key)
        futures: list[tuple[tuple[int, int], Future[PackedChunk | None]]] = []
        for key in keys:
            with self._lock:
                chunk = self._chunks.get(key)
            if chunk is not None:
                continue
            futures.append((key, self._executor.submit(self._load_for_worker, *key)))
        if futures:
            wait([future for _key, future in futures])
        for key, future in futures:
            chunk = future.result()
            if chunk is None:
                continue
            with self._lock:
                self._chunks[key] = chunk

    def service_prefetch(self, *, max_chunks: int = 1, collect_ready: bool = True) -> bool:
        did_work = False
        if collect_ready:
            for key, future in list(self._inflight.items()):
                if not future.done():
                    continue
                chunk = future.result()
                with self._lock:
                    if chunk is not None:
                        self._chunks[key] = chunk
                    self._inflight.pop(key, None)
                if chunk is not None:
                    did_work = True
        for _ in range(max(0, int(max_chunks))):
            if not self._prefetch_queue:
                break
            key = self._prefetch_queue.popleft()
            self._prefetch_queued.discard(key)
            with self._lock:
                if key in self._chunks or key in self._inflight:
                    continue
                self._inflight[key] = self._executor.submit(self._load_for_worker, *key)
            did_work = True
        return did_work

    def collect_ready_prefetch(self) -> bool:
        did_work = False
        for key, future in list(self._inflight.items()):
            if not future.done():
                continue
            chunk = future.result()
            with self._lock:
                if chunk is not None:
                    self._chunks[key] = chunk
                self._inflight.pop(key, None)
            if chunk is not None:
                did_work = True
        return did_work

    def evict_far_chunks(self, rect: WorldRect, *, flush_dirty: bool = False) -> None:
        if rect.is_empty:
            return
        chunk_min_x = max(0, rect.x) // self.chunk_size - self.prefetch_x
        chunk_max_x = max(0, rect.right - 1) // self.chunk_size + self.prefetch_x
        chunk_min_y = max(0, rect.y) // self.chunk_size - self.prefetch_y
        chunk_max_y = max(0, rect.bottom - 1) // self.chunk_size + self.prefetch_y
        for key, chunk in list(self._chunks.items()):
            chunk_x, chunk_y = key
            if chunk_min_x <= chunk_x <= chunk_max_x and chunk_min_y <= chunk_y <= chunk_max_y:
                continue
            if key in self._pinned_chunks:
                continue
            if chunk.dirty:
                if not flush_dirty:
                    continue
                self._save_to_disk(chunk_x, chunk_y, chunk)
            self._chunks.pop(key, None)

    def flush_dirty(self) -> None:
        for (chunk_x, chunk_y), chunk in list(self._chunks.items()):
            if chunk.dirty:
                self._save_to_disk(chunk_x, chunk_y, chunk)

    def flush_one_dirty(self) -> bool:
        for (chunk_x, chunk_y), chunk in list(self._chunks.items()):
            if chunk.dirty:
                self._save_to_disk(chunk_x, chunk_y, chunk)
                return True
        return False

    def shutdown(self) -> None:
        self.flush_dirty()
        self._prime_executor.shutdown(wait=False, cancel_futures=True)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def read_rect(self, rect: WorldRect):
        if rect.width <= 0 or rect.height <= 0:
            return None
        from .gpu_backend import PackedStateRegion

        result = PackedChunk(
            width=rect.width,
            height=rect.height,
            state_int=bytearray(rect.width * rect.height * GPU_STATE_PIXEL_BYTES),
            state_vec=bytearray(rect.width * rect.height * GPU_STATE_PIXEL_BYTES),
            state_misc=bytearray(rect.width * rect.height * GPU_STATE_PIXEL_BYTES),
        )
        for chunk_x, chunk_y in _chunks_for_rect(rect, self.chunk_size):
            chunk_rect = self._chunk_world_rect(chunk_x, chunk_y)
            overlap = rect.intersection(chunk_rect)
            if overlap is None:
                continue
            chunk = self.ensure_chunk_cached(chunk_x, chunk_y)
            _copy_packed_rect(
                chunk,
                result,
                src_x=overlap.x - chunk_rect.x,
                src_y=overlap.y - chunk_rect.y,
                dst_x=overlap.x - rect.x,
                dst_y=overlap.y - rect.y,
                width=overlap.width,
                height=overlap.height,
            )
        return PackedStateRegion(width=rect.width, height=rect.height, state_int=bytes(result.state_int), state_vec=bytes(result.state_vec), state_misc=bytes(result.state_misc))

    def read_rect_parts(self, rect: WorldRect) -> list[tuple[WorldRect, object | None]]:
        parts: list[tuple[WorldRect, object | None]] = []
        for chunk_x, chunk_y in _chunks_for_rect(rect, self.chunk_size):
            chunk_rect = self._chunk_world_rect(chunk_x, chunk_y)
            overlap = rect.intersection(chunk_rect)
            if overlap is None:
                continue
            key = (chunk_x, chunk_y)
            if self._can_fill_empty_chunk(chunk_x, chunk_y):
                parts.append((overlap, None))
                continue
            chunk = self.ensure_chunk_cached(chunk_x, chunk_y)
            if key in self._empty_chunks:
                with self._lock:
                    self._chunks.pop(key, None)
                parts.append((overlap, None))
                continue
            packed = _extract_packed_rect(
                chunk,
                x=overlap.x - chunk_rect.x,
                y=overlap.y - chunk_rect.y,
                width=overlap.width,
                height=overlap.height,
            )
            parts.append((overlap, packed))
        return parts

    def write_rect(self, rect: WorldRect, packed_region: object) -> None:
        if rect.width <= 0 or rect.height <= 0:
            return
        packed_region = _sanitize_packed_region_for_storage(
            packed_region,
            world_y0=rect.y,
            world_height=self.store.height,
            placeholder_variant_index=self._placeholder_variant_index,
            empty_variant_index=self.tables.empty_variant_index,
        )
        for chunk_x, chunk_y in _chunks_for_rect(rect, self.chunk_size):
            chunk_rect = self._chunk_world_rect(chunk_x, chunk_y)
            overlap = rect.intersection(chunk_rect)
            if overlap is None:
                continue
            key = (chunk_x, chunk_y)
            if key in self._empty_chunks:
                chunk = _blank_chunk(self.chunk_size, self.store.height, chunk_y, self.tables.empty_variant_index)
                with self._lock:
                    self._chunks[key] = chunk
                self._empty_chunks.discard(key)
            else:
                chunk = self.ensure_chunk_cached(chunk_x, chunk_y)
            _copy_packed_rect(
                packed_region,
                chunk,
                src_x=overlap.x - rect.x,
                src_y=overlap.y - rect.y,
                dst_x=overlap.x - chunk_rect.x,
                dst_y=overlap.y - chunk_rect.y,
                width=overlap.width,
                height=overlap.height,
            )
            self._empty_chunks.discard(key)

    def has_support_anchor_source(
        self,
        x: int,
        y: int,
        *,
        support_variant_indices: set[int],
    ) -> bool:
        if x < 0 or y < 0 or x >= self.store.width or y >= self.store.height:
            return False
        chunk_x = x // self.chunk_size
        chunk_y = y // self.chunk_size
        key = (chunk_x, chunk_y)
        if key in self._empty_chunks:
            return False
        local_x = x % self.chunk_size
        local_y = y % self.chunk_size
        local_index = local_y * self.chunk_size + local_x
        chunk = self.ensure_chunk_cached(chunk_x, chunk_y)
        int_values = array("i")
        int_start = local_index * GPU_STATE_PIXEL_BYTES
        int_values.frombytes(bytes(chunk.state_int[int_start:int_start + GPU_STATE_PIXEL_BYTES]))
        variant_index = int_values[0]
        flags = int_values[2]
        if variant_index not in support_variant_indices:
            return False
        if flags & int(CellFlag.FIXPOINT):
            return True
        misc_values = array("f")
        misc_values.frombytes(bytes(chunk.state_misc[int_start:int_start + GPU_STATE_PIXEL_BYTES]))
        return misc_values[1] > 0.0


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
            log.debug("[world] generating chunk (%d,%d) for the first time", chunk_x, chunk_y)
            self.chunk_generator(self, chunk_x, chunk_y, self.chunk_size, self.seed)
            log.debug("[world] chunk (%d,%d) generation complete, total_generated=%d",
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
        chunk_save_dir: str | Path = "artifacts/gpu_chunks",
        chunk_cache_prefetch_x: int = DEFAULT_CHUNK_CACHE_PREFETCH_X,
        chunk_cache_prefetch_y: int = DEFAULT_CHUNK_CACHE_PREFETCH_Y,
    ) -> None:
        self.store = store
        self.registry = registry
        from .gpu_backend import GpuMaterialTables

        self._gpu_tables = GpuMaterialTables.from_registry(registry)
        self.viewport_width = min(int(viewport_width), store.width)
        self.viewport_height = min(int(viewport_height), store.height)
        self._support_transmission_variant_indices = {
            self._gpu_tables.variant_index_by_key[(family_id, variant_id)]
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
        self.liquid_brownian_enabled = bool(liquid_brownian_enabled)
        self.blocked_impulse_enabled = bool(blocked_impulse_enabled)
        self.directional_fallback_enabled = bool(directional_fallback_enabled)
        self.directional_fallback_angle_limit_degrees = float(directional_fallback_angle_limit_degrees)
        self.external_support_anchors = [False for _ in range(self.active_width * self.active_height)]
        self.chunk_cache = GpuChunkCache(
            store,
            registry,
            save_dir=chunk_save_dir,
            prefetch_x=chunk_cache_prefetch_x,
            prefetch_y=chunk_cache_prefetch_y,
        )
        self.gpu_simulator = None
        self._pending_gpu_writebacks: list[_PendingGpuWriteback] = []
        self._pending_flush_cooldown_steps = 0
        self._camera_idle_elapsed_seconds = self.idle_flush_cooldown_seconds
        self._last_background_io_flush_idle_seconds = 0.0
        self._camera_recently_moved = False
        self.paging_stats = PagingStats()
        log.debug("[world] ActiveWorldWindow init: store=%dx%d viewport=%dx%d active=%dx%d halo=%d",
                 store.width, store.height, self.viewport_width, self.viewport_height,
                 self.active_width, self.active_height, self.halo_cells)
        log.debug("[world] camera=(%d,%d) active_origin=(%d,%d)",
                 self.camera_x, self.camera_y, self.active_origin_x, self.active_origin_y)
        if ctx is None:
            raise RuntimeError("GPU context required — CPU backend removed")
        from .gpu_backend import GpuSimulator
        self.gpu_simulator = GpuSimulator(
            ctx,
            self.active_width,
            self.active_height,
            self.registry,
            liquid_brownian_enabled=self.liquid_brownian_enabled,
            blocked_impulse_enabled=self.blocked_impulse_enabled,
            directional_fallback_enabled=self.directional_fallback_enabled,
            directional_fallback_angle_limit_degrees=self.directional_fallback_angle_limit_degrees,
        )
        self._load_incoming_rect_into_gpu_buffer(
            self.active_rect,
            target_buffer_index=self.gpu_simulator.front_index,
            target_origin_x=self.active_origin_x,
            target_origin_y=self.active_origin_y,
        )
        self.chunk_cache.schedule_prefetch_for_rect(self.active_rect)
        self.chunk_cache.service_prefetch(max_chunks=4, collect_ready=False)
        self._set_external_support_anchors()
        log.debug("[world] GPU simulator created")

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

    def _border_has_external_support_anchor(self, rect: WorldRect, local_x: int, local_y: int) -> bool:
        world_x = rect.x + local_x
        world_y = rect.y + local_y
        for dx, dy in NEIGHBORS_8:
            neighbor_x = world_x + dx
            neighbor_y = world_y + dy
            if rect.x <= neighbor_x < rect.right and rect.y <= neighbor_y < rect.bottom:
                continue
            if self.chunk_cache.has_support_anchor_source(
                neighbor_x,
                neighbor_y,
                support_variant_indices=self._support_transmission_variant_indices,
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
            if self.chunk_cache.has_support_anchor_source(
                world_x,
                world_y,
                support_variant_indices=self._support_transmission_variant_indices,
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
            if self.chunk_cache.has_support_anchor_source(
                world_x,
                world_y,
                support_variant_indices=self._support_transmission_variant_indices,
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
            if self.chunk_cache.has_support_anchor_source(
                world_x,
                world_y,
                support_variant_indices=self._support_transmission_variant_indices,
            ):
                positions.add(world_y)
        if rect.y > 0:
            corner_y = rect.y - 1
            for world_x_candidate in (rect.x - 1, rect.x):
                if self.chunk_cache.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_variant_indices=self._support_transmission_variant_indices,
                ):
                    positions.add(corner_y)
        if rect.bottom < self.store.height:
            corner_y = rect.bottom
            for world_x_candidate in (rect.x - 1, rect.x):
                if self.chunk_cache.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_variant_indices=self._support_transmission_variant_indices,
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
            if self.chunk_cache.has_support_anchor_source(
                world_x,
                world_y,
                support_variant_indices=self._support_transmission_variant_indices,
            ):
                positions.add(world_y)
        if rect.y > 0:
            corner_y = rect.y - 1
            for world_x_candidate in (rect.right - 1, rect.right):
                if self.chunk_cache.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_variant_indices=self._support_transmission_variant_indices,
                ):
                    positions.add(corner_y)
        if rect.bottom < self.store.height:
            corner_y = rect.bottom
            for world_x_candidate in (rect.right - 1, rect.right):
                if self.chunk_cache.has_support_anchor_source(
                    world_x_candidate,
                    corner_y,
                    support_variant_indices=self._support_transmission_variant_indices,
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
        anchors = self.external_support_anchors
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
        self.external_support_anchors = anchors
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
        budget = max(1, self.pending_writeback_slice_cell_budget)
        if flush_along_y:
            slice_width = pending.rect.width
            slice_height = max(1, min(pending.rect.height - pending.major_axis_offset, budget // max(1, slice_width)))
            src_x = 0
            src_y = pending.major_axis_offset
            world_x = pending.rect.x
            world_y = pending.rect.y + pending.major_axis_offset
            pending.major_axis_offset += slice_height
            flush_complete = pending.major_axis_offset >= pending.rect.height
        else:
            slice_height = pending.rect.height
            slice_width = max(1, min(pending.rect.width - pending.major_axis_offset, budget // max(1, slice_height)))
            src_x = pending.major_axis_offset
            src_y = 0
            world_x = pending.rect.x + pending.major_axis_offset
            world_y = pending.rect.y
            pending.major_axis_offset += slice_width
            flush_complete = pending.major_axis_offset >= pending.rect.width
        packed = self.gpu_simulator.read_staged_region_bytes(
            pending.staged_region,
            x=src_x,
            y=src_y,
            width=slice_width,
            height=slice_height,
        )
        self.chunk_cache.write_rect(WorldRect(world_x, world_y, slice_width, slice_height), packed)
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
        for pending in consumed_pending:
            if pending in self._pending_gpu_writebacks:
                self._pending_gpu_writebacks.remove(pending)
                self.gpu_simulator.release_staged_region(pending.staged_region)
        if remaining:
            self.chunk_cache.collect_ready_prefetch()
        for missing in remaining:
            for part, packed in self.chunk_cache.read_rect_parts(missing):
                if packed is None:
                    self.gpu_simulator.fill_empty_region(
                        part.x - target_origin_x,
                        part.y - target_origin_y,
                        part.width,
                        part.height,
                        world_row_offset=part.y,
                        world_height=self.store.height,
                        buffer_index=target_buffer_index,
                    )
                    continue
                self.gpu_simulator.write_region_bytes(
                    part.x - target_origin_x,
                    part.y - target_origin_y,
                    part.width,
                    part.height,
                    packed.state_int,
                    packed.state_vec,
                    packed.state_misc,
                    buffer_index=target_buffer_index,
                )

    def _shift_active_window(self, new_origin_x: int, new_origin_y: int) -> None:
        log.debug("[world] shifting active window: (%d,%d) -> (%d,%d), camera=(%d,%d)",
                 self.active_origin_x, self.active_origin_y, new_origin_x, new_origin_y,
                 self.camera_x, self.camera_y)
        shift_started_at = perf_counter()
        chunk_stats_before = self.chunk_cache.snapshot_stats()
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
            raise RuntimeError("GPU simulator required.")
        for evicted_rect in evicted_rects:
            self._stage_evicted_region(evicted_rect)
        evict_elapsed = perf_counter() - evict_started_at
        self.paging_stats.last_evict_stage_seconds = evict_elapsed
        self.paging_stats.max_evict_stage_seconds = max(self.paging_stats.max_evict_stage_seconds, evict_elapsed)

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
        self.chunk_cache.schedule_prefetch_for_rect(self.active_rect)
        self.chunk_cache.service_prefetch(max_chunks=4, collect_ready=False)
        self.chunk_cache.evict_far_chunks(self.active_rect)
        self._set_external_support_anchors()
        chunk_stats_after = self.chunk_cache.snapshot_stats()
        self.paging_stats.last_shift_cache_hits = chunk_stats_after.cache_hits - chunk_stats_before.cache_hits
        self.paging_stats.last_shift_empty_hits = chunk_stats_after.empty_hits - chunk_stats_before.empty_hits
        self.paging_stats.last_shift_inflight_wait_hits = (
            chunk_stats_after.inflight_wait_hits - chunk_stats_before.inflight_wait_hits
        )
        self.paging_stats.last_shift_disk_loads = chunk_stats_after.disk_load_count - chunk_stats_before.disk_load_count
        self.paging_stats.last_shift_generates = chunk_stats_after.generate_count - chunk_stats_before.generate_count
        self.paging_stats.last_shift_saves = chunk_stats_after.save_count - chunk_stats_before.save_count
        self.paging_stats.last_shift_disk_load_seconds = (
            chunk_stats_after.disk_load_total_seconds - chunk_stats_before.disk_load_total_seconds
        )
        self.paging_stats.last_shift_generate_seconds = (
            chunk_stats_after.generate_total_seconds - chunk_stats_before.generate_total_seconds
        )
        self.paging_stats.last_shift_save_seconds = (
            chunk_stats_after.save_total_seconds - chunk_stats_before.save_total_seconds
        )
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

    def _direct_origin_for_camera(self) -> tuple[int, int]:
        max_origin_x = max(0, self.store.width - self.active_width)
        max_origin_y = max(0, self.store.height - self.active_height)
        min_origin_x = self.camera_x + self.viewport_width + self.safety_margin_cells - self.active_width
        min_origin_y = self.camera_y + self.viewport_height + self.safety_margin_cells - self.active_height
        max_origin_for_margin_x = self.camera_x - self.safety_margin_cells
        max_origin_for_margin_y = self.camera_y - self.safety_margin_cells
        low_x = max(0, min_origin_x)
        low_y = max(0, min_origin_y)
        high_x = min(max_origin_x, max_origin_for_margin_x)
        high_y = min(max_origin_y, max_origin_for_margin_y)
        if low_x > high_x:
            low_x = high_x = _clamp(self.camera_x - self.halo_cells, 0, max_origin_x)
        if low_y > high_y:
            low_y = high_y = _clamp(self.camera_y - self.halo_cells, 0, max_origin_y)
        target_x = _clamp(self.camera_x - self.halo_cells, low_x, high_x)
        target_y = _clamp(self.camera_y - self.halo_cells, low_y, high_y)
        return (
            _clamp(int(target_x), 0, max_origin_x),
            _clamp(int(target_y), 0, max_origin_y),
        )

    def ensure_resident_for_camera(self, *, force_jump: bool = False) -> None:
        log.debug("[world] ensure_resident: camera=(%d,%d) active_origin=(%d,%d) active_size=%dx%d",
                  self.camera_x, self.camera_y, self.active_origin_x, self.active_origin_y,
                  self.active_width, self.active_height)
        if force_jump:
            new_origin_x, new_origin_y = self._direct_origin_for_camera()
            if new_origin_x != self.active_origin_x or new_origin_y != self.active_origin_y:
                self._shift_active_window(new_origin_x, new_origin_y)
            return

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
            log.debug("[world] pan_camera: new camera=(%d,%d) (dx=%d, dy=%d)",
                     self.camera_x, self.camera_y, dx, dy)
        self.mark_camera_activity(moved)
        self.ensure_resident_for_camera()

    def set_camera(self, camera_x: int, camera_y: int) -> None:
        next_camera_x = _clamp(int(camera_x), 0, max(0, self.store.width - self.viewport_width))
        next_camera_y = _clamp(int(camera_y), 0, max(0, self.store.height - self.viewport_height))
        moved = next_camera_x != self.camera_x or next_camera_y != self.camera_y
        self.camera_x = next_camera_x
        self.camera_y = next_camera_y
        self.mark_camera_activity(moved)
        self.ensure_resident_for_camera(force_jump=True)

    def _active_rect_for_camera(self, camera_x: int, camera_y: int) -> WorldRect:
        next_camera_x = _clamp(int(camera_x), 0, max(0, self.store.width - self.viewport_width))
        next_camera_y = _clamp(int(camera_y), 0, max(0, self.store.height - self.viewport_height))
        saved_camera_x = self.camera_x
        saved_camera_y = self.camera_y
        self.camera_x = next_camera_x
        self.camera_y = next_camera_y
        try:
            origin_x, origin_y = self._direct_origin_for_camera()
        finally:
            self.camera_x = saved_camera_x
            self.camera_y = saved_camera_y
        return WorldRect(origin_x, origin_y, self.active_width, self.active_height)

    def prefetch_camera_region(self, camera_x: int, camera_y: int, *, submit_chunks: int = 8) -> None:
        rect = self._active_rect_for_camera(camera_x, camera_y)
        self.chunk_cache.schedule_prefetch_for_rect(rect, margin_x=0, margin_y=0, prioritize=True)
        self.chunk_cache.service_prefetch(max_chunks=max(1, int(submit_chunks)), collect_ready=False)

    def prime_camera_region(self, camera_x: int, camera_y: int) -> None:
        self.chunk_cache.prime_rect_on_disk(self._active_rect_for_camera(camera_x, camera_y))

    def prime_camera_region_sync(self, camera_x: int, camera_y: int) -> None:
        self.chunk_cache.prime_rect_now(self._active_rect_for_camera(camera_x, camera_y))

    def prepare_camera_region_sync(self, camera_x: int, camera_y: int) -> None:
        self.chunk_cache.prepare_rect_now(self._active_rect_for_camera(camera_x, camera_y))

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
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        self.gpu_simulator.paint_circle(local_x, local_y, radius, family_id, variant_id, overrides=overrides)

    def inject_pressure_world(self, world_x: int, world_y: int, radius: int, pressure_value: float) -> None:
        """Inject high pressure at a world position for explosion effects."""
        if not self.active_rect.x <= world_x < self.active_rect.right or not self.active_rect.y <= world_y < self.active_rect.bottom:
            return
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        self.gpu_simulator.inject_pressure(local_x, local_y, radius, pressure_value)

    def inject_pressure_ring_world(
        self,
        world_x: int,
        world_y: int,
        inner_radius: int,
        outer_radius: int,
        pressure_value: float,
    ) -> None:
        """Inject a ring-shaped pressure shell at a world position."""
        if not self.active_rect.x <= world_x < self.active_rect.right or not self.active_rect.y <= world_y < self.active_rect.bottom:
            return
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        self.gpu_simulator.inject_pressure_ring(local_x, local_y, inner_radius, outer_radius, pressure_value)

    def request_projectile_feedback_world(
        self,
        projectiles: list[tuple[int, int, int]],
    ) -> object:
        """Submit projectile positions (world coords) for GPU collision query."""
        if self.gpu_simulator is None:
            return None
        local_projectiles: list[tuple[int, int, int]] = []
        for wx, wy, ptype in projectiles:
            lx = wx - self.active_origin_x
            ly = wy - self.active_origin_y
            local_projectiles.append((lx, ly, ptype))
        return self.gpu_simulator.request_projectile_feedback(local_projectiles)

    def poll_projectile_feedback_world(self, token: object) -> list | None:
        """Poll projectile feedback results from GPU."""
        if self.gpu_simulator is None or token is None:
            return None
        return self.gpu_simulator.poll_projectile_feedback(token)

    def set_liquid_brownian_enabled(self, enabled: bool) -> None:
        self.liquid_brownian_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_liquid_brownian_enabled(enabled)

    def set_blocked_impulse_enabled(self, enabled: bool) -> None:
        self.blocked_impulse_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_blocked_impulse_enabled(enabled)

    def set_directional_fallback_enabled(self, enabled: bool) -> None:
        self.directional_fallback_enabled = bool(enabled)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_directional_fallback_enabled(enabled)

    def set_directional_fallback_angle_limit_degrees(self, angle_limit_degrees: float) -> None:
        self.directional_fallback_angle_limit_degrees = float(angle_limit_degrees)
        if self.gpu_simulator is not None:
            self.gpu_simulator.set_directional_fallback_angle_limit_degrees(angle_limit_degrees)

    def service_background_io(self) -> None:
        if self.gpu_simulator is None:
            return
        if self._camera_recently_moved:
            self.chunk_cache.service_prefetch(max_chunks=1, collect_ready=False)
            return
        if self._camera_idle_elapsed_seconds - self._last_background_io_flush_idle_seconds < self.idle_flush_service_interval_seconds:
            return
        if self._pending_gpu_writebacks:
            if self._flush_one_pending_gpu_writeback():
                self._last_background_io_flush_idle_seconds = self._camera_idle_elapsed_seconds
                return
        if self.chunk_cache.service_prefetch(max_chunks=1):
            self._last_background_io_flush_idle_seconds = self._camera_idle_elapsed_seconds
            return
        if self.chunk_cache.flush_one_dirty():
            self._last_background_io_flush_idle_seconds = self._camera_idle_elapsed_seconds

    def step(self, dt: float) -> None:
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required — CPU backend removed")
        self.gpu_simulator.step(dt)

    def render(self, view_mode: DebugViewMode = DebugViewMode.MATERIAL):
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        return self.gpu_simulator.render(view_mode)

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

    def readback_gpu_snapshot(self) -> Grid:
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        grid = self.gpu_simulator.readback_grid()
        grid.external_support_anchors = list(self.external_support_anchors)
        return grid

    def readback_pressure_region_world(self, world_x: int, world_y: int, width: int, height: int) -> list[list[float]]:
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        return self.gpu_simulator.readback_pressure_region(local_x, local_y, width, height)

    def snapshot_cells_region_world(
        self,
        *,
        entity_id: str,
        world_x: int,
        world_y: int,
        width: int,
        height: int,
    ) -> LocalCellsSnapshot:
        if self.gpu_simulator is None:
            raise RuntimeError("GPU simulator required.")
        local_x = world_x - self.active_origin_x
        local_y = world_y - self.active_origin_y
        return self.gpu_simulator.snapshot_cells_region(
            entity_id=entity_id,
            world_x=world_x,
            world_y=world_y,
            gx=local_x,
            gy=local_y,
            gw=width,
            gh=height,
        )

    def close(self) -> None:
        if self.gpu_simulator is not None:
            self._stage_evicted_region(self.active_rect)
            flush_budget = max(1, self.active_width * self.active_height)
            self.pending_writeback_slice_cell_budget = flush_budget
            while self._pending_gpu_writebacks:
                if not self._flush_one_pending_gpu_writeback():
                    break
        self.chunk_cache.shutdown()
