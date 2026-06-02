"""Phase 5: non-blocking readiness-driven paging primitives.

The runtime contract from goal.md §4 is:

- main thread only submits *camera intent* (where it wants the camera to
  go) — it never waits for chunks,
- a planner derives the target active rect,
- if a needed chunk is still being loaded / generated, the corresponding
  region is treated as temporarily empty,
- patches arrive later when the chunk lands.

This module provides the small primitives:

- ``CameraIntentMailbox``: lock-free mailbox holding the latest intent.
  Old intents are overwritten by newer ones; the consumer only sees the
  freshest target.
- ``ChunkReadiness``: enum describing whether a chunk is ready, being
  prepared, or should be treated as empty for this frame.
- ``PageRequestPlanner``: pure function that, given the camera intent
  and the cache state, produces (target_rect, ready_keys,
  pending_keys, empty_keys). It never blocks.

Phase 6 wires these into the GPU owner thread; for now they stand alone
so we can prove the non-blocking guarantee in isolation.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass
class CameraIntent:
    camera_x: int
    camera_y: int
    viewport_width: int
    viewport_height: int
    halo_cells: int = 0
    tick_id: int = 0


class CameraIntentMailbox:
    """Single-slot mailbox; newer intents overwrite older ones."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intent: Optional[CameraIntent] = None
        self._tick = 0

    def submit(self, intent: CameraIntent) -> None:
        with self._lock:
            self._tick += 1
            self._intent = CameraIntent(
                camera_x=int(intent.camera_x),
                camera_y=int(intent.camera_y),
                viewport_width=int(intent.viewport_width),
                viewport_height=int(intent.viewport_height),
                halo_cells=int(intent.halo_cells),
                tick_id=self._tick,
            )

    def peek(self) -> Optional[CameraIntent]:
        with self._lock:
            return self._intent

    def consume(self) -> Optional[CameraIntent]:
        with self._lock:
            intent = self._intent
            self._intent = None
            return intent


class ChunkReadiness(enum.IntEnum):
    READY = 0
    PENDING = 1
    EMPTY = 2


@dataclass(frozen=True)
class PagePlan:
    target_rect: tuple[int, int, int, int]  # (x, y, width, height)
    ready_keys: tuple[tuple[int, int], ...]
    pending_keys: tuple[tuple[int, int], ...]
    empty_keys: tuple[tuple[int, int], ...]


class PageRequestPlanner:
    """Pure planner that maps a camera intent to a per-chunk readiness map.

    The planner asks the cache for chunk residency *without* triggering
    any synchronous loads. Chunks that aren't resident are reported as
    PENDING (so the caller can schedule prefetch) and the renderer
    treats them as temporarily empty.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        world_width: int,
        world_height: int,
        residency_probe: Callable[[int, int], object],
        ready_predicate: Callable[[object], bool],
        empty_predicate: Callable[[object], bool] | None = None,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.world_width = int(world_width)
        self.world_height = int(world_height)
        self._probe = residency_probe
        self._ready = ready_predicate
        self._empty = empty_predicate or (lambda _state: False)

    def _viewport_rect(self, intent: CameraIntent) -> tuple[int, int, int, int]:
        vw = max(0, intent.viewport_width)
        vh = max(0, intent.viewport_height)
        halo = max(0, intent.halo_cells)
        cx = intent.camera_x - vw // 2 - halo
        cy = intent.camera_y - vh // 2 - halo
        # Clamp to world bounds.
        cx = max(0, min(self.world_width - 1, cx))
        cy = max(0, min(self.world_height - 1, cy))
        right = min(self.world_width, cx + vw + 2 * halo)
        bottom = min(self.world_height, cy + vh + 2 * halo)
        return (cx, cy, max(0, right - cx), max(0, bottom - cy))

    def _chunk_keys_for_rect(self, rect: tuple[int, int, int, int]) -> Iterable[tuple[int, int]]:
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return ()
        min_cx = max(0, x) // self.chunk_size
        max_cx = max(0, x + w - 1) // self.chunk_size
        min_cy = max(0, y) // self.chunk_size
        max_cy = max(0, y + h - 1) // self.chunk_size
        keys: list[tuple[int, int]] = []
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                keys.append((cx, cy))
        return keys

    def plan(self, intent: CameraIntent) -> PagePlan:
        rect = self._viewport_rect(intent)
        ready: list[tuple[int, int]] = []
        pending: list[tuple[int, int]] = []
        empty: list[tuple[int, int]] = []
        for key in self._chunk_keys_for_rect(rect):
            state = self._probe(*key)
            if self._empty(state):
                empty.append(key)
            elif self._ready(state):
                ready.append(key)
            else:
                pending.append(key)
        return PagePlan(
            target_rect=rect,
            ready_keys=tuple(ready),
            pending_keys=tuple(pending),
            empty_keys=tuple(empty),
        )
