"""Phase 7: async snapshot mailbox for entity local-cell readbacks.

Goal.md §2 requires that hero / EnemyA-C / projectile collision and
hazard checks no longer call ``snapshot_cells_region_world()``
synchronously. Instead:

- The GPU owner thread *submits* a snapshot for an entity each tick
  (or whenever it has fresh GPU state).
- Consumers (collision logic, hazard checks) read the latest *completed*
  snapshot from the mailbox without blocking.
- If no fresh snapshot exists, the consumer reuses the most recent one
  it has — never blocks waiting for a new one.

Each ``SnapshotEnvelope`` carries the required metadata:

- ``entity_id``
- world-space origin (``origin_x``, ``origin_y``) and size
- ``tick_id`` (monotonically increasing per entity)
- ``age_frames`` (how many frames behind the current frame this snapshot
  is — set by ``SnapshotMailbox.consume`` at read time)
- ``packed`` payload (opaque ``LocalCellsSnapshot``-like object)
- ``submitted_at`` perf-counter timestamp for instrumentation
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SnapshotEnvelope:
    entity_id: str
    origin_x: int
    origin_y: int
    width: int
    height: int
    tick_id: int
    packed: Any = None
    submitted_at: float = field(default_factory=time.perf_counter)
    age_frames: int = 0


class SnapshotMailbox:
    """Single-slot mailbox per entity. New snapshots overwrite older ones.

    Consumers call ``consume(current_tick)`` to peek the latest envelope
    and have its ``age_frames`` filled in relative to ``current_tick``.
    The envelope is NOT cleared on consume — collision checks may pull
    the same snapshot multiple times across substeps; only a newer
    submit replaces it. Use ``clear()`` to explicitly drop.
    """

    def __init__(self, entity_id: str) -> None:
        self._entity_id = entity_id
        self._lock = threading.Lock()
        self._envelope: Optional[SnapshotEnvelope] = None
        self._submit_count = 0
        self._consume_count = 0

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def submit_count(self) -> int:
        return self._submit_count

    @property
    def consume_count(self) -> int:
        return self._consume_count

    def submit(self, envelope: SnapshotEnvelope) -> None:
        with self._lock:
            self._envelope = envelope
            self._submit_count += 1

    def consume(self, current_tick: int = 0) -> Optional[SnapshotEnvelope]:
        with self._lock:
            env = self._envelope
            if env is None:
                return None
            self._consume_count += 1
            # Return a shallow copy so the caller can mutate age_frames
            # without affecting the cached envelope.
            return SnapshotEnvelope(
                entity_id=env.entity_id,
                origin_x=env.origin_x,
                origin_y=env.origin_y,
                width=env.width,
                height=env.height,
                tick_id=env.tick_id,
                packed=env.packed,
                submitted_at=env.submitted_at,
                age_frames=max(0, int(current_tick) - int(env.tick_id)),
            )

    def peek_tick(self) -> Optional[int]:
        with self._lock:
            return None if self._envelope is None else self._envelope.tick_id

    def clear(self) -> None:
        with self._lock:
            self._envelope = None


class SnapshotMailboxRegistry:
    """Holds per-entity mailboxes keyed by entity id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: dict[str, SnapshotMailbox] = {}

    def get_or_create(self, entity_id: str) -> SnapshotMailbox:
        with self._lock:
            box = self._boxes.get(entity_id)
            if box is None:
                box = SnapshotMailbox(entity_id)
                self._boxes[entity_id] = box
            return box

    def submit(self, envelope: SnapshotEnvelope) -> None:
        box = self.get_or_create(envelope.entity_id)
        box.submit(envelope)

    def consume(self, entity_id: str, current_tick: int = 0) -> Optional[SnapshotEnvelope]:
        with self._lock:
            box = self._boxes.get(entity_id)
        if box is None:
            return None
        return box.consume(current_tick=current_tick)

    def entity_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._boxes.keys())

    def freshness_report(self, current_tick: int = 0) -> dict[str, int]:
        """Per-entity ``age_frames`` value for F3 / debug overlays."""
        out: dict[str, int] = {}
        with self._lock:
            boxes = list(self._boxes.items())
        for entity_id, box in boxes:
            tick = box.peek_tick()
            if tick is None:
                out[entity_id] = -1  # never received
            else:
                out[entity_id] = max(0, int(current_tick) - int(tick))
        return out

    def drop(self, entity_id: str) -> None:
        with self._lock:
            self._boxes.pop(entity_id, None)
