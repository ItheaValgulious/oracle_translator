"""Phase 2 tests: two-ring residency policy and async save queue.

These tests exercise ``GpuChunkCache.service_residency`` without a GPU
context. The two-ring policy is:

- Inner ring (``inner_ring_x``, ``inner_ring_y``) around the active rect:
  never evict.
- Soft band between inner and outer ring: leave resident, no forced
  eviction on the gameplay thread.
- Outside the outer ring: clean chunks are evicted immediately, dirty
  chunks are queued for async save and only evicted after the save
  acknowledgement.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.materials import build_material_registry
from engine.types import CellState
from engine.world import (
    ChunkResidency,
    GpuChunkCache,
    WorldChunkStore,
    WorldRect,
)


def _make_cache(
    tmp: str,
    *,
    seed: int = 7,
    chunk_size: int = 4,
    inner_ring_x: int = 1,
    inner_ring_y: int = 1,
    outer_ring_x: int = 2,
    outer_ring_y: int = 2,
    save_executor: ThreadPoolExecutor | None = None,
):
    registry = build_material_registry()
    calls: list[tuple[int, int]] = []

    def generate(store: WorldChunkStore, cx: int, cy: int, size: int, _seed: int) -> None:
        calls.append((cx, cy))
        store.set_cell(
            cx * size,
            cy * size,
            CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
        )

    store = WorldChunkStore(64, 64, chunk_size=chunk_size, seed=seed, chunk_generator=generate)
    cache = GpuChunkCache(
        store,
        registry,
        save_dir=tmp,
        prefetch_x=0,
        prefetch_y=0,
        inner_ring_x=inner_ring_x,
        inner_ring_y=inner_ring_y,
        outer_ring_x=outer_ring_x,
        outer_ring_y=outer_ring_y,
        save_executor=save_executor,
    )
    return cache, calls


def _chunk_rect(cx: int, cy: int, chunk_size: int = 4) -> WorldRect:
    return WorldRect(cx * chunk_size, cy * chunk_size, chunk_size, chunk_size)


class TwoRingRingMembershipTests(unittest.TestCase):
    """The inner ring must protect chunks from eviction even when they
    are clean and stale."""

    def test_inner_ring_clean_chunk_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            # Make (0,0) clean and resident.
            chunk = cache.ensure_chunk_cached(0, 0)
            cache._save_to_disk(0, 0, chunk)
            self.assertFalse(chunk.dirty)
            cache.service_residency(_chunk_rect(0, 0))
            self.assertEqual(
                cache.chunk_residency_state(0, 0),
                ChunkResidency.RESIDENT_CLEAN,
            )

    def test_soft_band_chunk_is_not_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # inner=1, outer=2 → chunk at (2,0) is in the soft band when
            # the active rect is chunk (0,0).
            cache, _ = _make_cache(tmp, inner_ring_x=1, outer_ring_x=2)
            chunk = cache.ensure_chunk_cached(2, 0)
            cache._save_to_disk(2, 0, chunk)
            cache.service_residency(_chunk_rect(0, 0))
            # Still resident, still clean.
            self.assertEqual(
                cache.chunk_residency_state(2, 0),
                ChunkResidency.RESIDENT_CLEAN,
            )


class TwoRingEvictionTests(unittest.TestCase):
    def test_clean_outer_chunk_is_evicted_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp, inner_ring_x=1, outer_ring_x=2)
            chunk = cache.ensure_chunk_cached(5, 0)
            cache._save_to_disk(5, 0, chunk)
            self.assertEqual(
                cache.chunk_residency_state(5, 0),
                ChunkResidency.RESIDENT_CLEAN,
            )
            cache.service_residency(_chunk_rect(0, 0))
            self.assertEqual(
                cache.chunk_residency_state(5, 0),
                ChunkResidency.EVICTED,
            )

    def test_dirty_outer_chunk_is_enqueued_then_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp, inner_ring_x=1, outer_ring_x=2)
            chunk = cache.ensure_chunk_cached(5, 0)
            self.assertTrue(chunk.dirty)
            cache.service_residency(_chunk_rect(0, 0))
            state = cache.chunk_residency_state(5, 0)
            # Either queued or already inflight depending on executor speed.
            self.assertIn(
                state,
                (ChunkResidency.QUEUED_SAVE, ChunkResidency.INFLIGHT_IO),
            )
            cache.wait_pending_saves(timeout=5.0)
            self.assertEqual(
                cache.chunk_residency_state(5, 0),
                ChunkResidency.EVICTED,
            )
            # And the file is on disk.
            self.assertTrue(cache._chunk_path(5, 0).exists())

    def test_pinned_chunk_is_not_evicted_even_if_outside_outer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp, inner_ring_x=1, outer_ring_x=2)
            chunk = cache.ensure_chunk_cached(5, 0)
            cache._save_to_disk(5, 0, chunk)
            cache._pinned_chunks.add((5, 0))
            cache.service_residency(_chunk_rect(0, 0))
            self.assertEqual(
                cache.chunk_residency_state(5, 0),
                ChunkResidency.RESIDENT_CLEAN,
            )


class TwoRingNonBlockingTests(unittest.TestCase):
    """service_residency must never block the caller, even when the save
    executor stalls."""

    def test_service_residency_does_not_block_on_slow_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = threading.Event()

            class _BlockingExecutor(ThreadPoolExecutor):
                def submit(self, fn, /, *args, **kwargs):
                    def _wrapped(*a, **kw):
                        release.wait(timeout=5.0)
                        return fn(*a, **kw)
                    return super().submit(_wrapped, *args, **kwargs)

            executor = _BlockingExecutor(max_workers=1, thread_name_prefix="test-slow-save")
            try:
                cache, _ = _make_cache(
                    tmp, inner_ring_x=1, outer_ring_x=2, save_executor=executor
                )
                cache.ensure_chunk_cached(5, 0)  # dirty
                start = time.perf_counter()
                cache.service_residency(_chunk_rect(0, 0))
                elapsed = time.perf_counter() - start
                # The call must not have waited for the save (which is
                # blocked on the event).
                self.assertLess(elapsed, 0.5)
                snapshot = cache.snapshot_stats()
                self.assertGreaterEqual(
                    snapshot.queued_write_count + snapshot.inflight_io_count,
                    1,
                )
            finally:
                release.set()
                executor.shutdown(wait=True)


class TwoRingSnapshotCountersTests(unittest.TestCase):
    def test_inflight_save_counts_under_inflight_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = threading.Event()

            class _BlockingExecutor(ThreadPoolExecutor):
                def submit(self, fn, /, *args, **kwargs):
                    def _wrapped(*a, **kw):
                        release.wait(timeout=5.0)
                        return fn(*a, **kw)
                    return super().submit(_wrapped, *args, **kwargs)

            executor = _BlockingExecutor(max_workers=1, thread_name_prefix="test-snap-save")
            try:
                cache, _ = _make_cache(
                    tmp, inner_ring_x=1, outer_ring_x=2, save_executor=executor
                )
                cache.ensure_chunk_cached(5, 0)
                cache.service_residency(_chunk_rect(0, 0))
                snapshot = cache.snapshot_stats()
                self.assertGreaterEqual(snapshot.inflight_io_count, 1)
            finally:
                release.set()
                executor.shutdown(wait=True)


class DiskBackedEvictionTests(unittest.TestCase):
    def test_evict_clean_disk_backed_chunk_keeps_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            chunk = cache.ensure_chunk_cached(3, 0)
            cache._save_to_disk(3, 0, chunk)

            result = cache.evict_clean_disk_backed_for_rect(_chunk_rect(3, 0))

            self.assertEqual(result["evicted"], 1)
            self.assertEqual(
                cache.chunk_residency_state(3, 0),
                ChunkResidency.EVICTED,
            )
            self.assertTrue(cache._chunk_path(3, 0).exists())

    def test_evict_clean_disk_backed_chunk_skips_dirty_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            cache.ensure_chunk_cached(3, 0)

            result = cache.evict_clean_disk_backed_for_rect(_chunk_rect(3, 0))

            self.assertEqual(result["evicted"], 0)
            self.assertEqual(result["dirty_skipped"], 1)
            self.assertEqual(
                cache.chunk_residency_state(3, 0),
                ChunkResidency.RESIDENT_DIRTY,
            )


if __name__ == "__main__":
    unittest.main()
