"""Phase 1 tests: chunk residency metadata and expanded F3/debug metrics.

These tests do not depend on a GPU context — they validate that the
chunk cache exposes the new residency state machine inputs and the
expanded `ChunkCacheDebugSnapshot` fields that Phases 2-7 will rely on.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.materials import build_material_registry
from engine.types import CellState
from engine.world import (
    ChunkCacheDebugSnapshot,
    ChunkResidency,
    GpuChunkCache,
    WorldChunkStore,
    WorldRect,
)


def _make_cache(tmp: str, *, seed: int = 7, chunk_size: int = 4):
    registry = build_material_registry()
    calls: list[tuple[int, int]] = []

    def generate(store: WorldChunkStore, cx: int, cy: int, size: int, _seed: int) -> None:
        calls.append((cx, cy))
        store.set_cell(
            cx * size,
            cy * size,
            CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
        )

    store = WorldChunkStore(16, 16, chunk_size=chunk_size, seed=seed, chunk_generator=generate)
    cache = GpuChunkCache(store, registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
    return cache, calls


class ChunkResidencyStateTests(unittest.TestCase):
    def test_enum_values_are_stable(self) -> None:
        # External monitoring may key off these names — keep them stable.
        self.assertEqual(ChunkResidency.UNKNOWN, 0)
        self.assertEqual(ChunkResidency.RESIDENT_CLEAN, 1)
        self.assertEqual(ChunkResidency.RESIDENT_DIRTY, 2)
        self.assertEqual(ChunkResidency.QUEUED_LOAD, 3)
        self.assertEqual(ChunkResidency.QUEUED_SAVE, 4)
        self.assertEqual(ChunkResidency.QUEUED_GENERATE, 5)
        self.assertEqual(ChunkResidency.INFLIGHT_IO, 6)
        self.assertEqual(ChunkResidency.INFLIGHT_GENERATION, 7)
        self.assertEqual(ChunkResidency.INFLIGHT_GPU_WRITEBACK, 8)
        self.assertEqual(ChunkResidency.EVICTED, 9)

    def test_unloaded_chunk_reports_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            self.assertEqual(cache.chunk_residency_state(0, 0), ChunkResidency.EVICTED)

    def test_generated_chunk_reports_resident_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            cache.ensure_chunk_cached(0, 0)
            self.assertEqual(cache.chunk_residency_state(0, 0), ChunkResidency.RESIDENT_DIRTY)

    def test_after_save_chunk_is_resident_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            chunk = cache.ensure_chunk_cached(0, 0)
            cache._save_to_disk(0, 0, chunk)
            self.assertFalse(chunk.dirty)
            self.assertEqual(cache.chunk_residency_state(0, 0), ChunkResidency.RESIDENT_CLEAN)


class ChunkCacheSnapshotTests(unittest.TestCase):
    def test_snapshot_exposes_phase1_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            snapshot = cache.snapshot_stats()
            self.assertIsInstance(snapshot, ChunkCacheDebugSnapshot)
            # New fields must exist and start at zero on a fresh cache.
            for field_name in (
                "clean_resident_chunks",
                "dirty_resident_chunks",
                "queued_read_count",
                "queued_write_count",
                "queued_generate_count",
                "inflight_io_count",
                "inflight_generation_count",
                "worker_fallback_count",
                "sync_blocking_fetch_count",
            ):
                self.assertTrue(hasattr(snapshot, field_name), field_name)
                self.assertEqual(getattr(snapshot, field_name), 0, field_name)

    def test_snapshot_counts_dirty_and_clean_residency_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            # Generate three chunks → all dirty until saved.
            for cx in range(3):
                cache.ensure_chunk_cached(cx, 0)
            snapshot = cache.snapshot_stats()
            self.assertEqual(snapshot.cached_chunks, 3)
            self.assertEqual(snapshot.dirty_resident_chunks, 3)
            self.assertEqual(snapshot.clean_resident_chunks, 0)
            # Save one → it becomes clean resident.
            cache._save_to_disk(0, 0, cache._chunks[(0, 0)])
            snapshot = cache.snapshot_stats()
            self.assertEqual(snapshot.cached_chunks, 3)
            self.assertEqual(snapshot.dirty_resident_chunks, 2)
            self.assertEqual(snapshot.clean_resident_chunks, 1)

    def test_prefetch_queue_is_split_between_read_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            cache.schedule_prefetch_for_rect(WorldRect(0, 0, 4, 4), margin_x=1, margin_y=0)
            snapshot = cache.snapshot_stats()
            self.assertEqual(
                snapshot.queued_read_count + snapshot.queued_generate_count,
                snapshot.prefetch_queued,
            )
            self.assertGreater(snapshot.queued_generate_count, 0)

    def test_worker_fallback_counter_increments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            cache.record_worker_fallback()
            cache.record_worker_fallback()
            snapshot = cache.snapshot_stats()
            self.assertEqual(snapshot.worker_fallback_count, 2)

    def test_sync_blocking_fetch_counter_increments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache, _ = _make_cache(tmp)
            cache.ensure_chunk_cached(0, 0)
            cache.ensure_chunk_cached(0, 0)
            snapshot = cache.snapshot_stats()
            self.assertEqual(snapshot.sync_blocking_fetch_count, 2)


class GpuWritebackDepthAliasTests(unittest.TestCase):
    """The ActiveWorldWindow `gpu_writeback_queue_depth` property must
    mirror the existing `pending_writeback_count` to disambiguate it
    from disk-save backlog at the F3/debug layer."""

    def test_alias_mirrors_pending_writeback_count(self) -> None:
        try:
            import moderngl
        except Exception:
            self.skipTest("moderngl not available")
        try:
            ctx = moderngl.create_standalone_context()
        except Exception:
            self.skipTest("standalone GL context unavailable")
        try:
            from engine.world import ActiveWorldWindow
            registry = build_material_registry()
            store = WorldChunkStore(16, 16, chunk_size=4, seed=99)
            world = ActiveWorldWindow(
                store,
                registry,
                viewport_width=8,
                viewport_height=8,
                halo_cells=0,
                ctx=ctx,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )
            self.assertEqual(
                world.gpu_writeback_queue_depth,
                world.pending_writeback_count,
            )
        finally:
            try:
                ctx.release()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
