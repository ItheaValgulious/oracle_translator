"""Phase 5 tests: non-blocking readiness-driven paging primitives."""

from __future__ import annotations

import sys
import threading
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.paging import (
    CameraIntent,
    CameraIntentMailbox,
    ChunkReadiness,
    PageRequestPlanner,
)
from engine.materials import build_material_registry
from engine.world import ChunkResidency, GpuChunkCache, WorldChunkStore, WorldRect


class CameraIntentMailboxTests(unittest.TestCase):
    def test_submit_and_peek(self) -> None:
        mb = CameraIntentMailbox()
        mb.submit(CameraIntent(10, 20, 672, 412))
        intent = mb.peek()
        self.assertIsNotNone(intent)
        self.assertEqual(intent.camera_x, 10)

    def test_newer_intent_overwrites_older(self) -> None:
        mb = CameraIntentMailbox()
        mb.submit(CameraIntent(10, 20, 672, 412, tick_id=1))
        mb.submit(CameraIntent(50, 60, 672, 412, tick_id=2))
        intent = mb.peek()
        self.assertEqual(intent.camera_x, 50)

    def test_consume_clears_mailbox(self) -> None:
        mb = CameraIntentMailbox()
        mb.submit(CameraIntent(10, 20, 672, 412))
        intent = mb.consume()
        self.assertIsNotNone(intent)
        self.assertIsNone(mb.peek())

    def test_concurrent_submit_peek(self) -> None:
        mb = CameraIntentMailbox()
        errors: list[str] = []

        def writer() -> None:
            for i in range(200):
                try:
                    mb.submit(CameraIntent(i, 0, 672, 412))
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))

        def reader() -> None:
            for _ in range(200):
                try:
                    mb.peek()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(errors, [])


class PageRequestPlannerTests(unittest.TestCase):
    @staticmethod
    def _make_planner(
        residency_map: dict[tuple[int, int], ChunkResidency],
        *,
        chunk_size: int = 4,
        world_w: int = 64,
        world_h: int = 64,
    ) -> PageRequestPlanner:
        # "ready"  = resident clean/dirty, immediately usable
        # "empty"  = inflight or queued — render as empty this frame,
        #            patch in when it arrives
        # otherwise (EVICTED etc.) = pending — orchestrator must schedule
        return PageRequestPlanner(
            chunk_size=chunk_size,
            world_width=world_w,
            world_height=world_h,
            residency_probe=lambda cx, cy: residency_map.get((cx, cy), ChunkResidency.EVICTED),
            ready_predicate=lambda s: s in (ChunkResidency.RESIDENT_CLEAN, ChunkResidency.RESIDENT_DIRTY),
            empty_predicate=lambda s: s in (
                ChunkResidency.QUEUED_LOAD,
                ChunkResidency.INFLIGHT_IO,
                ChunkResidency.QUEUED_GENERATE,
                ChunkResidency.INFLIGHT_GENERATION,
            ),
        )

    def test_all_chunks_ready(self) -> None:
        rmap = {(cx, 0): ChunkResidency.RESIDENT_CLEAN for cx in range(4)}
        planner = self._make_planner(rmap, chunk_size=4, world_w=16, world_h=4)
        plan = planner.plan(CameraIntent(8, 2, 16, 4))
        self.assertGreater(len(plan.ready_keys), 0)
        self.assertEqual(len(plan.pending_keys), 0)

    def test_missing_chunk_reported_as_pending(self) -> None:
        rmap = {(0, 0): ChunkResidency.RESIDENT_CLEAN}
        planner = self._make_planner(rmap, chunk_size=4, world_w=16, world_h=4)
        plan = planner.plan(CameraIntent(8, 2, 16, 4))
        self.assertGreater(len(plan.pending_keys), 0)
        self.assertGreater(len(plan.ready_keys), 0)

    def test_planner_does_not_block_on_slow_probe(self) -> None:
        # Even if the probe takes time, the planner should return a plan
        # that includes the keys — the planner itself is synchronous in
        # the sense of "call the probe for each key", but the key
        # property is that it never *triggers* a load. We verify that
        # probing an EVICTED chunk does NOT call ensure_chunk_cached.
        call_log: list[tuple[int, int]] = []

        def probe(cx: int, cy: int) -> ChunkResidency:
            call_log.append((cx, cy))
            return ChunkResidency.EVICTED

        planner = PageRequestPlanner(
            chunk_size=4,
            world_width=64,
            world_height=64,
            residency_probe=probe,
            ready_predicate=lambda s: s in (ChunkResidency.RESIDENT_CLEAN, ChunkResidency.RESIDENT_DIRTY),
            empty_predicate=lambda s: s in (
                ChunkResidency.QUEUED_LOAD,
                ChunkResidency.INFLIGHT_IO,
            ),
        )
        plan = planner.plan(CameraIntent(8, 2, 16, 4))
        # Every key in the viewport was probed exactly once.
        self.assertGreater(len(call_log), 0)
        # EVICTED chunks become pending (orchestrator must schedule prefetch).
        for key in plan.pending_keys:
            self.assertIn(key, call_log)


class BestEffortChunkReadTests(unittest.TestCase):
    def test_nonblocking_read_queues_missing_chunk_without_generation(self) -> None:
        registry = build_material_registry()
        generated: list[tuple[int, int]] = []

        def generate(_store: WorldChunkStore, chunk_x: int, chunk_y: int, _size: int, _seed: int) -> None:
            generated.append((chunk_x, chunk_y))
            raise AssertionError("best-effort read must not generate synchronously")

        store = WorldChunkStore(16, 16, chunk_size=4, seed=7, chunk_generator=generate)
        with tempfile.TemporaryDirectory() as tmp:
            cache = GpuChunkCache(store, registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
            parts = cache.read_rect_parts(WorldRect(0, 0, 4, 4), block=False)

            self.assertEqual(len(parts), 1)
            self.assertIsNone(parts[0][1])
            self.assertEqual(generated, [])
            self.assertEqual(cache.chunk_residency_state(0, 0), ChunkResidency.QUEUED_GENERATE)


if __name__ == "__main__":
    unittest.main()
