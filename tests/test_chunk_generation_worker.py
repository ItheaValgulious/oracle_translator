"""Phase 4 tests: multi-process chunk generation worker pool."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Register factories in the parent process so in-process fallback works.
from tests import _phase4_factory  # noqa: E402

_phase4_factory.register_test_factories()

from engine.chunk_generation_worker import (  # noqa: E402
    ChunkGenerationWorkerPool,
    GenerationResult,
    make_request,
)


class ChunkGenerationPoolRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = ChunkGenerationWorkerPool(
            max_workers=2,
            init_specs=["tests._phase4_factory:register_test_factories"],
        )
        self.addCleanup(lambda: self.pool.shutdown(wait=True))

    def test_generate_returns_packed_planes(self) -> None:
        req = make_request(
            factory_key="phase4.single_stone",
            chunk_x=1,
            chunk_y=2,
            chunk_size=4,
            world_width=64,
            world_height=64,
            seed=12345,
        )
        result: GenerationResult = self.pool.generate_blocking(req, timeout=20.0)
        self.assertTrue(result.ok, result.error)
        self.assertFalse(result.is_empty)
        # 4 * 4 cells, 16 bytes per pixel.
        self.assertEqual(len(result.state_int), 4 * 4 * 16)
        self.assertEqual(len(result.state_vec), 4 * 4 * 16)
        self.assertEqual(len(result.state_misc), 4 * 4 * 16)
        self.assertTrue(result.used_shared_memory)
        self.assertGreaterEqual(self.pool.stats.generate_count, 1)
        self.assertGreaterEqual(self.pool.stats.shared_generate_count, 1)

    def test_unknown_factory_returns_error_without_crashing_pool(self) -> None:
        req = make_request(
            factory_key="phase4.does_not_exist",
            chunk_x=0,
            chunk_y=0,
            chunk_size=4,
            world_width=64,
            world_height=64,
            seed=1,
        )
        result = self.pool.generate_blocking(req, timeout=20.0)
        self.assertFalse(result.ok)
        self.assertIn("not registered", result.error)

    def test_multiple_jobs_run_independently(self) -> None:
        futures = []
        for cx in range(4):
            req = make_request(
                factory_key="phase4.single_stone",
                chunk_x=cx,
                chunk_y=0,
                chunk_size=4,
                world_width=64,
                world_height=64,
                seed=42,
            )
            futures.append(self.pool.submit(req))
        results = [f.result(timeout=30.0) for f in futures]
        for r in results:
            self.assertTrue(r.ok)
            self.assertFalse(r.is_empty)
            self.assertTrue(r.used_shared_memory)
        # Same seed + same factory + same coords should be deterministic;
        # spot-check that all returned bytes are nonempty and same length.
        for r in results:
            self.assertEqual(len(r.state_int), 4 * 4 * 16)
        self.assertGreaterEqual(self.pool.stats.shared_generate_count, 4)

    def test_submit_without_shared_slot_uses_in_process_fallback(self) -> None:
        pool = ChunkGenerationWorkerPool(
            max_workers=1,
            init_specs=["tests._phase4_factory:register_test_factories"],
            shared_slot_count=0,
        )
        self.addCleanup(lambda: pool.shutdown(wait=True))
        req = make_request(
            factory_key="phase4.single_stone",
            chunk_x=0,
            chunk_y=0,
            chunk_size=4,
            world_width=64,
            world_height=64,
            seed=7,
        )

        result = pool.submit(req).result(timeout=20.0)

        self.assertTrue(result.ok, result.error)
        self.assertFalse(result.used_shared_memory)
        self.assertGreaterEqual(pool.stats.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
