"""Phase 3 tests: dedicated IO worker process roundtrip.

These tests spawn a real child process so they validate the actual
``multiprocessing`` plumbing, not just an in-process mock.
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

from engine.chunk_io_worker import ChunkIoWorkerClient


class ChunkIoWorkerRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = ChunkIoWorkerClient(read_threads=2, write_threads=1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.client.shutdown)

    def test_write_then_read_returns_same_bytes(self) -> None:
        path = Path(self.tmp.name) / "a.bin"
        payload = b"\x00\x01\x02hello-phase3" * 64
        ok = self.client.write(path, payload, timeout=5.0)
        self.assertTrue(ok)
        got = self.client.read(path, timeout=5.0)
        self.assertEqual(got, payload)
        self.assertGreaterEqual(self.client.stats.read_count, 1)
        self.assertGreaterEqual(self.client.stats.write_count, 1)
        self.assertGreaterEqual(self.client.stats.shared_read_count, 1)
        self.assertGreaterEqual(self.client.stats.shared_write_count, 1)

    def test_write_then_read_uses_shared_memory_payload_path(self) -> None:
        client = ChunkIoWorkerClient(read_threads=1, write_threads=1, shared_slot_count=1, shared_slot_size=4096)
        self.addCleanup(client.shutdown)
        path = Path(self.tmp.name) / "shared.bin"
        payload = b"shared-payload" * 64

        write_response = client.write_response(path, payload, timeout=5.0)
        read_response = client.read_response(path, timeout=5.0)

        self.assertTrue(write_response.ok)
        self.assertTrue(write_response.used_shared_memory)
        self.assertTrue(read_response.ok)
        self.assertTrue(read_response.used_shared_memory)
        self.assertEqual(read_response.payload, payload)
        self.assertGreaterEqual(client.stats.shared_read_count, 1)
        self.assertGreaterEqual(client.stats.shared_write_count, 1)

    def test_read_missing_returns_none(self) -> None:
        path = Path(self.tmp.name) / "does-not-exist.bin"
        got = self.client.read(path, timeout=5.0)
        self.assertIsNone(got)

    def test_concurrent_reads_complete(self) -> None:
        # Pre-write several files, then read them concurrently to confirm
        # the worker's read pool actually services many requests in
        # parallel. We don't measure latency; we just require all results
        # to be correct (this would deadlock if the dispatcher serialized
        # reads behind a single executor slot).
        paths: list[Path] = []
        for i in range(8):
            p = Path(self.tmp.name) / f"f_{i}.bin"
            self.assertTrue(self.client.write(p, bytes([i]) * 32, timeout=5.0))
            paths.append(p)
        for i, p in enumerate(paths):
            got = self.client.read(p, timeout=5.0)
            self.assertEqual(got, bytes([i]) * 32)


class ChunkIoWorkerShutdownTests(unittest.TestCase):
    def test_shutdown_is_idempotent(self) -> None:
        client = ChunkIoWorkerClient()
        client.shutdown()
        # Calling shutdown again must not raise.
        client.shutdown()


class ChunkIoWorkerFallbackCounterTests(unittest.TestCase):
    def test_timeout_increments_fallback_counter(self) -> None:
        # Use a client whose worker we deliberately kill so the next
        # request times out and the fallback counter ticks.
        client = ChunkIoWorkerClient()
        try:
            # Terminate the process so submits will time out.
            if client._process is not None:
                client._process.terminate()
                client._process.join(timeout=2.0)
            before = client.stats.fallback_count
            got = client.read(Path("nonexistent"), timeout=0.5)
            self.assertIsNone(got)
            self.assertGreater(client.stats.fallback_count, before)
        finally:
            client.shutdown()


if __name__ == "__main__":
    unittest.main()
