"""Runtime integration tests for chunk worker and residency wiring."""

from __future__ import annotations

import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.chunk_generation_worker import GenerationResult
from engine.chunk_io_worker import IoResponse
from engine.materials import build_material_registry
from engine.types import CellFlag, CellState
from engine.world import ActiveWorldWindow, GpuChunkCache, WorldChunkStore
from game.terrain import TerrainGenerator


class _WorkerBackedGenerator:
    supports_packed_chunk_generation = True
    chunk_generation_worker_factory_key = "tests.worker_backed"
    chunk_generation_worker_init_specs: list[str] = []

    def generate_chunk(
        self,
        store: WorldChunkStore,
        chunk_x: int,
        chunk_y: int,
        chunk_size: int,
        seed: int,
    ) -> None:
        del seed
        store.set_cell(
            chunk_x * chunk_size,
            chunk_y * chunk_size,
            CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
        )


class _FakeGenerationPool:
    def __init__(self) -> None:
        self.cache: GpuChunkCache | None = None
        self.requests: list[tuple[object, float]] = []
        self.stats = SimpleNamespace(fallback_count=0)

    def generate_blocking(self, request, *, timeout: float = 10.0) -> GenerationResult:
        assert self.cache is not None
        self.requests.append((request, timeout))
        chunk_size = int(request.chunk_size)
        cell_count = chunk_size * chunk_size
        empty_variant = self.cache.tables.empty_variant_index
        state_int = array("i", [empty_variant, 0, int(CellFlag.NONE), 0]) * cell_count
        state_vec = array("f", [0.0]) * (cell_count * 4)
        state_misc = array("f", [20.0, 0.0, 1.0, 0.0]) * cell_count
        state_int[0] = self.cache.tables.variant_index_by_key[("stone", "stone_platform")]
        return GenerationResult(
            request_id=request.request_id,
            ok=True,
            chunk_x=request.chunk_x,
            chunk_y=request.chunk_y,
            chunk_size=chunk_size,
            state_int=state_int.tobytes(),
            state_vec=state_vec.tobytes(),
            state_misc=state_misc.tobytes(),
            is_empty=False,
        )

    def shutdown(self, *, wait: bool = False) -> None:
        del wait


class _FakeIoWorkerClient:
    def __init__(self) -> None:
        self.stats = SimpleNamespace(fallback_count=0, read_count=0, write_count=0)
        self.read_calls: list[str] = []
        self.write_calls: list[str] = []
        self._payloads: dict[str, bytes] = {}

    def read_response(self, path: str | Path, *, timeout: float = 5.0) -> IoResponse:
        del timeout
        path_str = str(path)
        self.read_calls.append(path_str)
        return IoResponse(request_id="read", ok=True, payload=self._payloads.get(path_str, b""))

    def write_response(self, path: str | Path, data: bytes, *, timeout: float = 5.0) -> IoResponse:
        del timeout
        path_str = str(path)
        self.write_calls.append(path_str)
        self._payloads[path_str] = bytes(data)
        return IoResponse(request_id="write", ok=True)

    def shutdown(self) -> None:
        return


class _RecordingExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, bool]] = []

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({
            "wait": wait,
            "cancel_futures": cancel_futures,
        })


class _RecordingGenerationPool:
    def __init__(self) -> None:
        self.stats = SimpleNamespace(fallback_count=0)
        self.shutdown_waits: list[bool] = []

    def shutdown(self, *, wait: bool = False) -> None:
        self.shutdown_waits.append(wait)


class _RecordingIoWorkerClient:
    def __init__(self) -> None:
        self.stats = SimpleNamespace(fallback_count=0, read_count=0, write_count=0)
        self.shutdown_count = 0

    def shutdown(self) -> None:
        self.shutdown_count += 1


class TerrainGeneratorMetadataTests(unittest.TestCase):
    def test_terrain_generator_exposes_packed_and_worker_metadata(self) -> None:
        gen = TerrainGenerator(42, build_material_registry())
        self.assertTrue(gen.supports_packed_chunk_generation)
        self.assertTrue(gen.chunk_generation_worker_factory_key)
        self.assertIn(
            "src.game.terrain:register_chunk_generation_worker_factories",
            gen.chunk_generation_worker_init_specs,
        )


class GpuChunkCacheWorkerIntegrationTests(unittest.TestCase):
    def test_generation_worker_pool_is_used_for_chunk_generation(self) -> None:
        registry = build_material_registry()
        store = WorldChunkStore(
            16,
            16,
            chunk_size=4,
            seed=7,
            chunk_generator=_WorkerBackedGenerator().generate_chunk,
        )
        fake_pool = _FakeGenerationPool()
        with tempfile.TemporaryDirectory() as tmp:
            cache = GpuChunkCache(
                store,
                registry,
                save_dir=tmp,
                prefetch_x=0,
                prefetch_y=0,
                generation_worker_pool=fake_pool,
            )
            fake_pool.cache = cache
            chunk = cache._generate_chunk(0, 0)
            self.assertEqual(len(fake_pool.requests), 1)
            self.assertTrue(chunk.dirty)
            self.assertNotIn((0, 0), cache._empty_chunks)

    def test_io_worker_client_is_used_for_chunk_save_and_load(self) -> None:
        registry = build_material_registry()

        def generate(
            store: WorldChunkStore,
            chunk_x: int,
            chunk_y: int,
            chunk_size: int,
            seed: int,
        ) -> None:
            del seed
            store.set_cell(
                chunk_x * chunk_size,
                chunk_y * chunk_size,
                CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
            )

        store = WorldChunkStore(16, 16, chunk_size=4, seed=9, chunk_generator=generate)
        fake_io = _FakeIoWorkerClient()
        with tempfile.TemporaryDirectory() as tmp:
            cache = GpuChunkCache(
                store,
                registry,
                save_dir=tmp,
                prefetch_x=0,
                prefetch_y=0,
                io_worker_client=fake_io,
            )
            chunk = cache._generate_chunk_in_process(0, 0)
            cache._save_to_disk(0, 0, chunk)
            loaded = cache._load_from_disk(0, 0)
            self.assertIsNotNone(loaded)
            self.assertGreaterEqual(len(fake_io.write_calls), 1)
            self.assertGreaterEqual(len(fake_io.read_calls), 1)

    def test_shutdown_waits_for_owned_executors_and_worker_pools(self) -> None:
        registry = build_material_registry()
        store = WorldChunkStore(16, 16, chunk_size=4, seed=13)
        with tempfile.TemporaryDirectory() as tmp:
            cache = GpuChunkCache(
                store,
                registry,
                save_dir=tmp,
                prefetch_x=0,
                prefetch_y=0,
            )
            cache._prime_executor.shutdown(wait=True, cancel_futures=True)
            cache._executor.shutdown(wait=True, cancel_futures=True)
            if cache._owns_save_executor:
                cache._save_executor.shutdown(wait=True, cancel_futures=True)

            prime_executor = _RecordingExecutor()
            load_executor = _RecordingExecutor()
            save_executor = _RecordingExecutor()
            generation_pool = _RecordingGenerationPool()
            io_worker = _RecordingIoWorkerClient()
            cache._prime_executor = prime_executor  # type: ignore[assignment]
            cache._executor = load_executor  # type: ignore[assignment]
            cache._save_executor = save_executor  # type: ignore[assignment]
            cache._owns_save_executor = True
            cache._generation_worker_pool = generation_pool  # type: ignore[assignment]
            cache._owns_generation_worker_pool = True
            cache._io_worker = io_worker  # type: ignore[assignment]
            cache._owns_io_worker = True

            cache.shutdown()

            self.assertEqual(prime_executor.shutdown_calls, [{"wait": True, "cancel_futures": True}])
            self.assertEqual(load_executor.shutdown_calls, [{"wait": True, "cancel_futures": True}])
            self.assertEqual(save_executor.shutdown_calls, [{"wait": True, "cancel_futures": True}])
            self.assertEqual(generation_pool.shutdown_waits, [True])
            self.assertEqual(io_worker.shutdown_count, 1)


class ActiveWorldWindowRuntimeIntegrationTests(unittest.TestCase):
    def test_background_io_keeps_servicing_prefetch_while_idle(self) -> None:
        class _DummyChunkCache:
            def __init__(self) -> None:
                self.prefetch_calls: list[tuple[int, bool]] = []

            def service_prefetch(self, *, max_chunks: int = 1, collect_ready: bool = True) -> bool:
                self.prefetch_calls.append((max_chunks, collect_ready))
                return False

            def flush_one_dirty(self) -> bool:
                return False

        world = object.__new__(ActiveWorldWindow)
        world.gpu_simulator = object()
        world.chunk_cache = _DummyChunkCache()
        world._camera_recently_moved = False
        world._camera_idle_elapsed_seconds = 0.0
        world._last_background_io_flush_idle_seconds = 0.0
        world.idle_flush_service_interval_seconds = 300.0
        world._pending_gpu_writebacks = []
        world._flush_one_pending_gpu_writeback = lambda: False

        ActiveWorldWindow.service_background_io(world)

        self.assertEqual(world.chunk_cache.prefetch_calls, [(1, True)])

    def test_page_shift_uses_service_residency(self) -> None:
        try:
            import moderngl
        except Exception:
            self.skipTest("moderngl not available")
        try:
            ctx = moderngl.create_standalone_context()
        except Exception:
            self.skipTest("standalone GL context unavailable")
        try:
            registry = build_material_registry()
            store = WorldChunkStore(16, 8, chunk_size=4, seed=11)
            with tempfile.TemporaryDirectory() as tmp:
                world = ActiveWorldWindow(
                    store,
                    registry,
                    viewport_width=4,
                    viewport_height=4,
                    halo_cells=0,
                    page_shift_cells=4,
                    safety_margin_cells=0,
                    idle_flush_service_interval_seconds=0.0,
                    ctx=ctx,
                    initial_camera_x=0,
                    initial_camera_y=0,
                    chunk_save_dir=tmp,
                    chunk_cache_prefetch_x=0,
                    chunk_cache_prefetch_y=0,
                )
                try:
                    with patch.object(world.chunk_cache, "service_residency", wraps=world.chunk_cache.service_residency) as service_residency:
                        with patch.object(world.chunk_cache, "evict_far_chunks", wraps=world.chunk_cache.evict_far_chunks) as evict_far_chunks:
                            world.pan_camera(4, 0)
                            self.assertGreaterEqual(service_residency.call_count, 1)
                            self.assertEqual(evict_far_chunks.call_count, 0)
                finally:
                    world.close()
        finally:
            try:
                ctx.release()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
