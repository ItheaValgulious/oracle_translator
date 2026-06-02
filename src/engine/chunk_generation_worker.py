"""Phase 4: multi-process chunk generation worker pool.

The pool runs deterministic terrain generation in child processes so the
main process never CPU-stalls on chunk creation. Each job is described
by a generator factory key + chunk coords + seed. Results return as
packed binary planes (``state_int``, ``state_vec``, ``state_misc``) that
the main process drops straight into the chunk cache.

Determinism note: the worker only invokes generator factories registered
through ``register_generator_factory``. The main process and the worker
must register the same factory keys before the worker starts so the
seed-deterministic terrain matches exactly between in-process fallback
and out-of-process generation.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, replace
from typing import Callable, Optional

from .chunk_io_worker import SharedMemorySlot, SharedMemorySlotPool

log = logging.getLogger(__name__)


_FACTORY_REGISTRY: dict[str, Callable[..., Callable]] = {}


def register_generator_factory(key: str, factory: Callable[..., Callable]) -> None:
    """Register a generator factory under ``key``.

    The factory takes no positional args and returns a callable matching
    the ``WorldChunkStore.chunk_generator`` signature
    ``(store_or_writer, cx, cy, chunk_size, seed) -> None``.
    """
    _FACTORY_REGISTRY[key] = factory


def get_generator_factory(key: str) -> Callable[..., Callable]:
    return _FACTORY_REGISTRY[key]


@dataclass
class GenerationRequest:
    request_id: str
    factory_key: str
    chunk_x: int
    chunk_y: int
    chunk_size: int
    world_width: int
    world_height: int
    seed: int
    slot_name: str = ""
    slot_capacity: int = 0


@dataclass
class GenerationResult:
    request_id: str
    ok: bool
    chunk_x: int
    chunk_y: int
    chunk_size: int
    state_int: bytes = b""
    state_vec: bytes = b""
    state_misc: bytes = b""
    is_empty: bool = False
    error: str = ""
    used_shared_memory: bool = False
    slot_name: str = ""
    state_int_byte_count: int = 0
    state_vec_byte_count: int = 0
    state_misc_byte_count: int = 0


def _worker_init(init_specs: list[str] | None = None) -> None:
    """Child process initializer.

    ``init_specs`` is a list of ``"module:callable"`` strings the worker
    imports and calls (with no args) at startup. The expected callable
    populates the worker's ``_FACTORY_REGISTRY`` via
    ``register_generator_factory``. This is how the parent makes its
    factories available to the child without pickling closures.
    """
    if not init_specs:
        return
    import importlib

    for spec in init_specs:
        if ":" not in spec:
            continue
        module_name, attr = spec.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr)
            fn()
        except Exception:  # noqa: BLE001
            log.exception("[gen-worker] init spec %s failed", spec)


def _generate_one(request: GenerationRequest) -> GenerationResult:
    try:
        factory = _FACTORY_REGISTRY.get(request.factory_key)
        if factory is None:
            return GenerationResult(
                request_id=request.request_id,
                ok=False,
                chunk_x=request.chunk_x,
                chunk_y=request.chunk_y,
                chunk_size=request.chunk_size,
                error=f"factory '{request.factory_key}' not registered in worker",
            )
        generator = factory()
        # Import lazily so this module remains importable without
        # bringing in the rest of the engine package on every process.
        from .materials import build_material_registry
        from .world import _PackedChunkTerrainWriter
        from .gpu_backend import GpuMaterialTables

        registry = build_material_registry()
        tables = GpuMaterialTables.from_registry(registry)
        writer = _PackedChunkTerrainWriter(
            world_width=request.world_width,
            world_height=request.world_height,
            chunk_x=request.chunk_x,
            chunk_y=request.chunk_y,
            chunk_size=request.chunk_size,
            seed=request.seed,
            tables=tables,
        )
        generator(writer, request.chunk_x, request.chunk_y, request.chunk_size, request.seed)
        packed = writer.finalize()
        is_empty = writer._nondefault_writes <= 0  # noqa: SLF001
        state_int = bytes(packed.state_int)
        state_vec = bytes(packed.state_vec)
        state_misc = bytes(packed.state_misc)
        if request.slot_name:
            total_size = len(state_int) + len(state_vec) + len(state_misc)
            if total_size > request.slot_capacity:
                return GenerationResult(
                    request_id=request.request_id,
                    ok=False,
                    chunk_x=request.chunk_x,
                    chunk_y=request.chunk_y,
                    chunk_size=request.chunk_size,
                    error=f"shared slot too small: {total_size} > {request.slot_capacity}",
                )
            from multiprocessing import shared_memory

            shm = shared_memory.SharedMemory(name=request.slot_name)
            try:
                offset = 0
                shm.buf[offset:offset + len(state_int)] = state_int
                offset += len(state_int)
                shm.buf[offset:offset + len(state_vec)] = state_vec
                offset += len(state_vec)
                shm.buf[offset:offset + len(state_misc)] = state_misc
            finally:
                shm.close()
            return GenerationResult(
                request_id=request.request_id,
                ok=True,
                chunk_x=request.chunk_x,
                chunk_y=request.chunk_y,
                chunk_size=request.chunk_size,
                is_empty=is_empty,
                used_shared_memory=True,
                slot_name=request.slot_name,
                state_int_byte_count=len(state_int),
                state_vec_byte_count=len(state_vec),
                state_misc_byte_count=len(state_misc),
            )
        return GenerationResult(
            request_id=request.request_id,
            ok=True,
            chunk_x=request.chunk_x,
            chunk_y=request.chunk_y,
            chunk_size=request.chunk_size,
            state_int=state_int,
            state_vec=state_vec,
            state_misc=state_misc,
            is_empty=is_empty,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("[gen-worker] generation failed for (%d,%d)", request.chunk_x, request.chunk_y)
        return GenerationResult(
            request_id=request.request_id,
            ok=False,
            chunk_x=request.chunk_x,
            chunk_y=request.chunk_y,
            chunk_size=request.chunk_size,
            error=str(exc),
        )


class ChunkGenerationPoolStats:
    __slots__ = ("generate_count", "shared_generate_count", "fallback_count", "last_error")

    def __init__(self) -> None:
        self.generate_count = 0
        self.shared_generate_count = 0
        self.fallback_count = 0
        self.last_error: str = ""


class ChunkGenerationWorkerPool:
    """Multi-process generation pool.

    Submitting a job returns a ``Future[GenerationResult]``. If the
    underlying executor cannot accept the job (shutdown, crashed), the
    caller increments ``stats.fallback_count`` and runs the generation
    in-process via the same ``_generate_one`` codepath.
    """

    def __init__(
        self,
        *,
        max_workers: int = 2,
        init_specs: list[str] | None = None,
        shared_slot_count: int | None = None,
        shared_slot_size: int = 8 * 1024 * 1024,
    ) -> None:
        self._max_workers = max(1, int(max_workers))
        self._init_specs = list(init_specs or [])
        self._stats = ChunkGenerationPoolStats()
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._executor: Optional[ProcessPoolExecutor] = None
        self._shared_pool = SharedMemorySlotPool(
            slot_count=self._max_workers * 4 if shared_slot_count is None else int(shared_slot_count),
            slot_size=shared_slot_size,
        )

    @property
    def stats(self) -> ChunkGenerationPoolStats:
        return self._stats

    def start(self) -> None:
        if self._executor is not None:
            return
        self._executor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=self._ctx,
            initializer=_worker_init,
            initargs=(self._init_specs,),
        )

    def _hydrate_shared_result(self, result: GenerationResult, slot: SharedMemorySlot | None) -> GenerationResult:
        if result.ok and result.used_shared_memory and slot is not None:
            offset = 0
            state_int_end = offset + result.state_int_byte_count
            result.state_int = bytes(slot.shm.buf[offset:state_int_end])
            offset = state_int_end
            state_vec_end = offset + result.state_vec_byte_count
            result.state_vec = bytes(slot.shm.buf[offset:state_vec_end])
            offset = state_vec_end
            state_misc_end = offset + result.state_misc_byte_count
            result.state_misc = bytes(slot.shm.buf[offset:state_misc_end])
            with self._lock:
                self._stats.shared_generate_count += 1
        if result.ok:
            with self._lock:
                self._stats.generate_count += 1
        return result

    def submit(self, request: GenerationRequest) -> Future[GenerationResult]:
        if self._executor is None:
            self.start()
        assert self._executor is not None
        slot = self._shared_pool.acquire(min_size=1, timeout=0.0)
        worker_request = replace(request)
        outer: Future[GenerationResult] = Future()
        if slot is None:
            with self._lock:
                self._stats.fallback_count += 1
                self._stats.last_error = "shared generation slot unavailable"
            try:
                result = _generate_one(replace(request, slot_name="", slot_capacity=0))
                outer.set_result(self._hydrate_shared_result(result, None))
            except Exception as exc:  # noqa: BLE001
                outer.set_exception(exc)
            return outer
        if slot is not None:
            worker_request.slot_name = slot.name
            worker_request.slot_capacity = slot.capacity
        try:
            inner = self._executor.submit(_generate_one, worker_request)
        except Exception as exc:  # noqa: BLE001
            self._shared_pool.release(slot)
            with self._lock:
                self._stats.fallback_count += 1
                self._stats.last_error = str(exc)
            # Run in-process so the caller still gets a usable future.
            try:
                outer.set_result(_generate_one(replace(request, slot_name="", slot_capacity=0)))
            except Exception as inner:  # noqa: BLE001
                outer.set_exception(inner)
            return outer

        def _complete(done: Future[GenerationResult]) -> None:
            try:
                result = done.result()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._stats.fallback_count += 1
                    self._stats.last_error = str(exc)
                try:
                    result = _generate_one(replace(request, slot_name="", slot_capacity=0))
                except Exception as inner:  # noqa: BLE001
                    outer.set_exception(inner)
                    return
            try:
                hydrated = self._hydrate_shared_result(result, slot)
            except Exception as exc:  # noqa: BLE001
                outer.set_exception(exc)
            else:
                outer.set_result(hydrated)
            finally:
                self._shared_pool.release(slot)

        inner.add_done_callback(_complete)
        return outer

    def generate_blocking(self, request: GenerationRequest, *, timeout: float = 10.0) -> GenerationResult:
        future = self.submit(request)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats.fallback_count += 1
                self._stats.last_error = str(exc)
            return _generate_one(replace(request, slot_name="", slot_capacity=0))

    def shutdown(self, *, wait: bool = False) -> None:
        if self._executor is None:
            return
        try:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        self._executor = None
        self._shared_pool.close()


def make_request(
    *,
    factory_key: str,
    chunk_x: int,
    chunk_y: int,
    chunk_size: int,
    world_width: int,
    world_height: int,
    seed: int,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=uuid.uuid4().hex,
        factory_key=factory_key,
        chunk_x=chunk_x,
        chunk_y=chunk_y,
        chunk_size=chunk_size,
        world_width=world_width,
        world_height=world_height,
        seed=seed,
    )
