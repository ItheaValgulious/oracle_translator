"""Phase 3: dedicated IO worker process for chunk disk read/write.

The worker is a single child process driven by two queues:

- ``request_queue``: receives ``IoRequest`` records (READ / WRITE / SHUTDOWN).
- ``response_queue``: receives ``IoResponse`` records back.

Internally the worker dispatches READ ops to a multi-thread pool and
WRITE ops to a smaller pool, so reads never queue behind writes. Large
payload data flows through parent-owned shared-memory slots; metadata
flows by queue.

The client (``ChunkIoWorkerClient``) owns the process and exposes
``read(path)`` / ``write(path, data)`` that block the caller thread (not
the gameplay thread — Phase 5 will wire this in async). If the worker
dies or times out, the caller records a fallback hit and falls back to
in-process IO.
"""

from __future__ import annotations

import enum
import logging
import multiprocessing as mp
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class IoOpType(enum.IntEnum):
    READ = 1
    WRITE = 2
    SHUTDOWN = 99


@dataclass
class IoRequest:
    op: IoOpType
    request_id: str
    path: str = ""
    payload: bytes = b""
    slot_name: str = ""
    slot_capacity: int = 0
    byte_count: int = 0


@dataclass
class IoResponse:
    request_id: str
    ok: bool
    payload: bytes = b""
    error: str = ""
    slot_name: str = ""
    byte_count: int = 0
    used_shared_memory: bool = False


class SharedMemorySlot:
    __slots__ = ("name", "capacity", "shm")

    def __init__(self, *, capacity: int) -> None:
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, int(capacity)))
        self.name = self.shm.name
        self.capacity = int(capacity)

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class SharedMemorySlotPool:
    """Bounded parent-owned shared-memory slots for worker payloads."""

    def __init__(self, *, slot_count: int = 2, slot_size: int = 8 * 1024 * 1024) -> None:
        self._slots = [
            SharedMemorySlot(capacity=slot_size)
            for _ in range(max(0, int(slot_count)))
        ]
        self._available = list(self._slots)
        self._condition = threading.Condition()

    @property
    def enabled(self) -> bool:
        return bool(self._slots)

    @property
    def slot_size(self) -> int:
        return 0 if not self._slots else self._slots[0].capacity

    def acquire(self, *, min_size: int, timeout: float) -> SharedMemorySlot | None:
        if min_size > self.slot_size or not self.enabled:
            return None
        with self._condition:
            if not self._available:
                self._condition.wait(timeout=max(0.0, float(timeout)))
            if not self._available:
                return None
            return self._available.pop()

    def release(self, slot: SharedMemorySlot | None) -> None:
        if slot is None:
            return
        with self._condition:
            self._available.append(slot)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            slots = list(self._slots)
            self._available.clear()
            self._slots.clear()
        for slot in slots:
            slot.close()
            slot.unlink()


def _worker_main(
    request_queue: "mp.Queue[IoRequest]",
    response_queue: "mp.Queue[IoResponse]",
    *,
    read_threads: int = 2,
    write_threads: int = 1,
) -> None:
    reads = ThreadPoolExecutor(max_workers=max(1, read_threads), thread_name_prefix="io-read")
    writes = ThreadPoolExecutor(max_workers=max(1, write_threads), thread_name_prefix="io-write")

    def _do_read(req: IoRequest) -> None:
        try:
            data = Path(req.path).read_bytes()
            if req.slot_name:
                if len(data) > req.slot_capacity:
                    response_queue.put(IoResponse(
                        req.request_id,
                        False,
                        error=f"shared slot too small: {len(data)} > {req.slot_capacity}",
                    ))
                    return
                shm = shared_memory.SharedMemory(name=req.slot_name)
                try:
                    shm.buf[:len(data)] = data
                finally:
                    shm.close()
                response_queue.put(IoResponse(
                    req.request_id,
                    True,
                    slot_name=req.slot_name,
                    byte_count=len(data),
                    used_shared_memory=True,
                ))
            else:
                response_queue.put(IoResponse(req.request_id, True, payload=data))
        except FileNotFoundError:
            response_queue.put(IoResponse(
                req.request_id,
                True,
                payload=b"",
                slot_name=req.slot_name,
                byte_count=0,
                used_shared_memory=bool(req.slot_name),
            ))
        except Exception as exc:  # noqa: BLE001
            response_queue.put(IoResponse(req.request_id, False, error=str(exc)))

    def _do_write(req: IoRequest) -> None:
        try:
            path = Path(req.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            if req.slot_name:
                shm = shared_memory.SharedMemory(name=req.slot_name)
                try:
                    payload = bytes(shm.buf[:max(0, int(req.byte_count))])
                finally:
                    shm.close()
            else:
                payload = req.payload
            tmp.write_bytes(payload)
            os.replace(tmp, path)
            response_queue.put(IoResponse(
                req.request_id,
                True,
                slot_name=req.slot_name,
                byte_count=max(0, int(req.byte_count)) if req.slot_name else len(req.payload),
                used_shared_memory=bool(req.slot_name),
            ))
        except Exception as exc:  # noqa: BLE001
            response_queue.put(IoResponse(req.request_id, False, error=str(exc)))

    try:
        while True:
            req: IoRequest = request_queue.get()
            if req.op == IoOpType.SHUTDOWN:
                response_queue.put(IoResponse(req.request_id, True))
                break
            if req.op == IoOpType.READ:
                reads.submit(_do_read, req)
            elif req.op == IoOpType.WRITE:
                writes.submit(_do_write, req)
            else:
                response_queue.put(
                    IoResponse(req.request_id, False, error=f"unknown op {int(req.op)}")
                )
    finally:
        reads.shutdown(wait=True)
        writes.shutdown(wait=True)


@dataclass
class _PendingResult:
    event: threading.Event
    response: Optional[IoResponse] = None


class ChunkIoWorkerStats:
    __slots__ = (
        "read_count",
        "write_count",
        "shared_read_count",
        "shared_write_count",
        "fallback_count",
        "last_error",
    )

    def __init__(self) -> None:
        self.read_count = 0
        self.write_count = 0
        self.shared_read_count = 0
        self.shared_write_count = 0
        self.fallback_count = 0
        self.last_error: str = ""


class ChunkIoWorkerClient:
    def __init__(
        self,
        *,
        read_threads: int = 2,
        write_threads: int = 1,
        spawn: bool = True,
        shared_slot_count: int = 2,
        shared_slot_size: int = 8 * 1024 * 1024,
    ) -> None:
        self._stats = ChunkIoWorkerStats()
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingResult] = {}
        self._read_threads = read_threads
        self._write_threads = write_threads
        self._shared_pool = SharedMemorySlotPool(
            slot_count=shared_slot_count,
            slot_size=shared_slot_size,
        )
        self._ctx = mp.get_context("spawn")
        self._request_queue: "mp.Queue[IoRequest]" = self._ctx.Queue()
        self._response_queue: "mp.Queue[IoResponse]" = self._ctx.Queue()
        self._process = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        if spawn:
            self.start()

    @property
    def stats(self) -> ChunkIoWorkerStats:
        return self._stats

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stopped.clear()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._request_queue, self._response_queue),
            kwargs={"read_threads": self._read_threads, "write_threads": self._write_threads},
            name="chunk-io-worker",
            daemon=True,
        )
        self._process.start()
        self._reader_thread = threading.Thread(
            target=self._drain_responses, name="chunk-io-reader", daemon=True
        )
        self._reader_thread.start()

    def _drain_responses(self) -> None:
        while not self._stopped.is_set():
            try:
                resp: IoResponse = self._response_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                pending = self._pending.pop(resp.request_id, None)
            if pending is None:
                continue
            pending.response = resp
            pending.event.set()

    def _submit(self, request: IoRequest, timeout: float) -> IoResponse:
        pending = _PendingResult(event=threading.Event())
        with self._lock:
            self._pending[request.request_id] = pending
        try:
            self._request_queue.put(request)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._pending.pop(request.request_id, None)
            self._stats.fallback_count += 1
            self._stats.last_error = f"submit failed: {exc}"
            return IoResponse(request.request_id, False, error=str(exc))
        if not pending.event.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(request.request_id, None)
            self._stats.fallback_count += 1
            self._stats.last_error = f"timeout after {timeout}s"
            return IoResponse(request.request_id, False, error="timeout")
        assert pending.response is not None
        return pending.response

    def read_response(self, path: str | Path, *, timeout: float = 5.0) -> IoResponse:
        slot = self._shared_pool.acquire(min_size=1, timeout=timeout)
        try:
            req = IoRequest(
                op=IoOpType.READ,
                request_id=uuid.uuid4().hex,
                path=str(path),
                slot_name="" if slot is None else slot.name,
                slot_capacity=0 if slot is None else slot.capacity,
            )
            resp = self._submit(req, timeout=timeout)
            if resp.ok and resp.used_shared_memory and slot is not None and resp.byte_count > 0:
                resp.payload = bytes(slot.shm.buf[:resp.byte_count])
            if resp.used_shared_memory:
                self._stats.shared_read_count += 1
            return resp
        finally:
            self._shared_pool.release(slot)

    def read(self, path: str | Path, *, timeout: float = 5.0) -> bytes | None:
        resp = self.read_response(path, timeout=timeout)
        if not resp.ok:
            return None
        self._stats.read_count += 1
        return resp.payload if resp.payload else None

    def write_response(self, path: str | Path, data: bytes, *, timeout: float = 5.0) -> IoResponse:
        payload = bytes(data)
        slot = self._shared_pool.acquire(min_size=len(payload), timeout=timeout)
        if slot is not None:
            slot.shm.buf[:len(payload)] = payload
        req = IoRequest(
            op=IoOpType.WRITE,
            request_id=uuid.uuid4().hex,
            path=str(path),
            payload=b"" if slot is not None else payload,
            slot_name="" if slot is None else slot.name,
            slot_capacity=0 if slot is None else slot.capacity,
            byte_count=len(payload),
        )
        try:
            resp = self._submit(req, timeout=timeout)
            if resp.used_shared_memory:
                self._stats.shared_write_count += 1
            return resp
        finally:
            self._shared_pool.release(slot)

    def write(self, path: str | Path, data: bytes, *, timeout: float = 5.0) -> bool:
        resp = self.write_response(path, data, timeout=timeout)
        if not resp.ok:
            return False
        self._stats.write_count += 1
        return True

    def shutdown(self, *, timeout: float = 2.0) -> None:
        if self._process is None:
            self._shared_pool.close()
            return
        try:
            req = IoRequest(op=IoOpType.SHUTDOWN, request_id=uuid.uuid4().hex)
            self._submit(req, timeout=timeout)
        except Exception:  # noqa: BLE001
            pass
        self._stopped.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                try:
                    self._process.terminate()
                except Exception:  # noqa: BLE001
                    pass
        self._process = None
        self._shared_pool.close()
