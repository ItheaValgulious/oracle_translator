"""Phase 9: experiment harness for paging and disk benchmarks.

Implements the offline, programmatic side of goal.md §7. Two parts:

1. ``DiskBenchmark`` — measures sequential and pseudo-random chunk file
   read latency, reporting mean and p95 plus per-chunk byte size. Runs
   independently of the GPU so it can sit in the regular test suite.
2. ``PageShiftHarness`` — exercises ``GpuChunkCache`` against three
   well-defined scenarios:
   - ``page_shift_pure_cache``: chunks already resident in RAM; the
     "shift" only re-uses cache hits. Asserts ``0 generate / 0 load /
     0 save`` inside the measured window.
   - ``page_shift_disk_only``: chunks have been written to disk and
     evicted; the shift must produce ``> 0 disk loads`` and ``0
     generate``.
   - ``page_shift_with_generate``: chunks are neither resident nor on
     disk; the shift must produce ``> 0 generate``.

The live runtime experiment (game process + debug HTTP server) stays in
``scripts/run_paging_experiments.py``; this harness is its
process-internal complement.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class DiskReadResult:
    label: str
    sample_count: int
    bytes_per_chunk: int
    mean_seconds: float
    p95_seconds: float
    samples: list[float] = field(default_factory=list)


class DiskBenchmark:
    """Sequential / pseudo-random file-read benchmark."""

    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("DiskBenchmark needs at least one path")
        self._paths = [Path(p) for p in paths]

    @staticmethod
    def _p95(samples: list[float]) -> float:
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0]
        sorted_samples = sorted(samples)
        idx = int(round(0.95 * (len(sorted_samples) - 1)))
        return sorted_samples[idx]

    def _measure(self, ordered_paths: list[Path], label: str) -> DiskReadResult:
        samples: list[float] = []
        size = 0
        for path in ordered_paths:
            start = time.perf_counter()
            data = path.read_bytes()
            samples.append(time.perf_counter() - start)
            size = len(data) if size == 0 else size
        return DiskReadResult(
            label=label,
            sample_count=len(samples),
            bytes_per_chunk=size,
            mean_seconds=statistics.fmean(samples) if samples else 0.0,
            p95_seconds=self._p95(samples),
            samples=samples,
        )

    def sequential(self) -> DiskReadResult:
        return self._measure(self._paths, "sequential")

    def pseudo_random(self, *, seed: int = 0) -> DiskReadResult:
        rng = random.Random(seed)
        shuffled = list(self._paths)
        rng.shuffle(shuffled)
        return self._measure(shuffled, "pseudo_random")


@dataclass
class ShiftStats:
    case: str
    generate_count_delta: int
    disk_load_count_delta: int
    save_count_delta: int
    duration_seconds: float


class PageShiftHarness:
    """Drive a ``GpuChunkCache`` through pure-cache / disk-only / generate
    scenarios and report deltas in cache stats over the shift window.

    All paths are best-effort and synchronous within the harness — the
    caller is responsible for setting up the cache state before each run.
    """

    def __init__(self, cache, *, chunk_coords: list[tuple[int, int]]) -> None:
        if not chunk_coords:
            raise ValueError("PageShiftHarness needs at least one chunk coord")
        self._cache = cache
        self._coords = list(chunk_coords)

    def _snapshot_stats(self) -> tuple[int, int, int]:
        s = self._cache.stats
        return (s.generate_count, s.disk_load_count, s.save_count)

    def _shift(self, case_name: str) -> ShiftStats:
        before = self._snapshot_stats()
        start = time.perf_counter()
        for cx, cy in self._coords:
            self._cache.ensure_chunk_cached(cx, cy)
        elapsed = time.perf_counter() - start
        after = self._snapshot_stats()
        return ShiftStats(
            case=case_name,
            generate_count_delta=after[0] - before[0],
            disk_load_count_delta=after[1] - before[1],
            save_count_delta=after[2] - before[2],
            duration_seconds=elapsed,
        )

    def page_shift_pure_cache(self) -> ShiftStats:
        """All target chunks must already be resident.

        Caller responsibility: ensure each ``chunk_coord`` is in
        ``cache._chunks`` before calling.
        """
        return self._shift("page_shift_pure_cache")

    def page_shift_disk_only(self, *, evict_first: Callable[[], None]) -> ShiftStats:
        """Target chunks live on disk only.

        ``evict_first`` is called immediately before the measurement
        window to drop chunks from RAM (caller knows how to do this
        safely — typically by saving to disk then popping from
        ``cache._chunks``).
        """
        evict_first()
        return self._shift("page_shift_disk_only")

    def page_shift_with_generate(self) -> ShiftStats:
        """Target chunks are neither resident nor on disk."""
        return self._shift("page_shift_with_generate")
