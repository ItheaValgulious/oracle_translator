"""Phase 9 tests: experiment harness — disk benchmark + page-shift cases.

Covers goal.md §7 acceptance criteria:

- pure cache shift path produces 0 generate / 0 load / 0 save
- disk-only path produces nonzero disk loads with zero generation
- cold path isolates generation as the dominant source of stalls
- direct disk benchmark reports mean + p95 + per-chunk byte size
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.experiment_harness import DiskBenchmark, PageShiftHarness
from engine.materials import build_material_registry
from engine.types import CellState
from engine.world import GpuChunkCache, WorldChunkStore

_RUN_PAGING_SPEC = importlib.util.spec_from_file_location(
    "run_paging_experiments",
    ROOT / "scripts" / "run_paging_experiments.py",
)
assert _RUN_PAGING_SPEC is not None and _RUN_PAGING_SPEC.loader is not None
run_paging_experiments = importlib.util.module_from_spec(_RUN_PAGING_SPEC)
_RUN_PAGING_SPEC.loader.exec_module(run_paging_experiments)


def _make_cache(tmp: str, *, seed: int = 7, chunk_size: int = 4):
    registry = build_material_registry()

    def generate(store: WorldChunkStore, cx: int, cy: int, size: int, _seed: int) -> None:
        store.set_cell(
            cx * size,
            cy * size,
            CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
        )

    store = WorldChunkStore(64, 64, chunk_size=chunk_size, seed=seed, chunk_generator=generate)
    cache = GpuChunkCache(store, registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)
    return cache


class DiskBenchmarkTests(unittest.TestCase):
    def test_sequential_and_random_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths: list[Path] = []
            payload = b"X" * 4096
            for i in range(8):
                p = tmp_path / f"chunk_{i}.bin"
                p.write_bytes(payload)
                paths.append(p)
            bench = DiskBenchmark(paths)
            seq = bench.sequential()
            rnd = bench.pseudo_random(seed=42)
            for r in (seq, rnd):
                self.assertEqual(r.sample_count, 8)
                self.assertEqual(r.bytes_per_chunk, 4096)
                self.assertGreaterEqual(r.mean_seconds, 0.0)
                self.assertGreaterEqual(r.p95_seconds, r.mean_seconds * 0.5)


class PageShiftPureCacheTests(unittest.TestCase):
    def test_pure_cache_shift_zero_generate_load_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            coords = [(cx, 0) for cx in range(4)]
            # Pre-populate so all chunks are resident in RAM.
            for cx, cy in coords:
                cache.ensure_chunk_cached(cx, cy)
            harness = PageShiftHarness(cache, chunk_coords=coords)
            stats = harness.page_shift_pure_cache()
            self.assertEqual(stats.generate_count_delta, 0)
            self.assertEqual(stats.disk_load_count_delta, 0)
            self.assertEqual(stats.save_count_delta, 0)


class PageShiftDiskOnlyTests(unittest.TestCase):
    def test_disk_only_shift_has_disk_loads_no_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            coords = [(cx, 0) for cx in range(4)]
            # Generate, save to disk, then evict from RAM.
            for cx, cy in coords:
                chunk = cache.ensure_chunk_cached(cx, cy)
                cache._save_to_disk(cx, cy, chunk)

            def _evict() -> None:
                for cx, cy in coords:
                    cache._chunks.pop((cx, cy), None)
                # Also clear the empty-set in case a chunk was marked empty.
                for cx, cy in coords:
                    cache._empty_chunks.discard((cx, cy))

            harness = PageShiftHarness(cache, chunk_coords=coords)
            stats = harness.page_shift_disk_only(evict_first=_evict)
            self.assertGreater(stats.disk_load_count_delta, 0)
            self.assertEqual(stats.generate_count_delta, 0)


class EmptyChunkMarkerTests(unittest.TestCase):
    def test_empty_chunk_marker_prevents_generation_on_reload(self) -> None:
        registry = build_material_registry()

        def generate_empty(
            store: WorldChunkStore,
            cx: int,
            cy: int,
            size: int,
            seed: int,
        ) -> None:
            del store, cx, cy, size, seed

        with tempfile.TemporaryDirectory() as tmp:
            store = WorldChunkStore(16, 16, chunk_size=4, seed=11, chunk_generator=generate_empty)
            cache = GpuChunkCache(store, registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)

            loaded = cache._load_for_worker(0, 0)

            self.assertIsNone(loaded)
            self.assertTrue(cache._chunk_path(0, 0).exists())

            calls: list[tuple[int, int]] = []

            def fail_if_called(
                store: WorldChunkStore,
                cx: int,
                cy: int,
                size: int,
                seed: int,
            ) -> None:
                del store, size, seed
                calls.append((cx, cy))
                raise AssertionError("empty chunk marker should avoid generation")

            reloaded_store = WorldChunkStore(16, 16, chunk_size=4, seed=11, chunk_generator=fail_if_called)
            reloaded = GpuChunkCache(reloaded_store, registry, save_dir=tmp, prefetch_x=0, prefetch_y=0)

            loaded_again = reloaded._load_for_worker(0, 0)

            self.assertIsNone(loaded_again)
            self.assertEqual(calls, [])
            self.assertIn((0, 0), reloaded._empty_chunks)
            self.assertEqual(reloaded.stats.disk_load_count, 1)


class PageShiftWithGenerateTests(unittest.TestCase):
    def test_cold_path_has_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(tmp)
            # Use coords that have never been touched (and have no disk file).
            coords = [(cx, 5) for cx in range(4)]
            harness = PageShiftHarness(cache, chunk_coords=coords)
            stats = harness.page_shift_with_generate()
            self.assertGreater(stats.generate_count_delta, 0)


class PagingExperimentConclusionTests(unittest.TestCase):
    def test_summarize_step_main_thread_marks_small_step_records_within_budget(self) -> None:
        summary = run_paging_experiments.summarize_step_main_thread([
            {
                "status": {
                    "perf": {
                        "main_tick_total_last_ms": 4.2,
                        "update_game_total_last_ms": 3.7,
                        "world_step_submit_cpu_last_ms": 0.6,
                    }
                }
            },
            {
                "status": {
                    "perf": {
                        "main_tick_total_last_ms": 9.5,
                        "update_game_total_last_ms": 8.4,
                        "world_step_submit_cpu_last_ms": 0.8,
                    }
                }
            },
        ])

        self.assertTrue(summary["within_budget"])
        self.assertEqual(summary["max_main_tick_ms"], 9.5)
        self.assertEqual(summary["max_update_game_ms"], 8.4)
        self.assertEqual(summary["max_world_step_submit_cpu_ms"], 0.8)

    def test_build_derived_conclusions_reports_nonblocking_flags(self) -> None:
        def _page_shift_case(*, loads: int, generates: int, saves: int, gen_ms: float, disk_ms: float, tick_ms: float) -> dict:
            return {
                "page_shift": {
                    "shift_event_summary": {
                        "sum_last_shift_disk_loads": loads,
                        "sum_last_shift_generates": generates,
                        "sum_last_shift_saves": saves,
                        "sum_last_shift_disk_load_ms": disk_ms,
                        "sum_last_shift_generate_ms": gen_ms,
                        "max_shift_ms": 220.0,
                        "max_incoming_load_ms": 18.0,
                    },
                    "steps": [
                        {
                            "status": {
                                "perf": {
                                    "main_tick_total_last_ms": tick_ms,
                                    "update_game_total_last_ms": tick_ms - 1.0,
                                    "world_step_submit_cpu_last_ms": 0.5,
                                }
                            }
                        }
                    ],
                }
            }

        def _teleport_entry(main_tick_ms: float, update_ms: float) -> dict:
            return {
                "shifted_status": {
                    "perf": {
                        "main_tick_total_last_ms": main_tick_ms,
                        "update_game_total_last_ms": update_ms,
                        "world_step_submit_cpu_last_ms": 0.7,
                    }
                },
                "after_window": {
                    "max_main_tick_ms": max(main_tick_ms, 14.0),
                    "max_update_game_ms": max(update_ms, 11.0),
                },
            }

        results = {
            "cases": {
                "page_shift_with_generate": _page_shift_case(
                    loads=0, generates=2, saves=1, gen_ms=450.0, disk_ms=0.0, tick_ms=7.0
                ),
                "page_shift_disk_only": _page_shift_case(
                    loads=3, generates=0, saves=0, gen_ms=0.0, disk_ms=12.0, tick_ms=8.0
                ),
                "page_shift_pure_cache": _page_shift_case(
                    loads=0, generates=0, saves=0, gen_ms=0.0, disk_ms=0.0, tick_ms=5.0
                ),
                "existing_biome_teleports": {
                    "teleports": {
                        "plains": _teleport_entry(3.5, 3.1),
                    }
                },
                "existing_biome_teleports_alt_targets": {
                    "teleports": {
                        "alpine": _teleport_entry(6.4, 5.7),
                    }
                },
                "front_mountain_back_plains_teleports": {
                    "teleports": {
                        "underground": _teleport_entry(4.9, 4.6),
                    }
                },
            }
        }

        main_thread_summary, derived = run_paging_experiments.build_derived_conclusions(results)

        self.assertTrue(derived["page_shift_pure_cache_zero_generate_load_save_in_window"])
        self.assertTrue(derived["page_shift_disk_only_has_loads_zero_generate_in_window"])
        self.assertTrue(derived["page_shift_with_generate_zero_disk_loads_in_window"])
        self.assertTrue(derived["page_shift_with_generate_generation_time_dominates_disk_time"])
        self.assertTrue(derived["normal_movement_main_thread_nonblocking"])
        self.assertTrue(derived["teleport_main_thread_nonblocking"])
        self.assertEqual(
            main_thread_summary["page_shift_cases"]["page_shift_with_generate"]["max_main_tick_ms"],
            7.0,
        )
        self.assertEqual(
            main_thread_summary["teleport_cases"]["existing_biome_teleports"]["plains"]["max_update_game_ms"],
            3.1,
        )

    def test_build_derived_conclusions_marks_budget_exceedance(self) -> None:
        results = {
            "cases": {
                "page_shift_with_generate": {
                    "page_shift": {
                        "shift_event_summary": {
                            "sum_last_shift_disk_loads": 0,
                            "sum_last_shift_generates": 1,
                            "sum_last_shift_saves": 0,
                            "sum_last_shift_disk_load_ms": 0.0,
                            "sum_last_shift_generate_ms": 100.0,
                        },
                        "steps": [
                            {
                                "status": {
                                    "perf": {
                                        "main_tick_total_last_ms": 48.0,
                                        "update_game_total_last_ms": 40.0,
                                        "world_step_submit_cpu_last_ms": 0.5,
                                    }
                                }
                            }
                        ],
                    }
                },
                "page_shift_disk_only": {"page_shift": {"shift_event_summary": {}, "steps": []}},
                "page_shift_pure_cache": {"page_shift": {"shift_event_summary": {}, "steps": []}},
                "existing_biome_teleports": {
                    "teleports": {
                        "plains": {
                            "shifted_status": {
                                "perf": {
                                    "main_tick_total_last_ms": 60.0,
                                    "update_game_total_last_ms": 44.0,
                                    "world_step_submit_cpu_last_ms": 0.6,
                                }
                            },
                            "after_window": {
                                "max_main_tick_ms": 60.0,
                                "max_update_game_ms": 44.0,
                            },
                        }
                    }
                },
                "existing_biome_teleports_alt_targets": {"teleports": {}},
                "front_mountain_back_plains_teleports": {"teleports": {}},
            }
        }

        _summary, derived = run_paging_experiments.build_derived_conclusions(results)

        self.assertFalse(derived["normal_movement_main_thread_nonblocking"])
        self.assertFalse(derived["teleport_main_thread_nonblocking"])


if __name__ == "__main__":
    unittest.main()
