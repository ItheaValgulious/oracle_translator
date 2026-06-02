from __future__ import annotations

import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = 9123
BASE = f"http://127.0.0.1:{PORT}"
SAVE_ENV = "ORACLE_TRANSLATOR_SAVE_NAME"
BIOME_ORDER_ENV = "ORACLE_TRANSLATOR_EXPERIMENT_BIOME_ORDER"
DISK_BENCH_SAMPLE_LIMIT = 128
MAIN_THREAD_BEST_EFFORT_BUDGET_MS = 33.4


def get_json(
    path: str,
    *,
    timeout: float = 5.0,
    retries: int = 1,
    retry_delay: float = 0.2,
) -> dict:
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - runtime helper
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(max(0.0, retry_delay) * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {attempts} attempt(s): {last_error}")


def wait_for_server(timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            get_json("/status", timeout=1.0, retries=2)
            return
        except Exception as exc:  # pragma: no cover - runtime helper
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"debug server did not come up: {last_error}")


def port_is_listening(*, host: str = "127.0.0.1", port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port_free(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_is_listening():
            return
        time.sleep(0.1)
    raise RuntimeError(f"port {PORT} is still busy after {timeout} seconds")


def wait_for_world(timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json("/status", timeout=2.0, retries=3)
        if last.get("world_loaded"):
            return last
        time.sleep(0.2)
    raise RuntimeError(f"world did not load: {last}")


def press(key: str, *, timeout: float = 6.0) -> dict:
    query = urllib.parse.urlencode({"key": key})
    return get_json(f"/press?{query}", timeout=timeout)


def call(path: str, params: dict[str, object], *, timeout: float = 10.0) -> dict:
    query = urllib.parse.urlencode(params)
    return get_json(f"{path}?{query}", timeout=timeout)


def evict_disk_chunks(
    *,
    start_x: int,
    end_x: int,
    margin_x: int = 0,
    margin_y: int = 0,
    timeout: float = 15.0,
) -> dict:
    return call(
        "/evict_disk_chunks",
        {
            "start_x": int(start_x),
            "end_x": int(end_x),
            "margin_x": int(margin_x),
            "margin_y": int(margin_y),
        },
        timeout=timeout,
    )


def sample_window(seconds: float, *, interval: float = 0.2) -> dict:
    fps_samples: list[float] = []
    render_samples: list[float] = []
    statuses: list[dict] = []
    end_time = time.time() + seconds
    while time.time() < end_time:
        fps = fps_snapshot()
        status = status_snapshot()
        statuses.append(status)
        sim_fps = fps.get("sim_fps")
        render_fps = fps.get("render_fps")
        if sim_fps is not None:
            fps_samples.append(float(sim_fps))
        if render_fps is not None:
            render_samples.append(float(render_fps))
        time.sleep(interval)
    paging_samples = [entry.get("paging") or {} for entry in statuses]
    perf_samples = [entry.get("perf") or {} for entry in statuses]
    return {
        "samples": len(fps_samples),
        "avg_sim_fps": round(sum(fps_samples) / len(fps_samples), 2) if fps_samples else 0.0,
        "avg_render_fps": round(sum(render_samples) / len(render_samples), 2) if render_samples else 0.0,
        "max_shift_ms": round(max(float(p.get("shift_ms") or 0.0) for p in paging_samples), 3) if paging_samples else 0.0,
        "max_incoming_load_ms": round(max(float(p.get("incoming_load_ms") or 0.0) for p in paging_samples), 3) if paging_samples else 0.0,
        "max_update_game_ms": round(max(float(p.get("update_game_total_last_ms") or 0.0) for p in perf_samples), 3) if perf_samples else 0.0,
        "max_main_tick_ms": round(max(float(p.get("main_tick_total_last_ms") or 0.0) for p in perf_samples), 3) if perf_samples else 0.0,
        "last_status": statuses[-1] if statuses else {},
    }


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    sorted_samples = sorted(samples)
    return sorted_samples[int(round(0.95 * (len(sorted_samples) - 1)))]


def _measure_chunk_reads(paths: list[Path], *, label: str) -> dict:
    samples: list[float] = []
    bytes_per_chunk = 0
    for path in paths:
        start = time.perf_counter()
        data = path.read_bytes()
        samples.append(time.perf_counter() - start)
        if bytes_per_chunk == 0:
            bytes_per_chunk = len(data)
    return {
        "label": label,
        "sample_count": len(samples),
        "bytes_per_chunk": bytes_per_chunk,
        "mean_seconds": statistics.fmean(samples) if samples else 0.0,
        "p95_seconds": _p95(samples),
        "samples": samples,
    }


def run_direct_disk_benchmarks(*, max_samples: int = DISK_BENCH_SAMPLE_LIMIT) -> dict:
    chunk_paths = sorted((ROOT / "storage").glob("**/*.ogchunk"))
    if not chunk_paths:
        return {
            "sample_count": 0,
            "sequential": None,
            "pseudo_random": None,
            "note": "no chunk files found under storage",
        }
    selected = chunk_paths[: max(1, int(max_samples))]
    shuffled = list(selected)
    random.Random(42).shuffle(shuffled)
    return {
        "sample_count": len(selected),
        "source_file_count": len(chunk_paths),
        "sequential": _measure_chunk_reads(selected, label="sequential"),
        "pseudo_random": _measure_chunk_reads(shuffled, label="pseudo_random"),
    }


def start_game(save_name: str, *, biome_order: str | None = None) -> subprocess.Popen:
    env = {
        **os.environ,
        SAVE_ENV: save_name,
    }
    if biome_order:
        env[BIOME_ORDER_ENV] = biome_order
    else:
        env.pop(BIOME_ORDER_ENV, None)
    log_out = ROOT / f"{save_name}.out.log"
    log_err = ROOT / f"{save_name}.err.log"
    out_fp = log_out.open("wb")
    err_fp = log_err.open("wb")
    proc = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / "game_experiment_entry.py")],
        cwd=str(ROOT),
        env=env,
        stdout=out_fp,
        stderr=err_fp,
    )
    proc._codex_out_fp = out_fp  # type: ignore[attr-defined]
    proc._codex_err_fp = err_fp  # type: ignore[attr-defined]
    return proc


def stop_game(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            try:
                get_json("/shutdown", timeout=75.0)
            except Exception:
                pass
        if proc.poll() is None:
            proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    for attr in ("_codex_out_fp", "_codex_err_fp"):
        fp = getattr(proc, attr, None)
        if fp is not None:
            fp.close()
    wait_for_port_free(timeout=20.0)


def prepare_session(
    *,
    save_name: str,
    wipe_storage: bool,
    biome_order: str | None = None,
    settle_seconds: float = 2.0,
) -> subprocess.Popen:
    wait_for_port_free(timeout=5.0)
    if wipe_storage:
        shutil.rmtree(ROOT / "storage" / save_name, ignore_errors=True)
    proc = start_game(save_name, biome_order=biome_order)
    wait_for_server()
    time.sleep(1.0)
    last_error: Exception | None = None
    for _attempt in range(4):
        try:
            press("ENTER", timeout=12.0)
            last_error = None
            break
        except Exception as exc:  # pragma: no cover - runtime helper
            last_error = exc
            time.sleep(0.5)
    if last_error is not None:
        stop_game(proc)
        raise RuntimeError(f"failed to start world with ENTER: {last_error}")
    wait_for_world()
    time.sleep(max(0.0, settle_seconds))
    return proc


def status_snapshot() -> dict:
    return get_json("/status", timeout=2.0, retries=3)


def fps_snapshot() -> dict:
    return get_json("/fps", timeout=2.0, retries=3)


def wait_for_shift_count(previous_shift_count: int, *, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = status_snapshot()
        paging = last.get("paging") or {}
        if int(paging.get("shift_count") or 0) > previous_shift_count:
            return last
        time.sleep(0.05)
    raise RuntimeError(f"shift_count did not increase from {previous_shift_count}: last={last}")


def wait_for_background_idle(*, timeout: float = 45.0, interval: float = 0.25) -> dict:
    """Wait until async chunk/page queues visible in /status are drained."""
    queue_fields = (
        "chunk_prefetch_queued",
        "chunk_prefetch_inflight",
        "chunk_queued_read",
        "chunk_queued_write",
        "chunk_queued_generate",
        "chunk_inflight_io",
        "chunk_inflight_generation",
        "active_chunk_patch_queue_depth",
    )
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = status_snapshot()
        paging = last.get("paging") or {}
        if all(int(paging.get(name) or 0) == 0 for name in queue_fields):
            return last
        time.sleep(interval)
    return last


def set_experiment_camera(*, camera_x: int, camera_y: int, freeze: bool = True, timeout: float = 10.0) -> dict:
    return call(
        "/experiment_camera",
        {
            "x": int(camera_x),
            "y": int(camera_y),
            "freeze": 1 if freeze else 0,
        },
        timeout=timeout,
    )


def clear_experiment_camera(*, timeout: float = 10.0) -> dict:
    return get_json("/experiment_camera_clear", timeout=timeout, retries=3)


def summarize_shift_events(shift_events: list[dict]) -> dict:
    return {
        "event_count": len(shift_events),
        "sum_last_shift_disk_loads": sum(int(event.get("last_shift_disk_loads") or 0) for event in shift_events),
        "sum_last_shift_generates": sum(int(event.get("last_shift_generates") or 0) for event in shift_events),
        "sum_last_shift_saves": sum(int(event.get("last_shift_saves") or 0) for event in shift_events),
        "sum_last_shift_disk_load_ms": round(sum(float(event.get("last_shift_disk_load_ms") or 0.0) for event in shift_events), 3),
        "sum_last_shift_generate_ms": round(sum(float(event.get("last_shift_generate_ms") or 0.0) for event in shift_events), 3),
        "sum_last_shift_save_ms": round(sum(float(event.get("last_shift_save_ms") or 0.0) for event in shift_events), 3),
        "max_shift_ms": round(max((float(event.get("shift_ms") or 0.0) for event in shift_events), default=0.0), 3),
        "max_incoming_load_ms": round(max((float(event.get("incoming_load_ms") or 0.0) for event in shift_events), default=0.0), 3),
    }


def _perf_field_ms(perf: dict | None, field: str) -> float:
    if not isinstance(perf, dict):
        return 0.0
    return float(perf.get(field) or 0.0)


def summarize_step_main_thread(step_records: list[dict]) -> dict:
    max_main_tick_ms = 0.0
    max_update_game_ms = 0.0
    max_world_step_submit_cpu_ms = 0.0
    for record in step_records:
        perf = dict((record.get("status") or {}).get("perf") or {})
        max_main_tick_ms = max(max_main_tick_ms, _perf_field_ms(perf, "main_tick_total_last_ms"))
        max_update_game_ms = max(max_update_game_ms, _perf_field_ms(perf, "update_game_total_last_ms"))
        max_world_step_submit_cpu_ms = max(
            max_world_step_submit_cpu_ms,
            _perf_field_ms(perf, "world_step_submit_cpu_last_ms"),
        )
    within_budget = (
        max_main_tick_ms <= MAIN_THREAD_BEST_EFFORT_BUDGET_MS
        and max_update_game_ms <= MAIN_THREAD_BEST_EFFORT_BUDGET_MS
    )
    return {
        "sample_count": len(step_records),
        "budget_ms": MAIN_THREAD_BEST_EFFORT_BUDGET_MS,
        "max_main_tick_ms": round(max_main_tick_ms, 3),
        "max_update_game_ms": round(max_update_game_ms, 3),
        "max_world_step_submit_cpu_ms": round(max_world_step_submit_cpu_ms, 3),
        "within_budget": bool(within_budget),
    }


def summarize_shift_main_thread(page_shift: dict) -> dict:
    summary = summarize_step_main_thread(list(page_shift.get("steps") or []))
    shift_event_summary = dict(page_shift.get("shift_event_summary") or {})
    summary["max_shift_ms"] = round(float(shift_event_summary.get("max_shift_ms") or 0.0), 3)
    summary["max_incoming_load_ms"] = round(float(shift_event_summary.get("max_incoming_load_ms") or 0.0), 3)
    return summary


def summarize_single_teleport_main_thread(teleport_result: dict) -> dict:
    perf = dict((teleport_result.get("shifted_status") or {}).get("perf") or {})
    after_window = dict(teleport_result.get("after_window") or {})
    max_main_tick_ms = _perf_field_ms(perf, "main_tick_total_last_ms")
    max_update_game_ms = _perf_field_ms(perf, "update_game_total_last_ms")
    max_world_step_submit_cpu_ms = _perf_field_ms(perf, "world_step_submit_cpu_last_ms")
    within_budget = (
        max_main_tick_ms <= MAIN_THREAD_BEST_EFFORT_BUDGET_MS
        and max_update_game_ms <= MAIN_THREAD_BEST_EFFORT_BUDGET_MS
    )
    return {
        "budget_ms": MAIN_THREAD_BEST_EFFORT_BUDGET_MS,
        "max_main_tick_ms": round(max_main_tick_ms, 3),
        "max_update_game_ms": round(max_update_game_ms, 3),
        "max_world_step_submit_cpu_ms": round(max_world_step_submit_cpu_ms, 3),
        "after_window_max_main_tick_ms": round(float(after_window.get("max_main_tick_ms") or 0.0), 3),
        "after_window_max_update_game_ms": round(float(after_window.get("max_update_game_ms") or 0.0), 3),
        "within_budget": bool(within_budget),
    }


def summarize_teleport_group_main_thread(teleports: dict[str, dict]) -> dict[str, dict]:
    return {
        target_name: summarize_single_teleport_main_thread(result)
        for target_name, result in teleports.items()
    }


def build_derived_conclusions(results: dict) -> tuple[dict, dict]:
    cases = dict(results.get("cases") or {})
    cold = dict(((cases.get("page_shift_with_generate") or {}).get("page_shift") or {}).get("shift_event_summary") or {})
    disk_only = dict(((cases.get("page_shift_disk_only") or {}).get("page_shift") or {}).get("shift_event_summary") or {})
    pure_cache = dict(((cases.get("page_shift_pure_cache") or {}).get("page_shift") or {}).get("shift_event_summary") or {})

    movement_case_names = (
        "page_shift_with_generate",
        "page_shift_disk_only",
        "page_shift_pure_cache",
    )
    movement_summaries = {
        case_name: summarize_shift_main_thread(dict((cases.get(case_name) or {}).get("page_shift") or {}))
        for case_name in movement_case_names
    }

    teleport_case_names = (
        "existing_biome_teleports",
        "existing_biome_teleports_alt_targets",
        "front_mountain_back_plains_teleports",
    )
    teleport_summaries = {
        case_name: summarize_teleport_group_main_thread(dict((cases.get(case_name) or {}).get("teleports") or {}))
        for case_name in teleport_case_names
    }

    normal_movement_nonblocking = all(
        bool(summary.get("within_budget", False))
        for summary in movement_summaries.values()
    )
    teleport_nonblocking = all(
        bool(summary.get("within_budget", False))
        for group in teleport_summaries.values()
        for summary in group.values()
    )

    main_thread_nonblocking_summary = {
        "budget_ms": MAIN_THREAD_BEST_EFFORT_BUDGET_MS,
        "page_shift_cases": movement_summaries,
        "teleport_cases": teleport_summaries,
    }
    derived_conclusions = {
        "page_shift_pure_cache_zero_generate_load_save_in_window": (
            int(pure_cache.get("sum_last_shift_generates", 0)) == 0
            and int(pure_cache.get("sum_last_shift_disk_loads", 0)) == 0
            and int(pure_cache.get("sum_last_shift_saves", 0)) == 0
        ),
        "page_shift_disk_only_has_loads_zero_generate_in_window": (
            int(disk_only.get("sum_last_shift_disk_loads", 0)) > 0
            and int(disk_only.get("sum_last_shift_generates", 0)) == 0
        ),
        "page_shift_with_generate_zero_disk_loads_in_window": (
            int(cold.get("sum_last_shift_disk_loads", 0)) == 0
        ),
        "page_shift_with_generate_generation_time_dominates_disk_time": (
            float(cold.get("sum_last_shift_generate_ms", 0.0)) > float(cold.get("sum_last_shift_disk_load_ms", 0.0))
        ),
        "normal_movement_main_thread_nonblocking": bool(normal_movement_nonblocking),
        "teleport_main_thread_nonblocking": bool(teleport_nonblocking),
    }
    return main_thread_nonblocking_summary, derived_conclusions


def page_shift_probe(
    *,
    start_x: int,
    step_dx: int,
    max_steps: int,
    settle_seconds: float = 0.12,
    target_shift_events: int = 4,
) -> dict:
    baseline = status_snapshot()
    call("/teleport_surface", {"x": start_x}, timeout=15.0)
    time.sleep(0.6)
    before = status_snapshot()
    start_camera = tuple(before.get("camera") or (0, 0))
    target_camera_x = int(start_camera[0])
    target_camera_y = int(start_camera[1])
    set_experiment_camera(
        camera_x=target_camera_x,
        camera_y=target_camera_y,
        freeze=True,
        timeout=15.0,
    )
    base_shift = int(((before.get("paging") or {}).get("shift_count")) or 0)
    base_paging = before.get("paging") or {}
    base_generate_count = int(base_paging.get("chunk_generate_count") or 0)
    base_disk_load_count = int(base_paging.get("chunk_disk_load_count") or 0)
    base_save_count = int(base_paging.get("chunk_save_count") or 0)
    step_records: list[dict] = []
    shift_events: list[dict] = []
    last_seen_shift = base_shift
    for step_index in range(max_steps):
        target_camera_x += int(step_dx)
        camera_move = set_experiment_camera(
            camera_x=target_camera_x,
            camera_y=target_camera_y,
            freeze=True,
            timeout=15.0,
        )
        time.sleep(settle_seconds)
        fps = fps_snapshot()
        status = status_snapshot()
        step_records.append({
            "step": step_index,
            "camera_move": camera_move,
            "target_camera": {
                "x": int(target_camera_x),
                "y": int(target_camera_y),
            },
            "fps": fps,
            "status": status,
        })
        shift_count = int(((status.get("paging") or {}).get("shift_count")) or 0)
        if shift_count > last_seen_shift:
            paging = status.get("paging") or {}
            shift_events.append({
                "step": step_index,
                "shift_count": shift_count,
                "shift_ms": paging.get("shift_ms"),
                "incoming_load_ms": paging.get("incoming_load_ms"),
                "last_shift_disk_loads": paging.get("last_shift_disk_loads"),
                "last_shift_generates": paging.get("last_shift_generates"),
                "last_shift_saves": paging.get("last_shift_saves"),
                "last_shift_disk_load_ms": paging.get("last_shift_disk_load_ms"),
                "last_shift_generate_ms": paging.get("last_shift_generate_ms"),
                "last_shift_save_ms": paging.get("last_shift_save_ms"),
                "chunk_generate_count": paging.get("chunk_generate_count"),
                "chunk_disk_load_count": paging.get("chunk_disk_load_count"),
                "chunk_save_count": paging.get("chunk_save_count"),
                "sim_fps": fps.get("sim_fps"),
                "render_fps": fps.get("render_fps"),
            })
            last_seen_shift = shift_count
            if len(shift_events) >= target_shift_events:
                break
    final_status = step_records[-1]["status"] if step_records else before
    final_paging = final_status.get("paging") or {}
    return {
        "baseline": baseline,
        "start_status": before,
        "movement_summary": {
            "start_chunk_generate_count": base_generate_count,
            "end_chunk_generate_count": int(final_paging.get("chunk_generate_count") or 0),
            "delta_chunk_generate_count": int(final_paging.get("chunk_generate_count") or 0) - base_generate_count,
            "start_chunk_disk_load_count": base_disk_load_count,
            "end_chunk_disk_load_count": int(final_paging.get("chunk_disk_load_count") or 0),
            "delta_chunk_disk_load_count": int(final_paging.get("chunk_disk_load_count") or 0) - base_disk_load_count,
            "start_chunk_save_count": base_save_count,
            "end_chunk_save_count": int(final_paging.get("chunk_save_count") or 0),
            "delta_chunk_save_count": int(final_paging.get("chunk_save_count") or 0) - base_save_count,
            "shift_events_captured": len(shift_events),
        },
        "shift_event_summary": summarize_shift_events(shift_events),
        "shift_events": shift_events,
        "steps": step_records,
        "main_thread_summary": summarize_step_main_thread(step_records),
        "window_after_shift": sample_window(2.0) if shift_events else None,
    }


def movement_probe_from_current(
    *,
    step_dx: int,
    max_steps: int,
    settle_seconds: float = 0.12,
    target_shift_events: int = 4,
) -> dict:
    before = status_snapshot()
    start_camera = tuple(before.get("camera") or (0, 0))
    base_shift = int(((before.get("paging") or {}).get("shift_count")) or 0)
    base_paging = before.get("paging") or {}
    base_generate_count = int(base_paging.get("chunk_generate_count") or 0)
    base_disk_load_count = int(base_paging.get("chunk_disk_load_count") or 0)
    base_save_count = int(base_paging.get("chunk_save_count") or 0)
    step_records: list[dict] = []
    shift_events: list[dict] = []
    last_seen_shift = base_shift
    target_camera_x = int(start_camera[0])
    target_camera_y = int(start_camera[1])
    set_experiment_camera(camera_x=target_camera_x, camera_y=target_camera_y, freeze=True, timeout=15.0)
    for step_index in range(max_steps):
        target_camera_x += int(step_dx)
        camera_move = set_experiment_camera(
            camera_x=target_camera_x,
            camera_y=target_camera_y,
            freeze=True,
            timeout=15.0,
        )
        time.sleep(settle_seconds)
        fps = fps_snapshot()
        status = status_snapshot()
        step_records.append({
            "step": step_index,
            "camera_move": camera_move,
            "target_camera": {
                "x": int(target_camera_x),
                "y": int(target_camera_y),
            },
            "fps": fps,
            "status": status,
        })
        shift_count = int(((status.get("paging") or {}).get("shift_count")) or 0)
        if shift_count > last_seen_shift:
            paging = status.get("paging") or {}
            shift_events.append({
                "step": step_index,
                "shift_count": shift_count,
                "shift_ms": paging.get("shift_ms"),
                "incoming_load_ms": paging.get("incoming_load_ms"),
                "last_shift_disk_loads": paging.get("last_shift_disk_loads"),
                "last_shift_generates": paging.get("last_shift_generates"),
                "last_shift_saves": paging.get("last_shift_saves"),
                "last_shift_disk_load_ms": paging.get("last_shift_disk_load_ms"),
                "last_shift_generate_ms": paging.get("last_shift_generate_ms"),
                "last_shift_save_ms": paging.get("last_shift_save_ms"),
                "chunk_generate_count": paging.get("chunk_generate_count"),
                "chunk_disk_load_count": paging.get("chunk_disk_load_count"),
                "chunk_save_count": paging.get("chunk_save_count"),
                "sim_fps": fps.get("sim_fps"),
                "render_fps": fps.get("render_fps"),
            })
            last_seen_shift = shift_count
            if len(shift_events) >= target_shift_events:
                break
    final_status = step_records[-1]["status"] if step_records else before
    final_paging = final_status.get("paging") or {}
    return {
        "start_status": before,
        "movement_summary": {
            "start_chunk_generate_count": base_generate_count,
            "end_chunk_generate_count": int(final_paging.get("chunk_generate_count") or 0),
            "delta_chunk_generate_count": int(final_paging.get("chunk_generate_count") or 0) - base_generate_count,
            "start_chunk_disk_load_count": base_disk_load_count,
            "end_chunk_disk_load_count": int(final_paging.get("chunk_disk_load_count") or 0),
            "delta_chunk_disk_load_count": int(final_paging.get("chunk_disk_load_count") or 0) - base_disk_load_count,
            "start_chunk_save_count": base_save_count,
            "end_chunk_save_count": int(final_paging.get("chunk_save_count") or 0),
            "delta_chunk_save_count": int(final_paging.get("chunk_save_count") or 0) - base_save_count,
            "shift_events_captured": len(shift_events),
        },
        "shift_event_summary": summarize_shift_events(shift_events),
        "shift_events": shift_events,
        "steps": step_records,
        "main_thread_summary": summarize_step_main_thread(step_records),
        "window_after_shift": sample_window(2.0) if shift_events else None,
    }


def teleport_probe(target_x: int) -> dict:
    before = status_snapshot()
    fps_before = fps_snapshot()
    shift_count = int(((before.get("paging") or {}).get("shift_count")) or 0)
    teleport = call("/teleport_surface", {"x": target_x}, timeout=20.0)
    try:
        shifted = wait_for_shift_count(shift_count, timeout=30.0)
    except RuntimeError:
        time.sleep(1.0)
        shifted = status_snapshot()
    time.sleep(1.0)
    after_window = sample_window(3.0)
    return {
        "target_x": target_x,
        "before_status": before,
        "before_fps": fps_before,
        "teleport": teleport,
        "shifted_status": shifted,
        "main_thread_summary": summarize_single_teleport_main_thread({
            "shifted_status": shifted,
            "after_window": after_window,
        }),
        "after_window": after_window,
    }


def biome_centers() -> dict[str, int]:
    biome_width = 32000
    return {
        "plains": biome_width // 2,
        "hillside": biome_width + biome_width // 2,
        "alpine": 2 * biome_width + biome_width // 2,
        "underground": 3 * biome_width + biome_width // 2,
    }


def biome_flat_offset_targets() -> dict[str, int]:
    """Seed-42 alternate targets chosen from flatter, off-center surfaces."""
    return {
        "plains": 20_480,
        "hillside": 60_156,
        "alpine": 83_876,
        "underground": 120_377,
    }


def run_biome_teleports(targets: dict[str, int]) -> dict:
    results: dict[str, dict] = {}
    for biome_name, x in targets.items():
        results[biome_name] = teleport_probe(x)
    return results


def run_page_shift_case(
    *,
    label: str,
    start_x: int,
    save_name: str,
    wipe_storage: bool,
    biome_order: str | None = None,
    warmup_shift_events: int = 0,
    target_shift_events: int = 4,
) -> dict:
    warmup_result = None
    if warmup_shift_events > 0:
        warm_proc = prepare_session(
            save_name=save_name,
            wipe_storage=wipe_storage,
            biome_order=biome_order,
            settle_seconds=0.25,
        )
        try:
            warmup_result = page_shift_probe(
                start_x=start_x,
                step_dx=96,
                max_steps=160,
                target_shift_events=warmup_shift_events,
            )
            warmup_result["post_warmup_idle_status"] = wait_for_background_idle(timeout=60.0)
        finally:
            stop_game(warm_proc)
        wipe_storage = False

    settle_seconds = 0.25 if wipe_storage else 0.5
    proc = prepare_session(
        save_name=save_name,
        wipe_storage=wipe_storage,
        biome_order=biome_order,
        settle_seconds=settle_seconds,
    )
    try:
        normal = sample_window(2.0)
        page_shift = page_shift_probe(
            start_x=start_x,
            step_dx=96,
            max_steps=80,
            target_shift_events=target_shift_events,
        )
        return {
            "label": label,
            "warmup": warmup_result,
            "normal_window": normal,
            "page_shift": page_shift,
        }
    finally:
        clear_experiment_camera(timeout=10.0)
        stop_game(proc)


def run_page_shift_disk_only_case(*, label: str, save_name: str, wipe_storage: bool) -> dict:
    warm_proc = prepare_session(
        save_name=save_name,
        wipe_storage=wipe_storage,
        settle_seconds=0.25,
    )
    try:
        warmup = page_shift_probe(
            start_x=320,
            step_dx=96,
            max_steps=260,
            target_shift_events=32,
        )
        warmup["post_warmup_idle_status"] = wait_for_background_idle(timeout=90.0)
    finally:
        clear_experiment_camera(timeout=10.0)
        stop_game(warm_proc)

    proc = prepare_session(
        save_name=save_name,
        wipe_storage=False,
        settle_seconds=0.5,
    )
    try:
        normal = sample_window(2.0)
        call("/teleport_surface", {"x": 320}, timeout=15.0)
        time.sleep(0.6)
        idle_before_evict = wait_for_background_idle(timeout=60.0)
        evict_result = evict_disk_chunks(
            start_x=320,
            end_x=320 + 96 * 10,
            margin_x=3,
            margin_y=2,
        )
        page_shift = movement_probe_from_current(
            step_dx=96,
            max_steps=80,
            settle_seconds=0.25,
            target_shift_events=4,
        )
        page_shift["disk_only_setup"] = {
            "idle_before_evict": idle_before_evict,
            "evict_result": evict_result,
        }
        return {
            "label": label,
            "normal_window": normal,
            "warmup": warmup,
            "page_shift": page_shift,
        }
    finally:
        clear_experiment_camera(timeout=10.0)
        stop_game(proc)


def run_page_shift_cache_return_case(*, label: str, save_name: str, wipe_storage: bool) -> dict:
    proc = prepare_session(save_name=save_name, wipe_storage=wipe_storage, settle_seconds=0.25)
    try:
        normal = sample_window(2.0)
        warmup = page_shift_probe(start_x=320, step_dx=96, max_steps=220, target_shift_events=18)
        warmup["post_warmup_idle_status"] = wait_for_background_idle(timeout=60.0)
        backoff = movement_probe_from_current(step_dx=-96, max_steps=80, settle_seconds=0.25, target_shift_events=4)
        backoff["post_backoff_idle_status"] = wait_for_background_idle(timeout=60.0)
        page_shift = movement_probe_from_current(step_dx=-96, max_steps=80, settle_seconds=0.25, target_shift_events=4)
        return {
            "label": label,
            "normal_window": normal,
            "warmup": warmup,
            "backoff": backoff,
            "page_shift": page_shift,
        }
    finally:
        clear_experiment_camera(timeout=10.0)
        stop_game(proc)


def run_teleport_case(
    *,
    label: str,
    save_name: str,
    wipe_storage: bool,
    biome_order: str | None = None,
    teleport_targets: dict[str, int] | None = None,
) -> dict:
    proc = prepare_session(save_name=save_name, wipe_storage=wipe_storage, biome_order=biome_order)
    try:
        normal = sample_window(2.0)
        targets = dict(teleport_targets or biome_centers())
        teleports = run_biome_teleports(targets)
        return {
            "label": label,
            "normal_window": normal,
            "teleport_targets": targets,
            "teleports": teleports,
        }
    finally:
        stop_game(proc)


def main() -> int:
    run_id = uuid.uuid4().hex[:8]
    results = {
        "metric_notes": {
            "avg_sim_fps": "F3 S fps equivalent, based on completed sim ticks on the main thread.",
            "avg_render_fps": "F3 R fps equivalent, based on draw cadence.",
            "update_game_total_last_ms": "CPU-side update_game duration, not GPU completion time.",
            "world_step_submit": "Shown inside perf payload; CPU submit time only, not actual GPU compute duration.",
            "shift chunk counters": "Per-shift chunk cache hits, disk loads, generates, and saves recorded during the shift path.",
            "movement_summary": "Whole movement-window chunk counter deltas; includes async background residency work.",
            "shift_event_summary": "Only the captured active-window shift events; excludes unrelated background prefetch/generation between shifts.",
            "main_thread_nonblocking_summary": "Per-step / per-teleport main-thread timings compared against a best-effort budget to show paging stays off the gameplay thread.",
            "direct_disk_benchmarks": "Sequential and pseudo-random chunk-file read latency over persisted .ogchunk files.",
        },
        "cases": {},
        "direct_disk_benchmarks": {},
    }

    results["cases"]["page_shift_with_generate"] = run_page_shift_case(
        label="page_shift_with_generate",
        start_x=4096,
        save_name=f"exp_page_cold_{run_id}",
        wipe_storage=True,
        target_shift_events=2,
    )
    results["cases"]["page_shift_disk_only"] = run_page_shift_disk_only_case(
        label="page_shift_disk_only",
        save_name=f"exp_page_reload_{run_id}",
        wipe_storage=True,
    )
    results["cases"]["page_shift_pure_cache"] = run_page_shift_cache_return_case(
        label="page_shift_pure_cache",
        save_name=f"exp_page_cache_{run_id}",
        wipe_storage=True,
    )
    results["cases"]["existing_biome_teleports"] = run_teleport_case(
        label="existing_biome_teleports",
        save_name=f"exp_tele_existing_{run_id}",
        wipe_storage=True,
    )
    results["cases"]["existing_biome_teleports_alt_targets"] = run_teleport_case(
        label="existing_biome_teleports_alt_targets",
        save_name=f"exp_tele_existing_alt_{run_id}",
        wipe_storage=True,
        teleport_targets=biome_flat_offset_targets(),
    )
    results["cases"]["front_mountain_back_plains_teleports"] = run_teleport_case(
        label="front_mountain_back_plains_teleports",
        save_name=f"exp_tele_front_mountain_{run_id}",
        wipe_storage=True,
        biome_order="plains,alpine,hillside,underground",
    )
    results["direct_disk_benchmarks"] = run_direct_disk_benchmarks()
    main_thread_nonblocking_summary, derived_conclusions = build_derived_conclusions(results)
    results["main_thread_nonblocking_summary"] = main_thread_nonblocking_summary
    results["derived_conclusions"] = derived_conclusions

    output_path = ROOT / "artifacts" / f"paging_experiments_{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
