from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = 9123
BASE = f"http://127.0.0.1:{PORT}"
ENV = {
    **os.environ,
    "ORACLE_TRANSLATOR_SAVE_NAME": "measure_game_fps",
}


def get_json(path: str, *, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            get_json("/status", timeout=1.0)
            return
        except Exception as exc:  # pragma: no cover - runtime helper
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"debug server did not come up: {last_error}")


def press(key: str) -> dict:
    query = urllib.parse.urlencode({"key": key})
    return get_json(f"/press?{query}")


def wait_for_world(timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json("/status", timeout=2.0)
        if last.get("world_loaded"):
            return
        time.sleep(0.2)
    raise RuntimeError(f"world did not load: {last}")


def sample_window(seconds: float, *, interval: float = 0.2) -> dict:
    fps_samples: list[float] = []
    render_samples: list[float] = []
    shift_samples: list[float] = []
    incoming_samples: list[float] = []
    end_time = time.time() + seconds
    while time.time() < end_time:
        fps = get_json("/fps", timeout=2.0)
        status = get_json("/status", timeout=2.0)
        sim_fps = fps.get("sim_fps")
        render_fps = fps.get("render_fps")
        paging = status.get("paging") or {}
        if sim_fps is not None:
            fps_samples.append(float(sim_fps))
        if render_fps is not None:
            render_samples.append(float(render_fps))
        shift_samples.append(float(paging.get("shift_ms") or 0.0))
        incoming_samples.append(float(paging.get("incoming_load_ms") or 0.0))
        time.sleep(interval)
    avg_sim_fps = statistics.mean(fps_samples) if fps_samples else 0.0
    avg_render_fps = statistics.mean(render_samples) if render_samples else 0.0
    avg_ms = 1000.0 / avg_sim_fps if avg_sim_fps > 0.0 else 0.0
    return {
        "samples": len(fps_samples),
        "avg_sim_fps": round(avg_sim_fps, 2),
        "avg_render_fps": round(avg_render_fps, 2),
        "avg_ms": round(avg_ms, 3),
        "max_shift_ms": round(max(shift_samples) if shift_samples else 0.0, 3),
        "max_incoming_load_ms": round(max(incoming_samples) if incoming_samples else 0.0, 3),
    }


def wait_for_shift_count_increase(previous_shift_count: int, *, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json("/status", timeout=2.0)
        paging = last.get("paging") or {}
        if int(paging.get("shift_count") or 0) > previous_shift_count:
            return last
        time.sleep(0.05)
    raise RuntimeError(f"shift_count did not increase: last={last}")


def main() -> int:
    log_out = ROOT / "game_live_measure.out.log"
    log_err = ROOT / "game_live_measure.err.log"
    with log_out.open("wb") as out_fp, log_err.open("wb") as err_fp:
        proc = subprocess.Popen(
            [str(PYTHON), "-m", "src.game"],
            cwd=str(ROOT),
            env=ENV,
            stdout=out_fp,
            stderr=err_fp,
        )
        try:
            wait_for_server()
            press("ENTER")
            wait_for_world()
            time.sleep(2.0)

            normal = sample_window(4.0)

            status_before = get_json("/status", timeout=2.0)
            shift_count = int(((status_before.get("paging") or {}).get("shift_count")) or 0)
            biome_x = 3 * 32000 + 16000
            teleport = get_json(f"/teleport?{urllib.parse.urlencode({'x': biome_x, 'y': 0})}", timeout=15.0)
            shifted = wait_for_shift_count_increase(shift_count, timeout=20.0)
            teleport_window = sample_window(4.0)

            result = {
                "normal": normal,
                "teleport_call": teleport,
                "teleport_shift": shifted.get("paging"),
                "teleport": teleport_window,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
