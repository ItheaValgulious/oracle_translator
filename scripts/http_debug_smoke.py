from __future__ import annotations

import json
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


def get_json(path: str, *, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(path: str, *, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
        return response.read()


def wait_for_png(path: str, *, timeout: float = 10.0, interval: float = 0.25) -> bytes:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            payload = get_bytes(path, timeout=2.0)
            if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                return payload
            last_error = RuntimeError(f"non-PNG prefix {payload[:16]!r}")
        except Exception as exc:  # pragma: no cover - runtime helper
            last_error = exc
        time.sleep(interval)
    raise RuntimeError(f"PNG endpoint did not become ready: {last_error}")


def wait_for_server(timeout: float = 30.0) -> None:
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


def wait_for(predicate, *, timeout: float, interval: float = 0.1, message: str) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json("/status", timeout=2.0)
        if predicate(last):
            return last
        time.sleep(interval)
    raise RuntimeError(f"{message}: last={last}")


def wait_for_snapshot(entity_id: str, *, timeout: float = 10.0, interval: float = 0.1) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json("/snapshots", timeout=2.0)
        if entity_id in last.get("snapshots", {}):
            return last
        time.sleep(interval)
    raise RuntimeError(f"{entity_id} snapshot missing from snapshots endpoint: snapshots={last}")


def press(key: str) -> dict:
    query = urllib.parse.urlencode({"key": key})
    return get_json(f"/press?{query}")


def spawn(enemy_type: str) -> dict:
    query = urllib.parse.urlencode({"type": enemy_type})
    return get_json(f"/spawn?{query}")


def explode(x: int, y: int) -> dict:
    query = urllib.parse.urlencode({"x": x, "y": y})
    return get_json(f"/explode?{query}")


def gpu_inject_probe(x: int, y: int, *, radius: int = 6, pressure: float = 120.0) -> dict:
    query = urllib.parse.urlencode({"x": x, "y": y, "radius": radius, "pressure": pressure})
    return get_json(f"/gpu_inject_probe?{query}")


def gpu_pressure_timeline(x: int, y: int, *, radius: int = 12, frames: int = 8) -> dict:
    query = urllib.parse.urlencode({"x": x, "y": y, "radius": radius, "frames": frames})
    return get_json(f"/gpu_pressure_timeline?{query}")


def max_pressure_near(x: int, y: int, *, radius: int = 12) -> float:
    payload = get_json(
        f"/gpu_pressure?{urllib.parse.urlencode({'x': x - radius, 'y': y - radius, 'w': radius * 2 + 1, 'h': radius * 2 + 1})}"
    )
    return max((sample["pressure"] for sample in payload.get("samples", [])), default=0.0)


def nonzero_pressure_count_near(x: int, y: int, *, radius: int = 12) -> int:
    payload = get_json(
        f"/gpu_pressure?{urllib.parse.urlencode({'x': x - radius, 'y': y - radius, 'w': radius * 2 + 1, 'h': radius * 2 + 1})}"
    )
    return sum(1 for sample in payload.get("samples", []) if sample["pressure"] > 0.0)


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


def main() -> int:
    log_out = ROOT / "game_http_smoke.out.log"
    log_err = ROOT / "game_http_smoke.err.log"
    with log_out.open("wb") as out_fp, log_err.open("wb") as err_fp:
        proc = subprocess.Popen(
            [str(PYTHON), "-m", "src.game"],
            cwd=str(ROOT),
            stdout=out_fp,
            stderr=err_fp,
        )
        try:
            wait_for_server()
            press("ENTER")
            wait_for(
                lambda status: bool(status.get("world_loaded")),
                timeout=30.0,
                message="world did not load",
            )

            baseline = get_json("/status")
            snapshots = wait_for_snapshot("hero", timeout=10.0)
            screenshot = wait_for_png("/screenshot", timeout=10.0)
            hero_before = get_json("/hero")

            spawn("A")
            time.sleep(0.5)
            status_after_spawn = get_json("/status")
            if status_after_spawn.get("enemy_count", 0) <= baseline.get("enemy_count", 0):
                raise RuntimeError(f"enemy did not spawn: before={baseline} after={status_after_spawn}")

            blast_x = int(hero_before["x"])
            blast_y = int((hero_before["bottom"] + hero_before["top"]) * 0.5)
            raw_probe = gpu_inject_probe(blast_x, blast_y, radius=6, pressure=120.0)
            raw_probe_data = raw_probe.get("probe", {})
            if raw_probe_data.get("nonzero_samples", 0) <= 0:
                raise RuntimeError(f"raw gpu pressure probe failed: {raw_probe}")

            explode_result = explode(blast_x, blast_y)
            timeline_payload = gpu_pressure_timeline(blast_x, blast_y, radius=12, frames=8)
            timeline = timeline_payload.get("timeline", {})
            samples = timeline.get("samples", [])
            max_pressure = max((sample.get("max_pressure", 0.0) for sample in samples), default=0.0)
            nonzero_pressure = max((sample.get("nonzero_samples", 0) for sample in samples), default=0)
            explode_probe = {
                "ok": nonzero_pressure > 0,
                "max_pressure": max_pressure,
                "nonzero_pressure": nonzero_pressure,
                "timeline_samples": samples,
            }
            if not explode_probe["ok"]:
                debug = get_json("/gpu_debug")
                pressure = get_json(
                    f"/gpu_pressure?{urllib.parse.urlencode({'x': blast_x - 12, 'y': blast_y - 12, 'w': 25, 'h': 25})}"
                )
                samples = pressure.get("samples", [])
                explode_probe["debug"] = debug
                explode_probe["pressure_summary"] = {
                    "sample_count": len(samples),
                    "nonzero_samples": sum(1 for sample in samples if sample.get("pressure", 0.0) > 0.0),
                    "max_pressure": max((sample.get("pressure", 0.0) for sample in samples), default=0.0),
                    "window": [pressure.get("x"), pressure.get("y"), pressure.get("w"), pressure.get("h")],
                }

            hero_after = get_json("/hero")
            final_status = get_json("/status")
            summary = {
                "world_loaded": baseline.get("world_loaded"),
                "gpu_owner_active": baseline.get("gpu_owner_active"),
                "gpu_owner_created_world": baseline.get("gpu_owner_created_world"),
                "gpu_owner_commands_failed": baseline.get("gpu_owner_commands_failed"),
                "hero_snapshot_present": "hero" in snapshots.get("snapshots", {}),
                "screenshot_png_bytes": len(screenshot),
                "enemy_count_before": baseline.get("enemy_count"),
                "enemy_count_after_spawn": status_after_spawn.get("enemy_count"),
                "hero_before": {
                    "x": hero_before.get("x"),
                    "y": hero_before.get("y"),
                    "state": hero_before.get("state"),
                },
                "hero_after": {
                    "x": hero_after.get("x"),
                    "y": hero_after.get("y"),
                    "state": hero_after.get("state"),
                },
                "final_status": {
                    "enemy_count": final_status.get("enemy_count"),
                    "projectile_count": final_status.get("projectile_count"),
                    "active_pressure_bursts": final_status.get("active_pressure_bursts"),
                },
                "raw_pressure_probe": raw_probe_data,
                "explode_result": explode_result,
                "explode_pressure_probe": explode_probe,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if explode_probe["ok"] else 2
        finally:
            stop_game(proc)


if __name__ == "__main__":
    raise SystemExit(main())
