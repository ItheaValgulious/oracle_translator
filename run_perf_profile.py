"""Detailed performance profiling of the game loop."""

import subprocess
import time
import urllib.request
import json
import sys

proc = subprocess.Popen(
    [sys.executable, "-m", "src.game"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

try:
    time.sleep(4)
    try:
        urllib.request.urlopen("http://127.0.0.1:9123/press?key=ENTER", timeout=5)
    except Exception as e:
        print(f"ENTER failed: {e}")

    time.sleep(3)

    # Collect detailed perf data
    print("Collecting detailed perf data...")
    samples = []
    for i in range(30):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:9123/status", timeout=2) as resp:
                data = json.loads(resp.read().decode())
            with urllib.request.urlopen("http://127.0.0.1:9123/fps", timeout=2) as resp:
                fps_data = json.loads(resp.read().decode())
            with urllib.request.urlopen("http://127.0.0.1:9123/benchmark", timeout=2) as resp:
                bench_data = json.loads(resp.read().decode())
            samples.append({"status": data, "fps": fps_data, "bench": bench_data})
        except Exception:
            pass

    print("\n" + "="*60)
    print("DETAILED PERFORMANCE REPORT")
    print("="*60)

    if samples:
        sim_fps = [s["fps"].get("sim_fps", 0) for s in samples if s["fps"].get("sim_fps")]
        render_fps = [s["fps"].get("render_fps", 0) for s in samples if s["fps"].get("render_fps")]
        if sim_fps:
            print(f"Sim FPS:    min={min(sim_fps):.1f}, max={max(sim_fps):.1f}, avg={sum(sim_fps)/len(sim_fps):.1f}")
        if render_fps:
            print(f"Render FPS: min={min(render_fps):.1f}, max={max(render_fps):.1f}, avg={sum(render_fps)/len(render_fps):.1f}")

        # Extract perf breakdown
        print("\nPerf breakdown (last_ms, avg_ms):")
        perf_keys = [
            "update_game_total", "update_game_poll", "update_game_apply",
            "update_game_tick", "update_game_world_step_submit",
            "update_game_schedule_feedback", "update_game_snapshot",
            "main_tick_total", "main_tick_sim", "main_tick_camera"
        ]
        for key in perf_keys:
            last_vals = []
            avg_vals = []
            for s in samples:
                perf = s.get("bench", {}).get("perf", {})
                last = perf.get(f"{key}_last_ms", 0)
                avg = perf.get(f"{key}_avg_ms", 0)
                if last > 0:
                    last_vals.append(last)
                if avg > 0:
                    avg_vals.append(avg)
            if last_vals:
                print(f"  {key}: last={sum(last_vals)/len(last_vals):.2f}, avg={sum(avg_vals)/len(avg_vals):.2f}")

        # GPU flags
        print("\nGPU flags (from last sample):")
        gpu_info = samples[-1].get("bench", {}).get("gpu_info", {})
        for k, v in gpu_info.items():
            print(f"  {k}: {v}")

        # Paging stats
        print("\nPaging stats (from last sample):")
        paging = samples[-1].get("status", {}).get("paging") or {}
        for k, v in sorted(paging.items()):
            if "_ms" in k or "_count" in k:
                print(f"  {k}: {v}")

finally:
    print("\nStopping game...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
        proc.wait()
    print("Done.")
