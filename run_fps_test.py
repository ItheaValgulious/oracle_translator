"""Run game and collect detailed perf data via debug server."""

import subprocess
import time
import urllib.request
import json
import sys
import threading
import os

gpu_timing_lines = []

def run_test():
    global gpu_timing_lines
    gpu_timing_lines = []

    proc = subprocess.Popen(
        [sys.executable, "-B", "-m", "src.game"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    def read_stdout():
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            if "[GPU_TIMING]" in decoded:
                gpu_timing_lines.append(decoded.strip())

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stdout_thread.start()

    try:
        time.sleep(4)
        try:
            urllib.request.urlopen("http://127.0.0.1:9123/press?key=ENTER", timeout=5)
        except Exception as e:
            print(f"ENTER failed: {e}")

        time.sleep(3)

        # Check status first
        try:
            with urllib.request.urlopen("http://127.0.0.1:9123/status", timeout=2) as resp:
                status = json.loads(resp.read().decode())
                print("Status response keys:", list(status.keys()))
                print("world_loaded:", status.get("world_loaded"))
                print("screen:", status.get("screen"))
        except Exception as e:
            print(f"Status failed: {e}")

        # Collect FPS samples
        print("Collecting FPS data...")
        sim_fps_samples = []
        render_fps_samples = []

        for i in range(20):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://127.0.0.1:9123/fps", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    sim_fps = data.get("sim_fps")
                    render_fps = data.get("render_fps")
                    if sim_fps is not None:
                        sim_fps_samples.append(sim_fps)
                    if render_fps is not None:
                        render_fps_samples.append(render_fps)
            except Exception:
                pass

        print("\n" + "="*50)
        print("FPS REPORT")
        print("="*50)
        if sim_fps_samples:
            print(f"Sim FPS:    min={min(sim_fps_samples):.1f}, max={max(sim_fps_samples):.1f}, avg={sum(sim_fps_samples)/len(sim_fps_samples):.1f}")
        if render_fps_samples:
            print(f"Render FPS: min={min(render_fps_samples):.1f}, max={max(render_fps_samples):.1f}, avg={sum(render_fps_samples)/len(render_fps_samples):.1f}")

        if gpu_timing_lines:
            print("\nGPU Step Timing samples:")
            for line in gpu_timing_lines:
                print(line)

    finally:
        print("\nStopping game...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
            proc.wait()
        print("Done.")

if __name__ == "__main__":
    run_test()
