from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.materials import build_material_registry
from src.engine.paging import CameraIntentMailbox
from src.engine.render import DebugViewMode
from src.engine.snapshot_mailbox import SnapshotMailboxRegistry
from src.game import config as cfg
from src.game.app import GameApp
from src.game.entity_manager import EntityManager
from src.game.hero import Hero
from src.game.owner_frame_renderer import OwnerFrameRenderer
from src.game.stt import SpeechToText


def build_benchmark_app() -> GameApp:
    app = object.__new__(GameApp)
    app.seed = 42
    app.registry = build_material_registry()
    app.world = None
    app.terrain_gen = None
    app._snapshot_registry = SnapshotMailboxRegistry()
    app.hero = Hero()
    app.entity_manager = EntityManager(hero=app.hero, _snapshot_registry=app._snapshot_registry)
    app._camera_intent_mailbox = CameraIntentMailbox()
    app._latest_world_snapshots = {}
    app._latest_page_plan = None
    app.view_mode = DebugViewMode.MATERIAL
    app._camera_target_x = 0
    app._camera_target_y = 0
    app.active_streams = []
    app.active_pressure_bursts = []
    app.enemies = {}
    app.projectiles = []
    app.stt = SpeechToText()
    app._chant_started = False
    app.ctx = None
    app.renderer = OwnerFrameRenderer(cfg.VIEWPORT_WIDTH * cfg.CELL_SCALE, cfg.VIEWPORT_HEIGHT * cfg.CELL_SCALE)
    app._keys_pressed = set()
    app._last_dt = 1.0 / 60.0
    app._sim_accumulator = 0.0
    app._sim_fps = 0.0
    app._sim_fps_count = 0
    app._sim_fps_started_at = perf_counter()
    app._debug_overlay_enabled = False
    app._perf_samples = {}
    app._perf_last_ms = {}
    app._enable_chunk_worker_processes = False
    app._init_game_world()
    return app


def run_frame(app: GameApp, dt: float) -> float:
    started_at = perf_counter()
    app.update_game(dt)
    assert app.world is not None
    target_x = int(app.hero.x) - cfg.VIEWPORT_WIDTH // 2
    target_y = int(app.hero.y) - cfg.VIEWPORT_HEIGHT // 2
    app._submit_gpu_camera_follow(target_x, target_y, dt=dt)
    app._submit_gpu_background_io_service()
    return (perf_counter() - started_at) * 1000.0


def paging_snapshot(app: GameApp) -> dict:
    status = app.gpu_world_status_snapshot(block=True, timeout=5.0) or {}
    return dict(status.get("paging") or {})


def summarize(name: str, values: list[float]) -> str:
    avg_ms = statistics.mean(values)
    return (
        f"{name} avg_ms={avg_ms:.3f} fps={1000.0 / avg_ms:.2f} "
        f"median_ms={statistics.median(values):.3f} max_ms={max(values):.2f}"
    )


def main() -> None:
    os.environ[cfg.SAVE_NAME_ENV_VAR] = "benchmark_game_app"
    app = build_benchmark_app()
    dt = 1.0 / 60.0
    try:
        for _ in range(40):
            run_frame(app, dt)

        normal = [run_frame(app, dt) for _ in range(140)]
        print(summarize("normal", normal))

        teleport_x = 3 * cfg.BIOME_WIDTH + cfg.BIOME_WIDTH // 2
        app.teleport_to_surface_x(float(teleport_x))
        teleport = [run_frame(app, dt) for _ in range(140)]
        print(summarize("teleport", teleport))
        paging = paging_snapshot(app)
        print(
            "teleport_paging "
            f"shift_last_ms={float(paging.get('shift_ms') or 0.0):.2f} "
            f"incoming_load_last_ms={float(paging.get('incoming_load_ms') or 0.0):.2f} "
            f"evict_stage_last_ms={float(paging.get('evict_stage_ms') or 0.0):.2f}"
        )
    finally:
        app._close_game_world()
        release = getattr(getattr(app, "ctx", None), "release", None)
        if callable(release):
            release()


if __name__ == "__main__":
    main()
