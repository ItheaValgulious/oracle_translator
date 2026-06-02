"""Compatibility launcher for the retired GPU test window.

The previous test window created its own ModernGL context and stepped an
``ActiveWorldWindow`` on the UI thread. The owner-thread runtime is now
the only supported interactive GPU path, so this module keeps the old
``python src/game/gpu_test_app.py`` entry point and launches ``GameApp``.
"""

from __future__ import annotations

from src.game import config as cfg


def run_gpu_test() -> None:
    from src.game.app import run_game

    run_game(seed=42, cell_scale=cfg.CELL_SCALE)


if __name__ == "__main__":
    run_gpu_test()
