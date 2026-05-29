"""Entry point for running the game."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging to file + stderr
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("game_debug.log", mode="w"),
        logging.StreamHandler(sys.stderr),
    ],
)
# Suppress noisy external loggers
logging.getLogger("pyglet").setLevel(logging.WARNING)
logging.getLogger("moderngl").setLevel(logging.WARNING)

# Ensure src/ is on the path when running as a module
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[1]  # src/game/__main__.py -> src/ -> project_root/
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from game.app import run_game

if __name__ == "__main__":
    run_game()
