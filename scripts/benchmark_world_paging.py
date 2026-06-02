from __future__ import annotations

"""Compatibility entrypoint for the owner-boundary paging benchmark."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_game_app import main


if __name__ == "__main__":
    main()
