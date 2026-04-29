from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine.demo_app import run_demo


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the voxel engine desktop demo.")
    parser.add_argument("--grid-width", type=_positive_int, default=1080, help="Simulation grid width in cells.")
    parser.add_argument("--grid-height", type=_positive_int, default=760, help="Simulation grid height in cells.")
    parser.add_argument("--cell-scale", type=_positive_int, default=1, help="Default pixels per cell when window size is not given.")
    parser.add_argument("--window-width", type=_positive_int, help="Initial window width in pixels.")
    parser.add_argument("--window-height", type=_positive_int, help="Initial window height in pixels.")
    parser.add_argument("--substeps", type=_positive_int, default=1, help="Simulation substeps executed for each scheduled frame.")
    parser.add_argument("--no-liquid-brownian", action="store_true", help="Disable liquid Brownian motion jitter at startup.")
    parser.add_argument("--no-blocked-impulse", action="store_true", help="Disable blocked impulse residual intent at startup.")
    parser.add_argument(
        "--no-directional-fallback",
        action="store_true",
        help="Disable the nearest-angle fallback when the preferred motion direction is blocked at startup.",
    )
    parser.add_argument("--no-vsync", action="store_true", help="Disable vsync for the demo window.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        cell_scale=args.cell_scale,
        window_width=args.window_width,
        window_height=args.window_height,
        simulation_substeps=args.substeps,
        liquid_brownian_enabled=not args.no_liquid_brownian,
        blocked_impulse_enabled=not args.no_blocked_impulse,
        directional_fallback_enabled=not args.no_directional_fallback,
        vsync=not args.no_vsync,
    )
