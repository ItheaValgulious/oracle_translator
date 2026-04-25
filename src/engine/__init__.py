"""Voxel engine prototype core."""

from .gpu_backend import GpuSimulator
from .grid import Grid, create_grid
from .materials import build_material_registry
from .render import build_rgba_frame
from .scenarios import populate_demo_scene
from .sim import inject_cells, step

__all__ = [
    "Grid",
    "GpuSimulator",
    "build_material_registry",
    "build_rgba_frame",
    "create_grid",
    "inject_cells",
    "populate_demo_scene",
    "step",
]
