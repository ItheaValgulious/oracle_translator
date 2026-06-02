"""Phase 8 tests: snapshot-driven collision probe."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.snapshot_collision import (
    SnapshotCollisionProbe,
    build_probe_for_envelope,
)


@dataclass
class _FakeSnapshot:
    """LocalCellsSnapshot-compatible stub for unit tests."""
    world_x: int
    world_y: int
    width: int
    height: int
    variant_indices: tuple
    velocities: tuple
    temperatures: tuple
    variant_families: tuple
    empty_variant_index: int = 0
    entity_id: str = "test"
    issued_step: int = 0


def _make_snapshot(rows: list[list[int]], *, families: list[str], world_x: int = 100, world_y: int = 200, temps: list[list[float]] | None = None):
    height = len(rows)
    width = len(rows[0]) if rows else 0
    variant_indices = tuple(tuple(r) for r in rows)
    velocities = tuple(tuple((0.0, 0.0) for _ in range(width)) for _ in range(height))
    if temps is None:
        temps = [[20.0] * width for _ in range(height)]
    temperatures = tuple(tuple(r) for r in temps)
    return _FakeSnapshot(
        world_x=world_x,
        world_y=world_y,
        width=width,
        height=height,
        variant_indices=variant_indices,
        velocities=velocities,
        temperatures=temperatures,
        variant_families=tuple(families),
        empty_variant_index=0,
    )


class SnapshotCollisionProbeTests(unittest.TestCase):
    def test_is_solid_inside_snapshot(self) -> None:
        # families: 0=empty, 1=stone, 2=fire
        snap = _make_snapshot(
            rows=[
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            families=["empty", "stone", "fire"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        self.assertFalse(probe.is_solid(100, 200))  # local (0,0) -> empty
        self.assertTrue(probe.is_solid(102, 200))   # local (2,0) -> stone
        self.assertTrue(probe.is_solid(103, 201))   # local (3,1) -> stone

    def test_is_hazard(self) -> None:
        snap = _make_snapshot(
            rows=[
                [0, 2, 0],
            ],
            families=["empty", "stone", "fire"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        self.assertTrue(probe.is_hazard(101, 200))   # local (1,0) -> fire
        self.assertFalse(probe.is_hazard(100, 200))  # empty
        self.assertFalse(probe.is_hazard(102, 200))  # empty

    def test_out_of_bounds_returns_safe_defaults(self) -> None:
        snap = _make_snapshot(
            rows=[[1]],
            families=["empty", "stone"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        # Cell at the snapshot origin is solid (local 0,0).
        self.assertTrue(probe.is_solid(100, 200))
        # Outside the snapshot: not solid (the "treat as empty" rule).
        self.assertFalse(probe.is_solid(0, 0))
        self.assertFalse(probe.is_solid(999, 999))
        self.assertIsNone(probe.family_at(0, 0))

    def test_family_at_returns_string(self) -> None:
        snap = _make_snapshot(
            rows=[[1, 2]],
            families=["empty", "stone", "fire"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        self.assertEqual(probe.family_at(100, 200), "stone")
        self.assertEqual(probe.family_at(101, 200), "fire")
        self.assertIsNone(probe.family_at(50, 50))

    def test_temperature_lookup(self) -> None:
        snap = _make_snapshot(
            rows=[[1, 1]],
            families=["empty", "stone"],
            temps=[[20.0, 800.0]],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        self.assertEqual(probe.temperature_at(100, 200), 20.0)
        self.assertEqual(probe.temperature_at(101, 200), 800.0)
        self.assertIsNone(probe.temperature_at(50, 50))

    def test_local_cell_snapshot_origin_alias_and_velocity_lookup(self) -> None:
        snap = type("LocalLike", (), {
            "origin_x": 5,
            "origin_y": 7,
            "width": 1,
            "height": 1,
            "variant_indices": ((1,),),
            "velocities": (((2.5, -3.0),),),
            "temperatures": ((22.0,),),
            "variant_families": ("empty", "stone"),
            "empty_variant_index": 0,
        })()
        probe = SnapshotCollisionProbe(snapshot=snap)

        self.assertTrue(probe.is_solid(5, 7))
        self.assertEqual(probe.velocity_at(5, 7), (2.5, -3.0))
        self.assertIsNone(probe.velocity_at(4, 7))

    def test_any_solid_in_aabb(self) -> None:
        snap = _make_snapshot(
            rows=[
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            families=["empty", "stone"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        # AABB covering only empty cells.
        self.assertFalse(probe.any_solid_in_aabb(100, 200, 102, 200))
        # AABB covering the stone at local (1,1).
        self.assertTrue(probe.any_solid_in_aabb(100, 200, 102, 202))

    def test_hazard_cells_enumerated(self) -> None:
        snap = _make_snapshot(
            rows=[
                [0, 2, 0],
                [3, 0, 0],
            ],
            families=["empty", "stone", "fire", "acid"],
        )
        probe = SnapshotCollisionProbe(snapshot=snap)
        hazards = probe.hazard_cells(100, 200, 102, 201)
        # Sorted for determinism in assertion.
        hazards.sort()
        self.assertEqual(hazards, [(100, 201, "acid"), (101, 200, "fire")])


class SnapshotCollisionFactoryTests(unittest.TestCase):
    def test_build_probe_with_custom_family_sets(self) -> None:
        snap = _make_snapshot(rows=[[1]], families=["empty", "ice"])
        probe = build_probe_for_envelope(snap, solid_families={"ice"}, hazard_families=set())
        self.assertTrue(probe.is_solid(100, 200))
        self.assertFalse(probe.is_hazard(100, 200))


if __name__ == "__main__":
    unittest.main()
