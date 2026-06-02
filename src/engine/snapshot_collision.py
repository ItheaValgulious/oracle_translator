"""Phase 8: snapshot-driven collision and hazard probe.

Replaces the GridFeedback-primary path with a snapshot-driven path.
Given a ``LocalCellsSnapshot`` (or any object exposing the same
attributes) at world origin ``(world_x, world_y)`` and an opaque packed
plane, ``SnapshotCollisionProbe`` exposes the queries that
hero/EnemyA/B/C/projectile logic actually need:

- ``is_solid(world_x, world_y)`` — true if the cell belongs to a
  solid/structural family (used for hero ground checks, projectile
  terrain hits, enemy step-up logic).
- ``is_hazard(world_x, world_y)`` — true if the cell belongs to a
  hazard family (fire/acid/poison/tar etc.).
- ``family_at(world_x, world_y)`` — the underlying family id, or
  ``None`` if out of snapshot bounds / unknown.

The probe is a pure read against the snapshot envelope; it never calls
back into the GPU or the world.

This is the consumer side. The producer (GPU owner thread submitting a
new snapshot every tick) lives in ``snapshot_mailbox`` from Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# Family ids treated as solid for collision/ground checks.
DEFAULT_SOLID_FAMILIES: frozenset[str] = frozenset({
    "stone",
    "sand",
    "glass",
    "iron",
    "obsidian",
    "snow",
    "wood",
    "grass",
    "wood_plank",
    "entity_placeholder",
})

# Family ids treated as hazards for damage/burn checks.
DEFAULT_HAZARD_FAMILIES: frozenset[str] = frozenset({
    "fire",
    "acid",
    "poison",
    "tar",
})


@dataclass
class SnapshotCollisionProbe:
    """Pure read adapter over a snapshot envelope.

    The snapshot is treated as the source of truth for collision decisions
    in this region. Out-of-bounds queries return ``False`` (collisions)
    and ``None`` (family lookup), reflecting the "if no snapshot, treat
    as empty" policy from goal.md §4.
    """

    snapshot: object  # any LocalCellsSnapshot-like
    solid_families: frozenset[str] = DEFAULT_SOLID_FAMILIES
    hazard_families: frozenset[str] = DEFAULT_HAZARD_FAMILIES

    def _local(self, world_x: int, world_y: int) -> Optional[tuple[int, int]]:
        snap = self.snapshot
        wx0 = int(getattr(snap, "world_x", getattr(snap, "origin_x", 0)))
        wy0 = int(getattr(snap, "world_y", getattr(snap, "origin_y", 0)))
        lx = int(world_x) - wx0
        ly = int(world_y) - wy0
        if lx < 0 or ly < 0 or lx >= int(snap.width) or ly >= int(snap.height):
            return None
        return lx, ly

    def variant_at(self, world_x: int, world_y: int) -> int:
        coord = self._local(world_x, world_y)
        if coord is None:
            return int(self.snapshot.empty_variant_index)
        lx, ly = coord
        return int(self.snapshot.variant_indices[ly][lx])

    def family_at(self, world_x: int, world_y: int) -> Optional[str]:
        idx = self.variant_at(world_x, world_y)
        if idx == int(self.snapshot.empty_variant_index):
            return None
        families = self.snapshot.variant_families
        if 0 <= idx < len(families):
            return families[idx]
        return None

    def temperature_at(self, world_x: int, world_y: int) -> Optional[float]:
        coord = self._local(world_x, world_y)
        if coord is None:
            return None
        lx, ly = coord
        return float(self.snapshot.temperatures[ly][lx])

    def velocity_at(self, world_x: int, world_y: int) -> Optional[tuple[float, float]]:
        coord = self._local(world_x, world_y)
        if coord is None:
            return None
        lx, ly = coord
        vx, vy = self.snapshot.velocities[ly][lx]
        return float(vx), float(vy)

    def is_solid(self, world_x: int, world_y: int) -> bool:
        fam = self.family_at(world_x, world_y)
        return fam is not None and fam in self.solid_families

    def is_hazard(self, world_x: int, world_y: int) -> bool:
        fam = self.family_at(world_x, world_y)
        return fam is not None and fam in self.hazard_families

    def any_solid_in_aabb(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """True if any cell in the inclusive AABB is solid."""
        if x1 < x0 or y1 < y0:
            return False
        for wy in range(int(y0), int(y1) + 1):
            for wx in range(int(x0), int(x1) + 1):
                if self.is_solid(wx, wy):
                    return True
        return False

    def hazard_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        for wy in range(int(y0), int(y1) + 1):
            for wx in range(int(x0), int(x1) + 1):
                fam = self.family_at(wx, wy)
                if fam is not None and fam in self.hazard_families:
                    out.append((wx, wy, fam))
        return out


def build_probe_for_envelope(
    envelope_packed: object,
    *,
    solid_families: Optional[Iterable[str]] = None,
    hazard_families: Optional[Iterable[str]] = None,
) -> SnapshotCollisionProbe:
    """Build a probe from a snapshot envelope's ``packed`` payload.

    The packed payload is expected to be a ``LocalCellsSnapshot``-like
    object exposing the documented attributes from
    ``engine.gpu_backend.LocalCellsSnapshot``.
    """
    return SnapshotCollisionProbe(
        snapshot=envelope_packed,
        solid_families=(
            DEFAULT_SOLID_FAMILIES
            if solid_families is None
            else frozenset(solid_families)
        ),
        hazard_families=(
            DEFAULT_HAZARD_FAMILIES
            if hazard_families is None
            else frozenset(hazard_families)
        ),
    )
