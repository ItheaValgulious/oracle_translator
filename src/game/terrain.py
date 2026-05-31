"""Terrain generator: deterministic, seed-based, lazy chunk-level generation.

Generates 4 biomes:
  - Plains (x=0..BIOME_WIDTH): noise heightmap, grass surface, trees, ponds
  - Hillside (x=BIOME_WIDTH..2*BIOME_WIDTH): slope + noise, grass surface
  - Alpine (x=2*BIOME_WIDTH..3*BIOME_WIDTH): floating islands, bridges, ice/snow, shaft
  - Underground (x=3*BIOME_WIDTH..WORLD_WIDTH): chambers, corridors, obsidian walls, pools, boss
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from src.engine.types import CellState, CellFlag

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.engine.world import WorldChunkStore

from src.game import config as cfg


# ---------------------------------------------------------------------------
# Deterministic hash (same algorithm as motion._hash01 but step_id = seed)
# ---------------------------------------------------------------------------

def _hash01(seed: int, x: int, y: int, salt: int) -> float:
    value = (
        (seed + 1) * 374_761_393
        + (x + 11) * 668_265_263
        + (y + 17) * 2_147_483_647
        + (salt + 23) * 1_274_126_177
    ) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1_274_126_177) & 0xFFFFFFFF
    return value / 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Value noise (deterministic, seed-based)
# ---------------------------------------------------------------------------

def _value_noise(x: float, y: float, seed: int, salt: int = 0) -> float:
    """Multi-octave value noise with smoothing."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)  # smoothstep
    fy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _hash01(seed, ix, iy, salt)
    v10 = _hash01(seed, ix + 1, iy, salt)
    v01 = _hash01(seed, ix, iy + 1, salt)
    v11 = _hash01(seed, ix + 1, iy + 1, salt)
    v0 = v00 + (v10 - v00) * fx
    v1 = v01 + (v11 - v01) * fx
    return v0 + (v1 - v0) * fy


def _noise_octaves(x: float, y: float, seed: int,
                   octaves: int, persistence: float, salt: int = 0) -> float:
    """Sum multiple octaves of value noise, normalized to [0, 1]."""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for i in range(octaves):
        total += _value_noise(x * frequency, y * frequency, seed, salt + i * 1000) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    return total / max_val if max_val > 0 else 0.0


# ---------------------------------------------------------------------------
# Biome lookup
# ---------------------------------------------------------------------------

def _biome_for_x(world_x: int) -> str:
    if world_x < cfg.BIOME_WIDTH:
        return "plains"
    elif world_x < 2 * cfg.BIOME_WIDTH:
        return "hillside"
    elif world_x < 3 * cfg.BIOME_WIDTH:
        return "alpine"
    else:
        return "underground"


def _connect_mst(nodes: list[dict]) -> None:
    """Connect nodes via Kruskal's MST. Adds 'connections' list to each node.

    Each node gets a 'connections' key with a list of neighbor indices.
    Ensures full connectivity: every node is reachable from every other.
    """
    n = len(nodes)
    if n < 2:
        return
    # Initialize connections lists
    for node in nodes:
        node['connections'] = []
    # Build edge list with Euclidean distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = nodes[j]['cx'] - nodes[i]['cx']
            dy = nodes[j]['cy'] - nodes[i]['cy']
            d = math.sqrt(dx * dx + dy * dy)
            edges.append((d, i, j))
    edges.sort()
    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    # Kruskal's: add edges until all connected
    connected = 0
    for d, i, j in edges:
        if union(i, j):
            nodes[i]['connections'].append(j)
            nodes[j]['connections'].append(i)
            connected += 1
            if connected >= n - 1:
                break


# ---------------------------------------------------------------------------
# Cell factories (with FIXPOINT flag for non-falling platforms)
# ---------------------------------------------------------------------------

def _cell(family_id: str, variant_id: str, *,
          temp: float = 20.0, gen: int = 0, flags: int = 0) -> CellState:
    return CellState(
        family_id=family_id,
        variant_id=variant_id,
        temperature=temp,
        generation=gen,
        flags=flags,
    )


def _stone_platform(gen: int = 0) -> CellState:
    return _cell("stone", "stone_platform", gen=gen, flags=CellFlag.FIXPOINT)


def _stone_falling(gen: int = 0) -> CellState:
    return _cell("stone", "stone_falling", gen=gen)


def _grass_platform(gen: int = 0) -> CellState:
    return _cell("grass", "grass_platform", gen=gen, flags=CellFlag.FIXPOINT)


def _ice_cell(gen: int = 0) -> CellState:
    return _cell("water", "ice", temp=cfg.ALPINE_AMBIENT_TEMP, gen=gen, flags=CellFlag.FIXPOINT)


def _snow_cell(gen: int = 0) -> CellState:
    return _cell("snow", "snow_powder", temp=cfg.ALPINE_AMBIENT_TEMP, gen=gen)


def _wood_platform(gen: int = 0) -> CellState:
    return _cell("wood", "wood_platform", gen=gen, flags=CellFlag.FIXPOINT)


def _water_cell(gen: int = 0) -> CellState:
    return _cell("water", "water", gen=gen)


def _obsidian_platform(gen: int = 0) -> CellState:
    return _cell("obsidian", "obsidian_platform", gen=gen, flags=CellFlag.FIXPOINT)


def _oil_cell(gen: int = 0) -> CellState:
    return _cell("oil", "oil_liquid", gen=gen)


def _acid_cell(gen: int = 0) -> CellState:
    return _cell("magic_acid", "magic_acid_liquid", gen=gen)


def _fire_cell(gen: int = 0) -> CellState:
    return _cell("fire", "fire", temp=600.0, gen=gen)


def _poison_cell(gen: int = 0) -> CellState:
    return _cell("poison", "poison_liquid", gen=gen)


def _sand_surface_cell(gen: int = 0) -> CellState:
    return _cell("sand", "sand_powder", gen=gen, flags=CellFlag.FIXPOINT)


@dataclass(frozen=True)
class PlainsTreeFeature:
    center_x: int
    ground_y: float
    trunk_height: int
    canopy_radius_x: int
    canopy_radius_y: int
    canopy_center_y: float


@dataclass(frozen=True)
class PlainsPondFeature:
    center_x: int
    ground_y: float
    half_width: int
    max_depth: int
    water_surface_y: int


# ---------------------------------------------------------------------------
# TerrainGenerator
# ---------------------------------------------------------------------------

class TerrainGenerator:
    """Deterministic, seed-based terrain generator for lazy chunk creation.

    Usage:
        gen = TerrainGenerator(seed, registry)
        store = WorldChunkStore(..., chunk_generator=gen.generate_chunk)
    """

    def __init__(self, seed: int, registry) -> None:
        self.seed = seed
        self.registry = registry

    def generate_chunk(self, store: WorldChunkStore,
                       chunk_x: int, chunk_y: int,
                       chunk_size: int, seed: int) -> None:
        """WorldChunkStore chunk_generator callback.

        Fills every cell in the chunk based on biome.
        Precomputes heightmap per x-column and uses set-based feature placement.
        """
        t0 = perf_counter()
        x_start = chunk_x * chunk_size
        y_start = chunk_y * chunk_size
        x_end = x_start + chunk_size
        y_end = y_start + chunk_size
        biome_sample = _biome_for_x(x_start)
        log.debug("[terrain] generate_chunk cx=%d cy=%d range=[%d..%d) x [%d..%d) biome_sample=%s",
                 chunk_x, chunk_y, x_start, x_end, y_start, y_end, biome_sample)

        # Precompute heightmap for all x columns in this chunk
        # Optimization: compute raw heights first (1 noise eval per column),
        # then apply 3-point smoothing in a separate pass (0 noise evals).
        heights: dict[int, float] = {}
        biomes: dict[int, str] = {}
        raw_heights: dict[int, float] = {}
        for lx in range(x_start - 1, x_end + 1):  # +1 margin for smoothing
            biome = _biome_for_x(lx)
            if lx >= x_start:
                biomes[lx] = biome
            if biome == "plains":
                raw_heights[lx] = self._plains_raw_height(lx)
            elif biome == "hillside":
                raw_heights[lx] = self._hillside_raw_height(lx)
        # Apply 3-point smoothing
        for lx in range(x_start, x_end):
            biome = biomes[lx]
            if biome in ("plains", "hillside"):
                heights[lx] = (raw_heights.get(lx - 1, 0.0) +
                                raw_heights.get(lx, 0.0) +
                                raw_heights.get(lx + 1, 0.0)) / 3.0
            else:
                heights[lx] = 0.0

        # Precompute features for this chunk
        tree_cols: dict[int, PlainsTreeFeature] = {}
        pond_cols: dict[int, PlainsPondFeature] = {}
        for lx in range(x_start, x_end):
            if biomes[lx] == "plains":
                tc = self._plains_tree_feature(lx, heights)
                if tc is not None:
                    tree_cols[lx] = tc
                pc = self._plains_pond_feature(lx, heights)
                if pc is not None:
                    pond_cols[lx] = pc

        # Precompute alpine feature x-ranges for this chunk
        alpine_feature_xs: set[int] = set()
        has_alpine = any(b == "alpine" for b in biomes.values())
        alpine_islands = self._alpine_islands() if has_alpine else []
        if has_alpine:
            for isl in alpine_islands:
                ix0 = isl['cx'] - isl['w'] // 2
                ix1 = ix0 + isl['w']
                for x in range(max(x_start, ix0), min(x_end, ix1)):
                    alpine_feature_xs.add(x)
            for i, isl in enumerate(alpine_islands):
                for j in isl.get('connections', []):
                    if j <= i:
                        continue
                    other = alpine_islands[j]
                    bx0 = min(isl['cx'], other['cx'])
                    bx1 = max(isl['cx'], other['cx'])
                    for x in range(max(x_start, bx0), min(x_end, bx1 + 1)):
                        alpine_feature_xs.add(x)
            shaft_x = int(2 * cfg.BIOME_WIDTH + cfg.ALPINE_SHAFT_X_RATIO * cfg.BIOME_WIDTH)
            shaft_half = cfg.ALPINE_SHAFT_WIDTH
            for x in range(max(x_start, shaft_x - shaft_half), min(x_end, shaft_x + shaft_half + 1)):
                alpine_feature_xs.add(x)

        # Precompute underground feature x-ranges for this chunk
        ug_feature_xs: set[int] = set()
        has_underground = any(b == "underground" for b in biomes.values())
        ug_chambers = self._underground_chambers() if has_underground else []
        ug_y_min = y_end
        ug_y_max = y_start
        if has_underground:
            for ch in ug_chambers:
                half_w = ch['w'] // 2
                half_h = ch['h'] // 2
                for x in range(max(x_start, ch['cx'] - half_w), min(x_end, ch['cx'] + half_w)):
                    ug_feature_xs.add(x)
                ch_y0 = ch['cy'] - half_h
                ch_y1 = ch['cy'] + half_h
                if ch_y0 < y_end and ch_y1 > y_start:
                    ug_y_min = min(ug_y_min, max(y_start, ch_y0))
                    ug_y_max = max(ug_y_max, min(y_end, ch_y1))
            for i, ch in enumerate(ug_chambers):
                for j in ch.get('connections', []):
                    if j <= i:
                        continue
                    other = ug_chambers[j]
                    bx0 = min(ch['cx'], other['cx'])
                    bx1 = max(ch['cx'], other['cx'])
                    for x in range(max(x_start, bx0), min(x_end, bx1 + 1)):
                        ug_feature_xs.add(x)
                    corr_y_min = min(ch['cy'], other['cy']) - cfg.UNDERGROUND_CORRIDOR_HEIGHT
                    corr_y_max = max(ch['cy'], other['cy']) + cfg.UNDERGROUND_CORRIDOR_HEIGHT
                    if corr_y_min < y_end and corr_y_max > y_start:
                        ug_y_min = min(ug_y_min, max(y_start, corr_y_min))
                        ug_y_max = max(ug_y_max, min(y_end, corr_y_max))
            boss = ug_chambers[-1]
            boss_conns = boss.get('connections', [])
            if boss_conns:
                bn = ug_chambers[boss_conns[0]]
                bx0 = min(bn['cx'], boss['cx'])
                bx1 = max(bn['cx'], boss['cx'])
                for x in range(max(x_start, bx0), min(x_end, bx1 + 1)):
                    ug_feature_xs.add(x)

        # Chunk-level skip: pure alpine chunk with no features = all empty air
        only_alpine = has_alpine and not any(b != "alpine" for b in biomes.values())
        if only_alpine and not alpine_feature_xs:
            log.debug("[terrain] chunk (%d,%d) skipped: pure alpine, no features, %.1fms",
                     chunk_x, chunk_y, (perf_counter() - t0) * 1000)
            return  # entire chunk is empty air

        # Precompute y-ranges for alpine features to skip empty rows
        alpine_y_min = y_end
        alpine_y_max = y_start
        if has_alpine and alpine_feature_xs:
            for isl in alpine_islands:
                iy0 = isl['cy'] - isl['h'] // 2
                iy1 = iy0 + isl['h']
                if iy0 < y_end and iy1 > y_start:
                    alpine_y_min = min(alpine_y_min, max(y_start, iy0))
                    alpine_y_max = max(alpine_y_max, min(y_end, iy1))
            for i, isl in enumerate(alpine_islands):
                for j in isl.get('connections', []):
                    if j <= i:
                        continue
                    other = alpine_islands[j]
                    bridge_y_min = min(isl['cy'], other['cy']) - cfg.ALPINE_BRIDGE_THICKNESS
                    bridge_y_max = max(isl['cy'], other['cy']) + cfg.ALPINE_BRIDGE_THICKNESS
                    if bridge_y_min < y_end and bridge_y_max > y_start:
                        alpine_y_min = min(alpine_y_min, max(y_start, bridge_y_min))
                        alpine_y_max = max(alpine_y_max, min(y_end, bridge_y_max))
            shaft_y_min = cfg.ALPINE_ISLAND_Y_MAX + 20
            if shaft_y_min < y_end:
                alpine_y_min = min(alpine_y_min, max(y_start, shaft_y_min))
                alpine_y_max = max(alpine_y_max, y_end)

        # Precompute per-column min y (skip air above terrain) for plains/hillside
        col_y_min: dict[int, int] = {}
        for lx in range(x_start, x_end):
            biome = biomes[lx]
            ground = int(math.floor(heights[lx]))
            if biome == "plains":
                feature = tree_cols.get(lx)
                tree_above = 0
                if feature is not None:
                    tree_above = feature.trunk_height + feature.canopy_radius_y * 2
                col_y_min[lx] = max(y_start, ground - tree_above)
            elif biome == "hillside":
                col_y_min[lx] = max(y_start, ground)
            else:
                col_y_min[lx] = y_start

        # Process each biome group separately to avoid per-cell biome lookups
        plains_xs = [lx for lx in range(x_start, x_end) if biomes[lx] == "plains"]
        hillside_xs = [lx for lx in range(x_start, x_end) if biomes[lx] == "hillside"]
        alpine_xs = [lx for lx in range(x_start, x_end) if biomes[lx] == "alpine"]
        ug_xs = [lx for lx in range(x_start, x_end) if biomes[lx] == "underground"]

        # Plains columns — direct chunk writes (bypass store.set_cell overhead)
        grass_depth = cfg.PLAINS_GRASS_SURFACE_DEPTH
        floor_y = cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH
        cs = chunk_size
        span_writer = getattr(store, "_terrain_write_span", None)
        cell_writer = getattr(store, "_terrain_write_cell", None)

        def _direct_write(store, lx, y0, y1, cell):
            """Write cells directly to chunk dict, bypassing store.set_cell."""
            if callable(span_writer):
                span_writer(lx, y0, y1, cell)
                return
            cx = lx // cs
            local_x = lx % cs
            for ly in range(y0, y1):
                cy = ly // cs
                local_y = ly % cs
                key = (cx, cy)
                chunk = store._chunks.get(key)
                if chunk is None:
                    from src.engine.world import _WorldChunk
                    chunk = _WorldChunk()
                    store._chunks[key] = chunk
                chunk.cells[local_y * cs + local_x] = cell

        _stone = _stone_platform()
        _wood = _wood_platform()
        _water = _water_cell()
        _sand = _sand_surface_cell()
        _fire = _fire_cell()
        _poison = _poison_cell()
        _ice = _ice_cell()
        _snow = _snow_cell()
        _obsidian = _obsidian_platform()

        def _direct_set(store, lx: int, ly: int, cell: CellState) -> None:
            if callable(cell_writer):
                cell_writer(lx, ly, cell)
                return
            _direct_write(store, lx, ly, ly + 1, cell)

        def _store_set(store, lx: int, ly: int, cell: CellState) -> None:
            if callable(cell_writer):
                cell_writer(lx, ly, cell)
                return
            store.set_cell(lx, ly, cell)

        for lx in plains_xs:
            ground_y_f = heights[lx]
            tc = tree_cols.get(lx)
            pc = pond_cols.get(lx)
            pond_floor_y = self._pond_floor_y(lx, ground_y_f, pc)
            surface_y = int(math.floor(pond_floor_y))
            surface_cell = self._surface_cell_for_column(lx)

            # Build base terrain spans
            grass_top = max(y_start, surface_y)
            grass_bot = min(y_end, surface_y + grass_depth)
            if grass_top < grass_bot:
                _direct_write(store, lx, grass_top, grass_bot, _sand if surface_cell.family_id == "sand" else surface_cell)

            stone_top = max(y_start, surface_y + grass_depth)
            stone_bot = min(y_end, floor_y)
            if stone_top < stone_bot:
                _direct_write(store, lx, stone_top, stone_bot, _stone)

            deep_top = max(y_start, floor_y)
            if deep_top < y_end:
                _direct_write(store, lx, deep_top, y_end, _stone)

            # Pond overlay
            if pc is not None:
                wt = max(y_start, pc.water_surface_y)
                wb = min(y_end, surface_y + 1)
                if wt < wb:
                    _direct_write(store, lx, wt, wb, _water)
                st = max(y_start, surface_y + 1)
                sb = min(y_end, int(math.floor(ground_y_f)) + 1)
                if st < sb:
                    _direct_write(store, lx, st, sb, _stone)

            # Tree overlay
            elif tc is not None:
                trunk_base = int(math.floor(tc.ground_y))
                trunk_top = trunk_base - tc.trunk_height
                if lx == tc.center_x:
                    tt = max(y_start, trunk_top)
                    tb = min(y_end, trunk_base + 1)
                    if tt < tb:
                        _direct_write(store, lx, tt, tb, _wood)
                if abs(lx - tc.center_x) <= tc.canopy_radius_x:
                    leaf_y0 = max(y_start, int(math.floor(tc.canopy_center_y - tc.canopy_radius_y - 2)))
                    leaf_y1 = min(y_end, int(math.ceil(tc.canopy_center_y + tc.canopy_radius_y + 2)))
                    for ly in range(leaf_y0, leaf_y1):
                        if self._tree_leaf_present(tc, lx, ly):
                            _direct_set(store, lx, ly, _grass_platform())

        # Hillside columns — direct chunk writes
        hs_floor_y = cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH
        for lx in hillside_xs:
            ground_y_f = heights[lx]
            ground_y_i = int(math.floor(ground_y_f))
            gt = max(y_start, ground_y_i)
            gb = min(y_end, ground_y_i + 3)
            if gt < gb:
                _direct_write(store, lx, gt, gb, self._surface_cell_for_column(lx))
            st = max(y_start, ground_y_i + 3)
            sb = min(y_end, hs_floor_y)
            if st < sb:
                _direct_write(store, lx, st, sb, _stone)
            dt = max(y_start, hs_floor_y)
            if dt < y_end:
                _direct_write(store, lx, dt, y_end, _stone)

        # Alpine columns
        shaft_x = int(2 * cfg.BIOME_WIDTH + cfg.ALPINE_SHAFT_X_RATIO * cfg.BIOME_WIDTH)
        shaft_half = cfg.ALPINE_SHAFT_WIDTH
        shaft_y_min = cfg.ALPINE_ISLAND_Y_MAX + 20
        shaft_wall_thickness = max(1, shaft_half // 2)
        bridge_half_thickness = cfg.ALPINE_BRIDGE_THICKNESS // 2
        ice_depth = cfg.ALPINE_ICE_SURFACE_DEPTH
        for lx in alpine_xs:
            if lx not in alpine_feature_xs:
                continue
            for island in alpine_islands:
                ix0 = island['cx'] - island['w'] // 2
                iy0 = island['cy'] - island['h'] // 2
                iy1 = iy0 + island['h']
                if not (ix0 <= lx < ix0 + island['w']):
                    continue
                ice_top = max(y_start, iy0)
                ice_bottom = min(y_end, iy0 + ice_depth)
                if ice_top < ice_bottom:
                    _direct_write(store, lx, ice_top, ice_bottom, _ice)
                stone_top = max(y_start, iy0 + ice_depth)
                stone_bottom = min(y_end, iy1)
                if stone_top < stone_bottom:
                    _direct_write(store, lx, stone_top, stone_bottom, _stone)
                snow_y = iy0 + ice_depth
                if y_start <= snow_y < y_end and _hash01(self.seed, lx, snow_y, 600) < 0.3:
                    _direct_set(store, lx, snow_y, _snow)
            for i, island in enumerate(alpine_islands):
                for j in island.get('connections', []):
                    if j <= i:
                        continue
                    other = alpine_islands[j]
                    bx0 = min(island['cx'], other['cx'])
                    bx1 = max(island['cx'], other['cx'])
                    if not (bx0 <= lx <= bx1):
                        continue
                    span = bx1 - bx0
                    if span < 1:
                        continue
                    t = (lx - bx0) / span
                    bridge_y = int(island['cy'] + (other['cy'] - island['cy']) * t)
                    bridge_top = max(y_start, bridge_y - bridge_half_thickness)
                    bridge_bottom = min(y_end, bridge_y + bridge_half_thickness + 1)
                    for ly in range(bridge_top, bridge_bottom):
                        overlaps_island = False
                        for island_fill in alpine_islands:
                            island_x0 = island_fill['cx'] - island_fill['w'] // 2
                            island_y0 = island_fill['cy'] - island_fill['h'] // 2
                            if (island_x0 <= lx < island_x0 + island_fill['w']
                                    and island_y0 <= ly < island_y0 + island_fill['h']):
                                overlaps_island = True
                                break
                        if not overlaps_island:
                            _direct_set(store, lx, ly, _stone)
            dx = abs(lx - shaft_x)
            if dx <= shaft_half and dx > shaft_wall_thickness:
                shaft_top = max(y_start, shaft_y_min)
                if shaft_top < y_end:
                    _direct_write(store, lx, shaft_top, y_end, _obsidian)

        # Underground columns
        ug_cap_y = cfg.UNDERGROUND_Y_START + cfg.UNDERGROUND_STONE_CAP_DEPTH
        ug_fire_y = cfg.UNDERGROUND_SURFACE_FIRE_DEPTH
        ug_poison_y = cfg.UNDERGROUND_SURFACE_FIRE_DEPTH + cfg.UNDERGROUND_SURFACE_POISON_DEPTH
        for lx in ug_xs:
            fire_top = y_start
            fire_bottom = min(y_end, ug_fire_y)
            if fire_top < fire_bottom:
                _direct_write(store, lx, fire_top, fire_bottom, _fire)

            poison_top = max(y_start, ug_fire_y)
            poison_bottom = min(y_end, ug_poison_y)
            if poison_top < poison_bottom:
                _direct_write(store, lx, poison_top, poison_bottom, _poison)

            cap_top = max(y_start, ug_poison_y)
            cap_bottom = min(y_end, ug_cap_y)
            if cap_top < cap_bottom:
                _direct_write(store, lx, cap_top, cap_bottom, _stone)

            below_cap_top = max(y_start, ug_cap_y)
            if below_cap_top >= y_end:
                continue
            if lx not in ug_feature_xs or below_cap_top >= ug_y_max or y_end <= ug_y_min:
                _direct_write(store, lx, below_cap_top, y_end, _obsidian)
                continue

            breakpoints = self._underground_column_breakpoints(
                lx,
                below_cap_top,
                y_end,
                ug_chambers,
            )
            for seg_top, seg_bottom in zip(breakpoints, breakpoints[1:]):
                if seg_top >= seg_bottom:
                    continue
                cell = self._underground_cell(lx, seg_top)
                if cell is not None:
                    _direct_write(store, lx, seg_top, seg_bottom, cell)

        elapsed = (perf_counter() - t0) * 1000
        log.debug("[terrain] chunk (%d,%d) done in %.1fms", chunk_x, chunk_y, elapsed)

    # ── Surface height queries ──

    def ground_height_at(self, world_x: int) -> float:
        """Return the y coordinate of the topmost solid surface at world_x."""
        biome = _biome_for_x(world_x)
        if biome == "plains":
            return self._plains_height(world_x)
        elif biome == "hillside":
            return self._hillside_height(world_x)
        elif biome == "alpine":
            return self._alpine_surface_y(world_x)
        else:
            return self._underground_surface_y(world_x)

    def cell_at(self, wx: int, wy: int) -> CellState | None:
        biome = _biome_for_x(wx)
        if biome == "plains":
            return self._plains_cell_fast(
                wx,
                wy,
                self._plains_height(wx),
                tree_info=self._plains_tree_feature(wx),
                pond_info=self._plains_pond_feature(wx),
            )
        if biome == "hillside":
            return self._hillside_cell_fast(wx, wy, self._hillside_height(wx))
        if biome == "alpine":
            return self._alpine_cell(wx, wy)
        return self._underground_cell(wx, wy)

    def find_spawn_point_near(
        self,
        world_x: int,
        *,
        entity_width: float,
        entity_height: float,
        search_radius: int = 256,
        step: int = 4,
    ) -> tuple[float, float] | None:
        half_width = max(1, int(math.ceil(entity_width / 2.0)))
        height = max(1, int(math.ceil(entity_height)))
        max_x = cfg.WORLD_WIDTH - half_width - 2

        def _try_column(candidate_x: int) -> tuple[float, float] | None:
            candidate_x = max(half_width + 1, min(max_x, int(candidate_x)))
            base_floor_y = int(math.floor(self.ground_height_at(candidate_x)))
            floor_candidates = [base_floor_y]
            if _biome_for_x(candidate_x) == "underground":
                search_limit = min(cfg.WORLD_HEIGHT - 2, base_floor_y + cfg.UNDERGROUND_STONE_CAP_DEPTH + cfg.UNDERGROUND_CHAMBER_MAX_H * 2)
                floor_candidates = list(range(max(1, base_floor_y), search_limit + 1, 2))
            for floor_y in floor_candidates:
                spawn_y = floor_y - 1 - entity_height
                if spawn_y < 1:
                    continue
                solid_floor = False
                blocked = False
                for sample_x in range(candidate_x - half_width, candidate_x + half_width + 1):
                    sample_floor = self.cell_at(sample_x, floor_y)
                    if sample_floor is not None and not sample_floor.is_empty:
                        solid_floor = True
                    else:
                        blocked = True
                        break
                    for sample_y in range(int(math.floor(spawn_y)), int(math.ceil(spawn_y + height))):
                        cell = self.cell_at(sample_x, sample_y)
                        if cell is not None and not cell.is_empty:
                            blocked = True
                            break
                    if blocked:
                        break
                if solid_floor and not blocked:
                    return (float(candidate_x), float(spawn_y))
            return None

        offsets = [0]
        stride = max(1, step)
        for delta in range(stride, max(1, search_radius) + 1, stride):
            offsets.extend((delta, -delta))
        for delta in offsets:
            spawn = _try_column(int(world_x + delta))
            if spawn is not None:
                return spawn

        biome = _biome_for_x(int(world_x))
        if biome == "alpine":
            islands = sorted(self._alpine_islands(), key=lambda island: abs(island["cx"] - world_x))
            for island in islands:
                for candidate_x in range(island["cx"], island["cx"] + island["w"] // 3 + 1, max(4, island["w"] // 12)):
                    spawn = _try_column(candidate_x)
                    if spawn is not None:
                        return spawn
                    spawn = _try_column(island["cx"] - (candidate_x - island["cx"]))
                    if spawn is not None:
                        return spawn
        elif biome == "underground":
            chambers = sorted(self._underground_chambers(), key=lambda chamber: abs(chamber["cx"] - world_x))
            for chamber in chambers:
                span = max(8, chamber["w"] // 3)
                stride = max(4, chamber["w"] // 12)
                for delta in range(0, span + 1, stride):
                    for candidate_x in (chamber["cx"] + delta, chamber["cx"] - delta):
                        spawn = _try_column(candidate_x)
                        if spawn is not None:
                            return spawn
        return None

    def spawn_points(self) -> list[dict]:
        """Seed-based continuous enemy distribution across all biomes."""
        if hasattr(self, '_cached_spawn_points'):
            return self._cached_spawn_points
        points: list[dict] = []
        for wx in range(cfg.ENEMY_SPACING, cfg.WORLD_WIDTH - cfg.ENEMY_SPACING, cfg.ENEMY_SPACING):
            biome = _biome_for_x(wx)
            if biome == "underground":
                continue
            if _hash01(self.seed, wx, 0, cfg.ENEMY_TYPE_SALT_A) < cfg.ENEMY_A_PROBABILITY:
                ground_y = self.ground_height_at(wx)
                y = ground_y - 1 - cfg.ENEMY_A_HEIGHT
                points.append({"type": "A", "x": float(wx), "y": y, "biome": biome})
            if biome in ("hillside", "alpine") and _hash01(self.seed, wx, 0, cfg.ENEMY_TYPE_SALT_B) < cfg.ENEMY_B_PROBABILITY:
                ground_y = self.ground_height_at(wx)
                y = ground_y - 1 - cfg.ENEMY_B_HOVER_HEIGHT - cfg.ENEMY_B_HEIGHT
                points.append({"type": "B", "x": float(wx), "y": y, "biome": biome})

        # Underground: each chamber has 1-2 EnemyA
        chambers = self._underground_chambers()
        for i, ch in enumerate(chambers[:-1]):  # non-boss chambers
            count = 1 + int(_hash01(self.seed, i, 20, 850) < 0.5)
            for j in range(count):
                offset_x = int(_hash01(self.seed, j, i, 860) * ch['w'] * 0.4) - ch['w'] // 4
                x = ch['cx'] + offset_x
                # Chamber floor: cy + h/2 is bottom edge in world coords
                floor_y = ch['cy'] + ch['h'] // 2
                y = floor_y - 1 - cfg.ENEMY_A_HEIGHT
                points.append({"type": "A", "x": float(x), "y": float(y), "biome": "underground"})

        # Boss in last chamber
        boss_ch = chambers[-1]
        boss_floor_y = boss_ch['cy'] + boss_ch['h'] // 2
        points.append({
            "type": "C",
            "x": float(boss_ch['cx']),
            "y": float(boss_floor_y - 1 - cfg.ENEMY_C_HEIGHT),
            "biome": "underground",
        })

        self._cached_spawn_points = points
        return points

    # ── Plains ──

    def _plains_raw_height(self, wx: int) -> float:
        """Raw noise height without smoothing."""
        h = _noise_octaves(
            float(wx) / cfg.PLAINS_NOISE_SCALE, 0.0, self.seed,
            cfg.PLAINS_NOISE_OCTAVES, cfg.PLAINS_NOISE_PERSISTENCE, salt=100,
        )
        return cfg.PLAINS_GROUND_BASE_Y + (h - 0.5) * 2.0 * cfg.PLAINS_NOISE_AMPLITUDE

    def _plains_height(self, wx: int) -> float:
        """Smoothed height: 3-point mean to eliminate noise jaggedness."""
        h0 = self._plains_raw_height(wx - 1)
        h1 = self._plains_raw_height(wx)
        h2 = self._plains_raw_height(wx + 1)
        return (h0 + h1 + h2) / 3.0

    def _plains_cell(self, wx: int, wy: int) -> CellState | None:
        return self._plains_cell_fast(wx, wy, self._plains_height(wx))

    def _plains_cell_fast(self, wx: int, wy: int, ground_y: float,
                          tree_info: PlainsTreeFeature | None = None,
                          pond_info: PlainsPondFeature | None = None) -> CellState | None:
        natural_surface = self._pond_floor_y(wx, ground_y, pond_info)
        surface_y = int(math.floor(natural_surface))

        if pond_info is not None:
            if pond_info.water_surface_y <= wy <= surface_y:
                return _water_cell()
            if surface_y < wy <= int(math.floor(ground_y)):
                return _stone_platform()

        if tree_info is not None and pond_info is None:
            trunk_base = int(math.floor(tree_info.ground_y))
            trunk_top = trunk_base - tree_info.trunk_height
            if self._tree_leaf_present(tree_info, wx, wy):
                return _grass_platform()
            if trunk_top <= wy <= trunk_base and wx == tree_info.center_x:
                return _wood_platform()

        if wy >= cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH:
            return _stone_platform()
        if wy >= surface_y + cfg.PLAINS_GRASS_SURFACE_DEPTH:
            return _stone_platform()
        if wy >= surface_y:
            return self._surface_cell_for_column(wx)
        return None

    def _surface_cell_for_column(self, wx: int) -> CellState:
        surface_noise = _noise_octaves(
            float(wx) / cfg.PLAINS_SURFACE_PATCH_SCALE,
            0.0,
            self.seed,
            3,
            0.55,
            salt=140,
        )
        if surface_noise >= cfg.PLAINS_SURFACE_GRASS_THRESHOLD:
            return _grass_platform()
        if surface_noise >= cfg.PLAINS_SURFACE_SAND_THRESHOLD:
            return _sand_surface_cell()
        return _stone_platform()

    def _tree_leaf_present(self, feature: PlainsTreeFeature, wx: int, wy: int) -> bool:
        dx = (wx - feature.center_x) / max(1.0, float(feature.canopy_radius_x))
        dy = (wy - feature.canopy_center_y) / max(1.0, float(feature.canopy_radius_y))
        radial = dx * dx + dy * dy
        if radial > 1.08:
            return False
        edge_noise = _hash01(self.seed, wx, wy, 545)
        if radial > 0.82 and edge_noise < (radial - 0.82) * 1.4:
            return False
        return True

    def _pond_floor_y(self, wx: int, ground_y: float, pond_info: PlainsPondFeature | None) -> float:
        if pond_info is None:
            return ground_y
        dx = abs(wx - pond_info.center_x)
        if dx > pond_info.half_width:
            return ground_y
        edge = 1.0 - dx / max(1.0, float(pond_info.half_width))
        profile = edge * edge * (1.2 - 0.2 * edge)
        noise = 0.7 + 0.6 * _noise_octaves(float(wx) / 17.0, float(pond_info.center_x) / 43.0, self.seed, 2, 0.5, salt=530)
        return ground_y + pond_info.max_depth * profile * noise

    def _plains_tree_feature(self, wx: int, heights: dict[int, float] | None = None) -> PlainsTreeFeature | None:
        """If wx is the center column of a tree, return a deterministic profile."""
        spacing = max(1, cfg.PLAINS_TREE_CANOPY_W + 20)
        center = round(wx / spacing) * spacing
        if center == wx and 0 < wx < cfg.BIOME_WIDTH:
            if _hash01(self.seed, center, 0, 500) < 0.5:
                ground = heights[center] if heights is not None and center in heights else self._plains_height(center)
                trunk_height = cfg.PLAINS_TREE_HEIGHT + int((_hash01(self.seed, center, 4, 520) - 0.5) * 24.0)
                canopy_rx = max(8, cfg.PLAINS_TREE_CANOPY_W // 2 + int((_hash01(self.seed, center, 5, 521) - 0.5) * 16.0))
                canopy_ry = max(8, cfg.PLAINS_TREE_CANOPY_H // 2 + int((_hash01(self.seed, center, 6, 522) - 0.5) * 12.0))
                trunk_top = int(math.floor(ground)) - trunk_height
                canopy_center_y = trunk_top + canopy_ry * (0.55 + 0.2 * _hash01(self.seed, center, 7, 523))
                return PlainsTreeFeature(
                    center_x=center,
                    ground_y=ground,
                    trunk_height=trunk_height,
                    canopy_radius_x=canopy_rx,
                    canopy_radius_y=canopy_ry,
                    canopy_center_y=canopy_center_y,
                )
        return None

    def _plains_pond_feature(self, wx: int, heights: dict[int, float] | None = None) -> PlainsPondFeature | None:
        """If wx is inside a pond footprint, return a deterministic profile."""
        spacing = max(1, cfg.PLAINS_POND_WIDTH * 4)
        center = round(wx / spacing) * spacing
        if 0 < wx < cfg.BIOME_WIDTH:
            half_width = max(16, cfg.PLAINS_POND_WIDTH // 2 + int((_hash01(self.seed, center, 8, 531) + 0.6) * cfg.PLAINS_POND_WIDTH * 0.65))
        else:
            half_width = cfg.PLAINS_POND_WIDTH // 2
        if abs(wx - center) <= half_width and 0 < wx < cfg.BIOME_WIDTH:
            if _hash01(self.seed, center, 1, 510) < 0.3:
                ground = heights[center] if heights is not None and center in heights else self._plains_height(center)
                max_depth = max(cfg.PLAINS_POND_DEPTH, int(cfg.PLAINS_POND_DEPTH * (1.0 + _hash01(self.seed, center, 9, 532))))
                water_surface_y = int(math.floor(ground)) + max(2, int(max_depth * 0.28))
                return PlainsPondFeature(
                    center_x=center,
                    ground_y=ground,
                    half_width=half_width,
                    max_depth=max_depth,
                    water_surface_y=water_surface_y,
                )
        return None

    # ── Hillside ──

    def _hillside_raw_height(self, wx: int) -> float:
        t = (wx - cfg.BIOME_WIDTH) / cfg.BIOME_WIDTH
        base = cfg.HILLSIDE_GROUND_START_Y + (cfg.HILLSIDE_GROUND_END_Y - cfg.HILLSIDE_GROUND_START_Y) * t
        h = _noise_octaves(
            float(wx) / cfg.HILLSIDE_NOISE_SCALE, 0.0, self.seed,
            cfg.HILLSIDE_NOISE_OCTAVES, cfg.HILLSIDE_NOISE_PERSISTENCE, salt=200,
        )
        return base + (h - 0.5) * 2.0 * cfg.HILLSIDE_NOISE_AMPLITUDE

    def _hillside_height(self, wx: int) -> float:
        """Smoothed height: 3-point mean to eliminate noise jaggedness."""
        h0 = self._hillside_raw_height(wx - 1)
        h1 = self._hillside_raw_height(wx)
        h2 = self._hillside_raw_height(wx + 1)
        return (h0 + h1 + h2) / 3.0

    def _hillside_cell(self, wx: int, wy: int) -> CellState | None:
        return self._hillside_cell_fast(wx, wy, self._hillside_height(wx))

    def _hillside_cell_fast(self, wx: int, wy: int, ground_y: float) -> CellState | None:
        ground_y_int = int(math.floor(ground_y))
        if wy >= cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH:
            return _stone_platform()
        if wy >= ground_y_int + 3:
            return _stone_platform()
        if wy >= ground_y_int:
            return _grass_platform()
        return None

    # ── Alpine ──

    def _alpine_islands(self) -> list[dict]:
        """Generate deterministic floating island descriptors (cached)."""
        if hasattr(self, '_cached_alpine_islands'):
            return self._cached_alpine_islands
        islands = []
        for i in range(cfg.ALPINE_ISLAND_COUNT):
            cx = (cfg.BIOME_WIDTH * 2
                  + int(_hash01(self.seed, i, 0, 300) * cfg.BIOME_WIDTH))
            cy = cfg.ALPINE_ISLAND_Y_MIN + int(
                _hash01(self.seed, i, 1, 301) * (cfg.ALPINE_ISLAND_Y_MAX - cfg.ALPINE_ISLAND_Y_MIN))
            w = cfg.ALPINE_ISLAND_MIN_WIDTH + int(
                _hash01(self.seed, i, 2, 302) * (cfg.ALPINE_ISLAND_MAX_WIDTH - cfg.ALPINE_ISLAND_MIN_WIDTH))
            h = cfg.ALPINE_ISLAND_MIN_HEIGHT + int(
                _hash01(self.seed, i, 3, 303) * (cfg.ALPINE_ISLAND_MAX_HEIGHT - cfg.ALPINE_ISLAND_MIN_HEIGHT))
            islands.append({'cx': cx, 'cy': cy, 'w': w, 'h': h})
        # Connect via minimum spanning tree (Kruskal's) for full connectivity
        _connect_mst(islands)
        self._cached_alpine_islands = islands
        return islands

    def _alpine_cell(self, wx: int, wy: int) -> CellState | None:
        islands = self._alpine_islands()
        # Quick bounding-box check: skip if outside all features
        bounds = self._alpine_bounds()
        if not (bounds[0] <= wx <= bounds[1] and bounds[2] <= wy <= bounds[3]):
            return None
        # Floating islands
        ice_depth = cfg.ALPINE_ICE_SURFACE_DEPTH
        for island in islands:
            ix = island['cx'] - island['w'] // 2
            iy = island['cy'] - island['h'] // 2
            if (ix <= wx < ix + island['w'] and iy <= wy < iy + island['h']):
                if wy < iy + ice_depth:
                    return _ice_cell()
                # Snow scattered on ice surface
                if wy == iy + ice_depth and _hash01(self.seed, wx, wy, 600) < 0.3:
                    return _snow_cell()
                return _stone_platform()
        # Bridges between islands (MST connections)
        for i, island in enumerate(islands):
            for j in island.get('connections', []):
                if j <= i:
                    continue
                other = islands[j]
                bx0 = min(island['cx'], other['cx'])
                bx1 = max(island['cx'], other['cx'])
                if not (bx0 <= wx <= bx1):
                    continue
                span = bx1 - bx0
                if span < 1:
                    continue
                t = (wx - bx0) / span
                bridge_y = int(island['cy'] + (other['cy'] - island['cy']) * t)
                if abs(wy - bridge_y) <= cfg.ALPINE_BRIDGE_THICKNESS // 2:
                    return _stone_platform()
        # Shaft
        shaft_x = int(2 * cfg.BIOME_WIDTH + cfg.ALPINE_SHAFT_X_RATIO * cfg.BIOME_WIDTH)
        shaft_half = cfg.ALPINE_SHAFT_WIDTH
        dx = abs(wx - shaft_x)
        if dx <= shaft_half and wy >= cfg.ALPINE_ISLAND_Y_MAX + 20:
            wall_thickness = max(1, shaft_half // 2)
            if dx > wall_thickness:
                return _obsidian_platform()
            return None
        return None

    def _alpine_bounds(self) -> tuple[int, int, int, int]:
        """Cached bounding box (x_min, x_max, y_min, y_max) of all alpine features."""
        if hasattr(self, '_cached_alpine_bounds'):
            return self._cached_alpine_bounds
        islands = self._alpine_islands()
        x_min = min(i['cx'] - i['w'] // 2 for i in islands)
        x_max = max(i['cx'] + i['w'] // 2 for i in islands)
        y_min = min(i['cy'] - i['h'] // 2 for i in islands)
        y_max = max(i['cy'] + i['h'] // 2 for i in islands)
        # Include shaft
        shaft_x = int(2 * cfg.BIOME_WIDTH + cfg.ALPINE_SHAFT_X_RATIO * cfg.BIOME_WIDTH)
        x_min = min(x_min, shaft_x - cfg.ALPINE_SHAFT_WIDTH)
        x_max = max(x_max, shaft_x + cfg.ALPINE_SHAFT_WIDTH)
        y_max = max(y_max, cfg.WORLD_HEIGHT)
        # Include bridges (extend y range)
        y_min = min(y_min, min(i['cy'] for i in islands))
        y_max = max(y_max, max(i['cy'] for i in islands))
        # Add margin
        margin = max(cfg.ALPINE_BRIDGE_THICKNESS, cfg.ALPINE_ISLAND_MAX_HEIGHT) + 10
        self._cached_alpine_bounds = (x_min - margin, x_max + margin, y_min - margin, y_max + margin)
        return self._cached_alpine_bounds

    def _alpine_surface_y(self, wx: int) -> float:
        for island in self._alpine_islands():
            ix = island['cx'] - island['w'] // 2
            if ix <= wx < ix + island['w']:
                return float(island['cy'] - island['h'] // 2)
        return float(cfg.ALPINE_ISLAND_Y_MIN)

    # ── Underground ──

    def _underground_chambers(self) -> list[dict]:
        """Generate deterministic chamber descriptors (cached)."""
        if hasattr(self, '_cached_underground_chambers'):
            return self._cached_underground_chambers
        ug_start = 3 * cfg.BIOME_WIDTH
        chambers = []
        for i in range(cfg.UNDERGROUND_CHAMBER_COUNT):
            cx = ug_start + int(
                _hash01(self.seed, i, 10, 400) * cfg.BIOME_WIDTH * 0.8)
            cy = cfg.UNDERGROUND_Y_START + cfg.UNDERGROUND_STONE_CAP_DEPTH + int(
                _hash01(self.seed, i, 11, 401) * (cfg.WORLD_HEIGHT - cfg.UNDERGROUND_Y_START - cfg.UNDERGROUND_STONE_CAP_DEPTH - 200))
            w = cfg.UNDERGROUND_CHAMBER_MIN_W + int(
                _hash01(self.seed, i, 12, 402) * (cfg.UNDERGROUND_CHAMBER_MAX_W - cfg.UNDERGROUND_CHAMBER_MIN_W))
            h = cfg.UNDERGROUND_CHAMBER_MIN_H + int(
                _hash01(self.seed, i, 13, 403) * (cfg.UNDERGROUND_CHAMBER_MAX_H - cfg.UNDERGROUND_CHAMBER_MIN_H))
            has_water = _hash01(self.seed, i, 14, 404) < 0.35
            has_acid = _hash01(self.seed, i, 15, 405) < 0.2
            has_oil = _hash01(self.seed, i, 16, 406) < 0.25
            chambers.append({
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'has_water': has_water, 'has_acid': has_acid, 'has_oil': has_oil,
            })
        # Boss chamber at the rightmost position
        boss_x = ug_start + cfg.BIOME_WIDTH - 300
        boss_y = cfg.UNDERGROUND_Y_START + 400
        chambers.append({
            'cx': boss_x, 'cy': boss_y,
            'w': cfg.UNDERGROUND_BOSS_CHAMBER_W,
            'h': cfg.UNDERGROUND_BOSS_CHAMBER_H,
            'has_water': False, 'has_acid': False, 'has_oil': False,
            'is_boss': True,
        })
        # Connect via minimum spanning tree for full connectivity
        _connect_mst(chambers)
        self._cached_underground_chambers = chambers
        return chambers

    def _underground_column_breakpoints(
        self,
        wx: int,
        y0: int,
        y1: int,
        chambers: list[dict],
    ) -> list[int]:
        breakpoints = {int(y0), int(y1)}
        corridor_half_height = cfg.UNDERGROUND_CORRIDOR_HEIGHT / 2.0

        for ch in chambers:
            half_w = ch['w'] // 2
            if not (ch['cx'] - half_w <= wx < ch['cx'] + half_w):
                continue
            top_y = ch['cy'] - ch['h'] // 2
            bottom_y = ch['cy'] + ch['h'] // 2
            if bottom_y <= y0 or top_y >= y1:
                continue
            breakpoints.add(max(y0, top_y))
            breakpoints.add(min(y1, bottom_y))
            has_liquid = ch.get('has_water') or ch.get('has_acid') or ch.get('has_oil')
            if not ch.get('is_boss') and has_liquid:
                pool_y = ch['cy'] + ch['h'] // 2 - 5
                if y0 < pool_y < y1:
                    breakpoints.add(pool_y)

        for i, ch in enumerate(chambers):
            for j in ch.get('connections', []):
                if j <= i:
                    continue
                other = chambers[j]
                bx0 = min(ch['cx'], other['cx'])
                bx1 = max(ch['cx'], other['cx'])
                if not (bx0 <= wx <= bx1):
                    continue
                span = bx1 - bx0
                if span < 1:
                    continue
                t = (wx - bx0) / span
                center_y = ch['cy'] + (other['cy'] - ch['cy']) * t
                top_y = int(math.ceil(center_y - corridor_half_height))
                bottom_y = int(math.floor(center_y + corridor_half_height)) + 1
                if bottom_y <= y0 or top_y >= y1:
                    continue
                breakpoints.add(max(y0, top_y))
                breakpoints.add(min(y1, bottom_y))

        boss = chambers[-1]
        boss_conns = boss.get('connections', [])
        if boss_conns:
            bn = chambers[boss_conns[0]]
            bx0 = min(bn['cx'], boss['cx'])
            bx1 = max(bn['cx'], boss['cx'])
            if bx0 <= wx <= bx1 and abs(wx - boss['cx']) <= cfg.UNDERGROUND_CORRIDOR_WIDTH / 2:
                top_y = min(bn['cy'], boss['cy'])
                bottom_y = max(bn['cy'], boss['cy']) + 1
                if bottom_y > y0 and top_y < y1:
                    breakpoints.add(max(y0, top_y))
                    breakpoints.add(min(y1, bottom_y))

        return sorted(breakpoints)

    def _underground_cell(self, wx: int, wy: int) -> CellState | None:
        # Fast layer checks (most cells hit these early exits)
        if wy < cfg.UNDERGROUND_SURFACE_FIRE_DEPTH:
            return _fire_cell()
        if wy < cfg.UNDERGROUND_SURFACE_FIRE_DEPTH + cfg.UNDERGROUND_SURFACE_POISON_DEPTH:
            return _poison_cell()
        if wy < cfg.UNDERGROUND_Y_START + cfg.UNDERGROUND_STONE_CAP_DEPTH:
            return _stone_platform()

        # Below stone cap: check features
        chambers = self._underground_chambers()

        # Check regular chambers (not boss)
        for ch in chambers:
            if ch.get('is_boss'):
                continue
            half_w = ch['w'] // 2
            half_h = ch['h'] // 2
            if (ch['cx'] - half_w <= wx < ch['cx'] + half_w
                    and ch['cy'] - half_h <= wy < ch['cy'] + half_h):
                pool_y = ch['cy'] + ch['h'] // 2 - 5
                if wy >= pool_y:
                    if ch.get('has_water'):
                        return _water_cell()
                    if ch.get('has_acid'):
                        return _acid_cell()
                    if ch.get('has_oil'):
                        return _oil_cell()
                return None

        # Boss chamber
        boss = chambers[-1]
        half_w = boss['w'] // 2
        half_h = boss['h'] // 2
        if (boss['cx'] - half_w <= wx < boss['cx'] + half_w
                and boss['cy'] - half_h <= wy < boss['cy'] + half_h):
            return None

        # Corridors (only check if wx is in a corridor's x-range)
        for i, ch in enumerate(chambers):
            for j in ch.get('connections', []):
                if j <= i:
                    continue
                other = chambers[j]
                bx0 = min(ch['cx'], other['cx'])
                bx1 = max(ch['cx'], other['cx'])
                if bx0 <= wx <= bx1:
                    span = bx1 - bx0
                    if span >= 1:
                        t = (wx - bx0) / span
                        center_y = ch['cy'] + (other['cy'] - ch['cy']) * t
                        if abs(wy - center_y) <= cfg.UNDERGROUND_CORRIDOR_HEIGHT / 2:
                            return None

        # Boss corridor
        boss_conns = boss.get('connections', [])
        if boss_conns:
            bn = chambers[boss_conns[0]]
            bx0 = min(bn['cx'], boss['cx'])
            bx1 = max(bn['cx'], boss['cx'])
            if bx0 <= wx <= bx1 and abs(wx - boss['cx']) <= cfg.UNDERGROUND_CORRIDOR_WIDTH / 2:
                top_y = min(bn['cy'], boss['cy'])
                bot_y = max(bn['cy'], boss['cy'])
                if top_y <= wy <= bot_y:
                    return None

        return _obsidian_platform()

    def _underground_bounds(self) -> tuple[int, int, int, int]:
        """Cached bounding box (x_min, x_max, y_min, y_max) of all underground features."""
        if hasattr(self, '_cached_underground_bounds'):
            return self._cached_underground_bounds
        chambers = self._underground_chambers()
        x_min = min(ch['cx'] - ch['w'] // 2 for ch in chambers)
        x_max = max(ch['cx'] + ch['w'] // 2 for ch in chambers)
        y_min = min(ch['cy'] - ch['h'] // 2 for ch in chambers)
        y_max = max(ch['cy'] + ch['h'] // 2 for ch in chambers)
        # Include corridors (extend to cover full corridor width)
        cw = cfg.UNDERGROUND_CORRIDOR_WIDTH
        ch = cfg.UNDERGROUND_CORRIDOR_HEIGHT
        margin = max(cw, ch) + 20
        self._cached_underground_bounds = (x_min - margin, x_max + margin, y_min - margin, y_max + margin)
        return self._cached_underground_bounds

    def _underground_surface_y(self, wx: int) -> float:
        for ch in self._underground_chambers():
            if ch.get('is_boss'):
                continue
            half_w = ch['w'] // 2
            if ch['cx'] - half_w <= wx < ch['cx'] + half_w:
                return float(ch['cy'] - ch['h'] // 2)
        return float(cfg.UNDERGROUND_Y_START + cfg.UNDERGROUND_STONE_CAP_DEPTH)
