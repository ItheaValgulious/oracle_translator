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
        log.info("[terrain] generate_chunk cx=%d cy=%d range=[%d..%d) x [%d..%d) biome_sample=%s",
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
        tree_cols: dict[int, tuple[int, float]] = {}
        pond_cols: dict[int, tuple[int, float]] = {}
        for lx in range(x_start, x_end):
            if biomes[lx] == "plains":
                tc = self._plains_tree_center(lx, heights)
                if tc is not None:
                    tree_cols[lx] = tc
                pc = self._plains_pond_center(lx, heights)
                if pc is not None:
                    pond_cols[lx] = pc

        # Precompute alpine feature x-ranges for this chunk
        alpine_feature_xs: set[int] = set()
        has_alpine = any(b == "alpine" for b in biomes.values())
        if has_alpine:
            islands = self._alpine_islands()
            for isl in islands:
                ix0 = isl['cx'] - isl['w'] // 2
                ix1 = ix0 + isl['w']
                for x in range(max(x_start, ix0), min(x_end, ix1)):
                    alpine_feature_xs.add(x)
            for i, isl in enumerate(islands):
                for j in isl.get('connections', []):
                    if j <= i:
                        continue
                    other = islands[j]
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
            log.info("[terrain] chunk (%d,%d) skipped: pure alpine, no features, %.1fms",
                     chunk_x, chunk_y, (perf_counter() - t0) * 1000)
            return  # entire chunk is empty air

        # Precompute y-ranges for alpine features to skip empty rows
        alpine_y_min = y_end
        alpine_y_max = y_start
        if has_alpine and alpine_feature_xs:
            islands = self._alpine_islands()
            for isl in islands:
                iy0 = isl['cy'] - isl['h'] // 2
                iy1 = iy0 + isl['h']
                if iy0 < y_end and iy1 > y_start:
                    alpine_y_min = min(alpine_y_min, max(y_start, iy0))
                    alpine_y_max = max(alpine_y_max, min(y_end, iy1))
            for i, isl in enumerate(islands):
                for j in isl.get('connections', []):
                    if j <= i:
                        continue
                    other = islands[j]
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
                tree_above = cfg.PLAINS_TREE_HEIGHT + cfg.PLAINS_TREE_CANOPY_H
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
        tree_h = cfg.PLAINS_TREE_HEIGHT
        canopy_h = cfg.PLAINS_TREE_CANOPY_H
        canopy_hw = cfg.PLAINS_TREE_CANOPY_W // 2
        pond_water_d = cfg.PLAINS_POND_WATER_DEPTH
        pond_d = cfg.PLAINS_POND_DEPTH
        cs = chunk_size

        def _direct_write(store, lx, y0, y1, cell):
            """Write cells directly to chunk dict, bypassing store.set_cell."""
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

        _grass = _grass_platform()
        _stone = _stone_platform()
        _wood = _wood_platform()
        _water = _water_cell()

        for lx in plains_xs:
            ground_y_f = heights[lx]
            ground_y_i = int(math.floor(ground_y_f))
            tc = tree_cols.get(lx)
            pc = pond_cols.get(lx)

            # Build base terrain spans
            grass_top = max(y_start, ground_y_i)
            grass_bot = min(y_end, ground_y_i + grass_depth)
            if grass_top < grass_bot:
                _direct_write(store, lx, grass_top, grass_bot, _grass)

            stone_top = max(y_start, ground_y_i + grass_depth)
            stone_bot = min(y_end, floor_y)
            if stone_top < stone_bot:
                _direct_write(store, lx, stone_top, stone_bot, _stone)

            deep_top = max(y_start, floor_y)
            if deep_top < y_end:
                _direct_write(store, lx, deep_top, y_end, _stone)

            # Pond overlay
            if pc is not None:
                pcx, ground_at_pc = pc
                pc_top = int(math.floor(ground_at_pc))
                water_bot = pc_top + pond_water_d
                pond_bot = pc_top + pond_d
                wt = max(y_start, pc_top)
                wb = min(y_end, water_bot + 1)
                if wt < wb:
                    _direct_write(store, lx, wt, wb, _water)
                st = max(y_start, water_bot + 1)
                sb = min(y_end, pond_bot + 1)
                if st < sb:
                    _direct_write(store, lx, st, sb, _stone)

            # Tree overlay
            elif tc is not None:
                tcx, ground_at_tc = tc
                trunk_base = int(math.floor(ground_at_tc))
                trunk_top = trunk_base - tree_h
                canopy_top = trunk_top - canopy_h
                if abs(lx - tcx) <= canopy_hw:
                    ct = max(y_start, canopy_top)
                    cb = min(y_end, trunk_top)
                    if ct < cb:
                        _direct_write(store, lx, ct, cb, _grass)
                if lx == tcx:
                    tt = max(y_start, trunk_top)
                    tb = min(y_end, trunk_base + 1)
                    if tt < tb:
                        _direct_write(store, lx, tt, tb, _wood)

        # Hillside columns — direct chunk writes
        hs_floor_y = cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH
        for lx in hillside_xs:
            ground_y_f = heights[lx]
            ground_y_i = int(math.floor(ground_y_f))
            gt = max(y_start, ground_y_i)
            gb = min(y_end, ground_y_i + 3)
            if gt < gb:
                _direct_write(store, lx, gt, gb, _grass)
            st = max(y_start, ground_y_i + 3)
            sb = min(y_end, hs_floor_y)
            if st < sb:
                _direct_write(store, lx, st, sb, _stone)
            dt = max(y_start, hs_floor_y)
            if dt < y_end:
                _direct_write(store, lx, dt, y_end, _stone)

        # Alpine columns
        for lx in alpine_xs:
            if lx not in alpine_feature_xs:
                continue
            for ly in range(alpine_y_min, alpine_y_max + 1):
                cell = self._alpine_cell(lx, ly)
                if cell is not None:
                    store.set_cell(lx, ly, cell)

        # Underground columns
        ug_cap_y = cfg.UNDERGROUND_Y_START + cfg.UNDERGROUND_STONE_CAP_DEPTH
        _obsidian = _obsidian_platform()
        for lx in ug_xs:
            for ly in range(y_start, y_end):
                if ly >= ug_cap_y:
                    if lx not in ug_feature_xs or ly < ug_y_min or ly > ug_y_max:
                        _direct_write(store, lx, ly, ly + 1, _obsidian)
                        continue
                cell = self._underground_cell(lx, ly)
                if cell is not None:
                    store.set_cell(lx, ly, cell)

        elapsed = (perf_counter() - t0) * 1000
        log.info("[terrain] chunk (%d,%d) done in %.1fms", chunk_x, chunk_y, elapsed)

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
                          tree_info: tuple[int, float] | None = None,
                          pond_info: tuple[int, float] | None = None) -> CellState | None:
        ground_y_int = int(math.floor(ground_y))

        # Pond (precomputed)
        if pond_info is not None:
            pcx, ground_at_center = pond_info
            pond_top = int(math.floor(ground_at_center))
            water_bottom = pond_top + cfg.PLAINS_POND_WATER_DEPTH
            pond_bottom = pond_top + cfg.PLAINS_POND_DEPTH
            if pond_top <= wy <= water_bottom:
                return _water_cell()
            if water_bottom < wy <= pond_bottom:
                return _stone_platform()

        # Tree (precomputed, skip if pond)
        if tree_info is not None and pond_info is None:
            tcx, ground_at_trunk = tree_info
            trunk_base = int(math.floor(ground_at_trunk))
            trunk_top = trunk_base - cfg.PLAINS_TREE_HEIGHT
            canopy_top = trunk_top - cfg.PLAINS_TREE_CANOPY_H
            canopy_half_w = cfg.PLAINS_TREE_CANOPY_W // 2
            if (canopy_top <= wy < trunk_top
                    and abs(wx - tcx) <= canopy_half_w):
                return _grass_platform()
            if trunk_top <= wy <= trunk_base and wx == tcx:
                return _wood_platform()

        # Base terrain
        if wy >= cfg.WORLD_HEIGHT - cfg.PLAINS_FLOOR_DEPTH:
            return _stone_platform()
        if wy >= ground_y_int + cfg.PLAINS_GRASS_SURFACE_DEPTH:
            return _stone_platform()
        if wy >= ground_y_int:
            return _grass_platform()
        return None

    def _plains_tree_center(self, wx: int, heights: dict[int, float] | None = None) -> tuple[int, float] | None:
        """If wx is the center column of a tree, return (center_x, ground_y)."""
        spacing = max(1, cfg.PLAINS_TREE_CANOPY_W + 20)
        center = round(wx / spacing) * spacing
        if center == wx and 0 < wx < cfg.BIOME_WIDTH:
            if _hash01(self.seed, center, 0, 500) < 0.5:
                if heights is not None and center in heights:
                    return (center, heights[center])
                return (center, self._plains_height(center))
        return None

    def _plains_pond_center(self, wx: int, heights: dict[int, float] | None = None) -> tuple[int, float] | None:
        """If wx is within a pond's width, return (center_x, ground_y)."""
        spacing = max(1, cfg.PLAINS_POND_WIDTH * 3)
        center = round(wx / spacing) * spacing
        if abs(wx - center) <= cfg.PLAINS_POND_WIDTH // 2 and 0 < wx < cfg.BIOME_WIDTH:
            if _hash01(self.seed, center, 1, 510) < 0.3:
                if heights is not None and center in heights:
                    return (center, heights[center])
                return (center, self._plains_height(center))
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
