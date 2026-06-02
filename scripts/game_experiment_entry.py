from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("game_debug.log", mode="a"),
        logging.StreamHandler(sys.stderr),
    ],
)
logging.getLogger("pyglet").setLevel(logging.WARNING)
logging.getLogger("moderngl").setLevel(logging.WARNING)


DEFAULT_ORDER = ("plains", "hillside", "alpine", "underground")
ORDER_ENV = "ORACLE_TRANSLATOR_EXPERIMENT_BIOME_ORDER"


def install_biome_order_patch(order_text: str) -> None:
    order = tuple(part.strip() for part in order_text.split(",") if part.strip())
    if not order or order == DEFAULT_ORDER:
        return
    if set(order) != set(DEFAULT_ORDER) or len(order) != len(DEFAULT_ORDER):
        raise ValueError(f"invalid biome order: {order_text}")

    from src.engine.world import WorldChunkStore
    from src.game import config as cfg
    from src.game.terrain import TerrainGenerator

    biome_width = cfg.BIOME_WIDTH
    default_index_by_biome = {name: index for index, name in enumerate(DEFAULT_ORDER)}
    physical_index_by_biome = {name: index for index, name in enumerate(order)}

    original_generate_chunk = TerrainGenerator.generate_chunk
    original_ground_height_at = TerrainGenerator.ground_height_at
    original_cell_at = TerrainGenerator.cell_at

    def _physical_biome_for_x(world_x: int) -> str:
        segment_index = max(0, min(len(order) - 1, int(world_x) // biome_width))
        return order[segment_index]

    def _physical_to_logical_x(world_x: int) -> int:
        segment_index = max(0, min(len(order) - 1, int(world_x) // biome_width))
        local_x = int(world_x) - segment_index * biome_width
        logical_segment_index = default_index_by_biome[order[segment_index]]
        return logical_segment_index * biome_width + local_x

    def _logical_to_physical_x(world_x: int) -> int:
        segment_index = max(0, min(len(DEFAULT_ORDER) - 1, int(world_x) // biome_width))
        local_x = int(world_x) - segment_index * biome_width
        physical_segment_index = physical_index_by_biome[DEFAULT_ORDER[segment_index]]
        return physical_segment_index * biome_width + local_x

    def patched_generate_chunk(self, store: WorldChunkStore, chunk_x: int, chunk_y: int, chunk_size: int, seed: int) -> None:
        chunks_per_segment = max(1, biome_width // chunk_size)
        physical_segment_index = max(0, min(len(order) - 1, int(chunk_x) // chunks_per_segment))
        local_chunk_index = int(chunk_x) - physical_segment_index * chunks_per_segment
        logical_segment_index = default_index_by_biome[order[physical_segment_index]]
        logical_chunk_x = logical_segment_index * chunks_per_segment + local_chunk_index
        temp_store = WorldChunkStore(
            store.width,
            store.height,
            chunk_size=chunk_size,
            seed=seed,
        )
        original_generate_chunk(self, temp_store, logical_chunk_x, chunk_y, chunk_size, seed)
        source_chunk = temp_store._chunk(logical_chunk_x, chunk_y, create=False)  # noqa: SLF001
        if source_chunk is None:
            return
        target_chunk = store._chunk(chunk_x, chunk_y, create=True)  # noqa: SLF001
        assert target_chunk is not None
        target_chunk.cells = {local_index: cell.copy() for local_index, cell in source_chunk.cells.items()}
        target_chunk.anchored_support_indices = set(source_chunk.anchored_support_indices)

    def patched_ground_height_at(self, world_x: int) -> float:
        return original_ground_height_at(self, _physical_to_logical_x(int(world_x)))

    def patched_cell_at(self, wx: int, wy: int):
        return original_cell_at(self, _physical_to_logical_x(int(wx)), int(wy))

    def patched_find_spawn_point_near(
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
            candidate_x_clamped = max(half_width + 1, min(max_x, int(candidate_x)))
            base_floor_y = int(math.floor(self.ground_height_at(candidate_x_clamped)))
            floor_candidates = [base_floor_y]
            if _physical_biome_for_x(candidate_x_clamped) == "underground":
                search_limit = min(
                    cfg.WORLD_HEIGHT - 2,
                    base_floor_y + cfg.UNDERGROUND_STONE_CAP_DEPTH + cfg.UNDERGROUND_CHAMBER_MAX_H * 2,
                )
                floor_candidates = list(range(max(1, base_floor_y), search_limit + 1, 2))
            for floor_y in floor_candidates:
                spawn_y = floor_y - 1 - entity_height
                if spawn_y < 1:
                    continue
                solid_floor = False
                blocked = False
                for sample_x in range(candidate_x_clamped - half_width, candidate_x_clamped + half_width + 1):
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
                    return (float(candidate_x_clamped), float(spawn_y))
            return None

        offsets = [0]
        stride = max(1, step)
        for delta in range(stride, max(1, search_radius) + 1, stride):
            offsets.extend((delta, -delta))
        for delta in offsets:
            spawn = _try_column(int(world_x + delta))
            if spawn is not None:
                return spawn

        biome = _physical_biome_for_x(int(world_x))
        if biome == "alpine":
            islands = sorted(self._alpine_islands(), key=lambda island: abs(_logical_to_physical_x(island["cx"]) - world_x))
            for island in islands:
                center_x = _logical_to_physical_x(island["cx"])
                span = max(4, island["w"] // 12)
                for candidate_x in range(center_x, center_x + island["w"] // 3 + 1, span):
                    spawn = _try_column(candidate_x)
                    if spawn is not None:
                        return spawn
                    spawn = _try_column(center_x - (candidate_x - center_x))
                    if spawn is not None:
                        return spawn
        elif biome == "underground":
            chambers = sorted(
                self._underground_chambers(),
                key=lambda chamber: abs(_logical_to_physical_x(chamber["cx"]) - world_x),
            )
            for chamber in chambers:
                center_x = _logical_to_physical_x(chamber["cx"])
                span = max(8, chamber["w"] // 3)
                chamber_stride = max(4, chamber["w"] // 12)
                for delta in range(0, span + 1, chamber_stride):
                    for candidate_x in (center_x + delta, center_x - delta):
                        spawn = _try_column(candidate_x)
                        if spawn is not None:
                            return spawn
        return None

    TerrainGenerator.generate_chunk = patched_generate_chunk
    TerrainGenerator.ground_height_at = patched_ground_height_at
    TerrainGenerator.cell_at = patched_cell_at
    TerrainGenerator.find_spawn_point_near = patched_find_spawn_point_near


def main() -> int:
    order_text = os.environ.get(ORDER_ENV, "").strip()
    if order_text:
        install_biome_order_patch(order_text)
    from src.game.app import run_game

    run_game()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
