"""Phase 4 test helper: a top-level pickleable terrain factory.

Lives at module scope so the spawn-process worker can import it without
needing the test class. The factory writes a single stone cell so the
``_PackedChunkTerrainWriter._nondefault_writes`` counter is positive and
``is_empty`` ends up False.
"""

from __future__ import annotations

from engine.chunk_generation_worker import register_generator_factory
from engine.types import CellState


def _single_stone_generator(writer, cx: int, cy: int, chunk_size: int, seed: int) -> None:
    writer.set_cell(
        cx * chunk_size,
        cy * chunk_size,
        CellState(family_id="stone", variant_id="stone_platform", integrity=1.0),
    )


def _factory():
    return _single_stone_generator


def register_test_factories() -> None:
    register_generator_factory("phase4.single_stone", _factory)
