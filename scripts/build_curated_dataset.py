from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation import write_curated_dataset


if __name__ == "__main__":
    output = ROOT / "data" / "raw" / "curated_v1.jsonl"
    rows = write_curated_dataset(output)
    print(f"wrote {len(rows)} curated rows to {output}")
