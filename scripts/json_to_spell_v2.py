from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation_v2 import json_rows_to_spells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(ROOT / "data" / "source" / "json_to_spell_v2.jsonl"))
    parser.add_argument("--log", default=str(ROOT / "data" / "logs" / "json_to_spell_v2_log.jsonl"))
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    rows = json_rows_to_spells(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        model_name=args.model,
        max_retries=args.max_retries,
    )
    print(f"wrote {len(rows)} json->spell rows to {args.output}")


if __name__ == "__main__":
    main()
