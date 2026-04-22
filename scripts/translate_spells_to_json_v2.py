from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation_v2 import translate_spells_to_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "source" / "generated_spells_v2.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "source" / "spell_to_json_v2.jsonl"))
    parser.add_argument("--log", default=str(ROOT / "data" / "logs" / "spell_to_json_v2_log.jsonl"))
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    rows = translate_spells_to_json(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        model_name=args.model,
        max_retries=args.max_retries,
    )
    print(f"wrote {len(rows)} spell->json rows to {args.output}")


if __name__ == "__main__":
    main()
