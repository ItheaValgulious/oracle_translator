from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation_v2 import generate_json_to_spell_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data" / "source" / "random_json_to_spell_v2.jsonl"))
    parser.add_argument("--log", default=str(ROOT / "data" / "logs" / "random_json_to_spell_v2_log.jsonl"))
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=23)
    args = parser.parse_args()
    rows = generate_json_to_spell_dataset(
        output_path=args.output,
        log_path=args.log,
        target_count=args.target_count,
        model_name=args.model,
        max_retries=args.max_retries,
        rng_seed=args.rng_seed,
    )
    print(f"wrote {len(rows)} random json->spell rows to {args.output}")


if __name__ == "__main__":
    main()
