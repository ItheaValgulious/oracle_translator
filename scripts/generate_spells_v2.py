from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation_v2 import generate_spells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-examples", default=str(ROOT / "data" / "source" / "manual_spell_seeds_v2.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "source" / "generated_spells_v2.jsonl"))
    parser.add_argument("--log", default=str(ROOT / "data" / "logs" / "generated_spells_v2_log.jsonl"))
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=23)
    args = parser.parse_args()
    rows = generate_spells(
        seed_examples_path=args.seed_examples,
        output_path=args.output,
        log_path=args.log,
        target_count=args.target_count,
        model_name=args.model,
        max_retries=args.max_retries,
        rng_seed=args.rng_seed,
    )
    print(f"wrote {len(rows)} generated spells to {args.output}")


if __name__ == "__main__":
    main()
