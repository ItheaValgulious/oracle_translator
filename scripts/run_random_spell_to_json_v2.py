from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation_v2 import build_random_spell_to_json_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-examples", default=str(ROOT / "data" / "source" / "manual_spell_seeds_v2.jsonl"))
    parser.add_argument("--generated-spells", default=str(ROOT / "data" / "source" / "random_spells_v2.jsonl"))
    parser.add_argument("--generated-log", default=str(ROOT / "data" / "logs" / "random_spells_v2_log.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "source" / "random_spell_to_json_v2.jsonl"))
    parser.add_argument("--translation-log", default=str(ROOT / "data" / "logs" / "random_spell_to_json_v2_log.jsonl"))
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--spell-model", default="minimax-m2.5")
    parser.add_argument("--translation-model", default="minimax-m2.5")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=23)
    args = parser.parse_args()
    rows = build_random_spell_to_json_dataset(
        seed_examples_path=args.seed_examples,
        generated_spell_output_path=args.generated_spells,
        generated_spell_log_path=args.generated_log,
        translated_output_path=args.output,
        translated_log_path=args.translation_log,
        target_count=args.target_count,
        spell_model_name=args.spell_model,
        translation_model_name=args.translation_model,
        max_retries=args.max_retries,
        rng_seed=args.rng_seed,
    )
    print(f"wrote {len(rows)} random spell->json rows to {args.output}")


if __name__ == "__main__":
    main()
