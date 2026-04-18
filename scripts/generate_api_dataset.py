from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation import generate_api_augmentations
from oracle_translator.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "curated_v1.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "api_v1.jsonl"))
    parser.add_argument("--log", default=str(ROOT / "data" / "raw" / "api_generation_log.json"))
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--seeds-per-call", type=int, default=4)
    parser.add_argument("--variants-per-seed", type=int, default=10)
    args = parser.parse_args()
    curated = read_jsonl(args.input)
    rows = generate_api_augmentations(
        curated,
        output_path=args.output,
        request_log_path=args.log,
        model_name=args.model,
        seeds_per_call=args.seeds_per_call,
        variants_per_seed=args.variants_per_seed,
    )
    print(f"wrote {len(rows)} api-generated rows to {args.output}")


if __name__ == "__main__":
    main()
