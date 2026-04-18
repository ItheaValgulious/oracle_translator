from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oracle_translator.data_generation import merge_and_split_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated", default=str(ROOT / "data" / "raw" / "curated_v1.jsonl"))
    parser.add_argument("--api", default=str(ROOT / "data" / "raw" / "api_v1.jsonl"))
    parser.add_argument("--train", default=str(ROOT / "data" / "processed" / "train_v1.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data" / "processed" / "val_v1.jsonl"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "manifest_v1.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--no-prefixes", action="store_true")
    args = parser.parse_args()
    manifest = merge_and_split_datasets(
        args.curated,
        args.api,
        train_path=args.train,
        val_path=args.val,
        manifest_path=args.manifest,
        val_ratio=args.val_ratio,
        include_prefixes=not args.no_prefixes,
        seed=args.seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
