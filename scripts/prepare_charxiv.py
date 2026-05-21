"""Prepare chart-level CharXiv splits for EvoChartCode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.datasets import CharXivDataset, make_chart_level_split


def main():
    parser = argparse.ArgumentParser(description="Create reproducible chart-level CharXiv splits.")
    parser.add_argument("--charxiv_root", type=Path, default=Path("charxiv"))
    parser.add_argument("--source-split", default="val", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/splits/charxiv_chart_level_split.json"))
    args = parser.parse_args()

    dataset = CharXivDataset(args.charxiv_root)
    figure_ids = dataset.chart_types(args.source_split).keys()
    split = make_chart_level_split(figure_ids, seed=args.seed)
    payload = {
        "dataset": "CharXiv",
        "source_split": args.source_split,
        "seed": args.seed,
        "splits": split,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(ids) for name, ids in split.items()}, indent=2))


if __name__ == "__main__":
    main()
