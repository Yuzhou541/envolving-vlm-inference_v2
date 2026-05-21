"""Prepare a generic chart QA transfer manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.transfer import write_transfer_manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare a cross-dataset chart QA manifest.")
    parser.add_argument("--dataset", required=True, choices=["ChartQA", "PlotQA", "DVQA", "FigureQA"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = write_transfer_manifest(args.dataset, args.root, args.out, limit=args.limit)
    print(json.dumps({k: payload[k] for k in ("dataset", "status", "num_examples")}, indent=2))


if __name__ == "__main__":
    main()
