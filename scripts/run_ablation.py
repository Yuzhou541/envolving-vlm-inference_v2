"""Run a configured EvoChartCode ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import deep_update, load_config
from scripts.run_eval import evaluate_evochartcode


def main():
    parser = argparse.ArgumentParser(description="Run an EvoChartCode ablation config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ablation = load_config(args.config)
    base = load_config(ablation.get("base_config", "configs/charxiv_qwen3vl_2b.yaml"))
    config = deep_update(base, ablation.get("override", {}))
    metrics, _ = evaluate_evochartcode(
        config,
        method=ablation.get("method", "code_only"),
        split_name=ablation.get("split", "validation"),
        limit=ablation.get("limit"),
        policy=ablation.get("policy"),
    )
    if ablation.get("name"):
        metrics["method"] = ablation["name"]
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
