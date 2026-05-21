"""Run multi-seed EvoChartCode evaluations and aggregate bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.statistics import bootstrap_mean
from scripts.run_eval import evaluate_evochartcode


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation and bootstrap confidence intervals.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--method", default="code_only")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/multiseed_code_only.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        policy = {"seed": seed, "verifier": {"min_confidence": 0.4}}
        metrics, _ = evaluate_evochartcode(config, method=args.method, split_name=args.split, limit=args.limit, policy=policy)
        metrics["seed"] = seed
        rows.append(metrics)
    summary = {
        "method": args.method,
        "split": args.split,
        "limit": args.limit,
        "seeds": [row["seed"] for row in rows],
        "exact_match_ci": bootstrap_mean([row["exact_match"] for row in rows]),
        "relaxed_numeric_ci": bootstrap_mean([row["relaxed_numeric"] for row in rows]),
        "na_f1_ci": bootstrap_mean([row["na_f1"] for row in rows]),
        "mean_latency_ci": bootstrap_mean([row["mean_latency"] for row in rows]),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
