"""Aggregate run metrics, bootstrap intervals, and Pareto frontier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.statistics import bootstrap_mean, pareto_frontier


def main():
    parser = argparse.ArgumentParser(description="Analyze EvoChartCode run metrics.")
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/run_analysis.json"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "exact_match" in data and "mean_latency" in data:
            data["path"] = str(path)
            rows.append(data)
    analysis = {
        "num_runs": len(rows),
        "exact_match_ci": bootstrap_mean([row["exact_match"] for row in rows]),
        "relaxed_numeric_ci": bootstrap_mean([row.get("relaxed_numeric", 0.0) for row in rows]),
        "na_f1_ci": bootstrap_mean([row.get("na_f1", 0.0) for row in rows]),
        "pareto_frontier": pareto_frontier(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in analysis.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
