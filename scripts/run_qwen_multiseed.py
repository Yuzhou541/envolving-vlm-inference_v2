"""Run checkpointed multi-seed Qwen evaluation and bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.metrics import PredictionRecord, aggregate_records
from evochartcode.statistics import bootstrap_mean
from scripts.run_eval import evaluate_evochartcode
from evochartcode.config import load_config


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def records_from_prediction_dicts(items: list[dict[str, Any]]) -> list[PredictionRecord]:
    return [PredictionRecord(**{k: item.get(k) for k in PredictionRecord.__dataclass_fields__}) for item in items]


def bootstrap_records(records: list[PredictionRecord]) -> dict[str, Any]:
    exact = [1.0 if record.exact else 0.0 for record in records]
    relaxed = [1.0 if record.relaxed_numeric else 0.0 for record in records]
    invalid = [0.0 if record.prediction.strip() else 1.0 for record in records]
    latency = [record.latency for record in records]
    na_predictions = [1.0 if record.prediction.strip().lower() == "not applicable" else 0.0 for record in records]
    return {
        "exact_match_ci": bootstrap_mean(exact, samples=10000),
        "relaxed_numeric_ci": bootstrap_mean(relaxed, samples=10000),
        "invalid_rate_ci": bootstrap_mean(invalid, samples=10000),
        "na_prediction_rate_ci": bootstrap_mean(na_predictions, samples=10000),
        "mean_latency_ci": bootstrap_mean(latency, samples=10000),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Qwen multi-seed evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b_vlm.yaml"))
    parser.add_argument("--method", default="full_evochartcode")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/qwen_multiseed"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    all_predictions: list[dict[str, Any]] = []
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        metrics_path = args.output_dir / f"seed_{seed}.json"
        pred_path = args.output_dir / f"seed_{seed}_predictions.json"
        if args.resume and metrics_path.exists() and pred_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            policy = {"seed": seed, "verifier": {"min_confidence": 0.4}}
            metrics, predictions = evaluate_evochartcode(
                config,
                method=args.method,
                split_name=args.split,
                limit=args.limit,
                policy=policy,
            )
            metrics["seed"] = seed
            write_json(metrics_path, metrics)
            write_json(pred_path, predictions)
        rows.append(metrics)
        all_predictions.extend(predictions)

    records = records_from_prediction_dicts(all_predictions)
    aggregate = aggregate_records(records)
    summary = {
        "method": args.method,
        "split": args.split,
        "limit": args.limit,
        "seeds": [row["seed"] for row in rows],
        "num_seed_runs": len(rows),
        "per_seed": rows,
        "pooled_metrics": aggregate,
        "bootstrap": bootstrap_records(records),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
