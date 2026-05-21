"""Run the full Qwen-backed EvoChartCode ablation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from scripts.run_eval import evaluate_evochartcode


ABLATIONS: list[dict[str, Any]] = [
    {"name": "full_evochartcode", "method": "full_evochartcode", "policy": {"verifier": {"min_confidence": 0.4}}},
    {"name": "fixed_chart_code_code_only", "method": "code_only", "policy": {"verifier": {"min_confidence": 0.4}}},
    {"name": "no_verifier", "method": "full_evochartcode", "policy": {"verifier": {"min_confidence": 0.0}}},
    {"name": "no_uncertainty", "method": "full_evochartcode", "policy": {"selector": {"extra_blocks": []}, "verifier": {"min_confidence": 0.4}}},
    {"name": "no_derived_relations", "method": "full_evochartcode", "policy": {"selector": {"extra_blocks": []}, "verifier": {"min_confidence": 0.4}}},
    {"name": "raw_vlm_image_only", "method": "raw_vlm", "policy": {}},
    {"name": "code_only_qwen_chart_code", "method": "code_only", "policy": {"verifier": {"min_confidence": 0.4}}},
    {"name": "prompt_only", "method": "full_evochartcode", "policy": {"selector": {"max_serialized_chars": 1024}, "verifier": {"min_confidence": 0.0}}},
]


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen-backed ablations.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b_vlm.yaml"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs_qwen_ablation"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for ablation in ABLATIONS:
        metrics_path = args.output_dir / f"{ablation['name']}.json"
        pred_path = args.output_dir / f"{ablation['name']}_predictions.json"
        if args.resume and metrics_path.exists() and pred_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics, predictions = evaluate_evochartcode(
                config,
                method=ablation["method"],
                split_name=args.split,
                limit=args.limit,
                policy=ablation["policy"],
            )
            metrics["method"] = ablation["name"]
            metrics["base_method"] = ablation["method"]
            write_json(metrics_path, metrics)
            write_json(pred_path, predictions)
        rows.append(metrics)
    summary = {"split": args.split, "limit": args.limit, "num_ablations": len(rows), "runs": rows}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
