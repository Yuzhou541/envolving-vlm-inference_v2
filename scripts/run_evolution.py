"""Run lightweight verification-guided policy evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.evolution import run_evolution, write_evolution_outputs
from scripts.run_eval import evaluate_evochartcode


def main():
    parser = argparse.ArgumentParser(description="Run EvoChartCode policy evolution.")
    parser.add_argument("--config", type=Path, default=Path("configs/evolution_small.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    base_eval_config = load_config(config.get("eval_config", "configs/charxiv_qwen3vl_2b.yaml"))
    method = config.get("method", "code_only")
    split = config.get("split", "evolution_dev")
    limit = config.get("limit")
    runtime_weight = float(config.get("runtime_weight", 0.05))

    def evaluator(policy):
        metrics, _ = evaluate_evochartcode(base_eval_config, method=method, split_name=split, limit=limit, policy=policy)
        score = metrics["exact_match"] - runtime_weight * metrics["mean_latency"] - metrics["invalid_rate"]
        return {
            "exact_match": float(metrics["exact_match"]),
            "relaxed_numeric": float(metrics["relaxed_numeric"]),
            "na_f1": float(metrics["na_f1"]),
            "invalid_rate": float(metrics["invalid_rate"]),
            "mean_latency": float(metrics["mean_latency"]),
            "score": float(score),
        }

    seed_policy = config.get(
        "seed_policy",
        {
            "schema": {"optional_fields": ["uncertainty", "provenance"]},
            "extractor": {"prompt_style": "strict_json", "include_uncertainty": True},
            "selector": {"max_serialized_chars": 4096, "include_provenance": False},
            "reasoner": {"mode": method, "max_answer_tokens": 128},
            "verifier": {"min_confidence": 0.4, "numeric_rel_tol": 0.05, "rules": ["legend", "colorbar", "trend"]},
        },
    )
    archive, map_elites = run_evolution(
        seed_policy=seed_policy,
        evaluator=evaluator,
        generations=int(config.get("generations", 2)),
        candidates_per_generation=int(config.get("candidates_per_generation", 3)),
        population_size=int(config.get("population_size", 6)),
        seed=args.seed if args.seed is not None else int(config.get("seed", 0)),
    )
    out_dir = Path(config.get("output_dir", "outputs/evolution_small"))
    write_evolution_outputs(out_dir, archive, map_elites)
    print(json.dumps({"output_dir": str(out_dir), "best": archive.best().candidate_id}, indent=2))


if __name__ == "__main__":
    main()
