"""Run resumable full Qwen-backed transfer evaluation over sharded manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.extractor import ChartCodeExtractor
from evochartcode.metrics import PredictionRecord, aggregate_records
from evochartcode.reasoner import ChartCodeCache, EvoChartCodePipeline, QwenImageCodeReasoner
from evochartcode.statistics import bootstrap_mean


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_prediction_records(paths: list[Path]) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for path in paths:
        if not path.exists():
            continue
        for item in read_json(path):
            records.append(PredictionRecord(**{k: item.get(k) for k in PredictionRecord.__dataclass_fields__}))
    return records


def ci_from_records(records: list[PredictionRecord]) -> dict[str, Any]:
    exact = [1.0 if record.exact else 0.0 for record in records]
    relaxed = [1.0 if record.relaxed_numeric else 0.0 for record in records]
    invalid = [0.0 if record.prediction.strip() else 1.0 for record in records]
    latency = [record.latency for record in records]
    return {
        "exact_match_ci": bootstrap_mean(exact, samples=10000),
        "relaxed_numeric_ci": bootstrap_mean(relaxed, samples=10000),
        "invalid_rate_ci": bootstrap_mean(invalid, samples=10000),
        "mean_latency_ci": bootstrap_mean(latency, samples=10000),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate full transfer shards with Qwen-backed EvoChartCode.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b_vlm.yaml"))
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/transfer_qwen_full"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/chart_codes/transfer_qwen"))
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    extractor_cfg = config.get("extractor", {})
    model_cfg = config.get("model", {})
    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "qwen_vl_json"),
        model_name=extractor_cfg.get("model_name", model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct")),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )
    reasoner = QwenImageCodeReasoner(
        model_name=model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    cache = ChartCodeCache(args.cache_root / args.dataset)
    pipeline = EvoChartCodePipeline(extractor=extractor, cache=cache, reasoner=reasoner, mode="image_code")

    shard_paths = sorted(args.manifest_dir.glob("shard_*.json"))
    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]

    out_dir = args.output_root / args.dataset
    prediction_paths: list[Path] = []
    completed_shards = 0
    attempted_examples = 0
    for shard_path in shard_paths:
        shard = read_json(shard_path)
        shard_index = int(shard.get("shard_index", len(prediction_paths)))
        pred_path = out_dir / "predictions" / f"shard_{shard_index:05d}.json"
        prediction_paths.append(pred_path)
        if args.resume and pred_path.exists():
            attempted_examples += len(read_json(pred_path))
            completed_shards += 1
            if args.max_examples is not None and attempted_examples >= args.max_examples:
                break
            continue

        records: list[PredictionRecord] = []
        predictions: list[dict[str, Any]] = []
        for example in shard.get("examples", []):
            if args.max_examples is not None and attempted_examples >= args.max_examples:
                break
            attempted_examples += 1
            chart_id = f"{args.dataset}_{example['image_id']}"
            start = time.perf_counter()
            prediction, verification = pipeline.answer(
                example["image_path"],
                example["question"],
                chart_id=chart_id,
                chart_type=None,
                metadata={
                    "dataset": args.dataset,
                    "source_hf_id": example.get("source_hf_id"),
                    "split": example.get("split"),
                    "row_index": example.get("row_index"),
                    "conversation_index": example.get("conversation_index"),
                    "checksum": example.get("checksum"),
                },
            )
            latency = time.perf_counter() - start
            record = PredictionRecord(
                query_id=str(example.get("query_id") or f"{chart_id}_{example.get('conversation_index', 0)}"),
                question=example["question"],
                answer=str(example["answer"]),
                prediction=prediction,
                latency=latency,
                task=args.dataset,
                figure_id=example["image_id"],
                verifier=verification.model_dump(),
            )
            records.append(record)
            predictions.append(asdict(record))
        write_json(pred_path, predictions)
        completed_shards += 1
        if args.max_examples is not None and attempted_examples >= args.max_examples:
            break

    completed_prediction_paths = sorted((out_dir / "predictions").glob("shard_*.json"))
    all_records = load_prediction_records(completed_prediction_paths)
    manifest_examples = sum(len(read_json(path).get("examples", [])) for path in shard_paths)
    status = "complete" if args.max_examples is None and len(all_records) >= manifest_examples and completed_shards == len(shard_paths) else "partial"
    metrics = aggregate_records(all_records)
    metrics.update(
        {
            "dataset": args.dataset,
            "method": "full_evochartcode",
            "status": status,
            "completed_shards": completed_shards,
            "total_shards": len(shard_paths),
            "manifest_examples": manifest_examples,
            "chart_code_cache": str(args.cache_root / args.dataset),
            "prediction_dir": str(out_dir / "predictions"),
        }
    )
    metrics.update(ci_from_records(all_records))
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not str(k).endswith("_ci")}, indent=2))


if __name__ == "__main__":
    main()
