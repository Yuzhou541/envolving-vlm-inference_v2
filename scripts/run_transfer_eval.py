"""Evaluate EvoChartCode on prepared transfer manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.extractor import ChartCodeExtractor
from evochartcode.metrics import PredictionRecord, aggregate_records
from evochartcode.reasoner import ChartCodeCache, CodeOnlyReasoner, EvoChartCodePipeline, QwenImageCodeReasoner


def main():
    parser = argparse.ArgumentParser(description="Run transfer evaluation from a prepared manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", default="code_only", choices=["code_only", "full_evochartcode"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        metrics = {
            "dataset": manifest.get("dataset"),
            "method": args.method,
            "status": manifest.get("status", "missing"),
            "num_examples": 0,
            "error": manifest.get("error", "manifest is not ready"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    config = load_config(args.config)
    extractor_cfg = config.get("extractor", {})
    model_cfg = config.get("model", {})
    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "metadata"),
        model_name=extractor_cfg.get("model_name", model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct")),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )
    reasoner = CodeOnlyReasoner() if args.method == "code_only" else QwenImageCodeReasoner(
        model_name=model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    cache = ChartCodeCache(Path(config.get("chart_code_cache", "data/cache/chart_codes/transfer")) / manifest["dataset"])
    pipeline = EvoChartCodePipeline(
        extractor=extractor,
        cache=cache,
        reasoner=reasoner,
        mode="code_only" if args.method == "code_only" else "image_code",
    )

    records = []
    predictions = []
    for example in manifest["examples"]:
        start = time.perf_counter()
        prediction, verification = pipeline.answer(
            example["image_path"],
            example["question"],
            chart_id=f"{manifest['dataset']}_{example['figure_id']}",
            chart_type=None,
            metadata=example.get("metadata", {}),
        )
        latency = time.perf_counter() - start
        record = PredictionRecord(
            query_id=example["query_id"],
            question=example["question"],
            answer=example["answer"],
            prediction=prediction,
            latency=latency,
            task=manifest["dataset"],
            figure_id=example["figure_id"],
            verifier=verification.model_dump(),
        )
        records.append(record)
        predictions.append(asdict(record))

    metrics = aggregate_records(records)
    metrics["dataset"] = manifest["dataset"]
    metrics["method"] = args.method
    metrics["status"] = "ready"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
