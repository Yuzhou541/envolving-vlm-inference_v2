"""Run EvoChartCode evaluations."""

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
from evochartcode.datasets import CharXivDataset, filter_queries_by_split
from evochartcode.extractor import ChartCodeExtractor
from evochartcode.metrics import PredictionRecord, aggregate_records
from evochartcode.reasoner import ChartCodeCache, CodeOnlyReasoner, EvoChartCodePipeline, QwenImageCodeReasoner


def load_split_ids(path: str | Path | None, split_name: str) -> set[str] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    values = splits.get(split_name)
    return {str(item) for item in values} if values is not None else None


def build_queries(config: dict[str, Any], split_name: str, limit: int | None):
    dataset_cfg = config.get("dataset", {})
    dataset = CharXivDataset(dataset_cfg.get("root", "charxiv"))
    source_split = dataset_cfg.get("source_split", "val")
    task = dataset_cfg.get("task", "descriptive")
    queries = dataset.iter_queries(split=source_split, task=task, limit=None)
    split_ids = load_split_ids(dataset_cfg.get("split_file"), split_name)
    if split_ids is not None:
        queries = filter_queries_by_split(queries, split_ids)
    if limit is None:
        limit = dataset_cfg.get("limit")
    return dataset, source_split, queries[:limit] if limit is not None else queries


def evaluate_evochartcode(
    config: dict[str, Any],
    method: str,
    split_name: str,
    limit: int | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset, source_split, queries = build_queries(config, split_name, limit)
    extractor_cfg = config.get("extractor", {})
    model_cfg = config.get("model", {})
    cache = ChartCodeCache(config.get("chart_code_cache"))
    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "metadata"),
        model_name=extractor_cfg.get("model_name", model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct")),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )

    if method in {"code_only", "fixed_chart_code"}:
        reasoner = CodeOnlyReasoner()
        pipeline_mode = "code_only"
    elif method in {"image_code", "full_evochartcode"}:
        reasoner = QwenImageCodeReasoner(
            model_name=model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
        )
        pipeline_mode = "image_code"
    elif method == "raw_vlm":
        reasoner = QwenImageCodeReasoner(
            model_name=model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
        )
        pipeline_mode = "raw_vlm"
    else:
        raise ValueError(f"Unknown method: {method}")

    pipeline = EvoChartCodePipeline(extractor=extractor, cache=cache, reasoner=reasoner, mode=pipeline_mode if pipeline_mode != "raw_vlm" else "code_only")
    min_confidence = ((policy or {}).get("verifier") or {}).get("min_confidence")

    records: list[PredictionRecord] = []
    predictions: list[dict[str, Any]] = []
    for query in queries:
        chart_type = dataset.get_chart_type(query.figure_id, source_split)
        start = time.perf_counter()
        if method == "raw_vlm":
            prediction = reasoner.answer(query.image_path, query.question, None)  # type: ignore[union-attr]
            verifier_payload = None
        else:
            prediction, verifier = pipeline.answer(
                query.image_path,
                query.question,
                chart_id=query.figure_id,
                chart_type=chart_type,
                metadata=query.metadata,
            )
            verifier_payload = verifier.model_dump()
            if min_confidence is not None and verifier.verdict == "ambiguous" and verifier.confidence < float(min_confidence):
                prediction = "Not Applicable"
        latency = time.perf_counter() - start
        record = PredictionRecord(
            query_id=query.query_id,
            question=query.question,
            answer=query.answer,
            prediction=prediction,
            latency=latency,
            task=query.task,
            figure_id=query.figure_id,
            verifier=verifier_payload,
        )
        records.append(record)
        predictions.append(asdict(record))

    metrics = aggregate_records(records)
    metrics["method"] = method
    metrics["split"] = split_name
    metrics["policy"] = policy or {}
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate EvoChartCode.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--method", default="code_only")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    metrics, predictions = evaluate_evochartcode(config, args.method, args.split, args.limit)
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
