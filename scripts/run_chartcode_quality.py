"""Run the ChartCode-300 quality benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.datasets import CharXivDataset, filter_queries_by_split
from evochartcode.extractor import ChartCodeExtractor
from evochartcode.quality import score_chart_code
from evochartcode.schema import coerce_chart_code


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


def mean_bool(rows, key):
    values = [getattr(row, key) for row in rows if getattr(row, key) is not None]
    return sum(bool(value) for value in values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chart Code quality on up to 300 CharXiv charts.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/quality/chartcode_300.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    extractor_cfg = config.get("extractor", {})
    cache_dir = Path(config.get("chart_code_cache", ""))
    dataset = CharXivDataset(dataset_cfg.get("root", "charxiv"))
    source_split = dataset_cfg.get("source_split", "val")
    descriptive = dataset._read_json(f"descriptive_{source_split}.json")
    queries = dataset.descriptive_queries(source_split, limit=None)
    split_ids = load_split_ids(dataset_cfg.get("split_file"), args.split)
    if split_ids is not None:
        queries = filter_queries_by_split(queries, split_ids)
    figure_ids = []
    seen = set()
    for query in queries:
        if query.figure_id in seen:
            continue
        seen.add(query.figure_id)
        figure_ids.append(query.figure_id)
        if len(figure_ids) >= args.limit:
            break

    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "metadata"),
        model_name=extractor_cfg.get("model_name", "Qwen/Qwen3-VL-2B-Instruct"),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )

    rows = []
    for figure_id in figure_ids:
        image_path = dataset.images_dir / f"{figure_id}.jpg"
        chart_type = dataset.get_chart_type(figure_id, source_split)
        cache_path = cache_dir / f"{figure_id}.json" if cache_dir else Path()
        if cache_path.exists():
            chart_code = coerce_chart_code(json.loads(cache_path.read_text(encoding="utf-8")), chart_id=figure_id)
        else:
            chart_code = extractor.extract(
                image_path,
                chart_id=figure_id,
                chart_type=chart_type,
                metadata=dataset.image_metadata(source_split).get(figure_id, {}),
            )
        rows.append(score_chart_code(chart_code, expected_chart_type=chart_type, descriptive_entry=descriptive.get(figure_id, {})))

    summary = {
        "benchmark": "ChartCode-300",
        "split": args.split,
        "num_charts": len(rows),
        "mean_quality_score": statistics.mean(row.score() for row in rows) if rows else 0.0,
        "chart_type_accuracy": mean_bool(rows, "chart_type_correct"),
        "legend_exists_accuracy": mean_bool(rows, "legend_exists_correct"),
        "colorbar_exists_accuracy": mean_bool(rows, "colorbar_exists_correct"),
        "image_size_valid_rate": mean_bool(rows, "image_size_valid"),
        "plot_area_present_rate": mean_bool(rows, "plot_area_present"),
        "axis_evidence_present_rate": mean_bool(rows, "axis_evidence_present"),
        "series_or_mark_present_rate": mean_bool(rows, "series_or_mark_present"),
        "records": [asdict(row) | {"quality_score": row.score()} for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
