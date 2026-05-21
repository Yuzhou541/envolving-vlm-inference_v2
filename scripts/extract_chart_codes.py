"""Extract and cache Chart Code JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.datasets import CharXivDataset, filter_queries_by_split
from evochartcode.extractor import ChartCodeExtractor


def load_split_ids(path: Path | None, split_name: str) -> set[str] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    ids = splits.get(split_name)
    return {str(item) for item in ids} if ids is not None else None


def main():
    parser = argparse.ArgumentParser(description="Extract Chart Code cache for a dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--split", default="evolution_dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    extractor_cfg = config.get("extractor", {})
    cache_dir = args.output or Path(config.get("chart_code_cache", "data/cache/chart_codes/charxiv_evolution_dev"))
    split_file = dataset_cfg.get("split_file")

    dataset = CharXivDataset(dataset_cfg.get("root", "charxiv"))
    source_split = dataset_cfg.get("source_split", "val")
    task = dataset_cfg.get("task", "descriptive")
    queries = dataset.iter_queries(split=source_split, task=task, limit=None)
    split_ids = load_split_ids(Path(split_file) if split_file else None, args.split)
    if split_ids is not None:
        queries = filter_queries_by_split(queries, split_ids)
    if args.limit is not None:
        queries = queries[: args.limit]

    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "metadata"),
        model_name=extractor_cfg.get("model_name", "Qwen/Qwen3-VL-2B-Instruct"),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    for query in queries:
        if query.figure_id in seen:
            continue
        seen.add(query.figure_id)
        chart_type = dataset.get_chart_type(query.figure_id, source_split)
        chart_code = extractor.extract(
            query.image_path,
            chart_id=query.figure_id,
            chart_type=chart_type,
            metadata=query.metadata,
        )
        (cache_dir / f"{query.figure_id}.json").write_text(
            json.dumps(chart_code.compact_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written += 1

    print(json.dumps({"output": str(cache_dir), "chart_codes": written}, indent=2))


if __name__ == "__main__":
    main()
