"""Dataset loaders for EvoChartCode experiments."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ChartQuery:
    query_id: str
    figure_id: str
    image_path: str
    question: str
    answer: str
    task: str
    metadata: dict[str, Any]


class CharXivDataset:
    def __init__(self, root: str | Path = "charxiv"):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.images_dir = self.root / "images"
        src = str((self.root / "src").resolve())
        if src not in sys.path:
            sys.path.insert(0, src)

    def _read_json(self, name: str) -> dict[str, Any]:
        return json.loads((self.data_dir / name).read_text(encoding="utf-8"))

    def chart_types(self, split: str = "val") -> dict[str, Any]:
        return self._read_json(f"chart_types_{split}.json")

    def image_metadata(self, split: str = "val") -> dict[str, Any]:
        return self._read_json(f"image_metadata_{split}.json")

    def descriptive_queries(self, split: str = "val", limit: int | None = None) -> list[ChartQuery]:
        from descriptive_utils import build_descriptive_quries

        data = self._read_json(f"descriptive_{split}.json")
        metadata = self.image_metadata(split)
        queries = build_descriptive_quries(data, str(self.images_dir))
        rows: list[ChartQuery] = []
        for key, query in queries.items():
            figure_id, subq_idx = key.split("_")
            source = data[str(figure_id)]
            rows.append(
                ChartQuery(
                    query_id=key,
                    figure_id=str(figure_id),
                    image_path=query["figure_path"],
                    question=query["question"],
                    answer=source["answers"][int(subq_idx)],
                    task="descriptive",
                    metadata=metadata.get(str(figure_id), {}),
                )
            )
        return rows[:limit] if limit is not None else rows

    def reasoning_queries(self, split: str = "val", limit: int | None = None) -> list[ChartQuery]:
        from reasoning_utils import build_reasoning_queries

        data = self._read_json(f"reasoning_{split}.json")
        metadata = self.image_metadata(split)
        queries = build_reasoning_queries(data, str(self.images_dir))
        rows: list[ChartQuery] = []
        for figure_id, query in queries.items():
            source = data[str(figure_id)]
            rows.append(
                ChartQuery(
                    query_id=str(figure_id),
                    figure_id=str(figure_id),
                    image_path=query["figure_path"],
                    question=query["question"],
                    answer=source["answer"],
                    task="reasoning",
                    metadata=metadata.get(str(figure_id), {}),
                )
            )
        return rows[:limit] if limit is not None else rows

    def get_chart_type(self, figure_id: str, split: str = "val") -> str:
        item = self.chart_types(split).get(str(figure_id), {})
        types = item.get("chart_types") or []
        if not types:
            return "unknown"
        return str(types[0])

    def iter_queries(self, split: str = "val", task: str = "descriptive", limit: int | None = None) -> list[ChartQuery]:
        if task == "descriptive":
            return self.descriptive_queries(split=split, limit=limit)
        if task == "reasoning":
            return self.reasoning_queries(split=split, limit=limit)
        if task == "both":
            rows = self.descriptive_queries(split=split, limit=None) + self.reasoning_queries(split=split, limit=None)
            return rows[:limit] if limit is not None else rows
        raise ValueError(f"Unknown CharXiv task: {task}")


def make_chart_level_split(
    figure_ids: Iterable[str],
    seed: int = 0,
    evolution_fraction: float = 0.4,
    validation_fraction: float = 0.3,
) -> dict[str, list[str]]:
    ids = [str(item) for item in figure_ids]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_evolution = int(round(n * evolution_fraction))
    n_validation = int(round(n * validation_fraction))
    return {
        "evolution_dev": sorted(ids[:n_evolution], key=int),
        "validation": sorted(ids[n_evolution : n_evolution + n_validation], key=int),
        "heldout": sorted(ids[n_evolution + n_validation :], key=int),
    }


def filter_queries_by_split(queries: Iterable[ChartQuery], figure_ids: Iterable[str]) -> list[ChartQuery]:
    allowed = {str(item) for item in figure_ids}
    return [query for query in queries if query.figure_id in allowed]
