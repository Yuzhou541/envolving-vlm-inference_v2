"""Evaluation metrics for chart QA and EvoChartCode runs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from evochartcode.normalizer import extract_number, normalize_answer, numeric_close


@dataclass
class PredictionRecord:
    query_id: str
    question: str
    answer: str
    prediction: str
    latency: float
    task: str
    figure_id: str
    verifier: dict[str, Any] | None = None

    @property
    def exact(self) -> bool:
        return normalize_answer(self.prediction).lower() == normalize_answer(self.answer).lower()

    @property
    def relaxed_numeric(self) -> bool:
        if self.exact:
            return True
        if extract_number(self.answer) is None:
            return False
        return numeric_close(self.prediction, self.answer)


def aggregate_records(records: list[PredictionRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "num_examples": 0,
            "exact_match": 0.0,
            "relaxed_numeric": 0.0,
            "invalid_rate": 0.0,
            "na_precision": 0.0,
            "na_recall": 0.0,
            "na_f1": 0.0,
            "mean_latency": 0.0,
            "p95_latency": 0.0,
        }

    exact = sum(record.exact for record in records) / total
    relaxed = sum(record.relaxed_numeric for record in records) / total
    invalid = sum(not normalize_answer(record.prediction) for record in records) / total

    pred_na = [normalize_answer(record.prediction) == "Not Applicable" for record in records]
    gold_na = [normalize_answer(record.answer) == "Not Applicable" for record in records]
    tp = sum(p and g for p, g in zip(pred_na, gold_na))
    fp = sum(p and not g for p, g in zip(pred_na, gold_na))
    fn = sum((not p) and g for p, g in zip(pred_na, gold_na))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    latencies = [record.latency for record in records]
    p95_index = max(0, min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1))
    p95 = sorted(latencies)[p95_index]
    return {
        "num_examples": total,
        "exact_match": exact,
        "relaxed_numeric": relaxed,
        "invalid_rate": invalid,
        "na_precision": precision,
        "na_recall": recall,
        "na_f1": f1,
        "mean_latency": statistics.mean(latencies),
        "p95_latency": p95,
    }
