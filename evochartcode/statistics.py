"""Statistical analysis helpers."""

from __future__ import annotations

import random
from typing import Iterable


def bootstrap_mean(values: Iterable[float], seed: int = 0, samples: int = 1000, confidence: float = 0.95) -> dict[str, float]:
    data = list(values)
    if not data:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "samples": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(data) for _ in data]
        means.append(sum(draw) / len(draw))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = int(alpha * (samples - 1))
    high_idx = int((1.0 - alpha) * (samples - 1))
    return {
        "mean": sum(data) / len(data),
        "low": means[low_idx],
        "high": means[high_idx],
        "samples": len(data),
    }


def pareto_frontier(rows: list[dict], x_key: str = "mean_latency", y_key: str = "exact_match") -> list[dict]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if other.get(y_key, 0.0) >= row.get(y_key, 0.0) and other.get(x_key, float("inf")) <= row.get(x_key, float("inf")):
                if other.get(y_key, 0.0) > row.get(y_key, 0.0) or other.get(x_key, float("inf")) < row.get(x_key, float("inf")):
                    dominated = True
                    break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (-item.get(y_key, 0.0), item.get(x_key, float("inf"))))
