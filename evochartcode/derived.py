"""Programmatic derived Chart Code attributes."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from typing import Any

from evochartcode.schema import ChartCode, SeriesCode


def _clean_points(points: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        pair = list(point)
        if len(pair) < 2:
            continue
        x, y = float(pair[0]), float(pair[1])
        if isfinite(x) and isfinite(y):
            cleaned.append((x, y))
    return sorted(cleaned)


def derive_trend(points_data: Iterable[Iterable[float]], eps: float = 1e-6) -> dict[str, Any]:
    points = _clean_points(points_data)
    if len(points) < 2:
        return {"global_trend": "unknown", "segments": [], "confidence": 0.0}

    signs: list[str] = []
    segments: list[dict[str, Any]] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dy = y1 - y0
        if abs(dy) <= eps:
            sign = "flat"
        elif dy > 0:
            sign = "increasing"
        else:
            sign = "decreasing"
        signs.append(sign)
        segments.append(
            {
                "x_range": [x0, x1],
                "trend": sign,
                "slope_sign": {"increasing": "positive", "decreasing": "negative"}.get(sign, "zero"),
                "slope_magnitude": abs(dy / (x1 - x0)) if x1 != x0 else None,
            }
        )

    non_flat = [sign for sign in signs if sign != "flat"]
    if not non_flat:
        global_trend = "flat"
    elif all(sign == "increasing" for sign in non_flat):
        global_trend = "increasing"
    elif all(sign == "decreasing" for sign in non_flat):
        global_trend = "decreasing"
    else:
        global_trend = "nonmonotonic"

    agreement = max(signs.count("increasing"), signs.count("decreasing"), signs.count("flat")) / len(signs)
    return {
        "global_trend": global_trend,
        "segments": segments,
        "confidence": round(max(0.35, agreement), 3),
    }


def derive_extrema(series: SeriesCode) -> list[dict[str, Any]]:
    points = _clean_points(series.points_data)
    if not points:
        return []
    min_point = min(points, key=lambda p: p[1])
    max_point = max(points, key=lambda p: p[1])
    extrema = [
        {
            "series_id": series.id,
            "type": "minimum",
            "x": min_point[0],
            "y": min_point[1],
            "confidence": series.confidence,
        },
        {
            "series_id": series.id,
            "type": "maximum",
            "x": max_point[0],
            "y": max_point[1],
            "confidence": series.confidence,
        },
    ]
    return extrema


def derive_turning_points(series: SeriesCode) -> list[dict[str, Any]]:
    points = _clean_points(series.points_data)
    turning_points: list[dict[str, Any]] = []
    for prev_point, point, next_point in zip(points, points[1:], points[2:]):
        left = point[1] - prev_point[1]
        right = next_point[1] - point[1]
        if left > 0 and right < 0:
            kind = "local_maximum"
        elif left < 0 and right > 0:
            kind = "local_minimum"
        else:
            continue
        turning_points.append(
            {
                "series_id": series.id,
                "type": kind,
                "x": point[0],
                "y": point[1],
                "evidence_points": [list(prev_point), list(point), list(next_point)],
                "confidence": max(0.0, series.confidence - 0.1),
            }
        )
    return turning_points


def derive_comparisons(series_list: list[SeriesCode]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for index, left in enumerate(series_list):
        left_points = _clean_points(left.points_data)
        if not left_points:
            continue
        for right in series_list[index + 1 :]:
            right_points = _clean_points(right.points_data)
            if not right_points:
                continue
            length = min(len(left_points), len(right_points))
            if length == 0:
                continue
            diffs = [left_points[i][1] - right_points[i][1] for i in range(length)]
            positives = sum(diff > 0 for diff in diffs)
            negatives = sum(diff < 0 for diff in diffs)
            if positives and negatives:
                relation = "crosses"
            elif positives:
                relation = "a_above_b"
            elif negatives:
                relation = "a_below_b"
            else:
                relation = "similar"
            comparisons.append(
                {
                    "series_a": left.id,
                    "series_b": right.id,
                    "relation": relation,
                    "x_range": [left_points[0][0], left_points[length - 1][0]],
                    "confidence": min(left.confidence, right.confidence),
                }
            )
    return comparisons


def augment_derived_relations(chart_code: ChartCode) -> ChartCode:
    trends = []
    extrema = []
    turning_points = []
    for series in chart_code.series:
        trend = derive_trend(series.points_data)
        trend["series_id"] = series.id
        trends.append(trend)
        extrema.extend(derive_extrema(series))
        turning_points.extend(derive_turning_points(series))

    chart_code.derived_relations.trends = trends
    chart_code.derived_relations.extrema = extrema
    chart_code.derived_relations.turning_points = turning_points
    chart_code.derived_relations.comparisons = derive_comparisons(chart_code.series)
    return chart_code
