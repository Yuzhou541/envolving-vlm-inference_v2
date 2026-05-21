"""Question-conditioned Chart Code selection."""

from __future__ import annotations

from typing import Any

from evochartcode.routing import QuestionType
from evochartcode.schema import ChartCode


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


def minimal_series_metadata(chart_code: ChartCode) -> list[dict[str, Any]]:
    rows = []
    for series in chart_code.series:
        rows.append(
            {
                "id": series.id,
                "name": series.name,
                "visual_type": series.visual_type,
                "color": series.color,
                "start": series.start,
                "end": series.end,
                "confidence": series.confidence,
            }
        )
    return rows


def select_code_for_question(question_type: QuestionType, chart_code: ChartCode) -> dict[str, Any]:
    base = {
        "chart_id": chart_code.chart_id,
        "chart_type": chart_code.chart_type,
        "uncertainty": _dump(chart_code.uncertainty),
    }

    if question_type == "title":
        return {**base, "title": _dump(chart_code.title)}
    if question_type in {"axis_label", "axis_range", "tick_value", "tick_count"}:
        return {**base, "axes": _dump(chart_code.axes)}
    if question_type in {"legend_count", "legend_name"}:
        return {**base, "legend": _dump(chart_code.legend)}
    if question_type == "colorbar_min_max":
        return {**base, "colorbar": _dump(chart_code.colorbar)}
    if question_type in {"subplot_count", "chart_type"}:
        return {**base, "layout": chart_code.layout}
    if question_type in {"line_count", "bar_value", "point_value"}:
        return {
            **base,
            "series": [_dump(item) for item in chart_code.series],
            "marks": chart_code.marks,
        }
    if question_type in {"trend", "extremum", "turning_point", "intersection", "comparison", "ranking"}:
        return {
            **base,
            "series": minimal_series_metadata(chart_code),
            "derived_relations": _dump(chart_code.derived_relations),
        }
    if question_type in {"difference", "ratio"}:
        return {
            **base,
            "series": [_dump(item) for item in chart_code.series],
            "derived_relations": _dump(chart_code.derived_relations),
        }
    return chart_code.compact_dict()
