"""Question routing for Chart Code selection and verification."""

from __future__ import annotations

import re
from typing import Literal

QuestionType = Literal[
    "title",
    "axis_label",
    "axis_range",
    "tick_value",
    "tick_count",
    "legend_count",
    "legend_name",
    "colorbar_min_max",
    "chart_type",
    "subplot_count",
    "line_count",
    "bar_value",
    "point_value",
    "trend",
    "extremum",
    "turning_point",
    "intersection",
    "comparison",
    "ranking",
    "difference",
    "ratio",
    "not_applicable",
    "open_ended_reasoning",
]


def route_question(question: str) -> QuestionType:
    q = " ".join(question.lower().split())
    if "not applicable" in q:
        return "not_applicable"
    if "title" in q:
        return "title"
    if "chart type" in q or "type of chart" in q or "kind of chart" in q:
        return "chart_type"
    if "tick" in q and re.search(r"\bhow many\b|\bnumber\b|\bcount\b|\btotal\b", q):
        return "tick_count"
    if "tick" in q:
        return "tick_value"
    if "axis label" in q or "label of the x-axis" in q or "label of the y-axis" in q:
        return "axis_label"
    if "range" in q and ("axis" in q or "x-axis" in q or "y-axis" in q):
        return "axis_range"
    if "continuous legend" in q or "colorbar" in q or "color bar" in q:
        return "colorbar_min_max"
    if "difference" in q or "decline" in q or "increase from" in q:
        return "difference"
    if "ratio" in q or "times" in q:
        return "ratio"
    if "legend" in q and re.search(r"\bhow many\b|\bnumber\b|\bcount\b|\btotal\b", q):
        return "legend_count"
    if "legend" in q:
        return "legend_name"
    if "subplot" in q or "layout" in q:
        return "subplot_count"
    if "how many lines" in q or "number of lines" in q:
        return "line_count"
    if "trend" in q or "increasing" in q or "decreasing" in q:
        return "trend"
    if "maximum" in q or "minimum" in q or "highest" in q or "lowest" in q:
        return "extremum"
    if "turning point" in q or "local maximum" in q or "local minimum" in q:
        return "turning_point"
    if "intersect" in q or "cross" in q:
        return "intersection"
    if "greater" in q or "less" in q or "compare" in q or "above" in q or "below" in q:
        return "comparison"
    if "rank" in q or "order" in q:
        return "ranking"
    if "bar" in q:
        return "bar_value"
    if "point" in q or "value at" in q:
        return "point_value"
    return "open_ended_reasoning"
