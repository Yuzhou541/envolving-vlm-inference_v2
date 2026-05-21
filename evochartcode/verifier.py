"""Verifier and answer normalizer for Chart Code grounded QA."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from evochartcode.normalizer import extract_number, normalize_answer, normalize_int, numeric_close
from evochartcode.routing import QuestionType
from evochartcode.schema import ChartCode


class VerificationResult(BaseModel):
    verdict: Literal["supported", "unsupported", "ambiguous"]
    reason: str
    suggested_answer: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _supports_missing(answer: str) -> bool:
    return normalize_answer(answer) == "Not Applicable"


def _trend_terms(trend: str) -> set[str]:
    if trend == "increasing":
        return {"increasing", "increase", "rising", "upward", "positive"}
    if trend == "decreasing":
        return {"decreasing", "decrease", "declining", "downward", "negative"}
    if trend == "flat":
        return {"flat", "constant", "stable", "unchanged"}
    if trend == "nonmonotonic":
        return {"nonmonotonic", "varies", "mixed", "fluctuates"}
    return {"unknown"}


def _contains_text(answer: str, expected: str | None) -> bool:
    if not expected:
        return False
    normalized_answer = normalize_answer(answer).lower()
    normalized_expected = normalize_answer(expected).lower()
    return normalized_expected in normalized_answer or normalized_answer in normalized_expected


def _axis_name(question: str) -> str | None:
    q = question.lower()
    if "x-axis" in q or "x axis" in q or "horizontal" in q or "bottom axis" in q:
        return "x"
    if "y-axis" in q or "y axis" in q or "vertical" in q or "left axis" in q:
        return "y"
    return None


def _axis_ticks(chart_code: ChartCode, axis_name: str | None = None):
    axes = chart_code.axes
    if axis_name is not None:
        axis = axes.get(axis_name)
        return axis.ticks if axis is not None else []
    ticks = []
    for axis in axes.values():
        ticks.extend(axis.ticks)
    return ticks


def _count_lines(chart_code: ChartCode) -> int:
    return sum(1 for series in chart_code.series if str(series.visual_type).lower() == "line")


def _series_names(chart_code: ChartCode) -> set[str]:
    return {series.name.lower() for series in chart_code.series if series.name}


def _mark_categories(chart_code: ChartCode) -> set[str]:
    values = set()
    for mark in chart_code.marks:
        for key in ("category", "label", "series", "row", "col"):
            value = mark.get(key)
            if value is not None:
                values.add(str(value).lower())
    return values


def _rank_answer_supported(answer: str, rankings: list[dict]) -> bool:
    lowered = answer.lower()
    for ranking in rankings:
        order = ranking.get("order_high_to_low") or ranking.get("order") or []
        names = [str(item).lower() for item in order]
        if names and all(name in lowered for name in names[: min(3, len(names))]):
            return True
    return False


def verify_answer(
    question: str,
    question_type: QuestionType,
    chart_code: ChartCode,
    answer: str,
) -> VerificationResult:
    normalized = normalize_answer(answer)
    lowered = normalized.lower()

    if not normalized:
        return VerificationResult(
            verdict="unsupported",
            reason="empty answer",
            suggested_answer="Not Applicable",
            confidence=0.9,
        )

    if question_type in {"legend_count", "legend_name"}:
        if not chart_code.legend.exists:
            verdict = "supported" if _supports_missing(normalized) else "unsupported"
            return VerificationResult(verdict=verdict, reason="legend does not exist", suggested_answer="Not Applicable", confidence=0.9)
        if question_type == "legend_count":
            expected = len(chart_code.legend.items)
            actual = normalize_int(normalized)
            if actual == expected:
                return VerificationResult(verdict="supported", reason="legend count matches Chart Code", confidence=0.9)
            return VerificationResult(verdict="unsupported", reason="legend count mismatch", suggested_answer=str(expected), confidence=0.8)
        legend_names = [item.name for item in chart_code.legend.items if item.name]
        if legend_names and all(name.lower() in lowered for name in legend_names):
            return VerificationResult(verdict="supported", reason="legend names are present in answer", confidence=0.8)
        if legend_names:
            return VerificationResult(
                verdict="unsupported",
                reason="legend names mismatch",
                suggested_answer=", ".join(legend_names),
                confidence=0.75,
            )

    if question_type == "colorbar_min_max":
        if not chart_code.colorbar.exists:
            verdict = "supported" if _supports_missing(normalized) else "unsupported"
            return VerificationResult(verdict=verdict, reason="colorbar does not exist", suggested_answer="Not Applicable", confidence=0.9)
        if "max" in question.lower() and chart_code.colorbar.max is not None:
            ok = numeric_close(normalized, chart_code.colorbar.max)
            return VerificationResult(
                verdict="supported" if ok else "unsupported",
                reason="checked colorbar maximum",
                suggested_answer=str(chart_code.colorbar.max),
                confidence=0.8,
            )
        if "min" in question.lower() and chart_code.colorbar.min is not None:
            ok = numeric_close(normalized, chart_code.colorbar.min)
            return VerificationResult(
                verdict="supported" if ok else "unsupported",
                reason="checked colorbar minimum",
                suggested_answer=str(chart_code.colorbar.min),
                confidence=0.8,
            )

    if question_type == "chart_type":
        if chart_code.chart_type == "unknown":
            return VerificationResult(verdict="ambiguous", reason="chart type is unknown", confidence=0.4)
        ok = chart_code.chart_type.replace("_", " ") in lowered
        return VerificationResult(
            verdict="supported" if ok else "unsupported",
            reason="checked chart type",
            suggested_answer=chart_code.chart_type,
            confidence=0.8,
        )

    if question_type == "title":
        expected = chart_code.title.text
        if not expected:
            verdict = "supported" if _supports_missing(normalized) else "unsupported"
            return VerificationResult(verdict=verdict, reason="title is missing in Chart Code", suggested_answer="Not Applicable", confidence=0.8)
        return VerificationResult(
            verdict="supported" if _contains_text(normalized, expected) else "unsupported",
            reason="checked title text",
            suggested_answer=expected,
            confidence=chart_code.title.confidence,
        )

    if question_type == "axis_label":
        axis_name = _axis_name(question)
        axis = chart_code.axes.get(axis_name) if axis_name else None
        if axis is None:
            return VerificationResult(verdict="ambiguous", reason="axis not available in Chart Code", confidence=0.4)
        if not axis.label:
            verdict = "supported" if _supports_missing(normalized) else "unsupported"
            return VerificationResult(verdict=verdict, reason="axis label is missing", suggested_answer="Not Applicable", confidence=0.8)
        return VerificationResult(
            verdict="supported" if _contains_text(normalized, axis.label) else "unsupported",
            reason="checked axis label",
            suggested_answer=axis.label,
            confidence=axis.confidence,
        )

    if question_type == "axis_range":
        axis_name = _axis_name(question)
        axis = chart_code.axes.get(axis_name) if axis_name else None
        if axis is None or axis.range is None:
            return VerificationResult(verdict="ambiguous", reason="axis range unavailable", confidence=0.4)
        numbers = [extract_number(part) for part in normalized.replace("to", ",").split(",")]
        present = [number for number in numbers if number is not None]
        ok = len(present) >= 2 and numeric_close(present[0], axis.range[0]) and numeric_close(present[-1], axis.range[-1])
        return VerificationResult(
            verdict="supported" if ok else "unsupported",
            reason="checked axis range",
            suggested_answer=f"{axis.range[0]} to {axis.range[-1]}",
            confidence=axis.confidence,
        )

    if question_type == "tick_count":
        expected = len(_axis_ticks(chart_code))
        if expected == 0:
            return VerificationResult(verdict="ambiguous", reason="no tick evidence", confidence=0.4)
        actual = normalize_int(normalized)
        return VerificationResult(
            verdict="supported" if actual == expected else "unsupported",
            reason="checked total tick count",
            suggested_answer=str(expected),
            confidence=0.8,
        )

    if question_type == "tick_value":
        ticks = _axis_ticks(chart_code, _axis_name(question))
        if not ticks:
            return VerificationResult(verdict="ambiguous", reason="no tick evidence for requested axis", confidence=0.4)
        tick_values = [tick.value for tick in ticks]
        ok = any(str(value).lower() in lowered for value in tick_values)
        return VerificationResult(
            verdict="supported" if ok else "unsupported",
            reason="checked tick value against Chart Code ticks",
            suggested_answer=", ".join(tick_values),
            confidence=0.75,
        )

    if question_type == "subplot_count":
        count = chart_code.layout.get("subplot_count")
        grid = chart_code.layout.get("subplot_grid") or {}
        if count is None and grid:
            count = int(grid.get("rows", 1)) * int(grid.get("cols", 1))
        if count is None:
            return VerificationResult(verdict="ambiguous", reason="subplot count unavailable", confidence=0.4)
        actual = normalize_int(normalized)
        expected_layout = f"{grid.get('rows')} by {grid.get('cols')}" if grid.get("rows") and grid.get("cols") else str(count)
        ok = actual == int(count) or expected_layout.lower() in lowered
        return VerificationResult(
            verdict="supported" if ok else "unsupported",
            reason="checked subplot layout/count",
            suggested_answer=expected_layout,
            confidence=0.8,
        )

    if question_type == "line_count":
        expected = _count_lines(chart_code)
        if expected == 0:
            verdict = "supported" if _supports_missing(normalized) else "unsupported"
            return VerificationResult(verdict=verdict, reason="no line series in Chart Code", suggested_answer="Not Applicable", confidence=0.8)
        actual = normalize_int(normalized)
        return VerificationResult(
            verdict="supported" if actual == expected else "unsupported",
            reason="checked line-series count",
            suggested_answer=str(expected),
            confidence=0.8,
        )

    if question_type in {"bar_value", "point_value"}:
        value = extract_number(normalized)
        if value is None:
            return VerificationResult(verdict="ambiguous", reason="answer has no numeric value", confidence=0.4)
        mark_values = []
        for mark in chart_code.marks:
            for key in ("height_data", "value", "value_estimate"):
                if mark.get(key) is not None:
                    mark_values.append(mark[key])
        series_values = [point[1] for series in chart_code.series for point in series.points_data if len(point) >= 2]
        candidates = mark_values + series_values
        if not candidates:
            return VerificationResult(verdict="ambiguous", reason="no mark or point numeric evidence", confidence=0.4)
        ok = any(numeric_close(value, candidate) for candidate in candidates)
        return VerificationResult(
            verdict="supported" if ok else "unsupported",
            reason="checked numeric mark/point value",
            confidence=0.7,
        )

    if question_type == "trend":
        trends = chart_code.derived_relations.trends
        if not trends:
            return VerificationResult(verdict="ambiguous", reason="no derived trend evidence", confidence=0.4)
        accepted_terms: set[str] = set()
        for trend in trends:
            accepted_terms |= _trend_terms(str(trend.get("global_trend", "unknown")))
        ok = any(term in lowered for term in accepted_terms)
        return VerificationResult(
            verdict="supported" if ok else "ambiguous",
            reason="checked derived trend terms",
            confidence=0.65,
        )

    if question_type == "extremum":
        extrema = chart_code.derived_relations.extrema
        if not extrema:
            return VerificationResult(verdict="ambiguous", reason="no extrema evidence", confidence=0.4)
        value = extract_number(normalized)
        if value is None:
            ok = any(str(item.get("type", "")).replace("_", " ") in lowered for item in extrema)
        else:
            ok = any(numeric_close(value, item.get("x")) or numeric_close(value, item.get("y")) for item in extrema)
        return VerificationResult(verdict="supported" if ok else "ambiguous", reason="checked extrema evidence", confidence=0.65)

    if question_type == "turning_point":
        points = chart_code.derived_relations.turning_points
        if not points:
            return VerificationResult(verdict="ambiguous", reason="no turning point evidence", confidence=0.4)
        value = extract_number(normalized)
        ok = value is not None and any(numeric_close(value, item.get("x")) or numeric_close(value, item.get("y")) for item in points)
        return VerificationResult(verdict="supported" if ok else "ambiguous", reason="checked turning point evidence", confidence=0.65)

    if question_type == "intersection":
        intersections = chart_code.derived_relations.intersections
        comparisons = chart_code.derived_relations.comparisons
        if "yes" in lowered or "no" in lowered:
            has_crossing = bool(intersections) or any(item.get("relation") == "crosses" for item in comparisons)
            ok = ("yes" in lowered and has_crossing) or ("no" in lowered and not has_crossing)
            return VerificationResult(verdict="supported" if ok else "unsupported", reason="checked crossing relation", confidence=0.7)
        if not intersections:
            return VerificationResult(verdict="ambiguous", reason="no intersection evidence", confidence=0.4)
        value = extract_number(normalized)
        ok = value is not None and any(numeric_close(value, item.get("x")) or numeric_close(value, item.get("y")) for item in intersections)
        return VerificationResult(verdict="supported" if ok else "ambiguous", reason="checked intersection coordinates", confidence=0.65)

    if question_type == "comparison":
        names = _series_names(chart_code) | _mark_categories(chart_code)
        if names and any(name in lowered for name in names):
            return VerificationResult(verdict="supported", reason="answer names a known series or category", confidence=0.65)
        comparisons = chart_code.derived_relations.comparisons
        if comparisons and any(str(item.get("relation", "")).replace("_", " ") in lowered for item in comparisons):
            return VerificationResult(verdict="supported", reason="answer states a derived comparison relation", confidence=0.65)
        return VerificationResult(verdict="ambiguous", reason="comparison cannot be fully verified", confidence=0.45)

    if question_type == "ranking":
        rankings = chart_code.derived_relations.rankings
        if not rankings:
            return VerificationResult(verdict="ambiguous", reason="no ranking evidence", confidence=0.4)
        return VerificationResult(
            verdict="supported" if _rank_answer_supported(normalized, rankings) else "ambiguous",
            reason="checked ranking evidence",
            confidence=0.65,
        )

    if question_type in {"difference", "ratio"}:
        value = extract_number(normalized)
        if value is None:
            return VerificationResult(verdict="ambiguous", reason="answer has no numeric value", confidence=0.4)
        series_values = [point[1] for series in chart_code.series for point in series.points_data if len(point) >= 2]
        if len(series_values) < 2:
            return VerificationResult(verdict="ambiguous", reason="insufficient numeric series evidence", confidence=0.4)
        candidates = []
        for left in series_values:
            for right in series_values:
                candidates.append(abs(left - right))
                if question_type == "ratio" and right:
                    candidates.append(left / right)
        ok = any(numeric_close(value, candidate) for candidate in candidates)
        return VerificationResult(verdict="supported" if ok else "ambiguous", reason=f"checked numeric {question_type}", confidence=0.6)

    if question_type == "not_applicable":
        return VerificationResult(
            verdict="supported" if _supports_missing(normalized) else "ambiguous",
            reason="explicit Not Applicable question route",
            suggested_answer="Not Applicable",
            confidence=0.7,
        )

    if question_type == "open_ended_reasoning":
        evidence_terms = _series_names(chart_code) | _mark_categories(chart_code)
        if evidence_terms and any(term in lowered for term in evidence_terms):
            return VerificationResult(verdict="supported", reason="answer mentions Chart Code evidence", confidence=0.55)
        return VerificationResult(verdict="ambiguous", reason="open-ended answer requires semantic grading", confidence=0.4)

    return VerificationResult(verdict="ambiguous", reason=f"unrecognized question type: {question_type}", confidence=0.3)


def verify_and_normalize(question: str, question_type: QuestionType, chart_code: ChartCode, answer: str) -> tuple[str, VerificationResult]:
    normalized = normalize_answer(answer)
    result = verify_answer(question, question_type, chart_code, normalized)
    if result.verdict == "unsupported" and result.suggested_answer is not None:
        return result.suggested_answer, result
    return normalized, result
