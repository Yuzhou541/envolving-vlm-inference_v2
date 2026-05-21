"""Chart Code quality scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evochartcode.normalizer import normalize_answer
from evochartcode.schema import ChartCode


@dataclass
class ChartCodeQuality:
    chart_id: str
    chart_type_correct: bool | None
    image_size_valid: bool
    plot_area_present: bool
    legend_exists_correct: bool | None
    colorbar_exists_correct: bool | None
    axis_evidence_present: bool
    series_or_mark_present: bool
    global_confidence: float

    def score(self) -> float:
        values = [
            self.chart_type_correct,
            self.image_size_valid,
            self.plot_area_present,
            self.legend_exists_correct,
            self.colorbar_exists_correct,
            self.axis_evidence_present,
            self.series_or_mark_present,
        ]
        concrete = [float(value) for value in values if value is not None]
        return sum(concrete) / len(concrete) if concrete else 0.0


def expected_element_flags(descriptive_entry: dict[str, Any]) -> dict[str, bool | None]:
    flags: dict[str, bool | None] = {"legend": None, "colorbar": None}
    for qid, answer in zip(descriptive_entry.get("qids", []), descriptive_entry.get("answers", [])):
        normalized = normalize_answer(answer)
        if qid in {12, 13}:
            flags["legend"] = normalized != "Not Applicable"
        if qid in {14, 15}:
            flags["colorbar"] = normalized != "Not Applicable"
    return flags


def score_chart_code(
    chart_code: ChartCode,
    expected_chart_type: str | None = None,
    descriptive_entry: dict[str, Any] | None = None,
) -> ChartCodeQuality:
    expected_type = None
    if expected_chart_type:
        expected_type = expected_chart_type.lower().replace(" chart", "").replace("plot", "").strip()
    chart_type_correct = None if expected_type is None else chart_code.chart_type == expected_type
    flags = expected_element_flags(descriptive_entry or {})
    legend_correct = None if flags["legend"] is None else chart_code.legend.exists == flags["legend"]
    colorbar_correct = None if flags["colorbar"] is None else chart_code.colorbar.exists == flags["colorbar"]
    plot_area = chart_code.layout.get("plot_area_bbox")
    return ChartCodeQuality(
        chart_id=chart_code.chart_id,
        chart_type_correct=chart_type_correct,
        image_size_valid=chart_code.image_size.width > 0 and chart_code.image_size.height > 0,
        plot_area_present=bool(plot_area),
        legend_exists_correct=legend_correct,
        colorbar_exists_correct=colorbar_correct,
        axis_evidence_present=bool(chart_code.axes),
        series_or_mark_present=bool(chart_code.series or chart_code.marks),
        global_confidence=chart_code.uncertainty.global_confidence,
    )
