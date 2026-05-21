"""Pydantic schema for the Chart Code representation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BBox = list[float]
ChartType = Literal[
    "line",
    "bar",
    "scatter",
    "heatmap",
    "boxplot",
    "histogram",
    "pie",
    "area",
    "mixed",
    "table",
    "unknown",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ImageSize(StrictBaseModel):
    width: int = 0
    height: int = 0


class TextField(StrictBaseModel):
    text: str | None = None
    bbox: BBox | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Tick(StrictBaseModel):
    value: str
    position: list[float] | None = None
    numeric_value: float | None = None


class AxisCode(StrictBaseModel):
    label: str | None = None
    unit: str | None = None
    scale: Literal["linear", "log", "categorical", "time", "unknown"] = "unknown"
    range: list[float] | None = None
    ticks: list[Tick] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LegendItem(StrictBaseModel):
    name: str | None = None
    color: str | None = None
    marker: str | None = None
    line_style: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LegendCode(StrictBaseModel):
    exists: bool = False
    bbox: BBox | None = None
    items: list[LegendItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ColorbarCode(StrictBaseModel):
    exists: bool = False
    bbox: BBox | None = None
    min: float | None = None
    max: float | None = None
    ticks: list[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SeriesCode(StrictBaseModel):
    id: str
    name: str | None = None
    visual_type: str = "unknown"
    color: str | None = None
    marker: str | None = None
    points_pixel: list[list[float]] = Field(default_factory=list)
    points_data: list[list[float]] = Field(default_factory=list)
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DerivedRelations(StrictBaseModel):
    trends: list[dict[str, Any]] = Field(default_factory=list)
    turning_points: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    intersections: list[dict[str, Any]] = Field(default_factory=list)
    extrema: list[dict[str, Any]] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)


class UncertaintyCode(StrictBaseModel):
    low_confidence_fields: list[str] = Field(default_factory=list)
    ambiguous_regions: list[dict[str, Any]] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    global_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ChartCode(StrictBaseModel):
    chart_id: str
    image_size: ImageSize = Field(default_factory=ImageSize)
    chart_type: ChartType = "unknown"
    title: TextField = Field(default_factory=TextField)
    layout: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, AxisCode] = Field(default_factory=dict)
    legend: LegendCode = Field(default_factory=LegendCode)
    colorbar: ColorbarCode = Field(default_factory=ColorbarCode)
    series: list[SeriesCode] = Field(default_factory=list)
    marks: list[dict[str, Any]] = Field(default_factory=list)
    derived_relations: DerivedRelations = Field(default_factory=DerivedRelations)
    uncertainty: UncertaintyCode = Field(default_factory=UncertaintyCode)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chart_type", mode="before")
    @classmethod
    def normalize_chart_type(cls, value: Any) -> str:
        if value is None:
            return "unknown"
        text = str(value).strip().lower().replace(" chart", "")
        mapping = {
            "line chart": "line",
            "line": "line",
            "bar chart": "bar",
            "bar": "bar",
            "scatter plot": "scatter",
            "scatter": "scatter",
            "heat map": "heatmap",
            "heatmap": "heatmap",
            "box plot": "boxplot",
            "boxplot": "boxplot",
            "histogram": "histogram",
            "pie chart": "pie",
            "pie": "pie",
            "area chart": "area",
            "area": "area",
            "table": "table",
            "mixed": "mixed",
        }
        return mapping.get(text, "unknown")

    def compact_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def coerce_chart_code(raw: dict[str, Any] | ChartCode, chart_id: str | None = None) -> ChartCode:
    """Validate a raw dict and fill the minimum required Chart Code fields."""
    if isinstance(raw, ChartCode):
        return raw
    data = dict(raw or {})
    if chart_id is not None and not data.get("chart_id"):
        data["chart_id"] = str(chart_id)
    if not data.get("chart_id"):
        data["chart_id"] = "unknown"
    if not data.get("image_size"):
        data["image_size"] = {"width": 0, "height": 0}
    return ChartCode.model_validate(data)
