"""Chart Code extraction backends."""

from __future__ import annotations

import json
import re
import ast
from pathlib import Path
from typing import Any

from evochartcode.cv import detect_plot_area_bbox, dominant_colors, read_image_size
from evochartcode.derived import augment_derived_relations
from evochartcode.schema import ChartCode, coerce_chart_code

EXTRACTION_PROMPT = """You are a chart-code extractor. Convert the chart image into valid JSON.
Do not answer any question. Only extract visual and structural information.
Return fields for chart type, title, axes, ticks, legend, colorbar, series, marks,
trends, comparisons, uncertainty, and missing elements.
If a field is not visible, use null and set confidence below 0.5.
Return valid JSON only."""


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        candidates.append(match.group(0))
    repaired_candidates = []
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired_candidates.append(repaired)
        balanced = _balance_json_brackets(repaired)
        if balanced != repaired:
            repaired_candidates.append(balanced)
    for candidate in candidates + repaired_candidates:
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
                break
            except Exception:
                try:
                    import yaml

                    value = yaml.safe_load(candidate)
                    break
                except Exception:
                    value = None
    else:
        value = None
    if value is None:
        raise ValueError("Could not parse or repair JSON object from VLM output.")
    if not isinstance(value, dict):
        raise ValueError("Chart Code extraction must return a JSON object.")
    return value


def _balance_json_brackets(text: str) -> str:
    opens = []
    pairs = {"{": "}", "[": "]"}
    for char in text:
        if char in pairs:
            opens.append(char)
        elif char in {"}", "]"} and opens:
            opens.pop()
    return text + "".join(pairs[char] for char in reversed(opens))


def metadata_chart_type(raw_types: list[str] | str | None) -> str:
    if not raw_types:
        return "unknown"
    if isinstance(raw_types, list):
        text = raw_types[0] if raw_types else ""
    else:
        text = raw_types
    return str(text).lower().replace(" chart", "").replace("plot", "").strip() or "unknown"


def repair_raw_chart_code(raw: dict[str, Any], base: ChartCode | None = None) -> dict[str, Any]:
    repaired = dict(raw)
    if isinstance(repaired.get("title"), str):
        repaired["title"] = {"text": repaired["title"], "confidence": 0.7}
    elif repaired.get("title") is None:
        repaired["title"] = {}

    if repaired.get("colorbar") is None:
        repaired["colorbar"] = {"exists": False, "confidence": 0.8}
    if isinstance(repaired.get("legend"), list):
        repaired["legend"] = {
            "exists": bool(repaired["legend"]),
            "items": [
                item if isinstance(item, dict) else {"name": str(item), "confidence": 0.7}
                for item in repaired["legend"]
            ],
            "confidence": 0.7,
        }
    if repaired.get("legend") is None:
        repaired["legend"] = {"exists": False, "items": [], "confidence": 0.5}

    axes = repaired.get("axes")
    if isinstance(axes, list):
        axis_map: dict[str, Any] = {}
        for index, axis in enumerate(axes):
            if not isinstance(axis, dict):
                continue
            axis_name = str(axis.get("axis_type") or axis.get("name") or axis.get("id") or ("x" if index == 0 else "y")).lower()
            if "x" in axis_name:
                axis_map["x"] = axis
            elif "y" in axis_name:
                axis_map["y"] = axis
            else:
                axis_map[f"axis_{index}"] = axis
        repaired["axes"] = axis_map
        axes = axis_map
    if isinstance(axes, dict):
        for axis_name, axis in list(axes.items()):
            if isinstance(axis, list):
                merged_axis: dict[str, Any] = {}
                merged_ticks = []
                for candidate in axis:
                    if isinstance(candidate, dict):
                        if not merged_axis:
                            merged_axis.update(candidate)
                        ticks = candidate.get("ticks")
                        if isinstance(ticks, list):
                            merged_ticks.extend(ticks)
                if merged_ticks:
                    merged_axis["ticks"] = merged_ticks
                axes[axis_name] = merged_axis
                axis = merged_axis
            if axis is None:
                axes[axis_name] = {}
                continue
            if isinstance(axis, str):
                axes[axis_name] = {"label": axis, "confidence": 0.6}
                axis = axes[axis_name]
            ticks = axis.get("ticks") if isinstance(axis, dict) else None
            if isinstance(ticks, list):
                axis["ticks"] = [
                    tick
                    if isinstance(tick, dict)
                    else {"value": str(tick), "numeric_value": float(tick) if isinstance(tick, (int, float)) else None}
                    for tick in ticks
                ]

    series = repaired.get("series")
    if isinstance(series, list):
        fixed_series = []
        for index, item in enumerate(series):
            if not isinstance(item, dict):
                continue
            fixed = dict(item)
            fixed.setdefault("id", f"series_{index}")
            if "data" in fixed and "points_data" not in fixed:
                fixed["points_data"] = fixed.pop("data")
            points = fixed.get("points_data")
            if isinstance(points, list):
                fixed_points = []
                for point in points:
                    if isinstance(point, dict):
                        if "x" in point and "y" in point:
                            fixed_points.append([point["x"], point["y"]])
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        fixed_points.append([point[0], point[1]])
                fixed["points_data"] = fixed_points
            fixed_series.append(fixed)
        repaired["series"] = fixed_series

    if base is not None:
        repaired.setdefault("image_size", base.image_size.model_dump())
        repaired.setdefault("layout", base.layout)
        repaired.setdefault("chart_type", base.chart_type)
    if repaired.get("uncertainty") is None:
        repaired["uncertainty"] = {"global_confidence": 0.3}
    if repaired.get("provenance") is None:
        repaired["provenance"] = {}
    if repaired.get("derived_relations") is None:
        repaired["derived_relations"] = {}
    return repaired


def build_metadata_chart_code(
    image_path: str | Path,
    chart_id: str,
    chart_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChartCode:
    image_path = Path(image_path)
    image_size = read_image_size(image_path)
    plot_bbox = detect_plot_area_bbox(image_path)
    colors = dominant_colors(image_path)
    layout: dict[str, Any] = {
        "plot_area_bbox": plot_bbox,
        "subplot_count": 1,
        "subplot_grid": {"rows": 1, "cols": 1},
        "subplots": [
            {
                "id": "subplot_0",
                "bbox": plot_bbox,
                "title": None,
                "x_axis_id": "x",
                "y_axis_id": "y",
            }
        ],
    }
    raw: dict[str, Any] = {
        "chart_id": chart_id,
        "image_size": image_size,
        "chart_type": chart_type or "unknown",
        "layout": layout,
        "provenance": {
            "extractor_versions": {"cv": "opencv_bbox_color_v1"},
            "fields": {
                "image_size": "pillow",
                "layout.plot_area_bbox": "opencv_threshold_bbox",
                "provenance.dominant_colors": "opencv_kmeans",
            },
            "dominant_colors": colors,
        },
        "uncertainty": {
            "missing_elements": ["axes", "legend", "colorbar", "series"],
            "global_confidence": 0.25,
        },
    }
    if metadata:
        raw["provenance"]["metadata"] = metadata
    return coerce_chart_code(raw, chart_id=chart_id)


class QwenVLJSONExtractor:
    """Qwen-VL chart-to-JSON extractor with greedy decoding."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        local_files_only: bool = True,
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=False,
            local_files_only=self.local_files_only,
        )

    def extract_raw(self, image_path: str | Path) -> dict[str, Any]:
        self._load()
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        output = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_json_object(output)


class ChartCodeExtractor:
    def __init__(
        self,
        backend: str = "metadata",
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        local_files_only: bool = True,
        max_new_tokens: int = 2048,
    ):
        self.backend = backend
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._vlm = QwenVLJSONExtractor(model_name=model_name, local_files_only=local_files_only, max_new_tokens=max_new_tokens)

    def extract(
        self,
        image_path: str | Path,
        chart_id: str,
        chart_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChartCode:
        base = build_metadata_chart_code(image_path, chart_id, chart_type=chart_type, metadata=metadata)
        if self.backend == "metadata":
            return base
        if self.backend != "qwen_vl_json":
            raise ValueError(f"Unknown Chart Code extractor backend: {self.backend}")

        try:
            raw = self._vlm.extract_raw(image_path)
        except Exception as exc:
            raw = {
                "chart_id": chart_id,
                "image_size": base.image_size.model_dump(),
                "chart_type": base.chart_type,
                "layout": base.layout,
                "uncertainty": {
                    "missing_elements": ["vlm_json"],
                    "global_confidence": 0.1,
                    "low_confidence_fields": ["all"],
                },
                "provenance": {
                    "extractor_versions": {"vlm": self.model_name},
                    "invalid_vlm_json_error": str(exc),
                },
            }
        raw = repair_raw_chart_code(raw, base)
        raw.setdefault("chart_id", chart_id)
        raw.setdefault("image_size", base.image_size.model_dump())
        raw.setdefault("provenance", {})
        raw["provenance"]["metadata_cv_base"] = base.provenance
        try:
            chart_code = coerce_chart_code(raw, chart_id=chart_id)
        except Exception as exc:
            fallback = base.compact_dict()
            fallback.setdefault("provenance", {})
            fallback["provenance"]["validation_repair_error"] = str(exc)
            fallback.setdefault("uncertainty", {})
            fallback["uncertainty"]["global_confidence"] = 0.1
            fallback["uncertainty"]["low_confidence_fields"] = ["all"]
            chart_code = coerce_chart_code(fallback, chart_id=chart_id)
        return augment_derived_relations(chart_code)
