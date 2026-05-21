"""Code-grounded reasoning modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evochartcode.extractor import ChartCodeExtractor
from evochartcode.normalizer import normalize_answer
from evochartcode.routing import route_question
from evochartcode.schema import ChartCode, coerce_chart_code
from evochartcode.selector import select_code_for_question
from evochartcode.serializer import format_reasoning_prompt
from evochartcode.verifier import VerificationResult, verify_and_normalize


class ChartCodeCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None

    def load(self, chart_id: str) -> ChartCode | None:
        if self.root is None:
            return None
        path = self.root / f"{chart_id}.json"
        if not path.exists():
            return None
        return coerce_chart_code(json.loads(path.read_text(encoding="utf-8")), chart_id=chart_id)

    def save(self, chart_code: ChartCode):
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{chart_code.chart_id}.json"
        path.write_text(json.dumps(chart_code.compact_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class CodeOnlyReasoner:
    def answer(self, question: str, selected_code: dict[str, Any]) -> str:
        qtype = route_question(question)
        if qtype == "chart_type":
            return selected_code.get("chart_type") or "Not Applicable"
        if qtype == "title":
            title = selected_code.get("title", {}).get("text")
            return title or "Not Applicable"
        if qtype == "legend_count":
            legend = selected_code.get("legend", {})
            if not legend.get("exists"):
                return "Not Applicable"
            return str(len(legend.get("items", [])))
        if qtype == "legend_name":
            legend = selected_code.get("legend", {})
            names = [item.get("name") for item in legend.get("items", []) if item.get("name")]
            return ", ".join(names) if names else "Not Applicable"
        if qtype == "colorbar_min_max":
            colorbar = selected_code.get("colorbar", {})
            if not colorbar.get("exists"):
                return "Not Applicable"
            if "max" in question.lower() and colorbar.get("max") is not None:
                return str(colorbar["max"])
            if "min" in question.lower() and colorbar.get("min") is not None:
                return str(colorbar["min"])
            return "Not Applicable"
        if qtype == "subplot_count":
            layout = selected_code.get("layout", {})
            count = layout.get("subplot_count")
            if count is not None:
                return str(count)
        if qtype == "line_count":
            series = selected_code.get("series", [])
            count = sum(1 for item in series if str(item.get("visual_type", "")).lower() == "line")
            return str(count) if count else "Not Applicable"
        if qtype in {"axis_label", "axis_range", "tick_count", "tick_value"}:
            axes = selected_code.get("axes", {})
            if qtype == "tick_count":
                return str(sum(len(axis.get("ticks", [])) for axis in axes.values())) if axes else "Not Applicable"
            axis_key = "x" if "x-axis" in question.lower() or "x axis" in question.lower() else "y" if "y-axis" in question.lower() or "y axis" in question.lower() else None
            axis = axes.get(axis_key) if axis_key else None
            if not axis:
                return "Not Applicable"
            if qtype == "axis_label":
                return axis.get("label") or "Not Applicable"
            if qtype == "axis_range":
                values = axis.get("range")
                return f"{values[0]} to {values[-1]}" if values else "Not Applicable"
            ticks = axis.get("ticks", [])
            if not ticks:
                return "Not Applicable"
            if "leftmost" in question.lower() or "lowest" in question.lower():
                return str(ticks[0].get("value", "Not Applicable"))
            if "rightmost" in question.lower() or "highest" in question.lower():
                return str(ticks[-1].get("value", "Not Applicable"))
            return str(ticks[0].get("value", "Not Applicable"))
        if qtype == "trend":
            trends = selected_code.get("derived_relations", {}).get("trends", [])
            if trends:
                return str(trends[0].get("global_trend", "unknown"))
        if qtype in {"extremum", "turning_point", "intersection", "comparison", "ranking"}:
            relations = selected_code.get("derived_relations", {})
            key = {
                "extremum": "extrema",
                "turning_point": "turning_points",
                "intersection": "intersections",
                "comparison": "comparisons",
                "ranking": "rankings",
            }[qtype]
            values = relations.get(key, [])
            return json.dumps(values[0], ensure_ascii=False) if values else "Not Applicable"
        return "Not Applicable"


class QwenImageCodeReasoner:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", local_files_only: bool = True):
        self.model_name = model_name
        self.local_files_only = local_files_only
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

    def answer(self, image_path: str | Path, question: str, selected_code: dict[str, Any] | None = None) -> str:
        self._load()
        from qwen_vl_utils import process_vision_info

        prompt = question if selected_code is None else format_reasoning_prompt(question, selected_code)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
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
        generated = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class EvoChartCodePipeline:
    def __init__(
        self,
        extractor: ChartCodeExtractor,
        cache: ChartCodeCache | None = None,
        reasoner: CodeOnlyReasoner | QwenImageCodeReasoner | None = None,
        mode: str = "code_only",
    ):
        self.extractor = extractor
        self.cache = cache or ChartCodeCache(None)
        self.reasoner = reasoner or CodeOnlyReasoner()
        self.mode = mode

    def get_chart_code(
        self,
        image_path: str | Path,
        chart_id: str,
        chart_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChartCode:
        cached = self.cache.load(chart_id)
        if cached is not None:
            return cached
        chart_code = self.extractor.extract(image_path, chart_id=chart_id, chart_type=chart_type, metadata=metadata)
        self.cache.save(chart_code)
        return chart_code

    def answer(
        self,
        image_path: str | Path,
        question: str,
        chart_id: str,
        chart_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, VerificationResult]:
        chart_code = self.get_chart_code(image_path, chart_id=chart_id, chart_type=chart_type, metadata=metadata)
        question_type = route_question(question)
        selected_code = select_code_for_question(question_type, chart_code)

        if self.mode == "code_only":
            raw_answer = self.reasoner.answer(question, selected_code)  # type: ignore[arg-type]
        elif self.mode in {"image_code", "full_evochartcode"}:
            raw_answer = self.reasoner.answer(image_path, question, selected_code)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown reasoning mode: {self.mode}")

        final_answer, verification = verify_and_normalize(question, question_type, chart_code, normalize_answer(raw_answer))
        return final_answer, verification
