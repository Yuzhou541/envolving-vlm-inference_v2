"""Simple AlphaEvolve-style search loop for Qwen3-VL-2B-Instruct inference code."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import py_compile
import random
import re
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
try:
    from huggingface_hub.utils import logging as hf_logging
except ImportError:  # Optional dependency path.
    hf_logging = None
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
if hf_logging is not None:
    hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent
TMP_PACKAGE_DIR = ROOT / "evolution_tmp"
RUNS_DIR = ROOT / "evolution_runs"
EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"
TARGET_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
MUTATOR_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_RUNTIME_WEIGHT = 0.05

MUTATION_FOCI = [
    "tighten colorbar versus axis discrimination",
    "make legend-name formatting cleaner and more exact",
    "improve numeric tick extraction or normalization",
    "reduce unnecessary output length while keeping greedy decoding",
    "improve yes/no handling for line-intersection questions",
    "improve total tick counting without adding explanations",
    "strengthen Not Applicable behavior when the requested element is missing",
]

BANNED_PATTERNS = [
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "os.remove",
    "os.rmdir",
    "shutil.rmtree",
    "exec(",
]


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    source: str
    mutation_summary: str
    focus: str
    module_name: str
    file_path: str
    code_hash: str
    accuracy: float
    avg_time_per_query: float
    total_time: float
    num_errors: int
    score: float
    accepted: bool
    metrics_path: str = ""
    responses_path: str = ""
    prompt_path: str = ""
    mutation_path: str = ""
    validation_error: str = ""
    eval_error: str = ""


class LocalTextMutator:
    """Local LLM mutator using the cached Qwen3-VL-2B-Instruct model in text-only mode."""

    def __init__(self, model_name: str, local_files_only: bool = True):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto",
                local_files_only=self.local_files_only,
            )
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(self._model.generation_config, attr):
                    setattr(self._model.generation_config, attr, None)
            self._model.eval()
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=False,
                local_files_only=self.local_files_only,
            )

    def unload(self):
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_mutation(self, prompt: str, max_new_tokens: int = 512) -> str:
        self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are editing Python code for a chart-question answering module. "
                    "Return only a compact mutation proposal in the requested XML format."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.unload()
        return output


def ensure_runtime_dirs():
    TMP_PACKAGE_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    init_path = TMP_PACKAGE_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def extract_evolve_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(EVOLVE_BLOCK_START)}\s*(.*?)\s*{re.escape(EVOLVE_BLOCK_END)}",
        re.S,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return text


def wrap_candidate_code(code: str) -> str:
    return f"{EVOLVE_BLOCK_START}\n\n{code.rstrip()}\n\n{EVOLVE_BLOCK_END}\n"


def compute_score(metrics: dict, runtime_weight: float) -> float:
    return metrics["accuracy"] - runtime_weight * metrics["avg_time_per_query"]


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def coarse_question_type(question: str) -> str:
    q = question.lower()
    if "continuous legend" in q or "colorbar" in q:
        return "colorbar"
    if "legend" in q:
        return "legend"
    if "total number of explicitly labeled ticks" in q:
        return "total_ticks"
    if "difference between consecutive numerical tick values" in q:
        return "tick_diff"
    if "labeled tick" in q or "label of the x-axis" in q or "label of the y-axis" in q:
        return "axes_labels_ticks"
    if "how many lines" in q or "lines intersect" in q:
        return "line"
    if "layout of the subplots" in q or "number of subplots" in q:
        return "layout"
    if "title" in q:
        return "title"
    if "general trend" in q:
        return "trend"
    return "other"


def summarize_error_profile(responses_path: Path, max_examples: int = 4) -> dict:
    if not responses_path.exists():
        return {"num_wrong": 0, "counts": {}, "examples": []}
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    counts = Counter()
    examples = []
    for key, item in responses.items():
        if item.get("is_correct", False):
            continue
        q_type = coarse_question_type(item.get("question", ""))
        counts[q_type] += 1
        if len(examples) < max_examples:
            question = " ".join(item.get("question", "").split())
            if len(question) > 180:
                question = question[:177] + "..."
            examples.append(
                {
                    "query_key": key,
                    "type": q_type,
                    "question": question,
                    "response": item.get("response", ""),
                }
            )
    return {
        "num_wrong": sum(counts.values()),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "examples": examples,
    }


def render_error_profile(error_profile: dict) -> tuple[str, str]:
    counts = error_profile.get("counts", {})
    if counts:
        count_lines = "\n".join(f"- {name}: {value}" for name, value in counts.items())
    else:
        count_lines = "- none"
    examples = error_profile.get("examples", [])
    if examples:
        example_lines = "\n".join(
            f"- [{item['type']}] {item['query_key']}: model answered `{item['response']}` for `{item['question']}`"
            for item in examples
        )
    else:
        example_lines = "- none"
    return count_lines, example_lines


def make_mutation_prompt(
    parent_code: str,
    parent_metrics: dict,
    focus: str,
    top_summaries: list[str],
    error_profile: dict,
) -> str:
    history_block = "\n".join(f"- {line}" for line in top_summaries) if top_summaries else "- none yet"
    error_counts_block, error_examples_block = render_error_profile(error_profile)
    return textwrap.dedent(
        f"""
        Mutate the Python evolve block below to improve chart-question answering on CharXiv.

        Requirements:
        - Keep the module valid Python.
        - Keep the same public function: vlm_inference(image_path, question).
        - Keep greedy decoding: do_sample=False.
        - Keep the target model as Qwen/Qwen3-VL-2B-Instruct.
        - Do not add network calls, subprocess usage, or destructive file operations.
        - Prefer one focused edit, not a full rewrite.

        Current parent metrics:
        - accuracy: {parent_metrics["accuracy"]}
        - avg_time_per_query: {parent_metrics["avg_time_per_query"]}
        - score: {parent_metrics["score"]}

        Suggested mutation focus:
        - {focus}

        Current parent error profile:
        {error_counts_block}

        Example current misses:
        {error_examples_block}

        Current top candidates:
        {history_block}

        Return exactly one mutation in this XML format:
        <mutation>
        <summary>one short sentence</summary>
        <find><![CDATA[exact old text to replace]]></find>
        <replace><![CDATA[new text]]></replace>
        </mutation>

        The <find> text must appear exactly once in the current evolve block.

        Current evolve block:
        {EVOLVE_BLOCK_START}
        {parent_code.rstrip()}
        {EVOLVE_BLOCK_END}
        """
    ).strip()


def extract_mutation_xml(text: str) -> Optional[dict]:
    def decode_patch_text(value: str) -> str:
        return (
            value.replace("\r\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
        )

    summary_match = re.search(r"<summary>\s*(.*?)\s*</summary>", text, flags=re.S | re.I)
    find_match = re.search(r"<find>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</find>", text, flags=re.S | re.I)
    replace_match = re.search(r"<replace>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</replace>", text, flags=re.S | re.I)
    if not (summary_match and find_match and replace_match):
        return None
    return {
        "summary": summary_match.group(1).strip(),
        "find": decode_patch_text(find_match.group(1)),
        "replace": decode_patch_text(replace_match.group(1)),
    }


def extract_code_from_response(text: str) -> Optional[str]:
    block_match = re.search(
        rf"{re.escape(EVOLVE_BLOCK_START)}\s*(.*?)\s*{re.escape(EVOLVE_BLOCK_END)}",
        text,
        flags=re.S,
    )
    if block_match:
        return block_match.group(1).strip() + "\n"
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced[-1].strip() + "\n"
    if "def vlm_inference" in text and ("import torch" in text or "from transformers" in text):
        return text.strip() + "\n"
    return None


def validate_candidate_code(code: str) -> Optional[str]:
    if "def vlm_inference" not in code:
        return "missing vlm_inference"
    if TARGET_MODEL_NAME not in code:
        return "wrong target model"
    if "do_sample=False" not in code:
        return "greedy decoding requirement violated"
    lowered = code.lower()
    for pattern in BANNED_PATTERNS:
        if pattern.lower() in lowered:
            return f"banned pattern: {pattern}"
    return None


# Kept only so the archived Day 6 run artifacts remain explainable; the live search uses
# the cleaned heuristic_mutation implementation defined below.
def _legacy_heuristic_mutation(parent_code: str, rng: random.Random, focus: str, error_profile: dict) -> tuple[str, str]:
    focus_lower = focus.lower()
    dominant_error = next(iter(error_profile.get("counts", {})), "")
    options = []

    def relevance(tags: tuple[str, ...]) -> int:
        score = 0
        for tag in tags:
            if dominant_error == tag:
                score += 3
            if tag in focus_lower:
                score += 2
        return score

    def add_option(find: str, replace: str, summary: str, tags: tuple[str, ...]):
        if find in parent_code and find != replace:
            options.append((relevance(tags), parent_code.replace(find, replace, 1), summary))

    if "If the chart element is missing or unclear, prefer Not Applicable over guessing from nearby text.\\n" not in parent_code:
        add_option(
            "\"If the requested element does not exist, output Not Applicable exactly.\\n\"\n",
            "\"If the requested element does not exist, output Not Applicable exactly.\\n\"\n"
            "    \"If the chart element is missing or unclear, prefer Not Applicable over guessing from nearby text.\\n\"\n",
            "Added a stronger global Not Applicable rule to discourage guessing from nearby text.",
            ("colorbar", "legend", "line", "axes_labels_ticks"),
        )
    add_option(
        "    if question_type == \"title\":\n"
        "        return \"If the only title-like text is a subplot letter such as (a) or (b), output Not Applicable.\"\n",
        "    if question_type in {\"x_label\", \"y_label\"}:\n"
        "        return (\n"
        "            \"Return only the explicit axis label text. \"\n"
        "            \"Do not answer with tick labels, category names, or legend entries. \"\n"
        "            \"If there is no explicit axis label, output Not Applicable.\"\n"
        "        )\n"
        "    if question_type == \"title\":\n"
        "        return (\n"
        "            \"Return only the explicit title centered above the requested subplot. \"\n"
        "            \"Do not borrow titles from neighboring panels, row headers, or figure-wide captions. \"\n"
        "            \"If the only title-like text is a subplot letter such as (a) or (b), output Not Applicable.\"\n"
        "        )\n",
        "Added stricter axis-label and title hints to avoid borrowing ticks or neighboring captions.",
        ("axes_labels_ticks", "title"),
    )
    add_option(
        "            \"Only use a continuous legend or colorbar. Axis ticks are not a colorbar. \"\n"
        "            \"If there is no explicit relevant colorbar, output Not Applicable. \"\n"
        "            \"If you only see axes, subplot titles, or a surface with no separate colorbar scale, the answer is Not Applicable.\"",
        "            \"Only use a continuous legend or colorbar. Axis ticks are not a colorbar. \"\n"
        "            \"Do not use axis ranges, contour labels, or heatmap cell values as a substitute for a colorbar. \"\n"
        "            \"If there is no explicit relevant colorbar, output Not Applicable. \"\n"
        "            \"If you only see axes, subplot titles, or a surface with no separate colorbar scale, the answer is Not Applicable.\"",
        "Strengthened the colorbar hint to reject axis and heatmap-value substitutions.",
        ("colorbar",),
    )
    add_option(
        "            \"Use only discrete legend entries relevant to the requested subplot. \"\n"
        "            \"A continuous colorbar or gradient scale is not a discrete legend.\"",
        "            \"Use only discrete legend entries relevant to the requested subplot. \"\n"
        "            \"A continuous colorbar or gradient scale is not a discrete legend. \"\n"
        "            \"Captions, tables, method names, and panel labels are not legend entries.\"",
        "Tightened the legend hint so captions and table-like text do not count as legend entries.",
        ("legend",),
    )
    add_option(
        "            \"Count only actual plotted lines. Scatter markers, bars, heatmaps, surfaces, and grid lines do not count as lines. \"\n"
        "            \"Marker-only scatter plots are Not Applicable for line questions.\"",
        "            \"Count only actual plotted lines. Scatter markers, bars, heatmaps, surfaces, and grid lines do not count as lines. \"\n"
        "            \"Scatter plots, bar charts, histograms, heatmaps, contour maps, and image grids are Not Applicable for line questions. \"\n"
        "            \"Marker-only scatter plots are Not Applicable for line questions.\"",
        "Made line-question guidance more conservative for non-line plot types.",
        ("line",),
    )
    add_option(
        "            \"Use rows by columns order: n by m. \"\n"
        "            \"Two side-by-side subplots means 1 by 2. Two stacked subplots means 2 by 1. \"\n"
        "            \"Three side-by-side subplots means 1 by 3.\"",
        "            \"Use rows by columns order: n by m. \"\n"
        "            \"Two side-by-side subplots means 1 by 2. Two stacked subplots means 2 by 1. \"\n"
        "            \"Three side-by-side subplots means 1 by 3. \"\n"
        "            \"Grouped bars, many categories, multiple legend entries, or repeated line styles do not create subplots; a single axes panel is 1 by 1.\"",
        "Clarified that grouped marks and many categories do not imply a subplot layout.",
        ("layout",),
    )
    add_option(
        "    text = text.strip().strip(\"\\\"'`\").rstrip(\".\")\n",
        "    text = text.strip().strip(\"\\\"'`\").rstrip(\".\")\n"
        "    text = re.sub(r\"\\(\\s*e\\s*=\\s*([^)]+)\\)\", r\"(ε = \\1)\", text)\n",
        "Normalized parenthetical e = ... labels into the epsilon symbol when applicable.",
        ("axes_labels_ticks",),
    )
    add_option(
        "def _extract_layout(text):\n"
        "    match = re.search(r\"(\\d+)\\s*by\\s*(\\d+)\", text.lower())\n"
        "    if match:\n"
        "        return f\"{match.group(1)} by {match.group(2)}\"\n"
        "    return None\n\n\n"
        "def _postprocess_response(response, question):\n",
        "def _extract_layout(text):\n"
        "    match = re.search(r\"(\\d+)\\s*by\\s*(\\d+)\", text.lower())\n"
        "    if match:\n"
        "        return f\"{match.group(1)} by {match.group(2)}\"\n"
        "    return None\n\n\n"
        "def _looks_like_tick_code_sequence(text):\n"
        "    tokens = [re.sub(r\"[^A-Za-z0-9-]\", \"\", token) for token in text.replace(\",\", \" \").split()]\n"
        "    tokens = [token for token in tokens if token]\n"
        "    if len(tokens) < 4:\n"
        "        return False\n"
        "    upperish = [token for token in tokens if token.isupper() and len(token) <= 6]\n"
        "    return len(upperish) >= max(4, len(tokens) - 1)\n\n\n"
        "def _postprocess_response(response, question):\n",
        "Added a guard that detects uppercase tick-code sequences mistaken for axis labels.",
        ("axes_labels_ticks",),
    )
    add_option(
        "    if q_type in {\n"
        "        \"x_left_tick\",\n"
        "        \"x_right_tick\",\n"
        "        \"y_low_tick\",\n"
        "        \"y_high_tick\",\n"
        "        \"x_tick_diff\",\n"
        "        \"y_tick_diff\",\n"
        "        \"colorbar_range\",\n"
        "        \"colorbar_max\",\n"
        "    }:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        return last_line\n\n"
        "    if q_type in {\"title\", \"x_label\", \"y_label\", \"trend\"}:\n"
        "        return last_line\n",
        "    if q_type in {\"x_tick_diff\", \"y_tick_diff\"}:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is None:\n"
        "            number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return re.sub(r\"^([-+]?\\d+)\\.0+$\", r\"\\1\", number)\n"
        "        return last_line\n\n"
        "    if q_type in {\n"
        "        \"x_left_tick\",\n"
        "        \"x_right_tick\",\n"
        "        \"y_low_tick\",\n"
        "        \"y_high_tick\",\n"
        "        \"colorbar_range\",\n"
        "        \"colorbar_max\",\n"
        "    }:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        return last_line\n\n"
        "    if q_type in {\"x_label\", \"y_label\"}:\n"
        "        if _looks_like_tick_code_sequence(last_line):\n"
        "            return \"Not Applicable\"\n"
        "        return last_line\n\n"
        "    if q_type in {\"title\", \"trend\"}:\n"
        "        return last_line\n",
        "Separated tick-difference normalization from other numeric answers and added an axis-label tick-sequence guard.",
        ("tick_diff", "axes_labels_ticks"),
    )
    add_option(
        "        if \",\" in last_line:\n"
        "            return last_line\n",
        "        if \",\" in last_line:\n"
        "            return \", \".join(part.strip() for part in last_line.split(\",\") if part.strip())\n",
        "Normalized comma spacing in legend-name outputs.",
        ("legend",),
    )
    add_option(
        "padding=True, ",
        "",
        "Removed single-example padding to test lower preprocessing overhead.",
        ("speed",),
    )
    add_option(
        "max_new_tokens=64",
        "max_new_tokens=48",
        "Reduced max_new_tokens from 64 to 48 for shorter greedy generations.",
        ("speed",),
    )

    if not options:
        return parent_code, "Fallback mutation made no code changes."

    options.sort(key=lambda item: item[0], reverse=True)
    best_score = options[0][0]
    top_options = [item for item in options if item[0] == best_score]
    _, candidate_code, mutation_summary = rng.choice(top_options)
    return candidate_code, mutation_summary


def heuristic_mutation(parent_code: str, rng: random.Random, focus: str, error_profile: dict) -> tuple[str, str]:
    focus_lower = focus.lower()
    dominant_error = next(iter(error_profile.get("counts", {})), "")
    epsilon_char = "\u03b5"
    options = []
    seen_codes = set()

    def relevance(tags: tuple[str, ...]) -> int:
        score = 0
        for tag in tags:
            if dominant_error == tag:
                score += 3
            if tag in focus_lower:
                score += 2
        return score

    def add_option(find: str, replace: str, summary: str, tags: tuple[str, ...]):
        if find not in parent_code or find == replace:
            return
        candidate_code = parent_code.replace(find, replace, 1)
        if candidate_code in seen_codes:
            return
        seen_codes.add(candidate_code)
        options.append((relevance(tags), candidate_code, summary))

    if "If the chart element is missing or unclear, prefer Not Applicable over guessing from nearby text.\\n" not in parent_code:
        add_option(
            "\"If the requested element does not exist, output Not Applicable exactly.\\n\"\n",
            "\"If the requested element does not exist, output Not Applicable exactly.\\n\"\n"
            "    \"If the chart element is missing or unclear, prefer Not Applicable over guessing from nearby text.\\n\"\n",
            "Added a stronger global Not Applicable rule to discourage guessing from nearby text.",
            ("colorbar", "legend", "line", "axes_labels_ticks"),
        )
    add_option(
        "    if question_type == \"title\":\n"
        "        return \"If the only title-like text is a subplot letter such as (a) or (b), output Not Applicable.\"\n",
        "    if question_type in {\"x_label\", \"y_label\"}:\n"
        "        return (\n"
        "            \"Return only the explicit axis label text. \"\n"
        "            \"Do not answer with tick labels, category names, or legend entries. \"\n"
        "            \"If there is no explicit axis label, output Not Applicable.\"\n"
        "        )\n"
        "    if question_type == \"title\":\n"
        "        return (\n"
        "            \"Return only the explicit title centered above the requested subplot. \"\n"
        "            \"Do not borrow titles from neighboring panels, row headers, or figure-wide captions. \"\n"
        "            \"If the only title-like text is a subplot letter such as (a) or (b), output Not Applicable.\"\n"
        "        )\n",
        "Added stricter axis-label and title hints to avoid borrowing ticks or neighboring captions.",
        ("axes_labels_ticks", "title"),
    )
    add_option(
        "            \"Only use a continuous legend or colorbar. Axis ticks are not a colorbar. \"\n"
        "            \"If there is no explicit relevant colorbar, output Not Applicable. \"\n"
        "            \"If you only see axes, subplot titles, or a surface with no separate colorbar scale, the answer is Not Applicable.\"",
        "            \"Only use a continuous legend or colorbar. Axis ticks are not a colorbar. \"\n"
        "            \"Do not use axis ranges, contour labels, or heatmap cell values as a substitute for a colorbar. \"\n"
        "            \"If there is no explicit relevant colorbar, output Not Applicable. \"\n"
        "            \"If you only see axes, subplot titles, or a surface with no separate colorbar scale, the answer is Not Applicable.\"",
        "Strengthened the colorbar hint to reject axis and heatmap-value substitutions.",
        ("colorbar",),
    )
    add_option(
        "            \"Use only discrete legend entries relevant to the requested subplot. \"\n"
        "            \"A continuous colorbar or gradient scale is not a discrete legend.\"",
        "            \"Use only discrete legend entries relevant to the requested subplot. \"\n"
        "            \"A continuous colorbar or gradient scale is not a discrete legend. \"\n"
        "            \"Captions, tables, method names, and panel labels are not legend entries.\"",
        "Tightened the legend hint so captions and table-like text do not count as legend entries.",
        ("legend",),
    )
    add_option(
        "            \"Count only actual plotted lines. Scatter markers, bars, heatmaps, surfaces, and grid lines do not count as lines. \"\n"
        "            \"Marker-only scatter plots are Not Applicable for line questions.\"",
        "            \"Count only actual plotted lines. Scatter markers, bars, heatmaps, surfaces, and grid lines do not count as lines. \"\n"
        "            \"Scatter plots, bar charts, histograms, heatmaps, contour maps, and image grids are Not Applicable for line questions. \"\n"
        "            \"Marker-only scatter plots are Not Applicable for line questions.\"",
        "Made line-question guidance more conservative for non-line plot types.",
        ("line",),
    )
    add_option(
        "            \"Use rows by columns order: n by m. \"\n"
        "            \"Two side-by-side subplots means 1 by 2. Two stacked subplots means 2 by 1. \"\n"
        "            \"Three side-by-side subplots means 1 by 3.\"",
        "            \"Use rows by columns order: n by m. \"\n"
        "            \"Two side-by-side subplots means 1 by 2. Two stacked subplots means 2 by 1. \"\n"
        "            \"Three side-by-side subplots means 1 by 3. \"\n"
        "            \"Grouped bars, many categories, multiple legend entries, or repeated line styles do not create subplots; a single axes panel is 1 by 1.\"",
        "Clarified that grouped marks and many categories do not imply a subplot layout.",
        ("layout",),
    )
    add_option(
        "    text = text.strip().strip(\"\\\"'`\").rstrip(\".\")\n",
        "    text = text.strip().strip(\"\\\"'`\").rstrip(\".\")\n"
        f"    text = re.sub(r\"\\(\\s*e\\s*=\\s*([^)]+)\\)\", r\"({epsilon_char} = \\1)\", text)\n",
        "Normalized parenthetical e = ... labels into the epsilon symbol when applicable.",
        ("axes_labels_ticks",),
    )
    add_option(
        "def _extract_layout(text):\n"
        "    match = re.search(r\"(\\d+)\\s*by\\s*(\\d+)\", text.lower())\n"
        "    if match:\n"
        "        return f\"{match.group(1)} by {match.group(2)}\"\n"
        "    return None\n\n\n"
        "def _postprocess_response(response, question):\n",
        "def _extract_layout(text):\n"
        "    match = re.search(r\"(\\d+)\\s*by\\s*(\\d+)\", text.lower())\n"
        "    if match:\n"
        "        return f\"{match.group(1)} by {match.group(2)}\"\n"
        "    return None\n\n\n"
        "def _looks_like_tick_code_sequence(text):\n"
        "    tokens = [re.sub(r\"[^A-Za-z0-9-]\", \"\", token) for token in text.replace(\",\", \" \").split()]\n"
        "    tokens = [token for token in tokens if token]\n"
        "    if len(tokens) < 4:\n"
        "        return False\n"
        "    upperish = [token for token in tokens if token.isupper() and len(token) <= 6]\n"
        "    return len(upperish) >= max(4, len(tokens) - 1)\n\n\n"
        "def _postprocess_response(response, question):\n",
        "Added a guard that detects uppercase tick-code sequences mistaken for axis labels.",
        ("axes_labels_ticks",),
    )
    add_option(
        "    if q_type in {\n"
        "        \"x_left_tick\",\n"
        "        \"x_right_tick\",\n"
        "        \"y_low_tick\",\n"
        "        \"y_high_tick\",\n"
        "        \"x_tick_diff\",\n"
        "        \"y_tick_diff\",\n"
        "        \"colorbar_range\",\n"
        "        \"colorbar_max\",\n"
        "    }:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        return last_line\n\n"
        "    if q_type in {\"title\", \"x_label\", \"y_label\", \"trend\"}:\n"
        "        return last_line\n",
        "    if q_type in {\"x_tick_diff\", \"y_tick_diff\"}:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is None:\n"
        "            number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return re.sub(r\"^([-+]?\\d+)\\.0+$\", r\"\\1\", number)\n"
        "        return last_line\n\n"
        "    if q_type in {\n"
        "        \"x_left_tick\",\n"
        "        \"x_right_tick\",\n"
        "        \"y_low_tick\",\n"
        "        \"y_high_tick\",\n"
        "        \"colorbar_range\",\n"
        "        \"colorbar_max\",\n"
        "    }:\n"
        "        if \"not applicable\" in lowered_full:\n"
        "            return \"Not Applicable\"\n"
        "        number = _extract_number(last_line)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        number = _extract_number(normalized_full)\n"
        "        if number is not None:\n"
        "            return number\n"
        "        return last_line\n\n"
        "    if q_type in {\"x_label\", \"y_label\"}:\n"
        "        if _looks_like_tick_code_sequence(last_line):\n"
        "            return \"Not Applicable\"\n"
        "        return last_line\n\n"
        "    if q_type in {\"title\", \"trend\"}:\n"
        "        return last_line\n",
        "Separated tick-difference normalization from other numeric answers and added an axis-label tick-sequence guard.",
        ("tick_diff", "axes_labels_ticks"),
    )
    add_option(
        "        if \",\" in last_line:\n"
        "            return last_line\n",
        "        if \",\" in last_line:\n"
        "            return \", \".join(part.strip() for part in last_line.split(\",\") if part.strip())\n",
        "Normalized comma spacing in legend-name outputs.",
        ("legend",),
    )
    add_option(
        "padding=True, ",
        "",
        "Removed single-example padding to test lower preprocessing overhead.",
        ("speed",),
    )
    add_option(
        "max_new_tokens=64",
        "max_new_tokens=48",
        "Reduced max_new_tokens from 64 to 48 for shorter greedy generations.",
        ("speed",),
    )

    if not options:
        return parent_code, "Fallback mutation made no code changes."

    options.sort(key=lambda item: item[0], reverse=True)
    best_score = options[0][0]
    top_options = [item for item in options if item[0] == best_score]
    _, candidate_code, mutation_summary = rng.choice(top_options)
    return candidate_code, mutation_summary


def build_candidate_module_path(run_name: str, candidate_id: str) -> tuple[Path, str]:
    safe_run_name = re.sub(r"[^A-Za-z0-9_]+", "_", run_name).strip("_") or "run"
    module_basename = f"cand_{safe_run_name}_{candidate_id}"
    file_path = TMP_PACKAGE_DIR / f"{module_basename}.py"
    module_name = f"evolution_tmp.{module_basename}"
    return file_path, module_name


def write_candidate_files(run_dir: Path, run_name: str, candidate_id: str, code: str) -> tuple[Path, Path, str]:
    archive_dir = run_dir / "candidate_modules"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"{candidate_id}.py"
    archive_path.write_text(code.rstrip() + "\n", encoding="utf-8")

    file_path, module_name = build_candidate_module_path(run_name, candidate_id)
    file_path.write_text(wrap_candidate_code(code), encoding="utf-8")
    return archive_path, file_path, module_name


def compile_candidate(file_path: Path):
    py_compile.compile(str(file_path), doraise=True)


def evaluate_candidate(module_name: str, num_samples: int, run_dir: Path, candidate_id: str) -> dict:
    metrics_path = run_dir / f"{candidate_id}_metrics.json"
    responses_path = run_dir / f"{candidate_id}_responses.json"
    command = [
        sys.executable,
        "evaluate.py",
        module_name,
        "--hf-offline",
        "--num-samples",
        str(num_samples),
        "--output",
        str(metrics_path),
        "--save-responses",
        str(responses_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for {module_name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "responses_path": str(responses_path),
        "error_profile": summarize_error_profile(responses_path),
    }


def select_parent(population: list[CandidateRecord], rng: random.Random, top_k: int = 3) -> CandidateRecord:
    ranked = sorted(population, key=lambda c: (c.score, c.accuracy, -c.avg_time_per_query), reverse=True)
    return rng.choice(ranked[: min(top_k, len(ranked))])


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def dominates(left: CandidateRecord, right: CandidateRecord) -> bool:
    return (
        left.accuracy >= right.accuracy
        and left.avg_time_per_query <= right.avg_time_per_query
        and (
            left.accuracy > right.accuracy
            or left.avg_time_per_query < right.avg_time_per_query
        )
    )


def compute_pareto_frontier(records: list[CandidateRecord]) -> list[CandidateRecord]:
    accepted = [record for record in records if record.accepted]
    frontier = []
    for candidate in accepted:
        if any(
            dominates(other, candidate)
            for other in accepted
            if other.candidate_id != candidate.candidate_id
        ):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda c: (-c.accuracy, c.avg_time_per_query, c.candidate_id))


def refresh_population(
    accepted_records: list[CandidateRecord],
    population_size: int,
    elite_size: int,
) -> tuple[list[CandidateRecord], list[CandidateRecord], list[CandidateRecord]]:
    ranked = sorted(
        accepted_records,
        key=lambda c: (c.score, c.accuracy, -c.avg_time_per_query),
        reverse=True,
    )
    elites = ranked[: min(elite_size, len(ranked))]
    frontier = compute_pareto_frontier(accepted_records)
    population = []
    seen_ids = set()
    for group in (elites, frontier, ranked):
        for candidate in group:
            if candidate.candidate_id in seen_ids:
                continue
            population.append(candidate)
            seen_ids.add(candidate.candidate_id)
            if len(population) >= population_size:
                return population, elites, frontier
    return population, elites, frontier


def save_search_views(
    run_dir: Path,
    accepted_records: list[CandidateRecord],
    population: list[CandidateRecord],
    elite_size: int,
    generation: Optional[int] = None,
):
    _, elites, frontier = refresh_population(accepted_records, max(len(population), elite_size), elite_size)
    prefix = f"generation_{generation:02d}_" if generation is not None else ""
    save_json(run_dir / f"{prefix}population.json", [asdict(item) for item in population])
    save_json(run_dir / f"{prefix}elites.json", [asdict(item) for item in elites])
    save_json(run_dir / f"{prefix}pareto_frontier.json", [asdict(item) for item in frontier])


def export_best_candidate(best: CandidateRecord, destination: Path):
    source = Path(best.file_path)
    destination.write_text(extract_evolve_block(source), encoding="utf-8")


def run_evolution(args):
    ensure_runtime_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"instruct_proxy_{timestamp}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    elite_size = min(args.elite_size, args.population_size)

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    rng = random.Random(args.seed)
    seed_path = ROOT / f"{args.seed_module}.py"
    seed_code = extract_evolve_block(seed_path)
    seed_hash = code_hash(seed_code)

    config = {
        "run_name": run_name,
        "seed_module": args.seed_module,
        "num_samples": args.num_samples,
        "generations": args.generations,
        "candidates_per_generation": args.candidates_per_generation,
        "population_size": args.population_size,
        "elite_size": elite_size,
        "runtime_weight": args.runtime_weight,
        "mutation_backend": args.mutation_backend,
        "seed": args.seed,
        "hf_offline": args.hf_offline,
    }
    save_json(run_dir / "config.json", config)

    candidates_log = run_dir / "candidates.jsonl"
    generation_log = run_dir / "generation_summaries.jsonl"
    mutator = None
    if args.mutation_backend in {"local_qwen", "hybrid"}:
        mutator = LocalTextMutator(MUTATOR_MODEL_NAME, local_files_only=args.hf_offline)

    seed_archive_path, seed_tmp_path, seed_module_name = write_candidate_files(run_dir, run_name, "seed", seed_code)
    compile_candidate(seed_tmp_path)
    seed_eval = evaluate_candidate(seed_module_name, args.num_samples, run_dir, "seed")
    seed_metrics = seed_eval["metrics"]
    seed_metrics["score"] = compute_score(seed_metrics, args.runtime_weight)
    seed_record = CandidateRecord(
        candidate_id="seed",
        parent_id=None,
        source="seed",
        mutation_summary="Initial seed candidate.",
        focus="seed",
        module_name=seed_module_name,
        file_path=str(seed_archive_path),
        code_hash=seed_hash,
        accuracy=seed_metrics["accuracy"],
        avg_time_per_query=seed_metrics["avg_time_per_query"],
        total_time=seed_metrics["total_time"],
        num_errors=seed_metrics["num_errors"],
        score=seed_metrics["score"],
        accepted=True,
        metrics_path=seed_eval["metrics_path"],
        responses_path=seed_eval["responses_path"],
    )

    accepted_records = [seed_record]
    population, elites, frontier = refresh_population(accepted_records, args.population_size, elite_size)
    seen_hashes = {seed_hash}
    error_profiles = {"seed": seed_eval["error_profile"]}
    append_jsonl(candidates_log, asdict(seed_record))
    save_search_views(run_dir, accepted_records, population, elite_size)

    candidate_counter = 0
    for generation in range(args.generations):
        accepted_this_generation = []
        tried_this_generation = []
        for _ in range(args.candidates_per_generation):
            parent = select_parent(population, rng)
            parent_code = extract_evolve_block(Path(parent.file_path))
            focus = rng.choice(MUTATION_FOCI)
            candidate_counter += 1
            candidate_id = f"g{generation:02d}_c{candidate_counter:03d}"
            tried_this_generation.append(candidate_id)
            top_summaries = [
                f"{cand.candidate_id}: score={cand.score:.4f}, acc={cand.accuracy:.4f}, time={cand.avg_time_per_query:.4f}, summary={cand.mutation_summary}"
                for cand in sorted(population, key=lambda c: c.score, reverse=True)[:3]
            ]
            parent_error_profile = error_profiles.get(parent.candidate_id, {"num_wrong": 0, "counts": {}, "examples": []})
            prompt = make_mutation_prompt(
                parent_code=parent_code,
                parent_metrics={
                    "accuracy": parent.accuracy,
                    "avg_time_per_query": parent.avg_time_per_query,
                    "score": parent.score,
                },
                focus=focus,
                top_summaries=top_summaries,
                error_profile=parent_error_profile,
            )
            prompt_path = run_dir / f"{candidate_id}_prompt.txt"
            mutation_path = run_dir / f"{candidate_id}_raw_mutation.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            source = "heuristic_bank"
            mutation_summary = "Heuristic mutation"
            raw_mutation = ""
            use_llm = args.mutation_backend == "local_qwen" or (
                args.mutation_backend == "hybrid" and candidate_counter % 2 == 1
            )
            try:
                if use_llm and mutator is not None:
                    source = "local_llm"
                    raw_mutation = mutator.generate_mutation(prompt, max_new_tokens=args.mutator_max_new_tokens)
                else:
                    raw_mutation = ""
                mutation_path.write_text(raw_mutation, encoding="utf-8")
                if use_llm and raw_mutation:
                    parsed = extract_mutation_xml(raw_mutation)
                    find_count = parent_code.count(parsed["find"]) if parsed else 0
                    if parsed and find_count == 1:
                        candidate_code = parent_code.replace(parsed["find"], parsed["replace"], 1)
                        mutation_summary = parsed["summary"] or mutation_summary
                    elif parsed and 1 < find_count <= 3:
                        candidate_code = parent_code.replace(parsed["find"], parsed["replace"])
                        mutation_summary = (parsed["summary"] or mutation_summary) + " Applied to all matching occurrences."
                    else:
                        full_candidate = extract_code_from_response(raw_mutation)
                        if full_candidate:
                            candidate_code = full_candidate
                            mutation_summary = "LLM returned a full candidate block."
                        else:
                            raise ValueError("Could not extract a valid mutation from LLM output.")
                else:
                    candidate_code, mutation_summary = heuristic_mutation(parent_code, rng, focus, parent_error_profile)
            except Exception:
                candidate_code, mutation_summary = heuristic_mutation(parent_code, rng, focus, parent_error_profile)
                source = "heuristic_fallback"
                mutation_path.write_text(raw_mutation, encoding="utf-8")

            validation_error = validate_candidate_code(candidate_code)
            if validation_error is not None:
                candidate_code, mutation_summary = heuristic_mutation(parent_code, rng, focus, parent_error_profile)
                source = "heuristic_fallback"
                validation_error = validate_candidate_code(candidate_code)
                if validation_error is not None:
                    rejected = CandidateRecord(
                        candidate_id=candidate_id,
                        parent_id=parent.candidate_id,
                        source=source,
                        mutation_summary=f"Rejected before eval: {validation_error}",
                        focus=focus,
                        module_name="",
                        file_path="",
                        code_hash=code_hash(candidate_code),
                        accuracy=0.0,
                        avg_time_per_query=999.0,
                        total_time=999.0,
                        num_errors=1,
                        score=-999.0,
                        accepted=False,
                        prompt_path=str(prompt_path),
                        mutation_path=str(mutation_path),
                        validation_error=validation_error,
                    )
                    append_jsonl(candidates_log, asdict(rejected))
                    continue

            cand_hash = code_hash(candidate_code)
            if cand_hash in seen_hashes:
                duplicate_record = CandidateRecord(
                    candidate_id=candidate_id,
                    parent_id=parent.candidate_id,
                    source=source,
                    mutation_summary=f"{mutation_summary} Duplicate candidate skipped.",
                    focus=focus,
                    module_name="",
                    file_path="",
                    code_hash=cand_hash,
                    accuracy=0.0,
                    avg_time_per_query=999.0,
                    total_time=999.0,
                    num_errors=0,
                    score=-999.0,
                    accepted=False,
                    prompt_path=str(prompt_path),
                    mutation_path=str(mutation_path),
                )
                append_jsonl(candidates_log, asdict(duplicate_record))
                continue
            seen_hashes.add(cand_hash)

            archive_path, tmp_path, module_name = write_candidate_files(run_dir, run_name, candidate_id, candidate_code)
            metrics_path = ""
            responses_path = ""
            eval_error = ""

            try:
                compile_candidate(tmp_path)
                eval_result = evaluate_candidate(module_name, args.num_samples, run_dir, candidate_id)
                metrics = eval_result["metrics"]
                metrics["score"] = compute_score(metrics, args.runtime_weight)
                metrics_path = eval_result["metrics_path"]
                responses_path = eval_result["responses_path"]
                error_profiles[candidate_id] = eval_result["error_profile"]
                accepted = True
            except Exception as exc:
                metrics = {
                    "accuracy": 0.0,
                    "avg_time_per_query": 999.0,
                    "total_time": 999.0,
                    "num_errors": 1,
                    "score": -999.0,
                }
                mutation_summary = f"{mutation_summary} Eval failure: {exc}"
                accepted = False
                eval_error = str(exc)

            record = CandidateRecord(
                candidate_id=candidate_id,
                parent_id=parent.candidate_id,
                source=source,
                mutation_summary=mutation_summary,
                focus=focus,
                module_name=module_name,
                file_path=str(archive_path),
                code_hash=cand_hash,
                accuracy=metrics["accuracy"],
                avg_time_per_query=metrics["avg_time_per_query"],
                total_time=metrics["total_time"],
                num_errors=metrics["num_errors"],
                score=metrics["score"],
                accepted=accepted,
                metrics_path=metrics_path,
                responses_path=responses_path,
                prompt_path=str(prompt_path),
                mutation_path=str(mutation_path),
                validation_error=validation_error or "",
                eval_error=eval_error,
            )
            append_jsonl(candidates_log, asdict(record))

            if accepted:
                accepted_records.append(record)
                accepted_this_generation.append(record.candidate_id)
                population, elites, frontier = refresh_population(accepted_records, args.population_size, elite_size)

        save_search_views(run_dir, accepted_records, population, elite_size, generation=generation)
        generation_summary = {
            "generation": generation,
            "tried_candidates": tried_this_generation,
            "accepted_candidates": accepted_this_generation,
            "best_candidate_id": elites[0].candidate_id if elites else "",
            "elite_ids": [candidate.candidate_id for candidate in elites],
            "pareto_frontier_ids": [candidate.candidate_id for candidate in frontier],
        }
        append_jsonl(generation_log, generation_summary)

    best = max(accepted_records, key=lambda c: (c.score, c.accuracy, -c.avg_time_per_query))
    export_best_candidate(best, ROOT / args.export_best)
    final_population, final_elites, final_frontier = refresh_population(accepted_records, args.population_size, elite_size)
    save_search_views(run_dir, accepted_records, final_population, elite_size)
    non_seed_records = [record for record in accepted_records if record.candidate_id != "seed"]
    best_non_seed = None
    if non_seed_records:
        best_non_seed = max(non_seed_records, key=lambda c: (c.score, c.accuracy, -c.avg_time_per_query))

    summary = {
        "run_name": run_name,
        "best_candidate": asdict(best),
        "best_non_seed_candidate": asdict(best_non_seed) if best_non_seed is not None else None,
        "population": [asdict(cand) for cand in final_population],
        "elites": [asdict(cand) for cand in final_elites],
        "pareto_frontier": [asdict(cand) for cand in final_frontier],
        "accepted_candidate_count": len(accepted_records),
    }
    save_json(run_dir / "summary.json", summary)
    if mutator is not None:
        mutator.unload()
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run a simple local evolution loop for manual Instruct inference code.")
    parser.add_argument("--seed-module", default="manual_instruct")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of development queries to use during evolution.")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--candidates-per-generation", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--runtime-weight", type=float, default=DEFAULT_RUNTIME_WEIGHT)
    parser.add_argument("--mutation-backend", choices=["local_qwen", "heuristic_only", "hybrid"], default="hybrid")
    parser.add_argument("--mutator-max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--export-best", default="evolved_instruct.py")
    parser.add_argument("--hf-offline", action="store_true", default=True)
    parser.add_argument("--no-hf-offline", action="store_false", dest="hf_offline")
    args = parser.parse_args()
    run_evolution(args)


if __name__ == "__main__":
    main()
