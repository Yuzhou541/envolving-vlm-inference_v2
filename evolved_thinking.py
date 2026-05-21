"""Accuracy-focused manual optimization for Qwen3-VL-2B-Thinking on CharXiv."""
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

_model = None
_processor = None
_model_name = "Qwen/Qwen3-VL-2B-Thinking"
_SYSTEM_PROMPT = (
    "Reason internally if needed, but do not reveal your reasoning. "
    "Reply with exactly one line in the form <answer>FINAL_ANSWER</answer> and nothing else."
)

_BASE_RULES = (
    "You are graded by exact string match.\n"
    "Think silently if needed, but do not output your reasoning.\n"
    "Output exactly one line in the form <answer>FINAL_ANSWER</answer>.\n"
    "Do not output any text before or after the <answer> tags.\n"
    "If the requested element does not exist, output <answer>Not Applicable</answer>.\n"
    "Copy text, symbols, capitalization, and tick values exactly when possible.\n"
    "Preserve exact numeric formatting such as 1.00, 10^1, or 10^-6 instead of simplifying them."
)


def _normalize_text(text):
    text = text.strip().replace("\r", "")
    for src, dst in (
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u2212", "-"),
    ):
        text = text.replace(src, dst)
    text = text.strip().strip("\"'`").rstrip(".")
    text = text.replace("$", "")
    text = re.sub(r"10\s*\^\s*\{\s*([-+]?\d+)\s*\}", r"10^\1", text)
    text = re.sub(r"10\^\{\s*([-+]?\d+)\s*\}", r"10^\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _question_type(question):
    q = question.lower()
    if "what is its title" in q:
        return "title"
    if "label of the x-axis" in q:
        return "x_label"
    if "label of the y-axis" in q:
        return "y_label"
    if "leftmost labeled tick on the x-axis" in q:
        return "x_left_tick"
    if "rightmost labeled tick on the x-axis" in q:
        return "x_right_tick"
    if "spatially lowest labeled tick on the y-axis" in q:
        return "y_low_tick"
    if "spatially highest labeled tick on the y-axis" in q:
        return "y_high_tick"
    if "difference between consecutive numerical tick values on the x-axis" in q:
        return "x_tick_diff"
    if "difference between consecutive numerical tick values on the y-axis" in q:
        return "y_tick_diff"
    if "how many lines are there" in q:
        return "line_count"
    if "do any lines intersect" in q:
        return "line_intersection"
    if "how many discrete labels are there in the legend" in q:
        return "legend_count"
    if "what are the names of the labels in the legend" in q:
        return "legend_names"
    if "difference between the maximum and minimum values of the tick labels on the continuous legend" in q:
        return "colorbar_range"
    if "maximum value of the tick labels on the continuous legend" in q:
        return "colorbar_max"
    if "what is the general trend of data from left to right" in q:
        return "trend"
    if "what is the total number of explicitly labeled ticks across all axes" in q:
        return "total_ticks"
    if "what is the layout of the subplots" in q:
        return "layout"
    if "what is the number of subplots" in q:
        return "subplot_count"
    return "generic"


def _task_hint(question_type):
    if question_type == "title":
        return "If the only title-like text is a subplot letter such as (a) or (b), output Not Applicable."
    if question_type in {"colorbar_range", "colorbar_max"}:
        return (
            "Only use a continuous legend or colorbar. Axis ticks are not a colorbar. "
            "Do not use axis ranges, contour labels, or heatmap cell values as a substitute for a colorbar. "
            "If there is no explicit relevant colorbar, output Not Applicable. "
            "If you only see axes, subplot titles, or a surface with no separate colorbar scale, the answer is Not Applicable."
        )
    if question_type in {"legend_count", "legend_names"}:
        return (
            "Use only discrete legend entries relevant to the requested subplot. "
            "A continuous colorbar or gradient scale is not a discrete legend. "
            "Captions, tables, method names, and panel labels are not legend entries."
        )
    if question_type in {"line_count", "line_intersection"}:
        return (
            "Count only actual plotted lines. Scatter markers, bars, heatmaps, surfaces, and grid lines do not count as lines. "
            "Scatter plots, bar charts, histograms, heatmaps, contour maps, and image grids are Not Applicable for line questions. "
            "Marker-only scatter plots are Not Applicable for line questions."
        )
    if question_type == "total_ticks":
        return (
            "Count only explicitly labeled axis ticks for the requested subplot. "
            "Do not count labels from other subplots, data labels, or legend entries. "
            "Add the visible x-axis tick labels and y-axis tick labels for that subplot only. "
            "Return only the final total integer, not the per-axis subtotals."
        )
    if question_type == "layout":
        return (
            "Use rows by columns order: n by m. "
            "Two side-by-side subplots means 1 by 2. Two stacked subplots means 2 by 1. "
            "Three side-by-side subplots means 1 by 3."
        )
    if question_type == "subplot_count":
        return "Return only the final integer."
    if question_type in {
        "x_left_tick",
        "x_right_tick",
        "y_low_tick",
        "y_high_tick",
        "x_tick_diff",
        "y_tick_diff",
    }:
        return (
            "Return the exact written tick value format from the chart. "
            "For example, keep 1.00 as 1.00 and keep scientific-style ticks such as 10^1 or 10^-6."
        )
    return ""


def _build_prompt(question):
    q_type = _question_type(question)
    hint = _task_hint(q_type)
    parts = [_BASE_RULES]
    if hint:
        parts.append(hint)
    parts.append(question.strip())
    return "\n\n".join(parts)


def _extract_bullets(text):
    bullets = []
    for line in text.splitlines():
        item = line.strip()
        if not item.startswith("-"):
            continue
        item = item[1:].strip()
        if ":" in item:
            item = item.split(":", 1)[0].strip()
        item = _normalize_text(item)
        if item:
            bullets.append(item)
    return bullets


def _extract_number(text):
    power_tokens = re.findall(r"10\^-?\d+", text)
    if power_tokens:
        return power_tokens[-1]
    number_tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![A-Za-z])", text, re.I)
    if number_tokens:
        return number_tokens[-1]
    return None


def _extract_yes_no(text):
    lower = text.lower()
    if "not applicable" in lower:
        return "Not Applicable"
    if re.search(r"\byes\b", lower):
        return "Yes"
    if re.search(r"\bno\b", lower):
        return "No"
    return None


def _extract_layout(text):
    match = re.search(r"(\d+)\s*by\s*(\d+)", text.lower())
    if match:
        return f"{match.group(1)} by {match.group(2)}"
    return None


def _extract_final_answer_marker(text):
    patterns = (
        r"<answer>\s*(.*?)\s*</answer>",
        r"<final_answer>\s*(.*?)\s*</final_answer>",
        r"\bFINAL_ANSWER\s*:\s*(.+)",
        r"\bFinal answer\s*:\s*(.+)",
        r"\bAnswer\s*:\s*(.+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I | re.S)
        if matches:
            candidate = matches[-1].strip()
            if "\n" in candidate:
                candidate = candidate.splitlines()[0].strip()
            return candidate
    return None


def _postprocess_response(response, question):
    q_type = _question_type(question)
    text = response.strip()
    if not text:
        return text

    tagged_answer = _extract_final_answer_marker(text)
    if tagged_answer:
        text = tagged_answer

    lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
    last_line = lines[-1] if lines else _normalize_text(text)
    normalized_full = _normalize_text(text)
    lowered_full = normalized_full.lower()

    if "not applicable" in lowered_full or "no title" in lowered_full:
        return "Not Applicable"

    if q_type == "title" and re.fullmatch(r"\(?[a-zA-Z]\)?", last_line):
        return "Not Applicable"

    if q_type in {"line_intersection"}:
        answer = _extract_yes_no(normalized_full)
        if answer is not None:
            return answer

    if q_type == "layout":
        answer = _extract_layout(normalized_full)
        if answer is not None:
            return answer

    if q_type == "legend_names":
        bullets = _extract_bullets(text)
        if bullets:
            return ", ".join(bullets)
        if "," in last_line:
            return ", ".join(part.strip() for part in last_line.split(",") if part.strip())
        return normalized_full

    if q_type in {"subplot_count", "line_count", "legend_count", "total_ticks"}:
        if "not applicable" in lowered_full:
            return "Not Applicable"
        numbered = re.search(
            r"(?:there are|there is|total number of .*? is|number of .*? is|answer:)\s*([-+]?\d+)",
            lowered_full,
        )
        if numbered:
            return numbered.group(1)
        number = _extract_number(last_line)
        if number is not None:
            return number
        number = _extract_number(normalized_full)
        if number is not None:
            return number
        return last_line

    if q_type in {
        "x_left_tick",
        "x_right_tick",
        "y_low_tick",
        "y_high_tick",
        "x_tick_diff",
        "y_tick_diff",
        "colorbar_range",
        "colorbar_max",
    }:
        if "not applicable" in lowered_full:
            return "Not Applicable"
        number = _extract_number(last_line)
        if number is not None:
            return number
        number = _extract_number(normalized_full)
        if number is not None:
            return number
        return last_line

    if q_type in {"title", "x_label", "y_label", "trend"}:
        return last_line

    return last_line if len(last_line.split()) <= 12 else normalized_full


def _get_model():
    global _model, _processor
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(
            _model_name,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=local_files_only,
        )
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(_model.generation_config, attr):
                setattr(_model.generation_config, attr, None)
        _model.eval()
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(_model_name, use_fast=False, local_files_only=local_files_only)
    return _model, _processor


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n<think>\n</think>\n\n"
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return _postprocess_response(raw_output, question)
