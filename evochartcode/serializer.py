"""Compact serialization for selected Chart Code evidence."""

from __future__ import annotations

import json
from typing import Any


def serialize_code(code: dict[str, Any], max_chars: int | None = None) -> str:
    text = json.dumps(code, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def format_reasoning_prompt(question: str, selected_code: dict[str, Any]) -> str:
    return (
        "Answer the chart question using the Chart Code evidence. "
        "Return only the final short answer.\n\n"
        f"Question: {question}\n"
        f"Chart Code: {serialize_code(selected_code)}"
    )
