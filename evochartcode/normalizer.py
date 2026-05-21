"""Answer normalization utilities."""

from __future__ import annotations

import math
import re

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    text = re.sub(r"^<answer>\s*|\s*</answer>$", "", text, flags=re.I)
    text = text.strip().strip("\"'`").strip()
    if text.lower() in {"n/a", "na", "not applicable.", "not-applicable"}:
        return "Not Applicable"
    if "not applicable" in text.lower():
        return "Not Applicable"
    return text.rstrip(".")


def extract_number(answer: str | None) -> float | None:
    text = normalize_answer(answer)
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def normalize_int(answer: str | None) -> int | None:
    value = extract_number(answer)
    if value is None:
        return None
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        return None
    return int(rounded)


def numeric_close(left: str | float | None, right: str | float | None, rel_tol: float = 0.05, abs_tol: float = 1e-4) -> bool:
    left_value = extract_number(str(left)) if not isinstance(left, (int, float)) else float(left)
    right_value = extract_number(str(right)) if not isinstance(right, (int, float)) else float(right)
    if left_value is None or right_value is None:
        return False
    return math.isclose(left_value, right_value, rel_tol=rel_tol, abs_tol=abs_tol)
