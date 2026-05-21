"""Cross-dataset chart QA adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
QUESTION_KEYS = ("question", "query", "Question", "Q")
ANSWER_KEYS = ("answer", "label", "Answer", "A")
IMAGE_KEYS = ("image", "img", "image_path", "filename", "file", "figure_path")


@dataclass
class TransferExample:
    dataset: str
    query_id: str
    figure_id: str
    image_path: str
    question: str
    answer: str
    metadata: dict[str, Any]


def _flatten_records(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_records(item)
    elif isinstance(value, dict):
        if any(key in value for key in QUESTION_KEYS) and any(key in value for key in ANSWER_KEYS):
            yield value
        for key in ("data", "annotations", "examples", "qa_pairs", "questions"):
            child = value.get(key)
            if child is not None:
                yield from _flatten_records(child)


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _index_images(root: Path) -> dict[str, Path]:
    images = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images[path.name.lower()] = path
        images[path.stem.lower()] = path
        try:
            images[str(path.relative_to(root)).replace("\\", "/").lower()] = path
        except ValueError:
            pass
    return images


def _resolve_image(root: Path, image_index: dict[str, Path], value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).replace("\\", "/")
    direct = root / text
    if direct.exists():
        return direct
    return image_index.get(text.lower()) or image_index.get(Path(text).name.lower()) or image_index.get(Path(text).stem.lower())


def discover_transfer_examples(dataset: str, root: str | Path, limit: int | None = None) -> list[TransferExample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset} root does not exist: {root}")
    image_index = _index_images(root)
    examples: list[TransferExample] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for record in _flatten_records(payload):
            question = _first(record, QUESTION_KEYS)
            answer = _first(record, ANSWER_KEYS)
            image_value = _first(record, IMAGE_KEYS)
            image_path = _resolve_image(root, image_index, image_value)
            if question is None or answer is None or image_path is None:
                continue
            query_id = str(record.get("id") or record.get("qid") or f"{json_path.stem}_{len(examples)}")
            figure_id = str(record.get("figure_id") or record.get("image_id") or image_path.stem)
            examples.append(
                TransferExample(
                    dataset=dataset,
                    query_id=query_id,
                    figure_id=figure_id,
                    image_path=str(image_path),
                    question=str(question),
                    answer=str(answer),
                    metadata={
                        "source_json": str(json_path),
                        "raw_image_field": image_value,
                    },
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def write_transfer_manifest(dataset: str, root: str | Path, out: str | Path, limit: int | None = None) -> dict[str, Any]:
    out = Path(out)
    try:
        examples = discover_transfer_examples(dataset, root, limit=limit)
        payload = {
            "dataset": dataset,
            "root": str(root),
            "status": "ready",
            "num_examples": len(examples),
            "examples": [asdict(example) for example in examples],
        }
    except FileNotFoundError as exc:
        payload = {
            "dataset": dataset,
            "root": str(root),
            "status": "missing",
            "num_examples": 0,
            "error": str(exc),
            "examples": [],
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
