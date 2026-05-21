"""Materialize Hugging Face chart QA datasets into local transfer manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _qa_from_conversations(conversations: list[dict[str, Any]]) -> tuple[str, str] | None:
    question = None
    for turn in conversations:
        role = str(turn.get("role", "")).lower()
        content = str(turn.get("content", "")).replace("<image>", "").strip()
        if role in {"user", "human"} and question is None:
            question = content
        elif role in {"assistant", "gpt"} and question:
            return question, content
    return None


def _extract_qa(row: dict[str, Any], dataset: str) -> tuple[str, str] | None:
    if "query" in row and "label" in row:
        return str(row["query"]), _first(row["label"])
    if "question" in row and "answer" in row:
        return str(row["question"]), _first(row["answer"])
    if "Question" in row and "Answer" in row:
        return str(row["Question"]), _first(row["Answer"])
    if "conversations" in row and isinstance(row["conversations"], list):
        return _qa_from_conversations(row["conversations"])
    if dataset.lower() == "plotqa" and "text" in row:
        return "Extract the chart data table.", str(row["text"])
    return None


def main():
    parser = argparse.ArgumentParser(description="Materialize an HF chart QA dataset sample.")
    parser.add_argument("--dataset", required=True, choices=["ChartQA", "PlotQA", "DVQA", "FigureQA"])
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--out-root", type=Path, default=Path("data/external"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    out_root = args.out_root / args.dataset
    image_dir = out_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    status = "ready"
    error = ""
    try:
        dataset = load_dataset(args.hf_id, split=args.split, streaming=True)
        for row_index, row in enumerate(dataset):
            qa = _extract_qa(row, args.dataset)
            image = row.get("image")
            if qa is None or image is None:
                continue
            question, answer = qa
            image_path = image_dir / f"{row_index:08d}.png"
            if hasattr(image, "save"):
                image.save(image_path)
            else:
                continue
            examples.append(
                {
                    "dataset": args.dataset,
                    "query_id": str(row.get("id") or row.get("qid") or row_index),
                    "figure_id": image_path.stem,
                    "image_path": str(image_path),
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "hf_id": args.hf_id,
                        "split": args.split,
                        "row_index": row_index,
                    },
                }
            )
            if len(examples) >= args.limit:
                break
        if not examples:
            status = "no_qa_rows"
            error = "No rows with both image and question/answer fields were found."
    except Exception as exc:
        status = "error"
        error = str(exc)

    payload = {
        "dataset": args.dataset,
        "root": str(out_root),
        "hf_id": args.hf_id,
        "split": args.split,
        "status": status,
        "num_examples": len(examples),
        "error": error,
        "examples": examples,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("dataset", "hf_id", "status", "num_examples", "error")}, indent=2))


if __name__ == "__main__":
    main()
