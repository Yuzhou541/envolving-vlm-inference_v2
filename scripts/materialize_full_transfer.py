"""Materialize full proposal-scale transfer manifests in resumable shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASETS: dict[str, dict[str, str | int]] = {
    "ChartQA": {"hf_id": "HuggingFaceM4/ChartQA", "split": "test", "expected_rows": 2500},
    "PlotQA": {"hf_id": "jinaai/plotqa", "split": "test", "expected_rows": 1000},
    "DVQA": {"hf_id": "sionic-ai/dvqa", "split": "train", "expected_rows": 200000},
    "FigureQA": {"hf_id": "sionic-ai/figureqa", "split": "train", "expected_rows": 100000},
}


def first_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def conversation_pairs(conversations: list[dict[str, Any]]) -> Iterable[tuple[int, str, str]]:
    pending: tuple[int, str] | None = None
    pair_index = 0
    for turn in conversations:
        role = str(turn.get("role", "")).lower()
        content = str(turn.get("content", "")).replace("<image>", "").strip()
        if not content:
            continue
        if role in {"user", "human"}:
            pending = (pair_index, content)
        elif role in {"assistant", "gpt"} and pending is not None:
            yield pending[0], pending[1], content
            pair_index += 1
            pending = None


def qa_pairs(row: dict[str, Any], dataset: str) -> Iterable[tuple[int, str, str]]:
    if "query" in row and "label" in row:
        yield 0, str(row["query"]), first_answer(row["label"])
        return
    if "query" in row and "answer" in row:
        yield 0, str(row["query"]), first_answer(row["answer"])
        return
    if "question" in row and "answer" in row:
        yield 0, str(row["question"]), first_answer(row["answer"])
        return
    if "Question" in row and "Answer" in row:
        yield 0, str(row["Question"]), first_answer(row["Answer"])
        return
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        yield from conversation_pairs(conversations)
        return
    if dataset == "PlotQA" and "text" in row:
        yield 0, "Extract the chart data table.", str(row["text"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_image_once(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if not hasattr(image, "save"):
            raise ValueError("row image is not saveable")
        image.save(path)
    return file_sha256(path)


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Materialize full transfer manifests from Hugging Face datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-root", type=Path, default=Path("data/external_full"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/transfer_full_manifests"))
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    spec = DATASETS[args.dataset]
    hf_id = str(spec["hf_id"])
    split = str(spec["split"])
    expected_rows = int(spec["expected_rows"])
    row_limit = args.limit_rows or expected_rows
    out_root = args.out_root / args.dataset
    image_dir = out_root / "images"
    manifest_dir = args.manifest_root / args.dataset
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(hf_id, split=split, streaming=True)
    shard_examples: list[dict[str, Any]] = []
    shard_index = 0
    total_examples = 0
    total_rows_seen = 0
    total_images = 0
    errors: list[str] = []

    def flush_shard(final: bool = False):
        nonlocal shard_examples, shard_index
        if not shard_examples and not final:
            return
        shard_path = manifest_dir / f"shard_{shard_index:05d}.json"
        if args.resume and shard_path.exists():
            shard_index += 1
            shard_examples = []
            return
        payload = {
            "dataset": args.dataset,
            "hf_id": hf_id,
            "split": split,
            "shard_index": shard_index,
            "status": "ready",
            "num_examples": len(shard_examples),
            "examples": shard_examples,
        }
        write_json(shard_path, payload)
        shard_index += 1
        shard_examples = []

    try:
        for row_index, row in enumerate(dataset):
            if row_index >= row_limit:
                break
            total_rows_seen += 1
            image = row.get("image")
            if image is None:
                errors.append(f"row {row_index}: missing image")
                continue
            image_id = f"{row_index:08d}"
            image_path = image_dir / f"{image_id}.png"
            try:
                checksum = save_image_once(image, image_path)
            except Exception as exc:
                errors.append(f"row {row_index}: {exc}")
                continue
            total_images += 1
            row_pairs = list(qa_pairs(row, args.dataset))
            if not row_pairs:
                errors.append(f"row {row_index}: no QA pairs")
                continue
            for conversation_index, question, answer in row_pairs:
                query_id = f"{args.dataset}_{image_id}_{conversation_index:04d}"
                shard_examples.append(
                    {
                        "dataset": args.dataset,
                        "source_hf_id": hf_id,
                        "split": split,
                        "source_row_id": str(row.get("id") or row.get("qid") or row_index),
                        "row_index": row_index,
                        "image_id": image_id,
                        "figure_id": image_id,
                        "image_path": str(image_path),
                        "question": question,
                        "answer": answer,
                        "conversation_index": conversation_index,
                        "status": "ready",
                        "checksum": checksum,
                    }
                )
                total_examples += 1
                if len(shard_examples) >= args.shard_size:
                    flush_shard()
        flush_shard(final=True)
        status = "ready" if total_examples else "empty"
    except Exception as exc:
        status = "error"
        errors.append(str(exc))

    shard_paths = sorted(manifest_dir.glob("shard_*.json"))
    summary = {
        "dataset": args.dataset,
        "hf_id": hf_id,
        "split": split,
        "status": status,
        "expected_rows": expected_rows,
        "row_limit": row_limit,
        "rows_seen": total_rows_seen,
        "num_images": total_images,
        "num_examples": total_examples,
        "num_shards": len(shard_paths),
        "shards": [str(path) for path in shard_paths],
        "errors": errors[:100],
        "error_count": len(errors),
    }
    write_json(manifest_dir / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"shards", "errors"}}, indent=2))


if __name__ == "__main__":
    main()
