"""Extract Qwen Chart Code caches for sharded transfer manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evochartcode.config import load_config
from evochartcode.extractor import ChartCodeExtractor
from evochartcode.reasoner import ChartCodeCache


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def unique_images(manifest_dir: Path, max_images: int | None = None) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for shard_path in sorted(manifest_dir.glob("shard_*.json")):
        shard = read_json(shard_path)
        for example in shard.get("examples", []):
            image_id = str(example["image_id"])
            if image_id in seen:
                continue
            seen.add(image_id)
            rows.append(example)
            if max_images is not None and len(rows) >= max_images:
                return rows
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract transfer Chart Code caches.")
    parser.add_argument("--config", type=Path, default=Path("configs/charxiv_qwen3vl_2b_vlm.yaml"))
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/chart_codes/transfer_qwen"))
    parser.add_argument("--status-root", type=Path, default=Path("outputs/transfer_qwen_full"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    extractor_cfg = config.get("extractor", {})
    model_cfg = config.get("model", {})
    extractor = ChartCodeExtractor(
        backend=extractor_cfg.get("backend", "qwen_vl_json"),
        model_name=extractor_cfg.get("model_name", model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct")),
        local_files_only=bool(extractor_cfg.get("local_files_only", True)),
        max_new_tokens=int(extractor_cfg.get("max_new_tokens", 2048)),
    )
    cache = ChartCodeCache(args.cache_root / args.dataset)
    rows = unique_images(args.manifest_dir, max_images=args.max_images)
    completed = 0
    failed: list[dict[str, str]] = []
    for example in rows:
        chart_id = f"{args.dataset}_{example['image_id']}"
        if args.resume and cache.load(chart_id) is not None:
            completed += 1
            continue
        try:
            chart_code = extractor.extract(
                example["image_path"],
                chart_id=chart_id,
                chart_type=None,
                metadata={
                    "dataset": args.dataset,
                    "source_hf_id": example.get("source_hf_id"),
                    "split": example.get("split"),
                    "row_index": example.get("row_index"),
                    "checksum": example.get("checksum"),
                },
            )
            cache.save(chart_code)
            completed += 1
        except Exception as exc:
            failed.append({"chart_id": chart_id, "error": str(exc)})
    status = {
        "dataset": args.dataset,
        "status": "complete" if completed == len(rows) and not failed else "partial",
        "requested_images": len(rows),
        "completed_images": completed,
        "failed_images": len(failed),
        "failures": failed[:100],
        "cache": str(args.cache_root / args.dataset),
    }
    write_json(args.status_root / args.dataset / "chart_code_extraction_status.json", status)
    print(json.dumps({k: v for k, v in status.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
