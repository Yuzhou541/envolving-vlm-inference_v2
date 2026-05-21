"""Export paper-style markdown tables from run metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export markdown tables from metrics JSON files.")
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "exact_match" not in data:
            continue
        rows.append(
            {
                "method": data.get("method", path.stem),
                "split": data.get("split", ""),
                "exact_match": data["exact_match"],
                "relaxed_numeric": data.get("relaxed_numeric", 0.0),
                "na_f1": data.get("na_f1", 0.0),
                "invalid_rate": data.get("invalid_rate", 0.0),
                "mean_latency": data.get("mean_latency", 0.0),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    header = "| Method | Split | EM | Relaxed Numeric | NA-F1 | Invalid | Mean Latency |\n"
    header += "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    lines = [
        f"| {row['method']} | {row['split']} | {row['exact_match']:.4f} | {row['relaxed_numeric']:.4f} | {row['na_f1']:.4f} | {row['invalid_rate']:.4f} | {row['mean_latency']:.4f} |"
        for row in rows
    ]
    (args.out / "main_results.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "main_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
