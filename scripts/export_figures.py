"""Export simple paper figures from run metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export accuracy-latency scatter figure.")
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "exact_match" in data and "mean_latency" in data:
            rows.append((data.get("method", path.stem), float(data["mean_latency"]), float(data["exact_match"])))

    args.out.mkdir(parents=True, exist_ok=True)
    width, height = 720, 420
    margin = 60
    if rows:
        max_x = max(x for _, x, _ in rows) or 1.0
    else:
        max_x = 1.0
    points = []
    for label, x, y in rows:
        px = margin + (width - 2 * margin) * (x / max_x)
        py = height - margin - (height - 2 * margin) * y
        points.append((label, px, py, x, y))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="Arial" font-size="14">Mean latency (s)</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="14">Exact match</text>',
    ]
    for label, px, py, x, y in points:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#2563eb"/>')
        svg.append(f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-family="Arial" font-size="12">{label}</text>')
    svg.append("</svg>")
    (args.out / "accuracy_latency.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"points": len(points), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
