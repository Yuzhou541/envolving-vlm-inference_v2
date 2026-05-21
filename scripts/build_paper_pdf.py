"""Build a paginated project PDF from generated artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def paragraph(canvas, text, x, y, width, leading=13, font="Helvetica", size=10):
    canvas.setFont(font, size)
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if canvas.stringWidth(candidate, font, size) <= width:
            line = candidate
        else:
            canvas.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        canvas.drawString(x, y, line)
        y -= leading
    return y


def main():
    parser = argparse.ArgumentParser(description="Build a project PDF with final local results.")
    parser.add_argument("--out", type=Path, default=Path("paper/evochartcode_report.pdf"))
    parser.add_argument("--summary", type=Path, default=Path("summary.md"))
    parser.add_argument("--table", type=Path, default=Path("paper/tables/main_results.md"))
    parser.add_argument("--quality", type=Path, default=Path("outputs/quality/chartcode_300.json"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/analysis/run_analysis.json"))
    args = parser.parse_args()

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    args.out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.out), pagesize=letter)
    width, height = letter
    margin = 54
    page_bottom = margin
    y = height - margin

    def ensure_space(current_y: float, needed: float = 36) -> float:
        if current_y - needed < page_bottom:
            c.showPage()
            return height - margin
        return current_y

    def draw_heading(text: str, current_y: float, size: int = 12) -> float:
        current_y = ensure_space(current_y, 30)
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, current_y, text[:95])
        return current_y - size - 6

    def draw_text(text: str, current_y: float, size: int = 9, font: str = "Helvetica", leading: int = 12) -> float:
        words = text.split()
        if not words:
            return ensure_space(current_y, leading) - leading
        c.setFont(font, size)
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if c.stringWidth(candidate, font, size) <= width - 2 * margin:
                line = candidate
                continue
            current_y = ensure_space(current_y, leading)
            c.drawString(margin, current_y, line)
            current_y -= leading
            line = word
        if line:
            current_y = ensure_space(current_y, leading)
            c.drawString(margin, current_y, line)
            current_y -= leading
        return current_y

    def draw_markdown(path: Path, current_y: float) -> float:
        if not path.exists():
            return current_y
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if not line:
                current_y = ensure_space(current_y, 10) - 6
                continue
            if line.startswith("# "):
                current_y = draw_heading(line[2:].strip(), current_y, size=15)
            elif line.startswith("## "):
                current_y = draw_heading(line[3:].strip(), current_y, size=12)
            elif line.startswith("### "):
                current_y = draw_heading(line[4:].strip(), current_y, size=10)
            elif line.startswith("|") or line.startswith("```"):
                current_y = draw_text(line, current_y, size=7, font="Courier", leading=9)
            else:
                current_y = draw_text(line, current_y, size=8, leading=10)
        return current_y

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "EvoChartCode: Evolution-Guided Chart Code Induction")
    y -= 24
    c.setFont("Helvetica", 10)
    y = paragraph(
        c,
        "This report summarizes the executable EvoChartCode research package: explicit Chart Code extraction, "
        "question-conditioned selection, code-grounded reasoning, verification, evolution, ablations, and analysis.",
        margin,
        y,
        width - 2 * margin,
    )
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Method")
    y -= 16
    y = paragraph(
        c,
        "The system converts chart images into structured Chart Code, validates the schema, selects evidence for each "
        "question type, reasons over the selected code or image+code input, and verifies the answer against explicit evidence.",
        margin,
        y,
        width - 2 * margin,
    )
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Local Results")
    y -= 16
    if args.table.exists():
        for line in args.table.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("| ---"):
                continue
            y = paragraph(c, line, margin, y, width - 2 * margin, leading=12, size=8)
    if args.quality.exists():
        quality = json.loads(args.quality.read_text(encoding="utf-8"))
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "ChartCode-300")
        y -= 16
        y = paragraph(
            c,
            json.dumps({k: v for k, v in quality.items() if k != "records"}, ensure_ascii=False),
            margin,
            y,
            width - 2 * margin,
            size=8,
        )
    if args.analysis.exists():
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        y -= 8
        y = draw_heading("Bootstrap And Pareto Analysis", y)
        y = draw_text(json.dumps({k: v for k, v in analysis.items() if k != "runs"}, ensure_ascii=True), y, size=8)
    y -= 8
    y = draw_heading("Limitations", y)
    y = draw_text(
        "The deterministic metadata/CV backend is a reproducible local path. Paper-quality claims require model-backed "
        "Qwen extraction and image+code evaluation on available datasets; missing transfer roots are reported rather than inferred.",
        y,
    )
    c.showPage()
    y = height - margin
    y = draw_heading("Full Paper-Writing Summary", y, size=15)
    draw_markdown(args.summary, y)
    c.showPage()
    c.save()
    print(json.dumps({"pdf": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
