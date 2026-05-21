"""Build NeurIPS-style LaTeX paper assets from completed EvoChartCode outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(path: Path, key: str, default: Any = "n/a") -> Any:
    return read_json(path).get(key, default)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_main() -> str:
    rows = [
        ("Metadata/code-only", Path("outputs/runs/code_only_validation.json")),
        ("Qwen full EvoChartCode", Path("outputs/runs/qwen_full_evochartcode_validation128.json")),
    ]
    body = "\n".join(
        f"{name} & {metric(path, 'num_examples')} & {fmt(metric(path, 'exact_match'))} & "
        f"{fmt(metric(path, 'relaxed_numeric'))} & {fmt(metric(path, 'invalid_rate'))} & "
        f"{fmt(metric(path, 'mean_latency'))} \\\\"
        for name, path in rows
    )
    return "\\begin{tabular}{lrrrrr}\nMethod & N & EM & Relaxed & Invalid & Latency \\\\\n\\hline\n" + body + "\n\\end{tabular}\n"


def table_quality() -> str:
    rows = [
        ("Metadata", Path("outputs/quality/chartcode_300.json")),
        ("Qwen", Path("outputs/quality/chartcode_qwen_validation32.json")),
    ]
    body = "\n".join(
        f"{name} & {metric(path, 'num_charts')} & {fmt(metric(path, 'mean_quality_score'))} & "
        f"{fmt(metric(path, 'axis_evidence_present_rate'))} & {fmt(metric(path, 'series_or_mark_present_rate'))} \\\\"
        for name, path in rows
    )
    return "\\begin{tabular}{lrrrr}\nBackend & Charts & Quality & Axis evidence & Series/marks \\\\\n\\hline\n" + body + "\n\\end{tabular}\n"


def table_transfer() -> str:
    rows = []
    for dataset in ["ChartQA", "PlotQA", "DVQA", "FigureQA"]:
        qwen_path = Path("outputs/transfer_qwen_full") / dataset / "metrics.json"
        sample_path = Path("outputs/transfer") / f"{dataset.lower()}_code_only.json"
        path = qwen_path if qwen_path.exists() else sample_path
        rows.append((dataset, path))
    body = "\n".join(
        f"{name} & {metric(path, 'status', 'sample')} & {metric(path, 'num_examples')} & "
        f"{fmt(metric(path, 'exact_match'))} & {fmt(metric(path, 'relaxed_numeric'))} \\\\"
        for name, path in rows
    )
    return "\\begin{tabular}{llrrr}\nDataset & Status & N & EM & Relaxed \\\\\n\\hline\n" + body + "\n\\end{tabular}\n"


def write_pdf_figure(path: Path, title: str, boxes: list[str]):
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    width, height = landscape(letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 45, title)
    box_w = 120
    box_h = 46
    gap = 32
    start_x = (width - (len(boxes) * box_w + (len(boxes) - 1) * gap)) / 2
    y = height / 2 - box_h / 2
    for index, label in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        c.rect(x, y, box_w, box_h)
        c.setFont("Helvetica", 11)
        c.drawCentredString(x + box_w / 2, y + box_h / 2 - 4, label)
        if index < len(boxes) - 1:
            c.line(x + box_w, y + box_h / 2, x + box_w + gap, y + box_h / 2)
            c.line(x + box_w + gap - 8, y + box_h / 2 + 4, x + box_w + gap, y + box_h / 2)
            c.line(x + box_w + gap - 8, y + box_h / 2 - 4, x + box_w + gap, y + box_h / 2)
    c.showPage()
    c.save()


def write_figures(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        out_dir / "pipeline.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="160">
<rect width="900" height="160" fill="white"/>
<g font-family="Arial" font-size="18" text-anchor="middle">
<rect x="20" y="50" width="130" height="50" fill="#e8f1ff" stroke="#333"/><text x="85" y="82">Chart image</text>
<rect x="190" y="50" width="130" height="50" fill="#e7f7ec" stroke="#333"/><text x="255" y="82">Chart Code</text>
<rect x="360" y="50" width="130" height="50" fill="#fff4d8" stroke="#333"/><text x="425" y="82">Selector</text>
<rect x="530" y="50" width="130" height="50" fill="#f6e8ff" stroke="#333"/><text x="595" y="82">Reasoner</text>
<rect x="700" y="50" width="130" height="50" fill="#ffe8e8" stroke="#333"/><text x="765" y="82">Verifier</text>
<path d="M150 75 H190 M320 75 H360 M490 75 H530 M660 75 H700 M830 75 H880" stroke="#333" marker-end="url(#a)"/>
</g><defs><marker id="a" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>
</svg>
""",
    )
    write_text(
        out_dir / "evolution.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="160">
<rect width="900" height="160" fill="white"/>
<g font-family="Arial" font-size="17" text-anchor="middle">
<rect x="30" y="45" width="150" height="60" fill="#eef" stroke="#333"/><text x="105" y="80">Local Qwen Coder</text>
<rect x="220" y="45" width="150" height="60" fill="#efe" stroke="#333"/><text x="295" y="80">Unified diff</text>
<rect x="410" y="45" width="150" height="60" fill="#ffe" stroke="#333"/><text x="485" y="80">Throwaway eval</text>
<rect x="600" y="45" width="150" height="60" fill="#fee" stroke="#333"/><text x="675" y="80">MAP-Elites</text>
<path d="M180 75 H220 M370 75 H410 M560 75 H600 M750 75 H850" stroke="#333" marker-end="url(#a)"/>
</g><defs><marker id="a" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>
</svg>
""",
    )
    write_pdf_figure(out_dir / "pipeline.pdf", "EvoChartCode Pipeline", ["Chart image", "Chart Code", "Selector", "Reasoner", "Verifier"])
    write_pdf_figure(out_dir / "evolution.pdf", "Source Mutation Evolution", ["Local Qwen", "Unified diff", "Smoke eval", "MAP-Elites"])


def write_neurips_style(out_dir: Path):
    write_text(
        out_dir / "neurips_2026.sty",
        r"""
\ProvidesPackage{neurips_2026}[minimal local style]
\DeclareOption*{}
\ProcessOptions\relax
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{natbib}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
""".strip()
        + "\n",
    )


def build_tex(out_dir: Path):
    main_table = table_main()
    quality_table = table_quality()
    transfer_table = table_transfer()
    tex = rf"""
\documentclass{{article}}
\usepackage[preprint]{{neurips_2026}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\title{{EvoChartCode: Evolution-Guided Chart Code Induction for Reliable Chart Reasoning}}
\author{{Anonymous}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Vision-language models remain brittle on chart question answering because perception, reasoning, formatting, and abstention are entangled in a single generation. EvoChartCode introduces explicit Chart Code as an intermediate representation, then uses question-conditioned selection, code-grounded reasoning, verification, and evolution to make chart reasoning inspectable. This report is generated only from completed local artifacts and marks incomplete full-scale jobs as incomplete rather than inferring results.
\end{{abstract}}
\section{{Method}}
EvoChartCode maps a chart image to validated Chart Code containing layout, axes, legends, colorbars, series, derived relations, uncertainty, and provenance. A selector exposes relevant evidence to a code-only or image-plus-code reasoner, and a verifier checks support before normalization.
\begin{{figure}}[h]\centering\includegraphics[width=0.95\linewidth]{{figures/pipeline.pdf}}\caption{{EvoChartCode pipeline.}}\end{{figure}}
\section{{Main Results}}
\begin{{table}}[h]\centering
{main_table}
\caption{{CharXiv local validation results.}}
\end{{table}}
\begin{{table}}[h]\centering
{quality_table}
\caption{{Chart Code quality results.}}
\end{{table}}
\section{{Transfer}}
\begin{{table}}[h]\centering
{transfer_table}
\caption{{Transfer results. Full Qwen metrics are used when completed; otherwise existing sample outputs are labeled by status.}}
\end{{table}}
\section{{Evolution}}
\begin{{figure}}[h]\centering\includegraphics[width=0.95\linewidth]{{figures/evolution.pdf}}\caption{{Local-Qwen source-code mutation loop.}}\end{{figure}}
\section{{Limitations}}
Full benchmark-size transfer, multi-seed Qwen confidence intervals, and source-mutation evolution are checkpointed jobs. The paper should claim only completed measured results in the output JSON files.
\end{{document}}
"""
    write_text(out_dir / "main.tex", tex.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build NeurIPS-style paper artifacts.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/neurips"))
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    fig_dir = args.out_dir / "figures"
    write_figures(fig_dir)
    write_neurips_style(args.out_dir)
    build_tex(args.out_dir)
    result = {"tex": str(args.out_dir / "main.tex"), "pdf": None, "compiled": False}
    if args.compile:
        if shutil.which("pdflatex"):
            proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=args.out_dir, text=True, capture_output=True)
            result["compiled"] = proc.returncode == 0
            result["pdf"] = str(args.out_dir / "main.pdf") if proc.returncode == 0 else None
            write_text(args.out_dir / "pdflatex.log", proc.stdout + "\n" + proc.stderr)
        else:
            result["error"] = "pdflatex not found"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
