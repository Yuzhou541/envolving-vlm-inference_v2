"""Generate an SVG accuracy-latency scatter plot from the final reproduction metrics."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "repro_outputs" / "summary.json"
OUT_PATH = ROOT / "accuracy_latency_scatter.svg"

LABELS = {
    "starting_scripts": "Baseline",
    "manual_instruct": "Manual Instruct",
    "manual_thinking": "Manual Thinking",
    "evolved_instruct": "Evolved Instruct",
    "evolved_thinking": "Evolved Thinking",
}

COLORS = {
    "starting_scripts": "#6b7280",
    "manual_instruct": "#2563eb",
    "manual_thinking": "#f59e0b",
    "evolved_instruct": "#16a34a",
    "evolved_thinking": "#dc2626",
}

LABEL_OFFSETS = {
    "starting_scripts": (-12, -10),
    "manual_instruct": (12, 18),
    "manual_thinking": (12, 34),
    "evolved_instruct": (12, -10),
    "evolved_thinking": (12, 20),
}


def load_points():
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    points = [row for row in rows if not row.get("aliased_from")]
    return [row for row in points if row["module"] in LABELS]


def scale(value, domain_min, domain_max, range_min, range_max):
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def draw_circle(cx, cy, radius, fill):
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius}" fill="{fill}" stroke="white" stroke-width="2" />'


def draw_square(cx, cy, size, fill):
    half = size / 2
    return (
        f'<rect x="{cx - half:.2f}" y="{cy - half:.2f}" width="{size}" height="{size}" '
        f'fill="{fill}" stroke="white" stroke-width="2" rx="2" />'
    )


def draw_diamond(cx, cy, size, fill):
    half = size / 2
    points = [
        (cx, cy - half),
        (cx + half, cy),
        (cx, cy + half),
        (cx - half, cy),
    ]
    points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{points_str}" fill="{fill}" stroke="white" stroke-width="2" />'


def render():
    points = load_points()

    width = 980
    height = 620
    margin_left = 86
    margin_right = 54
    margin_top = 78
    margin_bottom = 84

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    xs = [point["avg_time_per_query"] for point in points]
    ys = [point["accuracy"] for point in points]

    x_min = min(xs) - 0.05
    x_max = max(xs) + 0.08
    y_min = min(ys) - 0.03
    y_max = max(ys) + 0.03

    x_ticks = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    y_ticks = [0.35, 0.45, 0.55, 0.65]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd" />',
        '<text x="50%" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">Accuracy-Latency Trade-off on Local CharXiv Dev Set</text>',
        '<text x="50%" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">128 descriptive validation queries, greedy decoding, lower latency is better</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d1d5db" stroke-width="1.5" rx="10" />',
    ]

    for tick in x_ticks:
        if x_min <= tick <= x_max:
            x = scale(tick, x_min, x_max, plot_left, plot_right)
            parts.append(
                f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#e5e7eb" stroke-width="1" />'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{tick:.1f}</text>'
            )

    for tick in y_ticks:
        if y_min <= tick <= y_max:
            y = scale(tick, y_min, y_max, plot_bottom, plot_top)
            parts.append(
                f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
            )
            parts.append(
                f'<text x="{plot_left - 16}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#374151">{tick:.2f}</text>'
            )

    parts.extend(
        [
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#6b7280" stroke-width="1.8" />',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#6b7280" stroke-width="1.8" />',
            f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="600" fill="#111827">Latency per query (seconds)</text>',
            f'<text x="28" y="{(plot_top + plot_bottom) / 2:.2f}" text-anchor="middle" transform="rotate(-90 28 {(plot_top + plot_bottom) / 2:.2f})" font-family="Arial, sans-serif" font-size="16" font-weight="600" fill="#111827">Accuracy</text>',
        ]
    )

    for point in points:
        module = point["module"]
        x = scale(point["avg_time_per_query"], x_min, x_max, plot_left, plot_right)
        y = scale(point["accuracy"], y_min, y_max, plot_bottom, plot_top)
        color = COLORS[module]

        if module == "starting_scripts":
            parts.append(draw_square(x, y, 14, color))
        elif module.startswith("evolved_"):
            parts.append(draw_diamond(x, y, 16, color))
        else:
            parts.append(draw_circle(x, y, 7, color))

        dx, dy = LABEL_OFFSETS[module]
        label_x = x + dx
        label_y = y + dy
        anchor = "end" if dx < 0 else "start"
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="14" font-weight="600" fill="#111827">{LABELS[module]}</text>'
        )

    legend_x = width - 248
    legend_y = 94
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="194" height="96" fill="#ffffff" stroke="#d1d5db" rx="10" />')
    parts.append(f'<text x="{legend_x + 16}" y="{legend_y + 24}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">Markers</text>')
    parts.append(draw_square(legend_x + 20, legend_y + 46, 12, "#6b7280"))
    parts.append(f'<text x="{legend_x + 38}" y="{legend_y + 50}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Baseline</text>')
    parts.append(draw_circle(legend_x + 20, legend_y + 68, 6, "#2563eb"))
    parts.append(f'<text x="{legend_x + 38}" y="{legend_y + 72}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Manual variants</text>')
    parts.append(draw_diamond(legend_x + 20, legend_y + 90, 14, "#16a34a"))
    parts.append(f'<text x="{legend_x + 38}" y="{legend_y + 94}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Evolved variants</text>')

    parts.append("</svg>")
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    render()
