"""Deterministic image evidence helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image_size(image_path: str | Path) -> dict[str, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return {"width": int(width), "height": int(height)}


def detect_plot_area_bbox(image_path: str | Path) -> list[int] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    height, width = gray.shape
    pad_x = max(2, width // 200)
    pad_y = max(2, height // 200)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    ]


def dominant_colors(image_path: str | Path, k: int = 5) -> list[str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape((-1, 3)).astype(np.float32)
    sample_step = max(1, len(pixels) // 20000)
    pixels = pixels[::sample_step]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    colors = []
    for idx in order:
        r, g, b = centers[idx].astype(int)
        if min(r, g, b) > 245:
            continue
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors
