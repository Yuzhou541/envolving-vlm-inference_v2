#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMIT="${LIMIT:-128}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cv2
import pydantic
import yaml
PY
then
  for candidate in python3 python.exe py.exe; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" - <<'PY' >/dev/null 2>&1
import cv2
import pydantic
import yaml
PY
    then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/prepare_charxiv.py \
  --charxiv_root charxiv \
  --out data/splits/charxiv_chart_level_split.json

"${PYTHON_BIN}" scripts/extract_chart_codes.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --split validation \
  --output data/cache/chart_codes/charxiv \
  --limit "${LIMIT}"
