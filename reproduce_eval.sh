#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMIT="${LIMIT:-128}"
OUT_DIR="outputs/runs"
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
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method code_only \
  --split validation \
  --limit "${LIMIT}" \
  --output "${OUT_DIR}/code_only_validation.json" \
  --save-predictions "${OUT_DIR}/code_only_validation_predictions.json"
