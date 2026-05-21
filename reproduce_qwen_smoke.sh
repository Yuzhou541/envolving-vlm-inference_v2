#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV_NAME:-vlm}"
LIMIT="${QWEN_LIMIT:-1}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  for candidate in conda.exe /mnt/c/Users/ROG/anaconda3/Scripts/conda.exe; do
    if command -v "${candidate}" >/dev/null 2>&1 || [ -x "${candidate}" ]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi

cd "${ROOT_DIR}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/extract_chart_codes.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --split validation \
  --output data/cache/chart_codes/charxiv_qwen_vl \
  --limit "${LIMIT}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --method full_evochartcode \
  --split validation \
  --limit "${LIMIT}" \
  --output outputs/runs/qwen_full_evochartcode_smoke.json \
  --save-predictions outputs/runs/qwen_full_evochartcode_smoke_predictions.json
