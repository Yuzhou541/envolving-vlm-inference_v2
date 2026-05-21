#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import pydantic
import yaml
PY
then
  for candidate in python3 python.exe py.exe; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" - <<'PY' >/dev/null 2>&1
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

"${PYTHON_BIN}" scripts/run_multiseed.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method code_only \
  --split validation \
  --limit "${MULTISEED_LIMIT:-128}" \
  --seeds "${SEEDS:-0,1,2}" \
  --output outputs/analysis/multiseed_code_only.json

"${PYTHON_BIN}" scripts/analyze_runs.py \
  --runs outputs/runs \
  --output outputs/analysis/run_analysis.json
