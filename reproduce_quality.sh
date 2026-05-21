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

"${PYTHON_BIN}" scripts/run_chartcode_quality.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --split validation \
  --limit "${CHARTCODE_LIMIT:-300}" \
  --output outputs/quality/chartcode_300.json
