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
mkdir -p data/transfer_manifests outputs/transfer

"${PYTHON_BIN}" scripts/prepare_chartqa.py --root "${CHARTQA_ROOT:-data/external/ChartQA}" --out data/transfer_manifests/chartqa.json --limit "${TRANSFER_LIMIT:-128}"
"${PYTHON_BIN}" scripts/prepare_plotqa.py --root "${PLOTQA_ROOT:-data/external/PlotQA}" --out data/transfer_manifests/plotqa.json --limit "${TRANSFER_LIMIT:-128}"
"${PYTHON_BIN}" scripts/prepare_dvqa.py --root "${DVQA_ROOT:-data/external/DVQA}" --out data/transfer_manifests/dvqa.json --limit "${TRANSFER_LIMIT:-128}"
"${PYTHON_BIN}" scripts/prepare_figureqa.py --root "${FIGUREQA_ROOT:-data/external/FigureQA}" --out data/transfer_manifests/figureqa.json --limit "${TRANSFER_LIMIT:-128}"

for dataset in chartqa plotqa dvqa figureqa; do
  "${PYTHON_BIN}" scripts/run_transfer_eval.py \
    --config configs/charxiv_qwen3vl_2b.yaml \
    --manifest "data/transfer_manifests/${dataset}.json" \
    --method code_only \
    --output "outputs/transfer/${dataset}_code_only.json"
done
