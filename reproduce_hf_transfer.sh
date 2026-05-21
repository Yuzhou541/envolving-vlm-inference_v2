#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV_NAME:-vlm}"
LIMIT="${TRANSFER_LIMIT:-32}"
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
mkdir -p data/transfer_manifests outputs/transfer

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/materialize_hf_transfer.py \
  --dataset ChartQA \
  --hf-id HuggingFaceM4/ChartQA \
  --split train \
  --limit "${LIMIT}" \
  --manifest data/transfer_manifests/chartqa.json

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/materialize_hf_transfer.py \
  --dataset DVQA \
  --hf-id sionic-ai/dvqa \
  --split train \
  --limit "${LIMIT}" \
  --manifest data/transfer_manifests/dvqa.json

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/materialize_hf_transfer.py \
  --dataset FigureQA \
  --hf-id sionic-ai/figureqa \
  --split train \
  --limit "${LIMIT}" \
  --manifest data/transfer_manifests/figureqa.json

"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/materialize_hf_transfer.py \
  --dataset PlotQA \
  --hf-id achang/plot_qa \
  --split train \
  --limit "${LIMIT}" \
  --manifest data/transfer_manifests/plotqa.json

for dataset in chartqa plotqa dvqa figureqa; do
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/run_transfer_eval.py \
    --config configs/charxiv_qwen3vl_2b.yaml \
    --manifest "data/transfer_manifests/${dataset}.json" \
    --method code_only \
    --output "outputs/transfer/${dataset}_code_only.json"
done
