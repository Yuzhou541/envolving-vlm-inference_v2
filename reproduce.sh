#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/repro_outputs"
ENV_NAME="${CONDA_ENV_NAME:-vlm}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"

mkdir -p "${OUT_DIR}"

MODULES=(
  starting_scripts
  manual_instruct
  manual_thinking
  evolved_instruct
  evolved_thinking
  best_accuracy
  best_speed
  best_overall
)

echo "Writing reproduction outputs to ${OUT_DIR}"
echo "Using conda env: ${ENV_NAME}"
echo "Using num_samples: ${NUM_SAMPLES}"

conda run -n "${ENV_NAME}" python "${ROOT_DIR}/reproduce.py" \
  --root-dir "${ROOT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --env-name "${ENV_NAME}" \
  --num-samples "${NUM_SAMPLES}" \
  "${MODULES[@]}"
