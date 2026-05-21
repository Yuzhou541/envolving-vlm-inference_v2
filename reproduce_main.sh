#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

bash reproduce_extract.sh
bash reproduce_eval.sh
bash reproduce_evolution_small.sh
bash reproduce_ablation.sh
bash reproduce_quality.sh
bash reproduce_analysis.sh
bash reproduce_hf_transfer.sh
bash reproduce_tables.sh
bash reproduce_pdf.sh
