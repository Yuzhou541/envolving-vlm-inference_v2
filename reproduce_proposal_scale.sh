#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_PYTHON="${CONDA_PYTHON:-conda run -n vlm python}"

echo "[1/8] Materialize full transfer manifests"
$CONDA_PYTHON scripts/materialize_full_transfer.py --dataset ChartQA --out-root data/external_full --manifest-root data/transfer_full_manifests --shard-size 1000 --resume
$CONDA_PYTHON scripts/materialize_full_transfer.py --dataset PlotQA --out-root data/external_full --manifest-root data/transfer_full_manifests --shard-size 1000 --resume
$CONDA_PYTHON scripts/materialize_full_transfer.py --dataset DVQA --out-root data/external_full --manifest-root data/transfer_full_manifests --shard-size 1000 --resume
$CONDA_PYTHON scripts/materialize_full_transfer.py --dataset FigureQA --out-root data/external_full --manifest-root data/transfer_full_manifests --shard-size 1000 --resume

echo "[2/8] Extract transfer Chart Code caches"
for dataset in ChartQA PlotQA DVQA FigureQA; do
  $CONDA_PYTHON scripts/extract_transfer_chart_codes.py \
    --config configs/charxiv_qwen3vl_2b_vlm.yaml \
    --manifest-dir "data/transfer_full_manifests/${dataset}" \
    --dataset "${dataset}" \
    --cache-root data/cache/chart_codes/transfer_qwen \
    --status-root outputs/transfer_qwen_full \
    --resume
done

echo "[3/8] Evaluate full transfer"
for dataset in ChartQA PlotQA DVQA FigureQA; do
  $CONDA_PYTHON scripts/run_transfer_qwen_full.py \
    --config configs/charxiv_qwen3vl_2b_vlm.yaml \
    --manifest-dir "data/transfer_full_manifests/${dataset}" \
    --dataset "${dataset}" \
    --output-root outputs/transfer_qwen_full \
    --cache-root data/cache/chart_codes/transfer_qwen \
    --resume
done

echo "[4/8] Run Qwen multi-seed CharXiv evaluation"
$CONDA_PYTHON scripts/run_qwen_multiseed.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --method full_evochartcode \
  --split validation \
  --seeds "0,1,2,3,4" \
  --output-dir outputs/analysis/qwen_multiseed \
  --resume

echo "[5/8] Run Qwen ablation suite"
$CONDA_PYTHON scripts/run_qwen_ablation_suite.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --split validation \
  --output-dir outputs/runs_qwen_ablation \
  --resume

echo "[6/8] Run local-Qwen source mutation evolution"
$CONDA_PYTHON scripts/run_source_mutation_evolution.py \
  --generations "${MUTATION_GENERATIONS:-8}" \
  --candidates-per-generation "${MUTATION_CANDIDATES:-4}" \
  --smoke-limit "${MUTATION_SMOKE_LIMIT:-32}" \
  --output-dir outputs/source_mutation_evolution

echo "[7/8] Refresh generated reports"
$PYTHON_BIN scripts/analyze_runs.py --runs outputs/runs --output outputs/analysis/run_analysis.json
$PYTHON_BIN scripts/export_tables.py --runs outputs/runs --out paper/tables
$PYTHON_BIN scripts/export_figures.py --runs outputs/runs --out paper/figures
$PYTHON_BIN scripts/build_neurips_paper.py --out-dir paper/neurips --compile

echo "[8/8] Compile checks"
$PYTHON_BIN -m compileall evochartcode scripts
