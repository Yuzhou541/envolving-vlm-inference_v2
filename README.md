# EvoChartCode

EvoChartCode is a reproducible research codebase for chart reasoning with explicit, verifiable intermediate representations. The pipeline is:

```text
chart image -> Chart Code -> question-conditioned code selection -> reasoner -> verifier -> answer
```

The older AlphaEvolve-style CharXiv submission files are still present (`manual_instruct.py`, `evolved_instruct.py`, `evaluate.py`, and related artifacts). The EvoChartCode implementation lives in `evochartcode/`, with runnable workflows in `scripts/` and configs in `configs/`.

## Setup

```bash
conda env create -f environment.yml
conda activate evochartcode
```

Or install into an existing environment:

```bash
pip install -r requirements.txt
```

## Reproduce The Main Local Workflow

```bash
bash reproduce_main.sh
```

This runs:

```bash
bash reproduce_extract.sh
bash reproduce_eval.sh
bash reproduce_evolution_small.sh
bash reproduce_ablation.sh
bash reproduce_quality.sh
bash reproduce_analysis.sh
bash reproduce_hf_transfer.sh
bash reproduce_tables.sh
bash reproduce_pdf.sh
```

By default the reproduction scripts use the deterministic metadata/CV extractor so they run without downloading model weights. Set `LIMIT=<N>` to control the number of CharXiv examples used by the extract/eval scripts.

On Windows through WSL, the scripts automatically choose an interpreter that can import the required packages. You can override it explicitly:

```bash
PYTHON_BIN=python.exe bash reproduce_main.sh
```

## Core Commands

Prepare chart-level splits:

```bash
python scripts/prepare_charxiv.py \
  --charxiv_root charxiv \
  --out data/splits/charxiv_chart_level_split.json
```

Extract Chart Code:

```bash
python scripts/extract_chart_codes.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --split validation \
  --output data/cache/chart_codes/charxiv
```

Evaluate code-only reasoning:

```bash
python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method code_only \
  --split validation \
  --output outputs/runs/code_only_validation.json
```

Run small policy evolution:

```bash
python scripts/run_evolution.py \
  --config configs/evolution_small.yaml \
  --seed 0
```

Export tables and figures:

```bash
python scripts/export_tables.py --runs outputs/runs --out paper/tables
python scripts/export_figures.py --runs outputs/runs --out paper/figures
```

## Model-Backed Experiments

For Qwen-backed extraction, edit `configs/charxiv_qwen3vl_2b.yaml`:

```yaml
extractor:
  backend: qwen_vl_json
  model_name: Qwen/Qwen3-VL-2B-Instruct
  local_files_only: true
```

For image+code reasoning:

```bash
python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method full_evochartcode \
  --split validation
```

Keep `local_files_only: true` when reproducing from a prepared model cache. Set it to `false` only when you intentionally want Hugging Face downloads.

The repository also includes a GPU-backed smoke path using the cached Qwen weights:

```bash
bash reproduce_qwen_smoke.sh
```

The completed local Qwen experiment used cached `Qwen/Qwen3-VL-2B-Instruct` weights on CUDA and covers 32 CharXiv validation charts / 128 descriptive validation queries:

```bash
conda run -n vlm python scripts/extract_chart_codes.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --split validation \
  --output data/cache/chart_codes/charxiv_qwen_vl \
  --limit 128

conda run -n vlm python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b_vlm.yaml \
  --method full_evochartcode \
  --split validation \
  --limit 128 \
  --output outputs/runs/qwen_full_evochartcode_validation128.json \
  --save-predictions outputs/runs/qwen_full_evochartcode_validation128_predictions.json
```

That run reports exact match `0.53125`, relaxed numeric accuracy `0.578125`, invalid rate `0.0`, Not Applicable F1 `0.4`, mean latency `13.28s`, and p95 latency `16.10s`. Full validation-scale Qwen extraction remains outside `reproduce_main.sh` because it is a GPU-time-heavy path.

## Transfer Datasets

Prepare and evaluate transfer manifests:

```bash
bash reproduce_hf_transfer.sh
```

This materializes bounded public Hugging Face samples for ChartQA, PlotQA, DVQA, and FigureQA into `data/external/` and writes transfer metrics under `outputs/transfer/`. Set `TRANSFER_LIMIT` to control the per-dataset sample count.

If you already have local dataset copies, `bash reproduce_transfer.sh` still supports `CHARTQA_ROOT`, `PLOTQA_ROOT`, `DVQA_ROOT`, and `FIGUREQA_ROOT`.

## Proposal-Scale Checkpointed Workflow

The proposal-scale pipeline is implemented as resumable scripts because the full Qwen transfer and multi-seed GPU jobs can run for days on the local RTX 4090 Laptop GPU:

```bash
bash reproduce_proposal_scale.sh
```

The workflow includes:

- full official-eval transfer manifest materialization,
- Qwen Chart Code extraction for transfer images,
- full image+code Qwen transfer evaluation,
- five-seed Qwen CharXiv evaluation with bootstrap intervals,
- Qwen-backed ablations,
- local-Qwen source-code mutation evolution,
- NeurIPS-style LaTeX paper generation.

Current proposal-scale status is recorded in `outputs/proposal_scale_status.json`. In this workspace, full ChartQA test, PlotQA test, DVQA public train, and FigureQA public train manifests are materialized. All four transfer datasets passed 2-example Qwen transfer smoke checks, Qwen ablation smoke passed for 8 ablations, multi-seed Qwen smoke passed for two seeds, a validated verifier source mutation was promoted, and `paper/neurips/main.pdf` compiles. Full Qwen transfer evaluation, full five-seed Qwen evaluation, and full Qwen ablations remain checkpointed long-running GPU jobs.

## Project Structure

- `evochartcode/schema.py`: Pydantic Chart Code schema.
- `evochartcode/extractor.py`: metadata/CV and Qwen JSON extraction backends.
- `evochartcode/derived.py`: trend, extrema, turning point, and comparison derivation.
- `evochartcode/routing.py`: question routing.
- `evochartcode/selector.py`: question-conditioned Chart Code selection.
- `evochartcode/reasoner.py`: code-only and image+code reasoning modes.
- `evochartcode/verifier.py`: verifier and answer normalizer.
- `evochartcode/evolution.py`: archive, MAP-Elites view, mutation, and policy evolution.
- `scripts/`: reproducible extraction, evaluation, evolution, ablation, table, and figure commands.
- `configs/`: CharXiv, evolution, and ablation configs.
- `MODEL_CARD.md` and `LIMITATIONS.md`: intended use and known limits.

## Legacy Assignment Reproduction

The original VLM inference submission path remains available:

```bash
bash reproduce.sh
python evaluate.py manual_instruct
python evaluate.py evolved_instruct
```
