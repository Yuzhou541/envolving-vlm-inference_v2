# EvoChartCode: Evolution-Guided Chart Code Induction for Reliable Chart Reasoning

## Abstract

Vision-language models have made rapid progress on chart understanding, yet direct image-to-answer inference remains brittle because perception, reasoning, answer formatting, and abstention are entangled in a single generation step. EvoChartCode converts a chart image into Chart Code, a structured representation containing layout, axes, visual marks, series, derived relations, uncertainty, and provenance. A question-conditioned selector exposes only relevant evidence to a code-grounded reasoner, while a verifier checks that the final answer is supported by extracted chart evidence. The implementation in this repository provides a CharXiv-first reproducible path with extraction, evaluation, policy evolution, ablations, and paper table/figure export.

## Contributions

1. Chart Code schema for explicit chart layout, axis, legend, colorbar, series, mark, derived relation, uncertainty, and provenance fields.
2. EvoChartCode pipeline for extraction, schema validation, routing, code selection, reasoning, verification, and normalization.
3. Verification-guided policy evolution with archive and MAP-Elites views.
4. Reproducible CharXiv evaluation workflow with exact match, relaxed numeric accuracy, Not Applicable F1, invalid-output rate, latency, tables, and figures.

## Current Experimental Slice

The checked-in reproducible run uses metadata/CV extraction for deterministic local execution. It is intended as the fast validation path for the codebase. A model-backed local GPU experiment has also been completed with cached `Qwen/Qwen3-VL-2B-Instruct` weights: 32 CharXiv validation charts were extracted into Chart Code and evaluated over 128 descriptive validation queries with image+code reasoning.

## Reproduction

```bash
bash reproduce_main.sh
```

Generated artifacts:

- `data/splits/charxiv_chart_level_split.json`
- `data/cache/chart_codes/charxiv/*.json`
- `data/cache/chart_codes/charxiv_qwen_vl/*.json`
- `outputs/runs/code_only_validation.json`
- `outputs/runs/ablation_no_verifier.json`
- `outputs/runs/qwen_full_evochartcode_smoke.json`
- `outputs/runs/qwen_full_evochartcode_validation128.json`
- `outputs/evolution_small/*.json`
- `outputs/quality/chartcode_300.json`
- `outputs/quality/chartcode_qwen_validation32.json`
- `outputs/analysis/run_analysis.json`
- `outputs/transfer/*.json`
- `paper/tables/main_results.md`
- `paper/figures/accuracy_latency.svg`
- `paper/evochartcode_report.pdf`

## Status Of Requested Full Experiments

Qwen-backed extraction and full EvoChartCode GPU evaluation have been run on 32 CharXiv validation charts / 128 descriptive validation queries with cached Qwen weights. The run reports exact match 0.53125, relaxed numeric accuracy 0.578125, invalid rate 0.0, Not Applicable F1 0.4, mean latency 13.28 seconds, and p95 latency 16.10 seconds. Transfer preparation/evaluation has been run on public Hugging Face samples for ChartQA, PlotQA, DVQA, and FigureQA.

The proposal-scale workflow has been implemented and smoke-tested. Full ChartQA test, PlotQA test, DVQA public train, and FigureQA public train manifests are materialized. Full Qwen transfer evaluation, five-seed Qwen evaluation, and full Qwen ablations are checkpointed long-running GPU jobs; they should not be claimed as final results until `outputs/proposal_scale_status.json` and the corresponding output directories show completed status. A minimal verifier source mutation has been validated and promoted.
