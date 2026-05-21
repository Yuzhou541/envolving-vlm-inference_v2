# Limitations

EvoChartCode makes chart reasoning more inspectable, but it does not solve chart understanding.

- Chart Code extraction can be wrong when OCR, legends, axis scales, or marks are visually ambiguous.
- The deterministic metadata backend exists for reproducible local testing; paper-quality experiments should use `qwen_vl_json` extraction and cached Chart Code.
- Code-only reasoning is fast and interpretable, but cannot recover information absent from Chart Code.
- Image+code reasoning is stronger but inherits VLM perception failures and GPU memory limits.
- Verification now covers the implemented question families in the routing layer, including title, axis labels and ranges, tick values and counts, legends, colorbars, chart type, subplot count, line count, point/bar values, trends, extrema, turning points, intersections, comparisons, rankings, differences, ratios, Not Applicable, and open-ended answers. It is still rule-based and can miss unsupported answers when the extracted evidence is incomplete.
- Transfer results are currently bounded public Hugging Face samples for ChartQA, PlotQA, DVQA, and FigureQA, not full benchmark-size paper results.
- The paper PDF in `paper/evochartcode_report.pdf` is a generated report from local experiments, not a final camera-ready NeurIPS submission.
- Proposal-scale runners now exist for full transfer, Qwen multi-seed evaluation, Qwen ablations, local-Qwen source mutation, and NeurIPS-style LaTeX generation. The full GPU jobs are checkpointed but not completed in the current local output set.

## Current Local Status

- Qwen-backed extraction and full image+code reasoning have been run on 32 CharXiv validation charts / 128 descriptive validation queries with cached `Qwen/Qwen3-VL-2B-Instruct` weights and CUDA.
- The 128-query Qwen run reports exact match `0.53125`, relaxed numeric accuracy `0.578125`, invalid rate `0.0`, Not Applicable F1 `0.4`, mean latency `13.28s`, and p95 latency `16.10s`.
- ChartQA, PlotQA, DVQA, and FigureQA transfer samples were materialized from public Hugging Face datasets into `data/external` and evaluated with the metadata/code-only transfer path; each sample run contains 32 examples.
- Full official/public manifests were materialized for ChartQA test (2,500 examples), PlotQA test (1,000 examples), DVQA train/public (2,325,316 QA examples from 200,000 images), and FigureQA train/public (1,327,368 QA examples from 100,000 images) under `data/transfer_full_manifests`.
- All four transfer datasets passed Qwen transfer smoke checks with two examples per dataset under `outputs/transfer_qwen_smoke`.
- Qwen-backed ablation smoke completed for all eight configured ablations with limit 8 under `outputs/runs_qwen_ablation_smoke`.
- Qwen multi-seed smoke completed for seeds 0 and 1 with limit 2 under `outputs/analysis/qwen_multiseed_smoke`.
- Local-Qwen source mutation smoke ran with Qwen2.5-Coder-7B and Qwen2.5-Coder-3B. The generated local-Qwen candidates were rejected by validation, and a minimal verifier source mutation was separately validated and promoted at `outputs/source_mutation_evolution/promoted_candidate.json`.
- A NeurIPS-style LaTeX artifact now compiles at `paper/neurips/main.pdf`.
- ChartCode-300 was run with the metadata backend on 300 CharXiv validation charts. A Qwen-backed quality pass was run on the 32 cached Qwen validation charts.
- Multi-seed/bootstrap and Pareto analysis were run for the metadata/code-only path. Multi-seed Qwen GPU evaluation remains a compute-scaling task.
- The metadata/CV backend is useful for reproducibility and pipeline testing, but it is not a substitute for paper-quality Chart Code extraction.
