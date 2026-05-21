# Day 2 Manual Instruct

Planned day: 2026-03-18
Completed on: 2026-03-18

## Objective

Build an accuracy-first manual version for `Qwen/Qwen3-VL-2B-Instruct` that improves exact-match accuracy over the Day 1 starter baseline while keeping greedy decoding.

## Final Day 2 File

- `manual_instruct.py`

Core design choices:

1. Stronger answer-only prompting
   - request one short final line only
   - forbid explanations and answer prefixes
   - preserve exact text and numeric formatting
   - normalize missing elements to `Not Applicable`

2. Question-aware post-processing
   - classify the descriptive question type from the question text
   - extract counts, tick values, layout answers, legend names, and yes/no answers with type-specific logic
   - strip verbose generations down to the exact string the evaluator expects

3. Greedy deterministic inference
   - same model family as required by the README
   - same `vlm_inference(image_path, question)` interface as the starter
   - greedy decoding only

## Final Canonical Day 2 Run

Command:

```powershell
python evaluate.py manual_instruct --hf-offline `
  --output manual_instruct_day2_final.json `
  --save-responses manual_instruct_day2_final_queries.json
```

Saved outputs:

- metrics: `manual_instruct_day2_final.json`
- per-query responses: `manual_instruct_day2_final_queries.json`

Result:

- accuracy: `0.6328125`
- evaluated: `128`
- errors: `0`
- total time: `46.12125015258789 s`
- avg/query: `0.3603222668170929 s`

Comparison to the Day 1 baseline:

- baseline accuracy: `0.3828125`
- Day 2 accuracy: `0.6328125`
- absolute accuracy gain: `+0.25`
- baseline avg/query: `1.4759445749223232 s`
- Day 2 avg/query: `0.3603222668170929 s`
- queries fixed vs starter: `35`
- regressions vs starter: `3`

## Final Error Categories

Remaining errors from `manual_instruct_day2_final_queries.json`:

- `colorbar_max`: `9`
- `total_ticks`: `7`
- `colorbar_range`: `3`
- `line_intersection`: `3`
- `legend_count`: `3`
- `x_left_tick`: `3`
- `y_tick_diff`: `3`
- `line_count`: `2`
- `layout`: `2`
- `legend_names`: `2`
- `y_high_tick`: `2`
- `x_tick_diff`: `2`
- each of `title`, `x_label`, `y_label`, `x_right_tick`, `y_low_tick`, and `trend`: `1`

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran `manual_instruct.py` through the validated `bash reproduce.sh` workflow.

Canonical fresh-process metrics from `repro_outputs/summary.md`:

- accuracy: `0.6328125`
- avg/query: `0.41409385576844215 s`
- errors: `0`

This is the final number to compare against `starting_scripts`, `manual_thinking`, and the evolved variants in the report.

## Day 2 Status

Day 2 is complete and clearly successful:

- exact-match accuracy improved substantially over the starter
- the answer formatting is much closer to the evaluator target
- the remaining weaknesses are now categorized for future work
- `manual_instruct.py` became the strong manual baseline for later speed and evolution work
