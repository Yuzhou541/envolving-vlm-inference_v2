# Day 4 Manual Thinking

Planned day: 2026-03-20
Completed on: 2026-03-20
Re-verified on: 2026-03-21

## Objective

Build a manual `Qwen/Qwen3-VL-2B-Thinking` inference module that reuses the working manual-Instruct infrastructure, but prevents the Thinking model from returning long reasoning traces to the evaluator.

## Final Day 4 File

- `manual_thinking.py`

Core design choices:

1. Reuse the proven manual-Instruct structure
   - same `vlm_inference(image_path, question)` API
   - same question-type detection and task-aware post-processing
   - same greedy-decoding requirement

2. Add Thinking-specific output control
   - stronger answer-only prompting
   - explicit `<answer>...</answer>` contract
   - strict final-answer extraction so the returned string stays evaluator-friendly

3. Override the default Thinking chat handoff
   - the stock template opens a visible `<think>` block when `add_generation_prompt=True`
   - this file renders the prompt differently so generation starts directly in the answer section

## Canonical Re-Run

Command:

```powershell
conda run -n vlm python evaluate.py manual_thinking --hf-offline `
  --num-samples 128 `
  --output manual_thinking_day4_final.json `
  --save-responses manual_thinking_day4_final_queries.json
```

Saved outputs:

- metrics: `manual_thinking_day4_final.json`
- per-query responses: `manual_thinking_day4_final_queries.json`

Final verified result:

- accuracy: `0.5546875`
- evaluated: `128`
- errors: `0`
- total time: `66.71908569335938 s`
- avg/query: `0.5212428569793701 s`

## What The Re-Run Confirmed

- the accuracy is stable versus the earlier Day 4 run
- the cached offline timing is better than the earlier colder measurement
- only `1` response in the full rerun exceeded `12` words
- both manual variants are working end to end:
  - `manual_instruct.py`
  - `manual_thinking.py`

Main remaining error categories from the fresh response dump:

- `colorbar_max`: `11`
- `total_ticks`: `7`
- `line_intersection`: `5`
- `y_tick_diff`: `5`
- `legend_names`: `4`
- `legend_count`: `4`
- `colorbar_range`: `3`
- `x_tick_diff`: `3`
- `y_high_tick`: `3`
- `title`: `3`
- `x_right_tick`: `3`

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran `manual_thinking.py` through the validated `bash reproduce.sh` workflow.

Canonical fresh-process metrics from `repro_outputs/summary.md`:

- accuracy: `0.5546875`
- avg/query: `0.6173870526254177 s`
- errors: `0`

This is the final number to use in the report-level four-way matrix.

## Day 4 Status

Day 4 is complete and README-aligned:

- `manual_thinking.py` runs end to end on the local 128-sample development benchmark
- the Thinking model now has a strict final-answer extraction path
- greedy decoding is preserved throughout
