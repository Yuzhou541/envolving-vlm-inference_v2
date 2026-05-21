# Day 6 Evolve Instruct

Planned day: 2026-03-22
Completed on: 2026-03-22
Re-verified on: 2026-03-22

## Objective

Scale the Instruct evolution search from the tiny proxy subset to the full 128-sample local dev set, keep explicit elites, track a Pareto frontier for accuracy versus speed, and freeze the best discovered evolved inference code as `evolved_instruct.py`.

## Framework Upgrades

The Day 5 prototype in `evolve_instruct.py` was extended so the Day 6 run includes:

- parent error-profile summaries in the mutation prompt
- per-run archived candidate modules under `evolution_runs/<run_name>/candidate_modules`
- explicit elite retention via `--elite-size`
- Pareto frontier tracking and saved `pareto_frontier.json`
- generation-by-generation population snapshots and `generation_summaries.jsonl`
- hybrid search support with both local-LLM and heuristic mutations
- a cleaned live heuristic mutation path so future reruns do not use the old broken Unicode replacement behavior

## Canonical Day 6 Run

Command:

```powershell
conda run -n vlm python evolve_instruct.py `
  --run-name instruct_day6_full `
  --num-samples 128 `
  --generations 4 `
  --candidates-per-generation 5 `
  --population-size 8 `
  --elite-size 3 `
  --mutation-backend hybrid
```

Saved run directory:

- `evolution_runs/instruct_day6_full`

## Final Evolved Result

Frozen best evolved module:

- `evolved_instruct.py`

Verification run:

```powershell
conda run -n vlm python evaluate.py evolved_instruct --hf-offline `
  --num-samples 128 `
  --output evolved_instruct_day6_final.json `
  --save-responses evolved_instruct_day6_final_queries.json
```

Final verified metrics:

- `accuracy = 0.6640625`
- `avg_time_per_query = 0.34761632792651653 s`
- `num_errors = 0`

Reference manual baseline from Day 3:

- `manual_instruct_day3_final.json`
- `accuracy = 0.6328125`
- `avg_time_per_query = 0.35397810116410255 s`

Net change versus the manual Instruct baseline:

- `+0.03125` absolute accuracy
- `+4 / 128` more exact-match correct answers
- `0` regressions on the local 128-sample dev set
- faster by about `0.0064 s/query`

## Best Candidates Found In The Canonical Full Run

Accuracy-best / exported best candidate from `evolution_runs/instruct_day6_full/summary.json`:

- candidate: `g00_c005`
- mutation summary: strengthened global `Not Applicable` behavior
- accuracy: `0.6640625`
- avg/query during the search: `0.40914330817759037 s`

Speed-leaning Pareto candidate:

- candidate: `g02_c015`
- mutation summary: simplified the no-prefix instruction
- accuracy: `0.65625`
- avg/query during the search: `0.3535063248127699 s`

The run-level Pareto frontier therefore preserved two useful tradeoffs:

- higher-accuracy evolved code: `g00_c005`
- faster search-time candidate with still-improved accuracy: `g02_c015`

## Re-Audit Result

A fresh full rerun with the same Day 6 settings was executed on 2026-03-18 to check stability. That rerun peaked at:

- candidate: `g00_c004`
- accuracy: `0.6484375`
- avg/query during the rerun: `0.3481218107044697 s`

Because that rerun underperformed the existing canonical best on accuracy, it was not promoted. The repository therefore keeps:

- the stronger canonical full-run directory: `evolution_runs/instruct_day6_full`
- the stronger exported artifact: `evolved_instruct.py`

This is the right final Day 6 state because the deliverable should freeze the best discovered inference code, not merely the most recent rerun.

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran `evolved_instruct.py` through the validated `bash reproduce.sh` workflow.

Canonical fresh-process metrics from `repro_outputs/summary.md`:

- accuracy: `0.6640625`
- avg/query: `0.41770545579493046 s`
- errors: `0`

An additional full evolution rerun was also executed with:

```powershell
conda run -n vlm python evolve_instruct.py `
  --run-name instruct_final_audit `
  --export-best evolution_tmp/evolved_instruct_final_audit.py
```

That fresh rerun completed cleanly but peaked at `0.65625` accuracy in `evolution_runs/instruct_final_audit/summary.json`, so the stronger frozen artifact remained `evolved_instruct.py`.

## What Improved

Comparing `manual_instruct_day3_final_queries.json` against `evolved_instruct_day6_final_queries.json`, the evolved winner fixed four dev-set questions and introduced no new regressions.

The fixed cases were concentrated in:

- legend false positives converted to `Not Applicable`
- colorbar false positives converted to `Not Applicable`
- one tick-difference answer corrected

## Day 6 Status

Day 6 is complete and README-aligned for the Instruct track:

- the search was scaled to the full 128-sample dev set
- every mutation was logged
- elites were kept explicitly
- a Pareto frontier was tracked and saved
- `evolved_instruct.py` is a real evolved variant, meaningfully different from `manual_instruct.py`
- the final frozen evolved file is both more accurate and slightly faster than the manual Instruct baseline on the local dev benchmark
