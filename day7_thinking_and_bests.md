# Day 7 Thinking Evolution And Global Bests

Planned day: 2026-03-23
Completed on: 2026-03-23
Re-verified on: 2026-03-23

## Objective

1. Adapt the Instruct evolution framework to `Qwen/Qwen3-VL-2B-Thinking` as `evolve_thinking.py`.
2. Run a smaller but disciplined Thinking evolution search and freeze `evolved_thinking.py`.
3. Choose `best_accuracy.py`, `best_speed.py`, and `best_overall.py`.
4. End the day with a full result matrix across manual versus evolved and Instruct versus Thinking.

## Day 7 Deliverables

- `evolve_thinking.py`
- `evolved_thinking.py`
- `best_accuracy.py`
- `best_speed.py`
- `best_overall.py`

## Canonical Thinking Evolution Run

Command:

```powershell
conda run -n vlm python evolve_thinking.py `
  --run-name thinking_day7_full `
  --num-samples 128 `
  --generations 2 `
  --candidates-per-generation 4 `
  --population-size 6 `
  --elite-size 2 `
  --mutation-backend hybrid `
  --export-best evolved_thinking.py
```

Saved run directory:

- `evolution_runs/thinking_day7_full`

Best candidate from `evolution_runs/thinking_day7_full/summary.json`:

- candidate: `g01_c008`
- source: `heuristic_bank`
- mutation summary: tightened line-question guidance for non-line plot types
- search accuracy: `0.578125`
- search avg/query: `0.607107363641262 s`
- search score: `0.5477696318179369`

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran the submitted modules through the validated `bash reproduce.sh` workflow in `repro_outputs/`.

Canonical four-way matrix from `repro_outputs/summary.md`:

| Variant | Accuracy | Avg/query (s) | Errors | Score |
| --- | ---: | ---: | ---: | ---: |
| Manual Instruct | `0.6328125` | `0.38159692101180553` | `0` | `0.6137326539494097` |
| Evolved Instruct | `0.6640625` | `0.3864690978080034` | `0` | `0.6447390451095998` |
| Manual Thinking | `0.5546875` | `0.570147329941392` | `0` | `0.5261801335029304` |
| Evolved Thinking | `0.578125` | `0.5762407723814249` | `0` | `0.5493129613809288` |

Score definition:

`score = accuracy - 0.05 * avg_time_per_query`

This final matrix is the one to use in the report because all four modules were rerun under the same fresh-process protocol after the offline-loading cleanup.

## Final Audit Rerun

A fresh end-to-end Thinking evolution rerun was also executed on `2026-03-21`:

```powershell
conda run -n vlm python evolve_thinking.py `
  --run-name thinking_final_audit `
  --export-best evolution_tmp/evolved_thinking_final_audit.py
```

What it showed:

- the search still completed cleanly end to end
- it rediscovered the same top-line search accuracy, `0.578125`
- the rerun is saved in `evolution_runs/thinking_final_audit`

Why the frozen submitted file did not change:

- the new rerun did not beat the existing submitted accuracy
- the exported audit candidate underperformed the frozen `evolved_thinking.py` on fresh standalone evaluation
- the repository therefore correctly keeps `evolved_thinking.py` as the stronger final artifact

## Global Best Choices

From the final fresh-process matrix:

- `best_accuracy.py` = copy of `evolved_instruct.py`
- `best_speed.py` = copy of `manual_instruct.py`
- `best_overall.py` = copy of `evolved_instruct.py`

Why the choices split this way:

- `evolved_instruct.py` has the highest accuracy
- `manual_instruct.py` is the fastest submitted file in the final fresh-process rerun
- `evolved_instruct.py` still has the best combined score on the fresh-process comparison

## Day 7 Status

Day 7 is complete and README-aligned:

- the Thinking evolution system exists and runs end to end
- a real evolved Thinking artifact exists as `evolved_thinking.py`
- the global best entry points have been selected
- the project now has a final verified manual-versus-evolved, Instruct-versus-Thinking matrix
