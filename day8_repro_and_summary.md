# Day 8 Reproducibility And Report Summary

Planned day: 2026-03-24
Completed on: 2026-03-24
Re-verified on: 2026-03-24

## Objective

1. Write `reproduce.sh`.
2. Clean up the evaluation and reproduction workflow.
3. Rerun the key reported numbers from scratch.
4. Summarize the design choices, manual optimizations, evolution design, results, failures, and the main lessons that should feed the final report.

## Final Reproduction Workflow

Submission files involved:

- `reproduce.sh`
- `reproduce.py`
- `evaluate.py`

What changed in the final cleanup:

- `reproduce.sh` now calls a normal helper script instead of piping a heredoc through `conda run`
- this makes the README-required `bash reproduce.sh` path reliable under Git Bash on Windows
- the module loaders now respect offline mode directly via `local_files_only`, so repeated evaluations no longer waste time on Hugging Face network retries

Validated command:

```bash
bash reproduce.sh
```

On this Windows machine, I validated the same command through Git Bash with:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' reproduce.sh
```

Outputs written by the final workflow:

- `repro_outputs/summary.md`
- `repro_outputs/summary.json`
- `repro_outputs/module_aliases.json`
- one JSON metrics file per submitted module

## Final Fresh-Process Results

Canonical final numbers from `repro_outputs/summary.md`:

| Module | Accuracy | Avg/query (s) | Errors | Score | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `starting_scripts` | `0.3828125` | `1.4922516774386168` | `0` | `0.3081999161280692` |  |
| `manual_instruct` | `0.6328125` | `0.38159692101180553` | `0` | `0.6137326539494097` |  |
| `manual_thinking` | `0.5546875` | `0.570147329941392` | `0` | `0.5261801335029304` |  |
| `evolved_instruct` | `0.6640625` | `0.3864690978080034` | `0` | `0.6447390451095998` |  |
| `evolved_thinking` | `0.578125` | `0.5762407723814249` | `0` | `0.5493129613809288` |  |
| `best_accuracy` | `0.6640625` | `0.3864690978080034` | `0` | `0.6447390451095998` | same as `evolved_instruct` |
| `best_speed` | `0.6328125` | `0.38159692101180553` | `0` | `0.6137326539494097` | same as `manual_instruct` |
| `best_overall` | `0.6640625` | `0.3864690978080034` | `0` | `0.6447390451095998` | same as `evolved_instruct` |

Score definition:

`score = accuracy - 0.05 * avg_time_per_query`

## Design Choices To Highlight In The Report

### Manual optimization design

For `manual_instruct.py`:

- answer-only prompting with exact-format constraints
- question-type-aware hints for the dominant CharXiv failure modes
- deterministic post-processing that trims generations down to the exact evaluator string
- conservative `Not Applicable` behavior when the chart element is absent

For `manual_thinking.py`:

- keep the required Thinking model but suppress visible reasoning
- use a strict `<answer>...</answer>` contract
- pre-close the model's visible `<think>` handoff before generation
- extract the final answer aggressively so the evaluator only sees one short line

### Evolution system design

For both evolution systems:

- seed from the best manual version
- use a local LLM mutation operator plus a heuristic mutation bank
- mutate the inference code directly rather than only changing scalar hyperparameters
- validate candidate code before evaluation
- maintain a scored population, explicit elites, and a Pareto frontier
- log prompts, raw mutations, candidate modules, metrics, and generation summaries

Why this is a strong story:

- it follows the core AlphaEvolve pattern of mutate, evaluate, select, and iterate
- it is concrete and reproducible under a small-project time budget
- it optimizes the exact part of the system that matters: the inference program used by the benchmark

## What Worked Best

The biggest gains came from precise output control rather than large architectural rewrites.

Most helpful manual improvements:

- exact-answer prompting
- question-type-specific hints
- strict post-processing
- stronger legend versus colorbar discrimination
- more conservative `Not Applicable` behavior

Most helpful evolved improvements:

- reinforcing conservative rejection of false positives
- tightening legend and colorbar hints around common benchmark confusions
- removing small preprocessing overheads only when accuracy was preserved

## Failures And Negative Results

The final report should keep the negative results explicit:

- the Day 5 proxy Instruct run did not beat the seed
- evolution is stochastic, and the final audit reruns did not beat the strongest previously discovered frozen artifacts
- a fresh Instruct evolution rerun on `2026-03-21` peaked at `0.65625`, below the submitted `0.6640625`
- the Thinking final audit rerun matched the submitted search accuracy but did not produce a stronger standalone exported file

These are useful results, not weaknesses to hide. They show that the search loop is real, that reruns are not cherry-picked, and that the repository freezes the best discovered artifact rather than blindly keeping the newest one.

## What Mattered Most

The strongest concise takeaway for the report is:

1. Start from a naive ~38% baseline.
2. Manual prompt and post-processing optimization delivers the biggest initial jump.
3. AlphaEvolve-style search gives an additional meaningful gain on the Instruct track.
4. Thinking evolution helps accuracy too, but the Instruct track remains the strongest final system.
5. On the local dev benchmark, `evolved_instruct.py` is the best file for accuracy and overall score, while `manual_instruct.py` is the fastest file and therefore becomes `best_speed.py`.

## Day 8 Status

Day 8 is complete for the code-and-results side of the submission:

- the reproduction workflow is now validated end to end
- the final cross-module numbers were rerun from scratch
- the project has a clean report-facing summary of design choices, results, failures, and lessons
- the next remaining step is to turn this material into the final `report.pdf`
