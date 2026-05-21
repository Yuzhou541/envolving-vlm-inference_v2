# Report Writing Summary

This file is the single-source summary.

It consolidates:

- the README requirements
- what was actually implemented
- the final verified numbers
- the design decisions worth explaining
- the negative results that should be reported honestly
- a practical outline for the final paper

## 1. Project Goal

The assignment is to optimize a fixed VLM inference pipeline for chart understanding, not to train a new model.

The required model family is:

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-2B-Thinking`

The required deliverables are:

1. Manual optimization for both models.
2. AlphaEvolve-style code evolution for both models.
3. A final comparison across the four variants.
4. `best_accuracy.py`, `best_speed.py`, and `best_overall.py`.
5. A reproducible `bash reproduce.sh` workflow.
6. A short research-style report.

Core constraint from the README:

- use greedy decoding throughout all development and all submitted scripts

## 2. What The Local Development Benchmark Actually Is

The local benchmark in this repo is not the full CharXiv leaderboard evaluation.

It is:

- the first `128` samples from `charxiv/data/descriptive_val.json`
- exact lowercase string match scoring
- no GPT-based grading
- a local speed-and-accuracy development harness

This matters for the report because we should be honest about what our numbers mean:

- the reported results are local dev results
- the held-out official evaluation may differ
- exact-match formatting control matters a lot in this harness

### 2.1 Exact runtime environment

Final verified environment:

- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- CUDA runtime: `12.8`
- PyTorch: `2.10.0+cu128`
- Transformers: `4.57.1`
- Conda env: `vlm`

Important implementation facts for the report:

- no quantization was used in the final submitted files
- final submitted inference scripts operate on one chart-question pair at a time
- the effective batch size in the submitted inference entry points is `1`
- greedy decoding was used throughout

## 3. Final Submission Files

Code files now present and ready:

- `manual_instruct.py`
- `manual_thinking.py`
- `evolved_instruct.py`
- `evolved_thinking.py`
- `evolve_instruct.py`
- `evolve_thinking.py`
- `best_accuracy.py`
- `best_speed.py`
- `best_overall.py`
- `evaluate.py`
- `reproduce.sh`
- `reproduce.py`

Supporting documentation:

- `day1_setup_baseline.md`
- `day2_manual_instruct.md`
- `day3_manual_instruct_speed.md`
- `day4_manual_thinking.md`
- `day5_evolve_instruct.md`
- `day6_evolve_instruct.md`
- `day7_thinking_and_bests.md`
- `day8_repro_and_summary.md`
- `full_pipeline_commands.md`

Canonical run folders:

- `evolution_runs/instruct_proxy_day5_canonical`
- `evolution_runs/instruct_day6_full`
- `evolution_runs/thinking_day7_full`
- `evolution_runs/instruct_final_audit`
- `evolution_runs/thinking_final_audit`

Canonical fresh-process summary:

- `repro_outputs/summary.md`
- `repro_outputs/summary.json`

## 4. Requirement-By-Requirement Compliance

### 4.1 Manual optimization for both models

Satisfied by:

- `manual_instruct.py`
- `manual_thinking.py`

Both keep:

- the required model family
- greedy decoding
- the same `vlm_inference(image_path, question)` API

### 4.2 AlphaEvolve-style evolution system

Satisfied by:

- `evolve_instruct.py`
- `evolve_thinking.py`

Both include:

- LLM-based mutation
- population management
- elite retention
- Pareto frontier tracking
- an evolution loop
- per-candidate evaluation
- logging of prompts, mutations, candidates, and metrics

### 4.3 Best evolved artifacts for both models

Satisfied by:

- `evolved_instruct.py`
- `evolved_thinking.py`

### 4.4 Best accuracy, speed, and overall entry points

Satisfied by:

- `best_accuracy.py` = `evolved_instruct.py`
- `best_speed.py` = `manual_instruct.py`
- `best_overall.py` = `evolved_instruct.py`

This split is important. After the final fresh-process rerun, `manual_instruct.py` was slightly faster than `evolved_instruct.py`, so `best_speed.py` was reassigned accordingly.

### 4.5 Self-contained reproduction script

Satisfied by:

- `reproduce.sh`
- `reproduce.py`

Validated command:

```bash
bash reproduce.sh
```

On this Windows machine it was validated via:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' reproduce.sh
```

## 5. Manual Optimization Design

### Baseline technical description

The starting baseline in `starting_scripts.py` is intentionally naive.

Its main technical characteristics are:

- single-example inference only
- one chart-question pair processed at a time
- generic chat-style prompt with no question-type routing
- no explicit output-control instructions beyond the raw question
- no deterministic post-processing or answer extraction
- greedy generation with `max_new_tokens=128`
- no task-specific handling for legends, colorbars, subplot layouts, or `Not Applicable`

### 5.1 `manual_instruct.py`

High-level idea:

- treat the local benchmark as exact-string chart QA
- reduce verbosity aggressively
- classify the question type
- add targeted hints for the most common error modes
- post-process deterministically to the exact answer form the evaluator wants

Important implementation details:

- one-line answer-only prompt
- forbid explanation prefixes such as `Answer:`
- preserve exact numeric formatting like `10^-6` and `1.00`
- normalize missing elements to `Not Applicable`
- question-aware logic for:
  - title
  - x/y axis labels
  - tick values
  - tick differences
  - line counts and line intersections
  - legend count and legend names
  - colorbar max and colorbar range
  - trend
  - total tick count
  - subplot layout and subplot count

Why it worked:

- the local harness heavily rewards formatting discipline
- chart QA errors often come from confusing nearby text with the requested chart element
- conservative rejection is often better than guessing

### 5.2 `manual_thinking.py`

High-level idea:

- keep the required Thinking model
- prevent visible reasoning from leaking into the answer
- use the same question-aware post-processing strategy as the Instruct version

Important implementation details:

- explicit system instruction to reason internally only
- strict `<answer>FINAL_ANSWER</answer>` contract
- pre-close the visible `<think>` handoff before generation
- extract only the tagged final answer
- retain the same question-type-specific logic used in the manual Instruct pipeline

Why it worked:

- the Thinking model's main local failure mode was over-generation
- strict answer tagging plus extraction made outputs evaluator-friendly

### 5.3 Representative manual changes to mention explicitly

If the report needs concrete examples instead of high-level descriptions, use these:

For `manual_instruct.py`:

- added a strict answer-only rule: one short final line, no explanations, no prefixes
- added question-type-specific hints so legend, colorbar, line, and layout questions are handled differently
- added deterministic normalization for numeric forms and short categorical answers
- biased the system toward `Not Applicable` when the requested chart element is absent or unclear

For `manual_thinking.py`:

- added a system prompt that forces hidden reasoning and visible `<answer>...</answer>` output only
- pre-closed the model's visible `<think>` handoff before generation
- extracted the final answer from tags and discarded any leaked reasoning text
- reused the same question-aware post-processing used in the Instruct pipeline

## 6. Evolution System Design

### 6.1 Shared architecture

Both `evolve_instruct.py` and `evolve_thinking.py` share the same core pattern:

1. Start from the best manual seed.
2. Ask a mutation model to propose a code edit.
3. Validate the candidate code before evaluation.
4. Evaluate the candidate on the local benchmark.
5. Score the candidate with a combined objective.
6. Keep elites and a Pareto frontier.
7. Repeat over multiple generations.

### 6.2 Mutation operator

Mutation backend:

- local text mutator based on cached `Qwen/Qwen3-VL-2B-Instruct`

Mutation strategy:

- ask for compact XML `find`/`replace` edits
- target the inference code directly
- use heuristic mutation fallback when the LLM proposal is invalid or unusable

Exact wording to use in the report:

- `local_qwen` means a local cached Qwen-based text mutator using `Qwen/Qwen3-VL-2B-Instruct`
- `hybrid` means the local cached Qwen-based mutator plus a heuristic mutation bank / fallback
- no external API-based mutation backend was used in the final submitted system

This is a strong point to emphasize in the report:

- we are evolving code, not just prompt strings
- the system is faithful to the AlphaEvolve idea of direct program optimization

### 6.3 Search bookkeeping

Key search features implemented:

- code-hash deduplication
- candidate validation before runtime evaluation
- archived candidate modules per run
- per-generation logs
- explicit elites
- Pareto frontier tracking
- final run summaries with population snapshots

### 6.4 Scoring objective

Both evolution scripts use:

`score = accuracy - 0.05 * avg_time_per_query`

This should be stated explicitly in the report because it explains why some slightly slower but more accurate candidates were retained, while still encouraging efficient inference.

### 6.5 Search settings

Proxy Day 5 Instruct run:

- `num_samples = 8`
- `generations = 2`
- `candidates_per_generation = 1`
- `population_size = 4`
- `elite_size = 2`
- `mutation_backend = local_qwen`

Full Day 6 Instruct run:

- `num_samples = 128`
- `generations = 4`
- `candidates_per_generation = 5`
- `population_size = 8`
- `elite_size = 3`
- `mutation_backend = hybrid`

Full Day 7 Thinking run:

- `num_samples = 128`
- `generations = 2`
- `candidates_per_generation = 4`
- `population_size = 6`
- `elite_size = 2`
- `mutation_backend = hybrid`

### 6.6 Representative evolved mutations to mention explicitly

For `evolved_instruct.py`:

- strengthened the global `Not Applicable` rule so the model is less likely to guess from nearby text
- tightened legend and colorbar rejection behavior for common false positives
- simplified answer-prefix behavior to keep outputs cleaner
- explored small preprocessing changes such as padding removal, but only retained them when accuracy stayed strong

For `evolved_thinking.py`:

- tightened line-question guidance for non-line plots
- strengthened legend filtering so captions and table-like text do not count as legend entries
- strengthened colorbar rejection so axis ranges and heatmap values are less likely to be mistaken for a continuous legend
- kept the strict answer-tag extraction pipeline while mutating the task hints around it

## 7. Final Verified Results

These are the final numbers to use in the paper.

Source:

- `repro_outputs/summary.md`

### 7.1 Main result table

| Variant | Accuracy | Avg/query (s) | Errors | Score |
| --- | ---: | ---: | ---: | ---: |
| Starting baseline | `0.3828125` | `1.4922516774386168` | `0` | `0.3081999161280692` |
| Manual Instruct | `0.6328125` | `0.38159692101180553` | `0` | `0.6137326539494097` |
| Manual Thinking | `0.5546875` | `0.570147329941392` | `0` | `0.5261801335029304` |
| Evolved Instruct | `0.6640625` | `0.3864690978080034` | `0` | `0.6447390451095998` |
| Evolved Thinking | `0.578125` | `0.5762407723814249` | `0` | `0.5493129613809288` |

### 7.2 Best-file table

| File | Chosen module | Reason |
| --- | --- | --- |
| `best_accuracy.py` | `evolved_instruct.py` | highest accuracy |
| `best_speed.py` | `manual_instruct.py` | fastest final fresh-process runtime |
| `best_overall.py` | `evolved_instruct.py` | highest combined score |

### 7.3 Improvement table

Against the starting baseline:

- `manual_instruct`: `+0.25` accuracy, about `3.91x` faster
- `manual_thinking`: `+0.171875` accuracy, about `2.62x` faster
- `evolved_instruct`: `+0.28125` accuracy, about `3.86x` faster
- `evolved_thinking`: `+0.1953125` accuracy, about `2.59x` faster

Manual-to-evolved gains:

- Instruct: `0.6328125 -> 0.6640625`, an absolute gain of `+0.03125`
- Thinking: `0.5546875 -> 0.578125`, an absolute gain of `+0.0234375`

Important nuance:

- the evolved variants improved accuracy for both models
- the evolved variants were not the fastest variants
- this is why `best_speed.py` and `best_overall.py` are different files

## 8. Best Candidates From Search

### 8.1 Instruct evolution

Canonical run directory:

- `evolution_runs/instruct_day6_full`

Best candidate:

- candidate id: `g00_c005`
- search accuracy: `0.6640625`
- search avg/query: `0.40914330817759037 s`
- mutation summary: strengthened global `Not Applicable` behavior

Speed-leaning Pareto candidate:

- candidate id: `g02_c015`
- search accuracy: `0.65625`
- search avg/query: `0.3535063248127699 s`
- mutation summary: simplified no-prefix behavior

### 8.2 Thinking evolution

Canonical run directory:

- `evolution_runs/thinking_day7_full`

Best candidate:

- candidate id: `g01_c008`
- search accuracy: `0.578125`
- search avg/query: `0.607107363641262 s`
- mutation summary: tightened line-question guidance for non-line plot types

## 9. What Worked Best

These are the most report-worthy technical takeaways.

### 9.1 Manual side

The largest improvements came from:

- exact-answer prompting
- output length control
- conservative `Not Applicable` behavior
- question-type-aware hints
- deterministic answer extraction
- legend-versus-colorbar discrimination

### 9.2 Evolution side

The most effective evolved changes were:

- reinforcing conservative rejection of false positives
- tightening task hints for legend, colorbar, and line questions
- preserving accuracy while shaving small preprocessing overheads

### 9.3 High-level lesson

For this local benchmark, exact-match formatting discipline mattered more than complex architectural changes.

This is a strong thesis sentence for the report.

## 10. Negative Results And Honest Limitations

These points should definitely appear in the report.

1. The Day 5 proxy Instruct evolution run did not beat the seed.
2. Evolution was stochastic; later audit reruns did not beat the strongest earlier discovered frozen artifacts.
3. The evolved variants were not the fastest submitted files, even though they had the best accuracy.
4. The development harness uses only `128` descriptive local samples, so overfitting risk remains.
5. The official held-out evaluation may not preserve the exact same relative ordering.

Why these points help instead of hurt:

- they show experiment rigor
- they show that the search loop is real
- they show that the final files were chosen honestly based on verified results

## 11. Originality And Strong Framing

If we want the report to sound strong but still honest, the originality claims should be:

1. We evolved the inference program itself, not just prompts or scalar hyperparameters.
2. We combined an LLM mutator with a heuristic mutation bank to keep the search productive under a small-project compute budget.
3. We used question-type-aware inference logic and post-processing tailored to realistic chart-reading errors.
4. We tracked both elites and a Pareto frontier, which gave a clean accuracy-speed tradeoff view instead of a single-number search.
5. We explicitly kept and reported negative reruns instead of hiding them.

Good phrasing:

- "program-level inference optimization"
- "error-aware mutation and selection"
- "deterministic exact-match optimization for chart understanding"
- "conservative false-positive suppression"

Avoid overstating:

- do not claim leaderboard-level CharXiv progress
- do not imply held-out test gains are known
- do not describe the system as fully autonomous research

## 12. Report Structure

### Abstract

Suggested content:

- state the task
- mention the two Qwen3-VL-2B models
- explain manual plus evolutionary optimization
- report the baseline and best final local numbers
- mention that the best overall system was evolved Instruct and the fastest system was manual Instruct

### 1. Introduction

What to say:

- chart QA is a realistic multimodal challenge
- exact-match local evaluation makes output formatting critical
- we study both manual optimization and AlphaEvolve-style code evolution
- our goal is to improve both accuracy and efficiency without changing the model family

### 2. Setup / Benchmark

What to include:

- local harness uses the first `128` descriptive validation queries
- exact lowercase string match
- greedy decoding in every experiment
- RTX 4090 Laptop GPU environment

### 3. Method

Split this into:

- baseline inference
- manual Instruct optimization
- manual Thinking optimization
- evolution framework

Points to emphasize:

- question-type-aware prompting and post-processing
- strict answer extraction
- LLM-based mutation
- heuristic mutation fallback
- elite and Pareto tracking
- combined scoring objective

### 4. Results

Include at least:

- the main five-row result table
- the best-file table
- maybe a small configuration table for the evolution runs

### 5. Analysis

Main points:

- why manual improvements helped so much
- why evolution helped more on Instruct than on Thinking
- why speed and accuracy winners were not the same file
- why negative reruns still strengthen credibility

### 6. Conclusion

Main message:

- deterministic output control plus modest program search can significantly improve a small VLM on local chart QA

## 13. Tables To Copy Into The Report

### Table A: Main results

| Variant | Accuracy | Avg/query (s) | Score |
| --- | ---: | ---: | ---: |
| Starting baseline | `0.3828125` | `1.4922516774386168` | `0.3081999161280692` |
| Manual Instruct | `0.6328125` | `0.38159692101180553` | `0.6137326539494097` |
| Manual Thinking | `0.5546875` | `0.570147329941392` | `0.5261801335029304` |
| Evolved Instruct | `0.6640625` | `0.3864690978080034` | `0.6447390451095998` |
| Evolved Thinking | `0.578125` | `0.5762407723814249` | `0.5493129613809288` |

### Table B: Final best files

| Submission file | Actual implementation |
| --- | --- |
| `best_accuracy.py` | `evolved_instruct.py` |
| `best_speed.py` | `manual_instruct.py` |
| `best_overall.py` | `evolved_instruct.py` |

### Table C: Evolution settings

| Run | Samples | Generations | Candidates/gen | Population | Elites | Backend |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Instruct proxy | `8` | `2` | `1` | `4` | `2` | `local_qwen` |
| Instruct full | `128` | `4` | `5` | `8` | `3` | `hybrid` |
| Thinking full | `128` | `2` | `4` | `6` | `2` | `hybrid` |

## 14. Files To Cite While Writing

Most useful files while drafting:

- `day8_repro_and_summary.md`
- `day7_thinking_and_bests.md`
- `day6_evolve_instruct.md`
- `day4_manual_thinking.md`
- `day3_manual_instruct_speed.md`
- `day2_manual_instruct.md`
- `repro_outputs/summary.md`
- `full_pipeline_commands.md`


## 15. Figures And Logs We Can Credibly Use

If we want figures in the report, these are the safest options because the underlying artifacts already exist:

1. Evolution score over generations
   - source files:
     - `evolution_runs/instruct_day6_full/generation_summaries.jsonl`
     - `evolution_runs/thinking_day7_full/generation_summaries.jsonl`

2. Pareto frontier figure
   - source files:
     - `evolution_runs/instruct_day6_full/pareto_frontier.json`
     - `evolution_runs/thinking_day7_full/pareto_frontier.json`

3. Candidate trajectory / population snapshot table
   - source files:
     - `evolution_runs/instruct_day6_full/population.json`
     - `evolution_runs/thinking_day7_full/population.json`

4. Failure-case examples
   - source files:
     - `manual_instruct_day2_final_queries.json`
     - `manual_thinking_day4_final_queries.json`
     - `evolved_instruct_day6_final_queries.json`
     - `evolved_thinking_day7_final_queries.json`
