# Day 5 Evolve Instruct

Planned day: 2026-03-21
Completed on: 2026-03-21
Re-verified on: 2026-03-21

## Objective

Implement a simple but complete AlphaEvolve-style search loop for `Qwen/Qwen3-VL-2B-Instruct` that supports:

- LLM-based mutation
- candidate generation
- evolve-block extraction
- population management
- evaluation on a proxy subset
- logging
- a score that balances accuracy and runtime

## Final Day 5 File

- `evolve_instruct.py`

Core design choices:

1. Seed from the best manual Instruct module
   - seed module: `manual_instruct`
   - the seed code is wrapped as an evolve block and used as the initial candidate

2. Use a local LLM mutation backend
   - mutator: cached `Qwen/Qwen3-VL-2B-Instruct`
   - mutation mode: text-only code mutation
   - mutation format: compact XML `find` / `replace` patch proposal

3. Keep the loop robust
   - validate generated code before evaluation
   - fall back to heuristic mutation when the local LLM does not yield a usable patch
   - skip duplicates by code hash
   - evaluate candidates in subprocesses with `evaluate.py`

4. Keep the logging explicit
   - per-run config
   - prompt logs
   - raw mutation logs
   - candidate metrics JSON
   - candidate history JSONL
   - archived candidate modules
   - summary, elites, population, and Pareto snapshots

## Scoring

The search score is:

`score = accuracy - 0.05 * avg_time_per_query`

This keeps accuracy primary while still penalizing slower candidates.

## Canonical Proxy Re-Run

Important note:

- `evolve_instruct.py` later gained Day 6 full-search defaults, so the Day 5 proxy rerun must use explicit proxy flags instead of relying on defaults.

Command:

```powershell
conda run -n vlm python evolve_instruct.py `
  --run-name instruct_proxy_day5_canonical `
  --num-samples 8 `
  --generations 2 `
  --candidates-per-generation 1 `
  --population-size 4 `
  --elite-size 2 `
  --mutation-backend local_qwen `
  --export-best evolved_instruct.py
```

Saved run directory:

- `evolution_runs/instruct_proxy_day5_canonical`

## Final Proxy Evolution Result

Best candidate from `evolution_runs/instruct_proxy_day5_canonical/summary.json`:

- candidate: `seed`
- accuracy: `0.75`
- avg/query: `0.6932363212108612 s`
- score: `0.7153381839394569`

Accepted evolved candidate:

- candidate: `g00_c001`
- source: `local_llm`
- summary: strengthened `Not Applicable` handling in `_postprocess_response`
- accuracy: `0.75`
- avg/query: `0.6979365944862366 s`
- score: `0.7151031702756881`

Duplicate candidate:

- candidate: `g01_c002`
- source: `local_llm`
- result: duplicate of the earlier LLM-mutated code hash, skipped and logged

## What Day 5 Proved

- the evolution loop runs end to end locally
- the system uses an LLM mutation operator and evaluates the resulting candidates
- the population, elite, and Pareto bookkeeping work on the proxy subset
- the proxy run finishes without manual intervention
- negative results are preserved honestly: on this proxy subset, the seed remained the best candidate

## Day 5 Status

Day 5 is complete and README-aligned for the Instruct track:

- `evolve_instruct.py` contains the required AlphaEvolve-style components
- the proxy evolution run is reproducible from the canonical command above
- the final repository keeps `evolved_instruct.py` reserved for the stronger Day 6 full-scale evolved result
