# Day 1 Setup and Baseline

Planned day: 2026-03-17
Completed on: 2026-03-17

## Objective

Make the repo runnable end to end, install the required software in a named conda environment, download the missing CharXiv images, read the AlphaEvolve paper at a practical level, and record a clean baseline for `starting_scripts.py`.

## Final Environment

- Conda env: `vlm`
- Python: `3.10`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- CUDA runtime: `12.8`
- PyTorch: `2.10.0+cu128`
- Transformers: `4.57.1`

Activation:

```powershell
conda activate vlm
```

## What Was Completed

1. Installed the project dependencies in the named conda env `vlm`.
2. Downloaded and extracted the CharXiv image archive.
3. Verified that `charxiv/images/` now contains all `2323` chart images.
4. Patched `evaluate.py` so the repo supports the README-style workflow:

```powershell
python evaluate.py starting_scripts
python evaluate.py manual_instruct
```

5. Patched `starting_scripts.py` only for compatibility and reproducibility:
   - current Qwen3-VL model import compatibility
   - greedy-decoding-safe generation config cleanup
   - `torch.inference_mode()`
6. Pinned `transformers==4.57.1` in `requirements.txt` because the starter pipeline needs the pre-v5 API surface while still supporting Qwen3-VL.

## AlphaEvolve Notes Relevant to This Project

Practical takeaways from the paper:

1. Optimize code directly, not just prompts or scalar hyperparameters.
2. Use an LLM as a mutation operator that proposes candidate program edits.
3. Maintain a set of candidate programs rather than a single current best.
4. Score candidates with the real evaluator, then keep the strongest ones and repeat.

Project mapping:

- candidate program: the editable inference code in `starting_scripts.py`
- evaluator: local CharXiv development evaluation
- fitness: accuracy, speed, or a combined score
- search loop: mutate, run, score, select, repeat

## Final Canonical Baseline Run

Command:

```powershell
python evaluate.py starting_scripts --hf-offline `
  --output baseline_starting_scripts_day1_final.json `
  --save-responses baseline_starting_scripts_day1_final_queries.json
```

Saved outputs:

- metrics: `baseline_starting_scripts_day1_final.json`
- per-query responses: `baseline_starting_scripts_day1_final_queries.json`

Result:

- accuracy: `0.3828125`
- evaluated: `128`
- errors: `0`
- total time: `188.92090559005737 s`
- avg/query: `1.4759445749223232 s`

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran the baseline through the validated `bash reproduce.sh` pipeline after fixing local offline loading in the module loaders.

Canonical fresh-process metrics from `repro_outputs/summary.md`:

- accuracy: `0.3828125`
- avg/query: `1.610166274011135 s`
- errors: `0`

This slightly slower number is the one to trust for final cross-module comparison because it was produced by the same fresh-process protocol used for every submitted module.

## Day 1 Status

Day 1 is complete and matches the project requirements:

- the repo runs end to end
- the benchmark assets are present locally
- the evaluator supports named modules
- the starter baseline is recorded with saved outputs
- the project is ready for manual optimization work
