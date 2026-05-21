# Full Pipeline Commands

This file lists the exact commands to run the whole local project pipeline end to end on this repository.

Repo root:

```powershell
cd C:\Users\ROG\Desktop\evolving-vlm-inference-main
```

## 1. Environment Setup

Create and activate the conda environment:

```powershell
conda create -n vlm python=3.10 -y
conda activate vlm
pip install -r requirements.txt
```

Check the core versions if you want a quick sanity check:

```powershell
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

## 2. CharXiv Images

Download and extract the chart images into `charxiv/images/`.

```powershell
New-Item -ItemType Directory -Force charxiv\images | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip" -OutFile "charxiv\images\images.zip"
Expand-Archive -Path "charxiv\images\images.zip" -DestinationPath "charxiv\images" -Force
Remove-Item "charxiv\images\images.zip"
```

Quick check:

```powershell
(Get-ChildItem charxiv\images -Filter *.jpg).Count
```

## 3. Day 1 Baseline

Run the naive baseline and save metrics plus per-query outputs:

```powershell
conda run -n vlm python evaluate.py starting_scripts --hf-offline `
  --num-samples 128 `
  --output baseline_starting_scripts_day1_final.json `
  --save-responses baseline_starting_scripts_day1_final_queries.json
```

## 4. Day 2 And Day 3 Manual Instruct

Run the manual Instruct module:

```powershell
conda run -n vlm python evaluate.py manual_instruct --hf-offline `
  --num-samples 128 `
  --output manual_instruct_day3_final.json `
  --save-responses manual_instruct_day3_final_queries.json
```

If you want the Day 2 accuracy-first artifact name as well:

```powershell
conda run -n vlm python evaluate.py manual_instruct --hf-offline `
  --num-samples 128 `
  --output manual_instruct_day2_final.json `
  --save-responses manual_instruct_day2_final_queries.json
```

## 5. Day 4 Manual Thinking

Run the manual Thinking module:

```powershell
conda run -n vlm python evaluate.py manual_thinking --hf-offline `
  --num-samples 128 `
  --output manual_thinking_day4_final.json `
  --save-responses manual_thinking_day4_final_queries.json
```

## 6. Day 5 Instruct Proxy Evolution

Run the small proxy evolution pass:

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

Outputs land in:

```text
evolution_runs/instruct_proxy_day5_canonical
```

## 7. Day 6 Instruct Full Evolution

Run the full Instruct evolution search:

```powershell
conda run -n vlm python evolve_instruct.py `
  --run-name instruct_day6_full `
  --num-samples 128 `
  --generations 4 `
  --candidates-per-generation 5 `
  --population-size 8 `
  --elite-size 3 `
  --mutation-backend hybrid `
  --export-best evolved_instruct.py
```

Verify the frozen evolved Instruct file:

```powershell
conda run -n vlm python evaluate.py evolved_instruct --hf-offline `
  --num-samples 128 `
  --output evolved_instruct_day6_final.json `
  --save-responses evolved_instruct_day6_final_queries.json
```

## 8. Day 7 Thinking Full Evolution

Run the full Thinking evolution search:

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

Verify the frozen evolved Thinking file:

```powershell
conda run -n vlm python evaluate.py evolved_thinking --hf-offline `
  --num-samples 128 `
  --output evolved_thinking_day7_final.json `
  --save-responses evolved_thinking_day7_final_queries.json
```

## 9. Global Best Files

The repository currently keeps:

- `best_accuracy.py` aligned with `evolved_instruct.py`
- `best_speed.py` aligned with `manual_instruct.py`
- `best_overall.py` aligned with `evolved_instruct.py`

Evaluate them exactly as the README expects:

```powershell
conda run -n vlm python evaluate.py best_accuracy --hf-offline --num-samples 128 --output repro_outputs\best_accuracy.json
conda run -n vlm python evaluate.py best_speed --hf-offline --num-samples 128 --output repro_outputs\best_speed.json
conda run -n vlm python evaluate.py best_overall --hf-offline --num-samples 128 --output repro_outputs\best_overall.json
```

## 10. Final Reproduction

README-style command:

```bash
bash reproduce.sh
```

On Windows PowerShell with Git Bash installed:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' reproduce.sh
```

This writes the canonical final comparison table into:

```text
repro_outputs/summary.md
repro_outputs/summary.json
repro_outputs/module_aliases.json
```

## 11. Optional Final Audit Reruns

These are the extra reruns used to confirm that the frozen evolved artifacts should remain unchanged:

```powershell
conda run -n vlm python evolve_instruct.py `
  --run-name instruct_final_audit `
  --export-best evolution_tmp/evolved_instruct_final_audit.py

conda run -n vlm python evolve_thinking.py `
  --run-name thinking_final_audit `
  --export-best evolution_tmp/evolved_thinking_final_audit.py
```

The final audit run directories are:

```text
evolution_runs/instruct_final_audit
evolution_runs/thinking_final_audit
```

## 12. Read The Final Summary

Open the final comparison table:

```powershell
Get-Content repro_outputs\summary.md
```

Open the final day notes:

```powershell
Get-Content day8_repro_and_summary.md
Get-Content day7_thinking_and_bests.md
Get-Content day6_evolve_instruct.md
```
