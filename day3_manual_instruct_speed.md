# Day 3 Manual Instruct Speed Study

Planned day: 2026-03-19
Completed on: 2026-03-19

## Objective

Benchmark deterministic speed optimizations for `manual_instruct.py`, keep greedy decoding throughout, and retain only the fastest stable configuration that preserves the Day 2 accuracy.

## Important Outcome

After rerunning the speed candidates on the cleaned repo, the fastest stable configuration on this machine remained the Day 2 configuration already in `manual_instruct.py`:

- `padding=True` in the processor call
- `max_new_tokens=64`
- greedy decoding

In other words, Day 3 still added value, but the value was selection discipline rather than a code change: the speed study rejected changes that looked promising earlier but did not win in the final reruns.

## Stable Baseline Repeats

The kept Day 2 / Day 3 configuration was rerun multiple times:

- repeat 1: `0.3603222668170929 s/query`
- repeat 2: `0.3568109776824713 s/query`
- repeat 3: `0.3570127375423908 s/query`
- repeat 4: `0.35397810116410255 s/query`

Accuracy stayed constant at `0.6328125` in every accepted run.

## Tested Day 3 Candidates

All candidate runs below kept accuracy at `0.6328125` and `0` errors:

| Candidate | Avg time/query | Decision |
| --- | ---: | --- |
| kept config: `padding=True`, `max_new_tokens=64` | `0.35397810116410255 s` | kept |
| candidate: `padding=False`, `max_new_tokens=64` | `0.35756939090788364 s` | rejected |
| candidate: `padding=True`, `max_new_tokens=48` | `0.35796312987804413 s` | rejected |
| candidate: `padding=True`, `max_new_tokens=32` | `0.39431656524538994 s` | rejected |

## Final Canonical Day 3 Run

Command:

```powershell
python evaluate.py manual_instruct --hf-offline `
  --output manual_instruct_day3_final.json `
  --save-responses manual_instruct_day3_final_queries.json
```

Saved outputs:

- metrics: `manual_instruct_day3_final.json`
- per-query responses: `manual_instruct_day3_final_queries.json`

Final selected result:

- accuracy: `0.6328125`
- evaluated: `128`
- errors: `0`
- total time: `45.30919694900513 s`
- avg/query: `0.35397810116410255 s`

Compared with the Day 1 starter baseline:

- baseline avg/query: `1.4759445749223232 s`
- final avg/query: `0.35397810116410255 s`
- overall speedup vs starter: about `4.17x`

## Final Fresh-Process Verification

The final submission audit on `2026-03-21` reran the kept Day 3 configuration through `bash reproduce.sh`.

Canonical fresh-process metrics from `repro_outputs/summary.md`:

- accuracy: `0.6328125`
- avg/query: `0.41409385576844215 s`
- errors: `0`

The fresh-process timing is a bit slower than the earlier warm-cache Day 3 repeat table, but the selection conclusion does not change: the current `manual_instruct.py` remains the right stable manual Instruct file to keep.

## Day 3 Status

Day 3 is complete:

- speed candidates were tested under the required greedy-decoding setup
- unhelpful changes were rejected instead of forced into the final code
- `manual_instruct.py` now reflects the fastest stable configuration verified in the final reruns
