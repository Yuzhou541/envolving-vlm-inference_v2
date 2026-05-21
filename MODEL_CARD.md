# EvoChartCode Model Card

## Intended Use

EvoChartCode is an inference framework for chart question answering. It converts chart images into explicit Chart Code, selects question-relevant evidence, reasons over that evidence, and verifies final answers against the structured representation.

## Primary Models

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-2B-Thinking`

The repository keeps greedy decoding as the reproducible default.

## Data

The included workflows target CharXiv validation data under `charxiv/`. The split generator creates chart-level `evolution_dev`, `validation`, and `heldout` splits in `data/splits/charxiv_chart_level_split.json`.

## Limitations

- Metadata-only extraction is a deterministic smoke backend, not a high-accuracy chart parser.
- The VLM JSON extractor requires locally available Qwen model weights when `local_files_only: true`.
- Numeric chart reasoning depends on extracted axis, tick, mark, and series quality.
- The verifier can reject unsupported legend, colorbar, chart-type, and trend answers, but it does not prove correctness for every open-ended reasoning question.

## Reproducibility

Run:

```bash
bash reproduce_main.sh
```

For model-backed extraction or image+code reasoning, make sure the Qwen weights are available in the Hugging Face cache or set `local_files_only: false` in the config.
