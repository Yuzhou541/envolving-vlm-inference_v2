# EvoChartCode: Evolution-Guided Chart Code Induction for Reliable Chart Reasoning

**Project type:** Research project + full reproducible codebase  
**Primary hardware:** NVIDIA RTX 4090 Laptop GPU, 16 GB VRAM  
**Software assumption:** CUDA 12.8, Conda virtual environment, PyTorch + HuggingFace Transformers  
**Core idea:** Convert chart images into explicit, structured, evolvable *Chart Code*, then use code-grounded reasoning and verification instead of directly asking a frozen VLM to answer from pixels alone.

---

## 0. One-Sentence Summary

This project proposes **EvoChartCode**, a verification-guided evolutionary framework that improves frozen VLM chart reasoning by evolving the intermediate representation, extraction program, serialization policy, reasoning prompt, and verifier for an explicit structured representation called **Chart Code**.

---

## 1. Motivation

Current multimodal chart understanding systems usually follow a direct inference path:

```text
chart image + question -> VLM -> answer
```

This direct path is brittle. Even when the VLM visually recognizes much of the chart, it may fail because of formatting errors, hallucinated legends, wrong colorbar interpretation, noisy axis reading, or weak numerical grounding. The preliminary AlphaEvolve-style project already provides evidence for this: under fixed Qwen3-VL-2B models and greedy decoding, carefully engineered inference programs substantially improved exact-match performance without training model weights. The largest gains came from deterministic output control, question-type-aware prompts, answer normalization, strict extraction, and conservative false-positive suppression.

However, this preliminary direction alone is not yet strong enough for a NeurIPS-level paper. It can be criticized as local benchmark engineering or prompt/post-processing optimization. To reach NeurIPS 2026 quality, the project should introduce a more general and novel abstraction:

```text
chart image -> explicit Chart Code -> code-grounded reasoning -> verified answer
```

The inspiration is the recent "Thinking with Spatial Code" idea: for physical-world video reasoning, the system transforms RGB video into explicit 3D spatial representations such as object labels and 3D oriented bounding boxes, allowing LLMs to reason over structured spatial variables rather than over raw pixels only. The key lesson is that **representation quality, not only model scale, can be the bottleneck for visual reasoning**.

This project adapts and extends that idea to scientific chart understanding:

```text
Spatial Code:
  objects, 3D boxes, positions, sizes, orientations, temporal tracks

Chart Code:
  chart type, plot layout, axes, scales, ticks, legends, marks, series,
  points, trends, extrema, turning points, intersections, comparisons,
  uncertainty, missing elements, and provenance
```

The novelty is not merely to define a hand-written chart schema. The stronger contribution is:

> **The Chart Code language itself and the inference program that produces and uses it are evolved under automatic evaluation.**

Thus, the project becomes a study of **evolvable explicit visual abstractions** for reliable multimodal reasoning.

---

## 2. Proposed Paper Title Options

### Recommended title

**EvoChartCode: Evolution-Guided Chart Code Induction for Reliable Chart Reasoning**

### Alternative titles

1. **Thinking with Chart Code: Explicit Visual Program Induction for Scientific Chart Reasoning**
2. **Evolving the Language of Charts: Programmatic Visual Abstractions for Reliable Multimodal Reasoning**
3. **Verification-Guided Evolution of Structured Chart Codes for Frozen Vision-Language Models**
4. **From Pixels to Chart Programs: Evolvable Intermediate Representations for Chart Understanding**

The recommended title is the most balanced: it is concrete, clear, and aligned with NeurIPS-style method papers.

---

## 3. Core Research Claim

Frozen VLMs often contain partial chart-understanding ability, but their raw image-to-answer interface is unreliable. Much of the recoverable performance lies not in further model scaling or full finetuning, but in the **inference interface**: the representation extracted from the image, the structured evidence given to the reasoning model, and the verification rules that constrain the final answer.

This project formulates chart reasoning as:

```text
c = E(I)
a = R(q, c, I_optional)
a_final = V(q, c, a)
```

where:

- `I` is the chart image,
- `q` is the question,
- `c` is explicit Chart Code,
- `E` is a chart-code extraction program,
- `R` is a code-grounded reasoner,
- `V` is a verifier and answer normalizer.

Instead of directly optimizing only prompts, EvoChartCode searches over a space of inference programs:

```text
π = {
  chart_code_schema,
  visual_extraction_program,
  derived_attribute_program,
  code_serializer,
  question_router,
  code_selector,
  reasoning_policy,
  verifier,
  answer_normalizer,
  fallback_policy
}
```

The framework then uses evolutionary program search to improve `π` under multi-objective automatic feedback.

---

## 4. Target Contributions

A NeurIPS 2026 version should aim for the following contributions.

### Contribution 1: Chart Code as an explicit intermediate representation

Introduce **Chart Code**, a structured representation for scientific charts that captures layout, scales, visual marks, data geometry, derived trends, comparisons, and uncertainty/provenance.

### Contribution 2: Evolution of representation and inference programs

Propose **EvoChartCode**, an evolution-guided framework that improves not only prompts, but also the Chart Code schema, extraction program, serialization, question-conditioned code selection, reasoning policy, verifier, and fallback strategy.

### Contribution 3: Verification-guided chart reasoning

Introduce a verifier that checks whether the answer is supported by Chart Code evidence. This enables reliable abstention, better `Not Applicable` handling, lower hallucination, and failure attribution.

### Contribution 4: Systematic evaluation across datasets, models, and objectives

Evaluate across CharXiv, ChartQA, PlotQA, DVQA/FigureQA-style controlled datasets, and held-out chart/question types. Report not only accuracy but also relaxed numeric accuracy, invalid-output rate, abstention calibration, latency, Pareto frontier, search efficiency, and overfitting gap.

### Contribution 5: Error attribution: perception vs reasoning vs format

Show that explicit Chart Code separates failure modes into:

```text
perception error: Chart Code extraction is wrong
reasoning error: Chart Code is correct but reasoner answers incorrectly
verification error: answer unsupported but verifier fails to reject
format error: answer is semantically correct but evaluator-incompatible
abstention error: system over- or under-uses Not Applicable
```

This is important because it makes the system diagnosable rather than a black-box VLM prompt.

---

## 5. Method Overview

## 5.1 Pipeline

The full EvoChartCode pipeline is:

```text
Input:
  chart image I
  natural-language question q

Stage 1: Chart Code extraction
  c_raw = Extractor(I)

Stage 2: Schema validation and repair
  c = ValidateAndRepair(c_raw)

Stage 3: Question routing
  t = RouteQuestion(q)

Stage 4: Code selection
  c_q = SelectRelevantCode(c, t, q)

Stage 5: Code-grounded reasoning
  a = Reasoner(q, c_q, optional_image=I)

Stage 6: Verification and answer normalization
  a_final = VerifierAndNormalizer(q, c, a)

Output:
  final short answer
```

A key difference from normal chart QA is that the LLM/VLM is not asked to answer directly from the image. It first sees a compact structured representation of the chart, and the final answer must be grounded in that representation.

---

## 5.2 Chart Code Schema

The Chart Code schema should be hierarchical and extensible. It should include both low-level visual evidence and high-level derived abstractions.

### 5.2.1 Top-level schema

```json
{
  "chart_id": "string",
  "image_size": {"width": 0, "height": 0},
  "chart_type": "line|bar|scatter|heatmap|boxplot|histogram|pie|area|mixed|table|unknown",
  "title": {"text": "string", "bbox": [0, 0, 0, 0], "confidence": 0.0},
  "layout": {},
  "axes": {},
  "legend": {},
  "colorbar": {},
  "series": [],
  "marks": [],
  "derived_relations": {},
  "uncertainty": {},
  "provenance": {}
}
```

### 5.2.2 Layout Code

```json
{
  "layout": {
    "plot_area_bbox": [x1, y1, x2, y2],
    "subplot_count": 1,
    "subplot_grid": {"rows": 1, "cols": 1},
    "subplots": [
      {
        "id": "subplot_0",
        "bbox": [x1, y1, x2, y2],
        "title": "string",
        "x_axis_id": "x0",
        "y_axis_id": "y0"
      }
    ],
    "caption_region_bbox": null,
    "legend_bbox": null,
    "colorbar_bbox": null
  }
}
```

### 5.2.3 Axis and Scale Code

```json
{
  "axes": {
    "x": {
      "label": "Epoch",
      "unit": null,
      "scale": "linear|log|categorical|time|unknown",
      "range": [0, 100],
      "ticks": [
        {"value": "0", "position": [120, 450], "numeric_value": 0.0},
        {"value": "50", "position": [300, 450], "numeric_value": 50.0}
      ],
      "confidence": 0.92
    },
    "y": {
      "label": "Accuracy",
      "unit": "%",
      "scale": "linear",
      "range": [0, 100],
      "ticks": [],
      "confidence": 0.87
    }
  }
}
```

### 5.2.4 Legend Code

```json
{
  "legend": {
    "exists": true,
    "bbox": [x1, y1, x2, y2],
    "items": [
      {
        "name": "Model A",
        "color": "blue",
        "marker": "circle",
        "line_style": "solid",
        "confidence": 0.88
      }
    ],
    "confidence": 0.85
  }
}
```

### 5.2.5 Series and Data Geometry Code

For line and scatter charts:

```json
{
  "series": [
    {
      "id": "series_0",
      "name": "Model A",
      "visual_type": "line",
      "color": "blue",
      "marker": "circle",
      "points_pixel": [[100, 400], [150, 360], [200, 300]],
      "points_data": [[0, 0.60], [10, 0.65], [20, 0.72]],
      "start": {"x": 0, "y": 0.60},
      "end": {"x": 20, "y": 0.72},
      "confidence": 0.76
    }
  ]
}
```

For bar charts:

```json
{
  "marks": [
    {
      "id": "bar_0",
      "type": "bar",
      "category": "A",
      "height_data": 12.5,
      "bbox": [x1, y1, x2, y2],
      "color": "orange",
      "series": "Method A",
      "confidence": 0.82
    }
  ]
}
```

For heatmaps:

```json
{
  "marks": [
    {
      "id": "cell_0_0",
      "type": "heatmap_cell",
      "row": "A",
      "col": "B",
      "color": "#34a853",
      "value_estimate": 0.73,
      "confidence": 0.70
    }
  ],
  "colorbar": {
    "exists": true,
    "min": 0.0,
    "max": 1.0,
    "ticks": [0.0, 0.5, 1.0],
    "confidence": 0.78
  }
}
```

### 5.2.6 Derived Relation Code

This is the most important part for reasoning-heavy questions.

```json
{
  "derived_relations": {
    "trends": [
      {
        "series_id": "series_0",
        "global_trend": "increasing|decreasing|flat|nonmonotonic|unknown",
        "segments": [
          {
            "x_range": [0, 20],
            "trend": "increasing",
            "slope_sign": "positive",
            "slope_magnitude": "moderate"
          }
        ],
        "confidence": 0.81
      }
    ],
    "turning_points": [
      {
        "series_id": "series_0",
        "type": "local_maximum|local_minimum|inflection|knee|elbow",
        "x": 50,
        "y": 0.92,
        "evidence_points": [[40, 0.88], [50, 0.92], [60, 0.91]],
        "confidence": 0.70
      }
    ],
    "comparisons": [
      {
        "series_a": "series_0",
        "series_b": "series_1",
        "relation": "a_above_b|a_below_b|crosses|similar|unknown",
        "x_range": [20, 100],
        "confidence": 0.76
      }
    ],
    "intersections": [
      {
        "series_a": "series_0",
        "series_b": "series_1",
        "x": 35,
        "y": 0.70,
        "confidence": 0.68
      }
    ],
    "extrema": [
      {
        "series_id": "series_0",
        "type": "maximum",
        "x": 80,
        "y": 0.94,
        "confidence": 0.84
      }
    ],
    "rankings": [
      {
        "at_x": 100,
        "order_high_to_low": ["series_0", "series_2", "series_1"],
        "confidence": 0.80
      }
    ]
  }
}
```

### 5.2.7 Uncertainty and Provenance Code

Each field should record confidence and provenance.

```json
{
  "uncertainty": {
    "low_confidence_fields": [
      "series[0].points_data",
      "colorbar.max"
    ],
    "ambiguous_regions": [
      {
        "bbox": [x1, y1, x2, y2],
        "reason": "overlapping legend and plot line"
      }
    ],
    "missing_elements": [
      "legend",
      "colorbar"
    ],
    "global_confidence": 0.74
  },
  "provenance": {
    "extractor_versions": {
      "ocr": "paddleocr_or_tesseract",
      "vlm": "qwen3-vl-2b-instruct",
      "cv": "opencv_pipeline_v1"
    },
    "fields": {
      "axes.x.label": "ocr",
      "series[0].global_trend": "derived_from_points",
      "legend.items": "vlm_and_ocr_consensus"
    }
  }
}
```

This enables verification and failure analysis.

---

## 5.3 Chart Code Extraction

The extractor should combine three sources:

```text
1. VLM-assisted semantic extraction
2. Deterministic computer vision and OCR extraction
3. Derived programmatic analysis
```

### 5.3.1 VLM-assisted extraction

Use Qwen3-VL-2B-Instruct or Qwen2.5-VL as a JSON extractor.

Prompt:

```text
You are a chart-code extractor. Convert the chart image into valid JSON.
Do not answer any question. Only extract visual and structural information.
Return fields for chart type, title, axes, ticks, legend, colorbar, series, marks,
trends, comparisons, uncertainty, and missing elements.
If a field is not visible, use null and set confidence below 0.5.
Return valid JSON only.
```

The VLM extractor is useful for:

```text
title
axis labels
legend names
chart type
high-level trends
subplot layout
qualitative comparisons
whether a requested element exists
```

### 5.3.2 Deterministic OCR and CV extraction

Use deterministic modules for low-level evidence:

```text
OCR:
  title, axis labels, tick labels, legend text, colorbar labels

OpenCV / skimage:
  plot area detection
  axis line detection
  Hough transform for straight lines
  contour detection for bars and markers
  color clustering for series separation
  connected components for legends
  heatmap grid detection
  subplot segmentation

Numeric conversion:
  map pixel coordinates to data coordinates using axis ticks
  estimate line/bar values from pixel position
```

Recommended libraries:

```text
opencv-python
pillow
numpy
scipy
scikit-image
pytesseract or easyocr
rapidfuzz
pydantic
jsonschema
```

### 5.3.3 Derived attribute program

Once approximate data points are available, compute higher-level fields:

```python
def derive_trend(points_data):
    # smooth points
    # compute slope signs
    # detect monotonicity
    # detect plateau
    # detect local extrema
    # detect turning points
    # return symbolic trend code
    pass

def derive_comparisons(series_list):
    # align by x
    # compare y values
    # detect crossings
    # compute rank order
    pass

def derive_chart_events(chart_code):
    # intersections
    # max/min
    # outliers
    # convergence/divergence
    # gap widening/narrowing
    pass
```

This makes Chart Code more than OCR. It becomes a compact executable representation of the chart.

---

## 5.4 Question Routing and Code Selection

The system should not feed the entire Chart Code to the LLM for every question. Full Chart Code can be long and noisy. Instead, route the question and select relevant fields.

### Question types

```text
title
axis_label
axis_range
tick_value
tick_count
legend_count
legend_name
colorbar_min_max
chart_type
subplot_count
line_count
bar_value
point_value
trend
extremum
turning_point
intersection
comparison
ranking
difference
ratio
not_applicable
open_ended_reasoning
```

### Code selector examples

```python
def select_code_for_question(question_type, chart_code):
    if question_type == "legend_count":
        return {
            "chart_type": chart_code.chart_type,
            "legend": chart_code.legend,
            "uncertainty": chart_code.uncertainty
        }

    if question_type == "trend":
        return {
            "chart_type": chart_code.chart_type,
            "series": minimal_series_metadata(chart_code.series),
            "trends": chart_code.derived_relations.trends,
            "turning_points": chart_code.derived_relations.turning_points
        }

    if question_type == "colorbar_min_max":
        return {
            "chart_type": chart_code.chart_type,
            "colorbar": chart_code.colorbar,
            "uncertainty": chart_code.uncertainty
        }
```

This module is also evolvable. EvoChartCode can discover that some question types need additional evidence fields.

---

## 5.5 Code-Grounded Reasoning Modes

### Mode A: Image-only VLM

Baseline:

```text
image + question -> VLM -> answer
```

This is the standard raw VLM inference.

### Mode B: Code-only LLM

```text
Chart Code + question -> text-only LLM -> answer
```

This tests whether Chart Code itself captures enough information to answer. It is cheap and interpretable.

### Mode C: Code + image VLM

```text
image + selected Chart Code + question -> VLM -> answer
```

This allows the model to fall back to visual evidence while still being guided by structured code.

### Mode D: Verifier-guided hybrid

```text
1. Try code-only if selected Chart Code confidence is high.
2. If confidence is low, ask VLM with image + code.
3. Verify answer against Chart Code.
4. If unsupported, return Not Applicable or trigger a repair pass.
```

The best final system will likely be Mode D.

---

## 5.6 Verifier and Answer Normalizer

The verifier prevents unsupported answers.

Examples:

```python
def verify_legend_answer(question, chart_code, answer):
    if asks_about_legend(question):
        if not chart_code.legend.exists:
            return answer == "Not Applicable"
        if asks_legend_count(question):
            return normalize_int(answer) == len(chart_code.legend.items)
```

```python
def verify_colorbar_answer(question, chart_code, answer):
    if asks_colorbar(question):
        if not chart_code.colorbar.exists:
            return answer == "Not Applicable"
        if asks_max(question):
            return numeric_close(answer, chart_code.colorbar.max)
```

```python
def verify_trend_answer(question, chart_code, answer):
    if asks_trend(question):
        trend = selected_series_trend(chart_code, question)
        return answer_supported_by_trend(answer, trend)
```

The verifier should output:

```json
{
  "verdict": "supported|unsupported|ambiguous",
  "reason": "legend does not exist",
  "suggested_answer": "Not Applicable",
  "confidence": 0.91
}
```

This supports both runtime inference and evolution feedback.

---

## 6. Evolution Framework

## 6.1 What is evolved?

Unlike prompt-only search, EvoChartCode evolves multiple program blocks:

```python
# EVOLVE-BLOCK-START: chart_code_schema
# Defines required and optional fields.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: vlm_extractor_prompt
# Prompt for chart-to-JSON extraction.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: cv_extraction_rules
# Deterministic OCR/CV rules.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: derived_attribute_program
# Trend, extrema, turning point, crossing, ranking computation.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: question_router
# Maps natural questions to types.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: code_selector
# Selects compact evidence subset for the reasoner.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: code_serializer
# Converts Chart Code to compact LLM-readable text.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: reasoning_policy
# Prompt and mode selection for code-only / image+code / fallback.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: verifier
# Evidence consistency and abstention rules.
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START: answer_normalizer
# Exact-match and numeric normalization.
# EVOLVE-BLOCK-END
```

This is the main novelty: the system evolves the *language of chart reasoning*, not only the final prompt.

---

## 6.2 Candidate representation

A candidate program is a full directory snapshot or a patch over selected files:

```text
candidate_id/
  schema.py
  extractor_vlm.py
  extractor_cv.py
  derive.py
  router.py
  selector.py
  serializer.py
  reasoner.py
  verifier.py
  normalizer.py
  config.yaml
```

Each candidate is scored by automatic evaluation.

---

## 6.3 Mutation operator

Use a local or API LLM as a mutation model. For local reproducibility on RTX 4090, use a small coding/instruct model for cheap mutations. For stronger experiments, optionally use an external frontier model to generate mutations, but the paper should clearly separate local-only and API-assisted settings.

Mutation prompt should include:

```text
- Current program block
- Previous best candidates
- Error clusters
- Metrics table
- Failed examples
- Constraints:
    * keep public interfaces unchanged
    * keep JSON valid
    * preserve deterministic decoding
    * do not hard-code dataset answers
    * improve generalization, not only dev accuracy
- Required output format: SEARCH/REPLACE blocks
```

Recommended mutation types:

```text
schema_add_field
schema_remove_noisy_field
extractor_prompt_refine
ocr_rule_refine
cv_rule_refine
derived_feature_addition
router_rule_refine
serializer_compression
verifier_rule_addition
normalizer_fix
fallback_policy_change
```

---

## 6.4 Evaluation cascade

To fit on one RTX 4090, avoid evaluating every candidate on full datasets. Use staged evaluation:

```text
Stage 0: static validation
  - import check
  - Pydantic schema check
  - no syntax error
  - public API check

Stage 1: smoke set
  - 16 examples
  - balanced over question types
  - must not crash
  - JSON validity > threshold

Stage 2: stratified mini-dev
  - 128 examples
  - includes descriptive and reasoning
  - measures exact match, relaxed numeric, invalid rate, latency

Stage 3: full dev
  - full selected development split
  - only top candidates enter

Stage 4: held-out validation
  - used for model selection
  - not directly used inside mutation prompt except aggregated metrics

Stage 5: OOD transfer
  - cross-dataset or chart-type holdout
  - run only for final candidates
```

This reduces compute cost and overfitting.

---

## 6.5 Error-aware feedback

For each candidate, store structured error feedback:

```json
{
  "candidate_id": "g03_c17",
  "metrics": {
    "exact_match": 0.68,
    "relaxed_numeric": 0.74,
    "invalid_rate": 0.03,
    "latency_mean": 0.62
  },
  "error_clusters": [
    {
      "cluster": "colorbar_missing_false_positive",
      "count": 8,
      "examples": [
        {
          "question": "What is the maximum value of the colorbar?",
          "gold": "Not Applicable",
          "pred": "1.0",
          "chart_code_field": "colorbar.exists=true",
          "diagnosis": "spurious vertical axis interpreted as colorbar"
        }
      ],
      "suggested_program_area": "verifier/colorbar_detector"
    }
  ]
}
```

The mutation LLM receives error clusters rather than raw aggregate accuracy only.

---

## 6.6 MAP-Elites and island-based population

Maintain separate islands for behavioral diversity:

```text
Island A: high exact-match accuracy
Island B: strong numeric reasoning
Island C: strong Not Applicable precision
Island D: low latency
Island E: robust trend/comparison reasoning
Island F: robust OCR/text extraction
Island G: OOD transfer candidates
```

Each candidate receives a behavior descriptor:

```python
descriptor = {
    "accuracy_bin": discretize(acc),
    "latency_bin": discretize(latency),
    "na_precision_bin": discretize(na_precision),
    "numeric_acc_bin": discretize(relaxed_numeric),
    "reasoning_acc_bin": discretize(reasoning_acc)
}
```

The program database stores elites per descriptor. Parent sampling balances:

```text
70% exploitation from high-score elites
20% diversity from underexplored islands
10% random archived candidates
```

This is important because a single scalar objective can collapse diversity.

---

## 6.7 Objective function

Use a multi-objective score:

```text
score =
    exact_match
  + 0.25 * relaxed_numeric_accuracy
  + 0.15 * reasoning_accuracy
  + 0.10 * not_applicable_f1
  - 0.05 * latency_mean_seconds
  - 0.03 * latency_p95_seconds
  - 0.10 * invalid_output_rate
  - 0.10 * verifier_contradiction_rate
  - 0.15 * overfit_gap
```

Where:

```text
overfit_gap = max(0, dev_score - heldout_score)
```

For evolution, use a cheaper approximate score on mini-dev. For final selection, use full validation and OOD transfer.

---

## 6.8 Anti-overfitting rules

Because the original project used 128 dev examples and warned against overfitting, this NeurIPS version must explicitly prevent overfitting.

Rules:

```text
1. Never put gold answers in mutation prompts.
2. Show error types and abstract diagnostics, not raw full answer lists, when possible.
3. Maintain separate evolution-dev, model-selection-val, and final-test splits.
4. Run at least 3 independent evolution seeds.
5. Report dev/val/test gaps.
6. Include prompt-only and random-mutation controls.
7. Use chart-type and question-type heldout splits.
```

---

## 7. Datasets

## 7.1 Primary dataset: CharXiv

CharXiv should be the main dataset because it focuses on scientific charts from papers and aligns with the preliminary project.

Use all available subsets:

```text
CharXiv descriptive questions
CharXiv reasoning questions
```

Suggested split:

```text
evolution-dev: 15% of training/development charts
validation: 15%
test: 70% or official held-out if available
```

If official test labels are unavailable, create a reproducible split by chart ID, not by question, to avoid leakage from the same chart appearing in both train/evolution and test.

Important split rule:

```text
All questions for the same chart must stay in the same split.
```

Question groups:

```text
title
axis
tick
legend
colorbar
subplot
chart type
value lookup
trend
comparison
intersection
reasoning
not applicable
```

---

## 7.2 Cross-dataset evaluation

The goal is not necessarily to beat all SOTA systems. The goal is to show that Chart Code and evolution improve reliability across chart styles.

### Dataset 1: ChartQA

Use for natural chart question answering and human-written/synthetic questions. It is useful for testing general chart QA transfer.

### Dataset 2: PlotQA

Use for synthetic plots with controlled values. It is useful for evaluating exact numerical reasoning, value lookup, extrema, comparisons, and axis-scale grounding.

### Dataset 3: DVQA

Use for bar-chart-heavy question answering. It is useful for testing bar geometry, categorical axes, legend handling, and synthetic OOD style.

### Dataset 4: FigureQA

Use for controlled figure reasoning. It is useful for testing relational and comparison questions.

### Optional Dataset 5: Chart-to-text / chart summarization data

Use only for auxiliary Chart Code extraction or trend summarization evaluation. This is optional and should not distract from the main QA experiments.

---

## 7.3 Manually audited Chart Code subset

Create a small high-quality annotated subset for intermediate representation evaluation.

Suggested scale:

```text
ChartCode-300:
  300 charts
  100 CharXiv
  80 ChartQA
  80 PlotQA
  40 DVQA/FigureQA
```

Annotate:

```text
chart type
title
axis labels
legend existence and items
colorbar existence and min/max
subplot count
series count
global trends
major extrema
major intersections
whether the requested element exists
```

This can be manually audited with VLM-assisted pre-labeling to reduce workload.

Metrics:

```text
chart_type_accuracy
axis_label_exact_or_fuzzy_match
legend_item_F1
series_count_accuracy
colorbar_exist_accuracy
colorbar_minmax_numeric_error
trend_accuracy
intersection_detection_F1
turning_point_detection_F1
```

This subset is critical for proving Chart Code quality directly, not only final answer accuracy.

---

## 8. Models

## 8.1 Primary frozen VLMs

Use the same model family as the preliminary project for continuity:

```text
Qwen3-VL-2B-Instruct
Qwen3-VL-2B-Thinking
```

Constraints:

```text
greedy decoding
temperature = 0
no weight training in the main method
batch size adapted to 16 GB VRAM
```

## 8.2 Additional transfer models

If feasible on RTX 4090 16 GB:

```text
Qwen2.5-VL-3B-Instruct
Qwen2.5-VL-7B-Instruct in 4-bit or 8-bit
InternVL small model
LLaVA-style small model
```

These are optional but valuable for generality.

## 8.3 Text-only reasoner

For code-only reasoning:

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct quantized if needed
```

The code-only reasoner can be much cheaper because it receives only selected Chart Code text, not images.

---

## 9. Baselines

A strong NeurIPS paper needs careful baselines.

## 9.1 Direct VLM baselines

```text
B1: Raw image-only VLM
  image + question -> answer

B2: Manual optimized VLM prompt
  question-type-aware prompt + answer normalization

B3: Manual optimized VLM with Not Applicable rules
  current strong manual baseline style
```

## 9.2 Evolution baselines

```text
B4: Prompt-only evolution
  only prompt strings evolve

B5: Post-processing-only evolution
  only normalizer/verifier evolves, no Chart Code

B6: Random code mutation
  random edits or heuristic mutation bank only

B7: Repeated LLM sampling without evolution
  ask mutation LLM repeatedly from same seed, keep best
```

## 9.3 Chart Code baselines

```text
B8: Fixed human-designed Chart Code
  hand-written schema, no evolution

B9: VLM-generated Chart Code only
  VLM extracts JSON, no deterministic CV/OCR

B10: CV/OCR Chart Code only
  deterministic extraction, no VLM semantic extraction

B11: Chart Code + code-only reasoner
  no image fallback

B12: Chart Code + image reasoner
  image and Chart Code both provided
```

## 9.4 EvoChartCode ablations

```text
A1: Full EvoChartCode
A2: No verifier
A3: No uncertainty/provenance fields
A4: No derived relations
A5: No MAP-Elites / island population
A6: No error-aware feedback
A7: No schema evolution
A8: No code selector; serialize full Chart Code
A9: No image fallback
A10: No code-only mode
```

These ablations directly test the contribution of each module.

---

## 10. Metrics

## 10.1 Answer accuracy

```text
Exact match
Case-normalized exact match
Relaxed numeric match
Unit-normalized numeric match
Multiple-choice accuracy if applicable
Semantic answer match using an evaluator model, reported separately
```

Relaxed numeric match:

```python
def relaxed_numeric_match(pred, gold):
    # exact string first
    # parse float/scientific notation/percent
    # allow absolute or relative tolerance
    # handle units
    pass
```

## 10.2 Reliability metrics

```text
invalid_output_rate
format_violation_rate
not_applicable_precision
not_applicable_recall
not_applicable_F1
verifier_contradiction_rate
unsupported_answer_rate
abstention_overuse_rate
```

## 10.3 Chart Code quality metrics

```text
chart_type_accuracy
axis_label_fuzzy_F1
tick_value_numeric_error
legend_item_F1
series_count_accuracy
colorbar_exist_accuracy
colorbar_minmax_error
trend_accuracy
comparison_accuracy
intersection_F1
turning_point_F1
```

## 10.4 Efficiency metrics

```text
mean latency
p50 latency
p95 latency
generated tokens
number of VLM calls
number of OCR/CV calls
GPU memory peak
throughput under batch evaluation
```

## 10.5 Generalization metrics

```text
dev-test gap
chart-type heldout accuracy
question-type heldout accuracy
cross-dataset transfer
cross-model transfer
bootstrap confidence intervals
multi-seed variance
```

---

## 11. Main Experiments

## 11.1 Experiment 1: Main results on CharXiv

Goal: show that EvoChartCode improves frozen VLMs on both descriptive and reasoning questions.

Table:

```text
Method                    Descriptive EM   Reasoning EM   Relaxed Num   NA-F1   Latency
Raw VLM
Manual prompt
Prompt-only evolution
Fixed Chart Code
VLM Chart Code
Full EvoChartCode
```

Report for:

```text
Qwen3-VL-2B-Instruct
Qwen3-VL-2B-Thinking
```

Expected outcome:

```text
Full EvoChartCode should improve reasoning, trend, legend, colorbar,
and Not Applicable questions more than raw image-only VLM.
```

---

## 11.2 Experiment 2: Cross-dataset transfer

Goal: show that the method is not overfit to CharXiv.

Train/evolve on CharXiv evolution-dev only, then test on:

```text
ChartQA
PlotQA
DVQA
FigureQA
```

Table:

```text
Method              CharXiv Test   ChartQA   PlotQA   DVQA   FigureQA
Raw VLM
Manual prompt
Prompt-only evo
Fixed Chart Code
Full EvoChartCode
```

Expected outcome:

```text
Full EvoChartCode should transfer best on structured questions:
axis, legend, value lookup, comparison, trend, and chart-type reasoning.
```

---

## 11.3 Experiment 3: Code-only vs image-only vs hybrid

Goal: prove that Chart Code is a meaningful intermediate representation.

Compare:

```text
Image-only:
  image + question -> answer

Code-only:
  Chart Code + question -> answer

Image + code:
  image + Chart Code + question -> answer

Verifier-guided hybrid:
  choose code-only or image+code based on confidence; verify final answer
```

Important analysis:

```text
code-only should be fastest and interpretable
image-only should handle visual ambiguity better
hybrid should be strongest overall
```

---

## 11.4 Experiment 4: Chart Code quality

Evaluate extracted Chart Code on ChartCode-300.

Metrics:

```text
chart type accuracy
legend item F1
axis label F1
series count accuracy
colorbar existence accuracy
trend accuracy
derived relation F1
```

Ablation:

```text
VLM-only extractor
CV/OCR-only extractor
VLM + CV/OCR extractor
Evolved extractor
Full evolved schema + extractor
```

This experiment is essential because final QA accuracy alone does not prove the intermediate representation is good.

---

## 11.5 Experiment 5: Evolution component ablation

Compare:

```text
Full EvoChartCode
No evolution
Prompt-only evolution
No schema evolution
No verifier evolution
No MAP-Elites
No error feedback
No cascade
Random mutation
Repeated sampling without evolution
```

Report:

```text
best validation score after fixed budget
test score
number of valid candidates
wall-clock search time
search efficiency
```

Plot:

```text
x-axis: evaluated candidates or GPU-hours
y-axis: validation score
curves: full vs ablations
```

---

## 11.6 Experiment 6: Search budget scaling

Run:

```text
generations: 2, 4, 8, 16
candidates per generation: 4, 8
seeds: 3
```

Report:

```text
mean best score
standard deviation
candidate validity rate
latency of evolution
```

This demonstrates whether evolution is actually useful beyond manual design.

---

## 11.7 Experiment 7: Failure attribution

For final systems, manually or semi-automatically classify failures:

```text
perception error
reasoning error
format error
verification error
abstention error
dataset ambiguity
```

Report before/after evolution:

```text
Error Type             Raw VLM   Manual   Fixed Code   EvoChartCode
Perception
Reasoning
Format
Verification
Abstention
```

Expected outcome:

```text
EvoChartCode should reduce format and unsupported-answer errors.
Some perception errors will remain, especially when OCR or visual parsing fails.
```

---

## 11.8 Experiment 8: Robustness to chart perturbations

Apply perturbations:

```text
image resolution downsampling
axis label font change
color palette change
legend relocation
line thickness variation
JPEG compression
cropping margins
```

Evaluate:

```text
accuracy under perturbation
Chart Code field stability
verifier rejection rate
```

This is useful for NeurIPS because it tests robustness beyond standard split accuracy.

---

## 12. Implementation Plan

## 12.1 Repository structure

```text
evochartcode/
  README.md
  environment.yml
  requirements.txt
  pyproject.toml

  configs/
    base.yaml
    charxiv_qwen3vl_2b.yaml
    chartqa_transfer.yaml
    plotqa_transfer.yaml
    ablation_no_verifier.yaml
    ablation_prompt_only.yaml
    evolution_small.yaml
    evolution_full.yaml

  data/
    README.md
    splits/
      charxiv_chart_level_split.json
      chartcode_300_manifest.json
    cache/
      chart_codes/
      ocr/
      resized_images/

  evochartcode/
    __init__.py

    models/
      qwen_vl_runner.py
      text_llm_runner.py
      model_cache.py

    schema/
      chart_code.py
      validators.py
      schema_versions.py

    extraction/
      vlm_extractor.py
      ocr_extractor.py
      cv_extractor.py
      fusion.py
      repair.py

    derived/
      trends.py
      extrema.py
      comparisons.py
      intersections.py
      rankings.py

    reasoning/
      router.py
      selector.py
      serializer.py
      code_reasoner.py
      hybrid_reasoner.py
      prompts.py

    verification/
      verifier.py
      rules_axis.py
      rules_legend.py
      rules_colorbar.py
      rules_trend.py
      normalizer.py

    evolution/
      controller.py
      candidate.py
      mutation.py
      diff_parser.py
      population.py
      map_elites.py
      islands.py
      cascade.py
      feedback.py
      archive.py

    evaluation/
      datasets.py
      metrics.py
      chartcode_metrics.py
      latency.py
      bootstrap.py
      error_analysis.py

    utils/
      image_utils.py
      json_utils.py
      logging_utils.py
      seed.py

  scripts/
    prepare_charxiv.py
    prepare_chartqa.py
    prepare_plotqa.py
    extract_chart_codes.py
    run_eval.py
    run_evolution.py
    run_ablation.py
    run_transfer.py
    export_tables.py
    export_figures.py

  tests/
    test_schema.py
    test_verifier.py
    test_router.py
    test_metrics.py
```

---

## 12.2 Conda environment

Recommended `environment.yml`:

```yaml
name: evochartcode
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - pillow
  - matplotlib
  - tqdm
  - pyyaml
  - pytest
  - pip:
      - torch
      - torchvision
      - transformers
      - accelerate
      - qwen-vl-utils
      - safetensors
      - sentencepiece
      - opencv-python
      - scikit-image
      - pydantic
      - jsonschema
      - rapidfuzz
      - datasets
      - evaluate
      - rich
      - loguru
      - bitsandbytes
      - pytesseract
      - easyocr
      - einops
```

Install:

```bash
conda env create -f environment.yml
conda activate evochartcode
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For RTX 4090 + CUDA 12.8, install the PyTorch build that matches CUDA 12.x. If an exact CUDA 12.8 wheel is unavailable, use the closest officially supported CUDA 12.x wheel and confirm runtime compatibility with `torch.cuda.is_available()`.

---

## 12.3 Memory strategy for 16 GB VRAM

Use:

```text
bf16 where possible
4-bit or 8-bit quantization for optional larger models
single-image evaluation for VLM
cache extracted Chart Code
cache OCR results
resize images to a controlled max dimension
short max_new_tokens for extraction and answering
separate extraction from reasoning
avoid loading multiple VLMs simultaneously
```

Recommended runtime modes:

```text
Mode 1: extraction cache generation
  load VLM extractor, process images, save chart_code.json

Mode 2: code-only reasoning
  unload VLM, load text model or use same model in text mode

Mode 3: image+code fallback evaluation
  load VLM only for examples requiring fallback
```

This makes the system feasible on 16 GB VRAM.

---

## 13. Core Class Definitions

Use Pydantic for schema validation.

```python
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

BBox = list[float]

class TextField(BaseModel):
    text: Optional[str] = None
    bbox: Optional[BBox] = None
    confidence: float = 0.0
    source: Optional[str] = None

class Tick(BaseModel):
    value: str
    position: Optional[list[float]] = None
    numeric_value: Optional[float] = None
    confidence: float = 0.0

class AxisCode(BaseModel):
    label: Optional[str] = None
    unit: Optional[str] = None
    scale: Literal["linear", "log", "categorical", "time", "unknown"] = "unknown"
    range: Optional[list[float]] = None
    ticks: list[Tick] = Field(default_factory=list)
    confidence: float = 0.0

class LegendItem(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    marker: Optional[str] = None
    line_style: Optional[str] = None
    confidence: float = 0.0

class LegendCode(BaseModel):
    exists: bool = False
    bbox: Optional[BBox] = None
    items: list[LegendItem] = Field(default_factory=list)
    confidence: float = 0.0

class SeriesCode(BaseModel):
    id: str
    name: Optional[str] = None
    visual_type: str = "unknown"
    color: Optional[str] = None
    marker: Optional[str] = None
    points_pixel: list[list[float]] = Field(default_factory=list)
    points_data: list[list[float]] = Field(default_factory=list)
    start: Optional[dict[str, Any]] = None
    end: Optional[dict[str, Any]] = None
    confidence: float = 0.0

class DerivedRelations(BaseModel):
    trends: list[dict[str, Any]] = Field(default_factory=list)
    turning_points: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    intersections: list[dict[str, Any]] = Field(default_factory=list)
    extrema: list[dict[str, Any]] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)

class UncertaintyCode(BaseModel):
    low_confidence_fields: list[str] = Field(default_factory=list)
    ambiguous_regions: list[dict[str, Any]] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    global_confidence: float = 0.0

class ChartCode(BaseModel):
    chart_id: str
    image_size: dict[str, int]
    chart_type: str = "unknown"
    title: TextField = Field(default_factory=TextField)
    layout: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, AxisCode] = Field(default_factory=dict)
    legend: LegendCode = Field(default_factory=LegendCode)
    colorbar: dict[str, Any] = Field(default_factory=dict)
    series: list[SeriesCode] = Field(default_factory=list)
    marks: list[dict[str, Any]] = Field(default_factory=list)
    derived_relations: DerivedRelations = Field(default_factory=DerivedRelations)
    uncertainty: UncertaintyCode = Field(default_factory=UncertaintyCode)
    provenance: dict[str, Any] = Field(default_factory=dict)
```

---

## 14. Evolution Controller Pseudocode

```python
def run_evolution(config):
    population = initialize_population(config.seed_programs)
    archive = ProgramArchive()
    map_elites = MapElites(config.behavior_bins)

    for gen in range(config.num_generations):
        parents = population.sample_parents(
            archive=archive,
            map_elites=map_elites,
            strategy=config.parent_sampling
        )

        candidates = []
        for parent in parents:
            feedback = archive.render_feedback(parent.id)
            patch = mutation_llm.propose_patch(
                parent_program=parent,
                feedback=feedback,
                constraints=config.constraints
            )
            child = apply_patch(parent, patch)

            if not static_validate(child):
                archive.add_invalid(child, reason="static validation failed")
                continue

            candidates.append(child)

        for child in candidates:
            result = evaluation_cascade(child, config)
            archive.add(child, result)
            map_elites.update(child, result)

        population = select_next_population(
            archive=archive,
            map_elites=map_elites,
            config=config
        )

        save_generation_summary(gen, archive, map_elites)

    return archive.best_by_metric("final_score")
```

---

## 15. Evaluation Commands

### Prepare data

```bash
conda activate evochartcode

python scripts/prepare_charxiv.py \
  --charxiv_root /path/to/CharXiv \
  --out data/splits/charxiv_chart_level_split.json

python scripts/prepare_chartqa.py --root /path/to/ChartQA
python scripts/prepare_plotqa.py --root /path/to/PlotQA
```

### Extract Chart Code cache

```bash
python scripts/extract_chart_codes.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --split evolution_dev \
  --output data/cache/chart_codes/charxiv_evolution_dev
```

### Run evaluation

```bash
python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method raw_vlm \
  --split validation

python scripts/run_eval.py \
  --config configs/charxiv_qwen3vl_2b.yaml \
  --method full_evochartcode \
  --split validation
```

### Run evolution

```bash
python scripts/run_evolution.py \
  --config configs/evolution_small.yaml \
  --seed 0

python scripts/run_evolution.py \
  --config configs/evolution_full.yaml \
  --seed 0
```

### Run ablations

```bash
python scripts/run_ablation.py \
  --config configs/ablation_no_verifier.yaml

python scripts/run_ablation.py \
  --config configs/ablation_prompt_only.yaml
```

### Export paper tables

```bash
python scripts/export_tables.py \
  --runs outputs/runs \
  --out paper/tables

python scripts/export_figures.py \
  --runs outputs/runs \
  --out paper/figures
```

---

## 16. Proposed Paper Figures and Tables

## 16.1 Figures

### Figure 1: EvoChartCode overview

Show:

```text
image -> Chart Code extractor -> structured code -> selector -> reasoner -> verifier -> answer
```

### Figure 2: Chart Code example

Show one scientific chart with a compact JSON-like Chart Code beside it.

### Figure 3: Evolution loop

Show:

```text
program database -> prompt sampler -> mutation LLM -> candidate patch -> cascade evaluator -> MAP-Elites archive
```

### Figure 4: Accuracy-latency Pareto frontier

Compare raw VLM, manual, prompt-only evolution, fixed Chart Code, full EvoChartCode.

### Figure 5: Error type breakdown

Show reduction in format errors, unsupported answers, and Not Applicable false positives.

### Figure 6: Search budget scaling

Show performance vs number of evaluated candidates.

---

## 16.2 Tables

### Table 1: Main CharXiv results

Columns:

```text
Method
Descriptive EM
Reasoning EM
Relaxed Numeric
NA-F1
Invalid Rate
Latency
```

### Table 2: Cross-dataset transfer

Columns:

```text
Method
CharXiv
ChartQA
PlotQA
DVQA
FigureQA
```

### Table 3: Ablation study

Rows:

```text
Full EvoChartCode
- verifier
- uncertainty
- derived relations
- schema evolution
- MAP-Elites
- error feedback
- cascade
```

### Table 4: Chart Code quality

Columns:

```text
Extractor
Chart Type Acc
Legend F1
Axis F1
Series Count Acc
Trend Acc
Colorbar Acc
```

### Table 5: Efficiency

Columns:

```text
Method
VLM Calls
Mean Latency
P95 Latency
Tokens
Peak VRAM
```

### Table 6: Failure attribution

Rows:

```text
Perception error
Reasoning error
Format error
Verifier error
Abstention error
Dataset ambiguity
```

---

## 17. Expected Results

The expected pattern is:

```text
1. Manual optimized inference improves over raw VLM.
2. Prompt-only evolution improves slightly but saturates.
3. Fixed Chart Code improves interpretability and some structured questions.
4. Full EvoChartCode improves most on:
   - legend/colorbar existence
   - trend questions
   - comparison questions
   - Not Applicable handling
   - exact output validity
5. Code-only is fastest but weaker on noisy perception.
6. Hybrid code+image is strongest overall.
7. Verifier reduces false positives and unsupported answers.
8. MAP-Elites improves search diversity and Pareto frontier quality.
```

The strongest honest claim should be:

> EvoChartCode does not solve all chart reasoning. It improves the reliability and diagnosability of frozen VLM chart QA by converting direct black-box image answering into explicit, verifiable code-grounded reasoning.

Avoid claiming:

```text
- solved chart understanding
- universal SOTA across all chart datasets
- perfect numerical extraction
- training-free approach always beats finetuning
- Chart Code extraction is always correct
```

---

## 18. Risk Analysis and Mitigation

## 18.1 Risk: Chart Code extraction is noisy

Mitigation:

```text
store confidence
store provenance
use verifier
fallback to image+code mode
evaluate Chart Code quality directly
```

## 18.2 Risk: System overfits to CharXiv dev

Mitigation:

```text
chart-level split
held-out validation
cross-dataset transfer
multi-seed evolution
overfit-gap penalty
no gold answers in mutation prompts
```

## 18.3 Risk: OCR/CV pipeline is brittle

Mitigation:

```text
combine VLM and OCR
use fuzzy matching
do not require perfect point extraction for qualitative trend questions
use uncertainty-aware fallback
```

## 18.4 Risk: Too much engineering, not enough research novelty

Mitigation:

```text
focus paper narrative on evolvable intermediate representations
show ablations proving schema evolution and verifier matter
analyze failure decomposition
compare to prompt-only and fixed-code baselines
```

## 18.5 Risk: RTX 4090 16 GB is insufficient for broad experiments

Mitigation:

```text
use 2B/3B primary models
use quantization for 7B transfer only
cache Chart Code
evaluate candidates with cascade
run code-only modes cheaply
reduce image resolution consistently
```

---

## 19. Timeline

Assuming an 8-10 week focused implementation cycle.

### Week 1: Repository and baseline

```text
- Set up repo and environment
- Implement dataset loaders
- Reproduce preliminary raw/manual/evolved baselines
- Implement metric suite
```

### Week 2: Fixed Chart Code v0

```text
- Define Pydantic schema
- Implement VLM JSON extractor
- Implement JSON validation/repair
- Cache Chart Code for CharXiv subset
```

### Week 3: Code-grounded reasoner

```text
- Implement question router
- Implement code selector
- Implement code-only and image+code reasoners
- Implement answer normalizer
```

### Week 4: Verifier

```text
- Implement legend/colorbar/axis/trend verifiers
- Implement Not Applicable handling
- Run first fixed Chart Code experiments
```

### Week 5: Evolution framework

```text
- Implement candidate archive
- Implement mutation LLM interface
- Implement diff parser and static validator
- Implement evaluation cascade
```

### Week 6: MAP-Elites and error feedback

```text
- Implement island population
- Implement error clustering
- Implement structured feedback prompts
- Run small evolution experiments
```

### Week 7: Main experiments

```text
- Run CharXiv main results
- Run ablations
- Run search budget scaling
```

### Week 8: Cross-dataset transfer

```text
- Prepare ChartQA/PlotQA/DVQA/FigureQA
- Run transfer experiments
- Run robustness perturbations
```

### Week 9: Analysis and paper draft

```text
- Error attribution
- Chart Code quality analysis
- Export tables and figures
- Write NeurIPS-style paper draft
```

### Week 10: Final polish

```text
- Re-run key experiments with 3 seeds
- Bootstrap confidence intervals
- Clean code and reproducibility scripts
- Finalize paper
```

---

## 20. NeurIPS 2026 Paper Outline

```text
Title:
  EvoChartCode: Evolution-Guided Chart Code Induction for Reliable Chart Reasoning

Abstract:
  Frozen VLMs struggle with reliable chart reasoning because direct image-to-answer inference
  entangles perception, reasoning, and answer formatting. We introduce Chart Code, an explicit
  structured representation of chart layout, scales, marks, trends, comparisons, and uncertainty,
  and EvoChartCode, a verification-guided evolutionary framework that optimizes the schema,
  extraction program, serialization policy, reasoner, and verifier for Chart Code. Across CharXiv
  and cross-dataset chart QA benchmarks, EvoChartCode improves frozen small VLMs under
  strict inference constraints, yields better accuracy-latency trade-offs than prompt-only evolution,
  and enables interpretable error attribution.

1 Introduction
  - Problem: direct VLM chart reasoning is brittle.
  - Observation: preliminary program optimization helps but remains implicit.
  - Key idea: explicit, evolvable Chart Code.
  - Contributions.

2 Related Work
  - Chart understanding
  - VLM reasoning and chart QA
  - Intermediate representations for visual reasoning
  - Spatial Code and structured scene representations
  - LLM-guided program evolution / AlphaEvolve / FunSearch
  - Verification and neuro-symbolic reasoning

3 Problem Formulation
  - Frozen VLM inference program optimization
  - Chart Code factorization
  - Multi-objective optimization

4 Method
  - Chart Code schema
  - Chart Code extraction
  - Question-conditioned selection
  - Code-grounded reasoning
  - Verifier and normalizer
  - Evolution algorithm

5 Experiments
  - Datasets
  - Models
  - Metrics
  - Baselines
  - Main results
  - Transfer
  - Ablations
  - Search scaling
  - Error attribution

6 Analysis
  - What fields evolve?
  - When does code-only work?
  - Where does image fallback help?
  - Failure cases

7 Limitations
  - extraction quality
  - OCR brittleness
  - compute
  - chart-type coverage

8 Conclusion
```

---

## 21. Minimal Viable Version vs Full NeurIPS Version

## 21.1 Minimal viable version

This version can be done quickly.

```text
- CharXiv only
- Qwen3-VL-2B-Instruct only
- fixed Chart Code schema
- VLM JSON extractor
- question router
- code-only and image+code reasoner
- verifier for legend/colorbar/axis/trend
- prompt-only vs Chart Code comparison
```

This can validate the idea.

## 21.2 Strong NeurIPS version

This is the target.

```text
- CharXiv + ChartQA + PlotQA + DVQA/FigureQA
- Qwen3-VL-2B-Instruct and Thinking
- optional Qwen2.5-VL transfer
- evolved schema/extractor/selector/reasoner/verifier
- MAP-Elites search
- error-aware feedback
- ChartCode-300 intermediate evaluation
- cross-dataset transfer
- multi-seed search
- latency and Pareto analysis
```

---

## 22. Practical First Experiments to Run

Start with these five experiments before building the full framework:

### Experiment A: VLM JSON extraction sanity check

For 50 CharXiv charts:

```text
image -> Qwen3-VL -> Chart Code JSON
```

Measure:

```text
valid JSON rate
chart type accuracy
legend exists accuracy
axis label quality
qualitative trend quality
```

### Experiment B: Code-only QA

Use generated Chart Code:

```text
question + Chart Code -> text-only LLM -> answer
```

Compare against raw VLM.

### Experiment C: Hybrid QA

```text
question + image + selected Chart Code -> VLM -> answer
```

Compare against raw image-only.

### Experiment D: Verifier impact

Add verifier:

```text
without verifier vs with verifier
```

Focus on:

```text
Not Applicable
legend
colorbar
subplot
trend
```

### Experiment E: Evolve one block first

Only evolve:

```text
code_selector + verifier
```

This is lower risk than evolving everything initially.

---

## 23. Why This Can Reach NeurIPS Level

The project has NeurIPS potential if the paper is framed around three deeper ideas:

### 23.1 Explicit representation

Chart Code makes the latent visual reasoning state visible and inspectable.

### 23.2 Evolvable abstraction

The representation is not fixed. The system can discover which fields are useful for answering questions.

### 23.3 Verification-guided inference

The final answer is checked against explicit evidence, reducing hallucination and false positives.

This combination is stronger than:

```text
- prompt engineering
- simple post-processing
- direct AlphaEvolve reproduction
- ordinary chart QA system
```

The strongest conceptual statement is:

> We evolve not only an inference program, but the intermediate language through which a frozen VLM sees and reasons about charts.

---

## 24. Final Deliverables

The final research package should contain:

```text
1. NeurIPS-style paper PDF
2. Full codebase
3. environment.yml
4. reproducible dataset split files
5. scripts for extraction/evaluation/evolution
6. cached or regenerable Chart Code examples
7. tables and figures
8. ablation configs
9. README with exact commands
10. model card / limitations
```

Recommended reproduction command:

```bash
bash reproduce_main.sh
```

with subcommands:

```bash
bash reproduce_extract.sh
bash reproduce_eval.sh
bash reproduce_evolution_small.sh
bash reproduce_tables.sh
```

---

## 25. Final Recommended Abstract Draft

**Abstract.**  
Vision-language models have made rapid progress on chart understanding, yet direct image-to-answer inference remains brittle: perception, reasoning, answer formatting, and abstention are entangled in a single black-box generation step. We propose **EvoChartCode**, a framework for reliable chart reasoning through explicit, evolvable intermediate representations. EvoChartCode first converts a chart image into **Chart Code**, a structured representation containing layout, axes, scales, visual marks, series, derived trends, comparisons, extrema, intersections, uncertainty, and provenance. A question-conditioned selector then exposes only relevant code to a code-grounded reasoner, while a verifier checks that the answer is supported by the extracted chart evidence. Rather than fixing this representation by hand, EvoChartCode uses verification-guided evolutionary program search to improve the Chart Code schema, extraction program, serialization strategy, reasoning policy, and verifier under automatic feedback. Across scientific chart QA and cross-dataset chart reasoning benchmarks, EvoChartCode improves frozen small VLMs over direct prompting, prompt-only evolution, and fixed-code baselines, while yielding stronger abstention behavior, better accuracy-latency trade-offs, and interpretable failure attribution. Our results suggest that for structured visual reasoning, a significant portion of recoverable capability lies not only in model weights, but in the explicit inference language through which models access visual evidence.

---

## 26. Final Recommended Contribution Paragraph

We make four contributions.  
First, we introduce **Chart Code**, an explicit intermediate representation for scientific charts that captures visual layout, axis scales, marks, series geometry, derived trend and comparison primitives, uncertainty, and evidence provenance.  
Second, we propose **EvoChartCode**, a verification-guided evolutionary framework that optimizes the Chart Code schema, extractor, serializer, reasoner, and verifier as a unified inference program for frozen VLMs.  
Third, we develop a multi-objective evaluation protocol that measures exact answer accuracy, relaxed numeric correctness, abstention calibration, invalid-output rate, latency, search efficiency, and Chart Code quality.  
Fourth, we show through cross-dataset experiments and ablations that evolving explicit chart representations improves reliability beyond prompt-only evolution and enables interpretable decomposition of perception, reasoning, verification, and formatting errors.

---

## 27. References to Include in the Paper

The final paper should cite at least:

```text
- AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery
- FunSearch
- Thinking with Spatial Code for Physical-World Video Reasoning
- CharXiv
- ChartQA
- PlotQA
- DVQA
- FigureQA
- DePlot / MatCha / Pix2Struct-style chart understanding if relevant
- LLaVA / Qwen-VL / InternVL for VLM backbones
- MAP-Elites / quality-diversity evolutionary algorithms
- Program synthesis / genetic programming / LLM-guided evolution
```

---

## 28. Bottom Line

The original project is a strong systems engineering result: frozen VLM inference can be improved by program-level optimization. The NeurIPS 2026 version should turn that into a broader research thesis:

> **Structured visual reasoning should not rely on raw image-to-answer generation alone. For charts, we can expose an explicit Chart Code, evolve the code language and its inference program, and verify answers against structured visual evidence.**

This makes the project novel, experimentally rich, feasible on a single RTX 4090 16 GB GPU, and much closer to a publishable NeurIPS-level research contribution.
