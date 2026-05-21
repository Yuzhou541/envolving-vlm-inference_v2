"""Evaluator for VLM inference using the local CharXiv development split."""
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    from huggingface_hub.utils import logging as hf_logging
except ImportError:  # Optional dependency path.
    hf_logging = None

try:
    from transformers.utils import logging as transformers_logging
except ImportError:  # Transformers is required at runtime, but keep import-time failure readable.
    transformers_logging = None

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

if transformers_logging is not None:
    transformers_logging.set_verbosity_error()
if hf_logging is not None:
    hf_logging.set_verbosity_error()

from descriptive_utils import build_descriptive_quries


def load_charxiv_data(num_samples=128):
    images_dir = CHARXIV_PATH / "images"
    if not any(images_dir.glob("*.jpg")):
        raise FileNotFoundError(
            f"No chart images found in {images_dir}. Download the CharXiv image archive first."
        )
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    queries = build_descriptive_quries(data, str(images_dir))
    if num_samples is not None:
        queries = dict(list(queries.items())[:num_samples])
    return queries, data


def evaluate(program, num_samples=128, return_queries=False):
    queries, ground_truth_data = load_charxiv_data(num_samples)

    num_errors = 0
    start_time = time.time()
    for query_key, query in queries.items():
        try:
            response = program.vlm_inference(
                image_path=query["figure_path"],
                question=query["question"],
            )
            query["response"] = response
        except Exception as e:
            print(f"Error on {query_key}: {e}")
            traceback.print_exc()
            query["response"] = "ERROR"
            num_errors += 1
    total_time = time.time() - start_time

    correct = 0
    total = 0
    for query_key, query in queries.items():
        if "response" not in query:
            continue
        figure_id, subq_idx = query_key.split("_")
        gt_entry = ground_truth_data.get(figure_id)
        if gt_entry is None:
            continue
        gt_answer = gt_entry["answers"][int(subq_idx)]
        model_response = query["response"].strip()
        if model_response.lower() == gt_answer.lower():
            correct += 1
        total += 1

        query["is_correct"] = model_response.lower() == gt_answer.lower()

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "num_evaluated": total,
        "num_errors": num_errors,
        "total_time": total_time,
        "avg_time_per_query": total_time / total if total > 0 else 0.0,
    }
    if return_queries:
        return metrics, queries
    return metrics


def load_program_module(module_name):
    module = importlib.import_module(module_name)
    if not hasattr(module, "vlm_inference"):
        raise AttributeError(
            f"Module '{module_name}' must define a vlm_inference(image_path, question) function."
        )
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM inference module on the local CharXiv development split."
    )
    parser.add_argument(
        "program",
        help="Python module name to evaluate, for example 'starting_scripts' or 'manual_instruct'.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of descriptive validation queries to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--save-responses",
        type=Path,
        help="Optional path to save per-query prompts, responses, and correctness flags as JSON.",
    )
    parser.add_argument(
        "--hf-offline",
        action="store_true",
        help="Force local cached model files only for cleaner repeated timing runs.",
    )
    args = parser.parse_args()

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    program = load_program_module(args.program)
    want_queries = args.save_responses is not None
    result = evaluate(program, num_samples=args.num_samples, return_queries=want_queries)
    if want_queries:
        metrics, queries = result
    else:
        metrics = result

    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.save_responses is not None:
        args.save_responses.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
