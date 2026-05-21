"""Local-Qwen source-code mutation evolution in throwaway workspaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_FILES = [
    Path("evochartcode/selector.py"),
    Path("evochartcode/verifier.py"),
    Path("evochartcode/serializer.py"),
    Path("evochartcode/extractor.py"),
]


@dataclass
class MutationCandidate:
    candidate_id: str
    parent_id: str | None
    model_name: str
    patch_path: str
    status: str
    compile_ok: bool
    smoke_ok: bool
    metrics: dict[str, Any]
    behavior: dict[str, str]
    patch_lines: int
    error: str = ""

    @property
    def score(self) -> float:
        if self.status != "accepted":
            return -1.0
        return float(self.metrics.get("exact_match", 0.0)) - 0.05 * float(self.metrics.get("mean_latency", 0.0))


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_target_context(max_chars_per_file: int = 16000) -> str:
    chunks = []
    for rel_path in TARGET_FILES:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n# ... truncated ...\n"
        chunks.append(f"### {rel_path}\n```python\n{text}\n```")
    return "\n\n".join(chunks)


def load_one_coder(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_patch(model_name: str, fallback_model: str, generation: int, candidate: int, mock: bool = False) -> tuple[str, str]:
    if mock:
        return "mock", ""
    prompt = f"""You are improving EvoChartCode chart reasoning.
Return only one unified diff. Do not explain it.
Allowed files:
- evochartcode/selector.py
- evochartcode/verifier.py
- evochartcode/serializer.py
- evochartcode/extractor.py

Goal for candidate g{generation:02d}_c{candidate:03d}:
Improve evidence selection, verification, or extraction prompts for chart QA while keeping behavior deterministic.
Avoid adding dependencies. Keep the patch small.

Current source:
{read_target_context()}
"""
    last_error = None
    for used_model in [model_name, fallback_model]:
        tokenizer = None
        model = None
        try:
            tokenizer, model = load_one_coder(used_model)
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=1600, do_sample=True, temperature=0.4, top_p=0.9)
            generated = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            return used_model, extract_diff(generated)
        except Exception as exc:
            last_error = exc
        finally:
            try:
                del model
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    raise RuntimeError(f"Could not generate patch with local Qwen Coder: {last_error}")


def extract_diff(text: str) -> str:
    if "```diff" in text:
        text = text.split("```diff", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start = text.find("diff --git")
    if start == -1:
        start = text.find("--- ")
    return text[start:].strip() if start != -1 else text.strip()


def patch_line_count(patch_text: str) -> int:
    return sum(1 for line in patch_text.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def behavior(metrics: dict[str, Any], patch_lines: int) -> dict[str, str]:
    exact = float(metrics.get("exact_match", 0.0))
    latency = float(metrics.get("mean_latency", 0.0))
    invalid = float(metrics.get("invalid_rate", 0.0))
    na_f1 = float(metrics.get("na_f1", 0.0))
    return {
        "accuracy_bin": "high" if exact >= 0.6 else "mid" if exact >= 0.3 else "low",
        "latency_bin": "fast" if latency <= 1.0 else "slow",
        "abstention_bin": "high" if na_f1 >= 0.7 else "mid" if na_f1 >= 0.3 else "low",
        "invalid_bin": "clean" if invalid == 0.0 else "dirty",
        "patch_size_bin": "small" if patch_lines <= 30 else "medium" if patch_lines <= 120 else "large",
    }


def copy_workspace(dest: Path):
    for rel in ["evochartcode", "scripts"]:
        shutil.copytree(ROOT / rel, dest / rel, ignore=shutil.ignore_patterns("__pycache__"))


def validate_candidate(workspace: Path, patch_text: str, eval_config: Path, limit: int) -> tuple[bool, bool, dict[str, Any], str]:
    if patch_text:
        patch_file = workspace / "candidate.patch"
        patch_file.write_text(patch_text + "\n", encoding="utf-8")
        check = subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=workspace, text=True, capture_output=True)
        if check.returncode != 0:
            return False, False, {}, check.stderr.strip() or check.stdout.strip()
        apply = subprocess.run(["git", "apply", str(patch_file)], cwd=workspace, text=True, capture_output=True)
        if apply.returncode != 0:
            return False, False, {}, apply.stderr.strip() or apply.stdout.strip()

    compile_cmd = [sys.executable, "-m", "compileall", str(workspace / "evochartcode"), str(workspace / "scripts")]
    compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
    if compiled.returncode != 0:
        return False, False, {}, compiled.stderr.strip() or compiled.stdout.strip()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    eval_cmd = [
        sys.executable,
        str(workspace / "scripts" / "run_eval.py"),
        "--config",
        str(eval_config),
        "--method",
        "code_only",
        "--split",
        "validation",
        "--limit",
        str(limit),
    ]
    smoke = subprocess.run(eval_cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
    if smoke.returncode != 0:
        return True, False, {}, smoke.stderr.strip() or smoke.stdout.strip()
    try:
        metrics = json.loads(smoke.stdout[smoke.stdout.find("{") :])
    except Exception as exc:
        return True, False, {}, f"could not parse smoke metrics: {exc}; output={smoke.stdout[-500:]}"
    return True, True, metrics, ""


def main():
    parser = argparse.ArgumentParser(description="Run source-code mutation evolution with local Qwen Coder.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--fallback-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--candidates-per-generation", type=int, default=1)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/charxiv_qwen3vl_2b.yaml"))
    parser.add_argument("--smoke-limit", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_mutation_evolution"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive: list[MutationCandidate] = []
    parent_id: str | None = None
    for generation in range(args.generations):
        for candidate_index in range(args.candidates_per_generation):
            candidate_id = f"g{generation:02d}_c{len(archive) + 1:03d}"
            started = time.perf_counter()
            try:
                used_model, patch_text = generate_patch(args.model, args.fallback_model, generation, len(archive) + 1, mock=args.mock)
                patch_path = args.output_dir / "patches" / f"{candidate_id}.patch"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_path.write_text(patch_text + "\n", encoding="utf-8")
                patch_lines = patch_line_count(patch_text)
                with tempfile.TemporaryDirectory(prefix="evochartcode_mutation_") as tmp:
                    workspace = Path(tmp)
                    copy_workspace(workspace)
                    compile_ok, smoke_ok, metrics, error = validate_candidate(workspace, patch_text, args.eval_config.resolve(), args.smoke_limit)
                status = "accepted" if compile_ok and smoke_ok else "rejected"
                if not patch_text.strip():
                    status = "baseline" if args.mock else "rejected"
                metrics.setdefault("wall_time", time.perf_counter() - started)
                record = MutationCandidate(
                    candidate_id=candidate_id,
                    parent_id=parent_id,
                    model_name=used_model,
                    patch_path=str(patch_path),
                    status=status,
                    compile_ok=compile_ok,
                    smoke_ok=smoke_ok,
                    metrics=metrics,
                    behavior=behavior(metrics, patch_lines),
                    patch_lines=patch_lines,
                    error=error,
                )
            except Exception as exc:
                record = MutationCandidate(
                    candidate_id=candidate_id,
                    parent_id=parent_id,
                    model_name=args.model,
                    patch_path="",
                    status="error",
                    compile_ok=False,
                    smoke_ok=False,
                    metrics={"wall_time": time.perf_counter() - started},
                    behavior={},
                    patch_lines=0,
                    error=str(exc),
                )
            archive.append(record)
            if record.status == "accepted":
                parent_id = record.candidate_id
            write_json(args.output_dir / "archive.json", [asdict(item) for item in archive])

    accepted = [item for item in archive if item.status == "accepted"]
    best = max(accepted, key=lambda item: item.score) if accepted else max(archive, key=lambda item: item.score)
    map_elites: dict[str, dict[str, Any]] = {}
    for item in archive:
        key = "|".join(item.behavior.get(name, "unknown") for name in ["accuracy_bin", "latency_bin", "abstention_bin", "invalid_bin", "patch_size_bin"])
        current = map_elites.get(key)
        if current is None or item.score > float(current.get("score", -1.0)):
            payload = asdict(item)
            payload["score"] = item.score
            map_elites[key] = payload
    write_json(args.output_dir / "best_candidate.json", asdict(best) | {"score": best.score})
    write_json(args.output_dir / "map_elites.json", map_elites)
    print(json.dumps({"output_dir": str(args.output_dir), "best": best.candidate_id, "accepted": len(accepted)}, indent=2))


if __name__ == "__main__":
    main()
