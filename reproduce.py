"""Reproduce the key local CharXiv evaluation numbers for all submission modules."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


DEFAULT_MODULES = [
    "starting_scripts",
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
    "best_accuracy",
    "best_speed",
    "best_overall",
]


def module_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dedupe_modules(root: Path, modules: list[str]):
    hash_to_canonical = {}
    aliases = {}
    file_hashes = {}

    for module in modules:
        module_path = root / f"{module}.py"
        digest = module_digest(module_path)
        file_hashes[module] = digest
        canonical = hash_to_canonical.setdefault(digest, module)
        aliases[module] = canonical

    unique_modules = []
    seen = set()
    for module in modules:
        canonical = aliases[module]
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_modules.append(canonical)

    return aliases, file_hashes, unique_modules


def run_evaluation(root: Path, env_name: str, module: str, num_samples: int, metrics_path: Path):
    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(root / "evaluate.py"),
        module,
        "--hf-offline",
        "--num-samples",
        str(num_samples),
        "--output",
        str(metrics_path),
    ]
    subprocess.run(command, cwd=root, check=True)


def build_summary_rows(out_dir: Path, modules: list[str]):
    rows = []
    for module in modules:
        metrics = json.loads((out_dir / f"{module}.json").read_text(encoding="utf-8"))
        score = metrics["accuracy"] - 0.05 * metrics["avg_time_per_query"]
        rows.append(
            {
                "module": module,
                "accuracy": metrics["accuracy"],
                "avg_time_per_query": metrics["avg_time_per_query"],
                "num_errors": metrics["num_errors"],
                "score": score,
                "aliased_from": metrics.get("aliased_from", ""),
            }
        )
    return rows


def write_summary_files(out_dir: Path, rows: list[dict]):
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    header = "| Module | Accuracy | Avg/query (s) | Errors | Score | Notes |\n| --- | ---: | ---: | ---: | ---: | --- |\n"
    lines = []
    for row in rows:
        note = f"same as {row['aliased_from']}" if row["aliased_from"] else ""
        lines.append(
            f"| {row['module']} | {row['accuracy']} | {row['avg_time_per_query']} | {row['num_errors']} | {row['score']} | {note} |"
        )
    summary_text = header + "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    print("\nReproduction summary:")
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the key local evaluation numbers for the submission modules.")
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "repro_outputs")
    parser.add_argument("--env-name", default="vlm")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("modules", nargs="*", default=DEFAULT_MODULES)
    args = parser.parse_args()

    root = args.root_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aliases, file_hashes, unique_modules = dedupe_modules(root, args.modules)
    (out_dir / "module_aliases.json").write_text(
        json.dumps(
            {
                "aliases": aliases,
                "file_hashes": file_hashes,
                "unique_modules": unique_modules,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for module in unique_modules:
        print(f"Evaluating {module}...")
        run_evaluation(root, args.env_name, module, args.num_samples, out_dir / f"{module}.json")

    for module in args.modules:
        canonical = aliases[module]
        source_path = out_dir / f"{canonical}.json"
        target_path = out_dir / f"{module}.json"
        if target_path == source_path:
            continue
        metrics = json.loads(source_path.read_text(encoding="utf-8"))
        metrics["aliased_from"] = canonical
        target_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    summary_rows = build_summary_rows(out_dir, args.modules)
    write_summary_files(out_dir, summary_rows)


if __name__ == "__main__":
    main()
