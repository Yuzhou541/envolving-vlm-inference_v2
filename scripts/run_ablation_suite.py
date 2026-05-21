"""Run the EvoChartCode ablation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    "configs/ablation_full_evochartcode.yaml",
    "configs/ablation_fixed_chart_code.yaml",
    "configs/ablation_no_verifier.yaml",
    "configs/ablation_no_uncertainty.yaml",
    "configs/ablation_no_derived_relations.yaml",
]


def main():
    parser = argparse.ArgumentParser(description="Run all configured ablations.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        name = Path(config).stem
        if name.startswith("ablation_"):
            out_name = name
        else:
            out_name = f"ablation_{name}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_ablation.py"),
            "--config",
            str(ROOT / config),
            "--output",
            str(args.out_dir / f"{out_name}.json"),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
