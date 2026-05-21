"""Lightweight verification-guided evolution utilities."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class CandidateProgram:
    candidate_id: str
    parent_id: str | None
    policy: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    behavior: dict[str, str] = field(default_factory=dict)
    mutation_summary: str = ""

    @property
    def score(self) -> float:
        return float(self.metrics.get("score", self.metrics.get("exact_match", 0.0)))


class ProgramArchive:
    def __init__(self):
        self.records: list[CandidateProgram] = []

    def add(self, candidate: CandidateProgram):
        self.records.append(candidate)

    def best(self) -> CandidateProgram:
        if not self.records:
            raise ValueError("Program archive is empty.")
        return max(self.records, key=lambda item: (item.score, item.metrics.get("exact_match", 0.0)))

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


class MapElites:
    def __init__(self, bins: list[str]):
        self.bins = bins
        self.elites: dict[str, CandidateProgram] = {}

    def key(self, candidate: CandidateProgram) -> str:
        return "|".join(candidate.behavior.get(bin_name, "unknown") for bin_name in self.bins)

    def update(self, candidate: CandidateProgram):
        key = self.key(candidate)
        current = self.elites.get(key)
        if current is None or candidate.score > current.score:
            self.elites[key] = candidate

    def to_json(self) -> dict[str, dict[str, Any]]:
        return {key: asdict(value) for key, value in self.elites.items()}


def mutate_policy(parent: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str]:
    child = json.loads(json.dumps(parent))
    schema = child.setdefault("schema", {"optional_fields": []})
    extractor = child.setdefault("extractor", {"prompt_style": "strict_json", "include_uncertainty": True})
    selector = child.setdefault("selector", {"max_serialized_chars": 4096, "include_provenance": False})
    reasoner = child.setdefault("reasoner", {"mode": "code_only", "max_answer_tokens": 128})
    verifier = child.setdefault("verifier", {"min_confidence": 0.4, "numeric_rel_tol": 0.05, "rules": []})
    choices = []

    def add_numeric(name: str, target: dict[str, Any], key: str, step: float, low: float, high: float):
        choices.append(("numeric", name, target, key, step, low, high))

    def add_toggle(name: str, target: dict[str, Any], key: str):
        choices.append(("toggle", name, target, key, None, None, None))

    def add_append(name: str, target: dict[str, Any], key: str, values: list[str]):
        choices.append(("append", name, target, key, values, None, None))

    add_append("schema.optional_fields", schema, "optional_fields", ["data_units", "text_regions", "visual_density", "axis_orientation"])
    add_toggle("extractor.include_uncertainty", extractor, "include_uncertainty")
    add_append("extractor.prompt_constraints", extractor, "prompt_constraints", ["return_null_for_missing", "quote_all_text", "separate_axis_and_colorbar"])
    add_numeric("selector.max_serialized_chars", selector, "max_serialized_chars", 256, 512, 8192)
    add_toggle("selector.include_provenance", selector, "include_provenance")
    add_append("selector.extra_blocks", selector, "extra_blocks", ["uncertainty", "provenance", "layout", "derived_relations"])
    add_numeric("reasoner.max_answer_tokens", reasoner, "max_answer_tokens", 16, 32, 256)
    add_numeric("verifier.min_confidence", verifier, "min_confidence", 0.05, 0.0, 1.0)
    add_numeric("verifier.numeric_rel_tol", verifier, "numeric_rel_tol", 0.01, 0.0, 0.5)
    add_append("verifier.rules", verifier, "rules", ["axis_label", "tick_value", "difference", "ranking", "open_ended"])

    kind, name, target, key, payload, low, high = rng.choice(choices)
    if kind == "numeric":
        current = target.get(key, 0.5 if "confidence" in key else 1024)
        next_value = float(current) + rng.choice([-1, 1]) * float(payload)
        next_value = max(float(low), min(float(high), next_value))
        target[key] = int(next_value) if isinstance(current, int) else round(next_value, 4)
    elif kind == "toggle":
        target[key] = not bool(target.get(key, False))
    elif kind == "append":
        values = target.setdefault(key, [])
        candidate = rng.choice(payload)
        if candidate in values:
            values.remove(candidate)
        else:
            values.append(candidate)
    return child, f"Mutated {name}."


def behavior_from_metrics(metrics: dict[str, float]) -> dict[str, str]:
    exact = metrics.get("exact_match", 0.0)
    latency = metrics.get("mean_latency", 0.0)
    return {
        "accuracy_bin": "high" if exact >= 0.6 else "mid" if exact >= 0.3 else "low",
        "latency_bin": "fast" if latency <= 1.0 else "slow",
        "abstention_bin": "high" if metrics.get("na_f1", 0.0) >= 0.7 else "mid" if metrics.get("na_f1", 0.0) >= 0.3 else "low",
    }


def run_evolution(
    seed_policy: dict[str, Any],
    evaluator: Callable[[dict[str, Any]], dict[str, float]],
    generations: int,
    candidates_per_generation: int,
    population_size: int,
    seed: int,
) -> tuple[ProgramArchive, MapElites]:
    rng = random.Random(seed)
    archive = ProgramArchive()
    map_elites = MapElites(["accuracy_bin", "latency_bin", "abstention_bin"])

    seed_metrics = evaluator(seed_policy)
    seed_candidate = CandidateProgram(
        candidate_id="seed",
        parent_id=None,
        policy=seed_policy,
        metrics=seed_metrics,
        behavior=behavior_from_metrics(seed_metrics),
        mutation_summary="Initial policy.",
    )
    archive.add(seed_candidate)
    map_elites.update(seed_candidate)
    population = [seed_candidate]

    counter = 0
    for generation in range(generations):
        for _ in range(candidates_per_generation):
            parent = rng.choice(sorted(population, key=lambda item: item.score, reverse=True)[: max(1, min(3, len(population)))])
            child_policy, summary = mutate_policy(parent.policy, rng)
            counter += 1
            metrics = evaluator(child_policy)
            candidate = CandidateProgram(
                candidate_id=f"g{generation:02d}_c{counter:03d}",
                parent_id=parent.candidate_id,
                policy=child_policy,
                metrics=metrics,
                behavior=behavior_from_metrics(metrics),
                mutation_summary=summary,
            )
            archive.add(candidate)
            map_elites.update(candidate)
            population = sorted(archive.records, key=lambda item: item.score, reverse=True)[:population_size]
    return archive, map_elites


def write_evolution_outputs(out_dir: str | Path, archive: ProgramArchive, map_elites: MapElites):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "archive.json").write_text(json.dumps(archive.to_json(), indent=2) + "\n", encoding="utf-8")
    (path / "map_elites.json").write_text(json.dumps(map_elites.to_json(), indent=2) + "\n", encoding="utf-8")
    (path / "best_policy.json").write_text(json.dumps(asdict(archive.best()), indent=2) + "\n", encoding="utf-8")
