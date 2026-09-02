from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

FAILURE_CLASSES = {
    "IMPLEMENTATION_FAILURE",
    "NUMERICAL_FAILURE",
    "SEARCH_FAILURE",
    "REPRESENTATION_FAILURE",
    "DATA_FAILURE",
    "BENCHMARK_CONFOUND",
    "CREDIT_ASSIGNMENT_FAILURE",
    "GENERALIZATION_FAILURE",
    "RESOURCE_FAILURE",
    "STATISTICAL_POWER_FAILURE",
    "UNKNOWN_FAILURE",
}

REPRODUCTION_CLASSES = {
    "EXACTLY_REPRODUCED",
    "STATISTICALLY_REPRODUCED",
    "NOT_REPRODUCED",
    "CANNOT_REPRODUCE_MISSING_DEPENDENCY",
    "CANNOT_REPRODUCE_MISSING_SOURCE",
}

RESULT_REQUIRED_FIELDS = {
    "version",
    "parent",
    "research_question",
    "hypothesis",
    "single_change",
    "baselines",
    "pass_gate",
    "development_result",
    "fresh_audit_result",
    "resource_accounting",
    "pass",
    "failure_classification",
    "caveats",
    "next_question",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inclusive_range(bounds: Iterable[int]) -> set[int]:
    start, end = list(bounds)
    if end < start:
        raise ValueError(f"invalid seed range {start}..{end}")
    return set(range(int(start), int(end) + 1))


def assert_seed_ranges_disjoint(seed_ranges: dict[str, list[int]]) -> None:
    expanded = {name: inclusive_range(bounds) for name, bounds in seed_ranges.items()}
    names = sorted(expanded)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = expanded[left] & expanded[right]
            if overlap:
                raise ValueError(f"seed ranges overlap: {left}/{right}: {sorted(overlap)[:5]}")


def validate_result_schema(data: dict[str, Any], *, config: dict[str, Any] | None = None) -> None:
    missing = sorted(RESULT_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(f"result missing required fields: {missing}")
    if not isinstance(data["pass"], bool):
        raise ValueError("result pass must be boolean")
    if not isinstance(data["resource_accounting"], dict):
        raise ValueError("resource_accounting must be an object")
    if not isinstance(data["failure_classification"], list):
        raise ValueError("failure_classification must be a list")
    invalid_classes = sorted(set(data["failure_classification"]) - FAILURE_CLASSES)
    if invalid_classes:
        raise ValueError(f"invalid failure classifications: {invalid_classes}")
    if data["pass"] is False and not data["failure_classification"]:
        raise ValueError("failed result requires at least one failure classification")
    if config is not None:
        if "pass_gate" not in config:
            raise ValueError("config missing pass_gate")
        if canonical_fingerprint(config["pass_gate"]) != canonical_fingerprint(data["pass_gate"]):
            raise ValueError("pass gate changed between config and result")
        if config.get("pass_gate_frozen_before_run") is not True:
            raise ValueError("config must freeze pass gate before run")
        if "seed_ranges" in config:
            assert_seed_ranges_disjoint(config["seed_ranges"])


def validate_integrity_manifest() -> None:
    manifest_path = ROOT / "experiments" / "v836_recovery" / "integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in ("preserved_files", "preserved_registry_files"):
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.exists():
                raise ValueError(f"integrity path missing: {relative}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"integrity mismatch: {relative}: {actual} != {expected}")


def validate_v836_recovery() -> None:
    forensic = json.loads((ROOT / "experiments" / "v836_recovery" / "V836_FORENSIC_SPEC.json").read_text(encoding="utf-8"))
    if forensic.get("historical_status") != "PASS":
        raise ValueError("historical V836 status must remain PASS")
    if forensic.get("original_pass_gate", {}).get("preserved_boolean") != "V836_PASS=true":
        raise ValueError("preserved V836 boolean gate evidence changed")

    required_recovery_docs = [
        ROOT / "experiments" / "v836_recovery" / "V836_FAILURE_TREE.md",
        ROOT / "experiments" / "v836_recovery" / "baseline" / "FAILURE.md",
    ]
    for path in required_recovery_docs:
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"required recovery documentation missing or empty: {path.relative_to(ROOT)}")

    reproduction = json.loads((ROOT / "experiments" / "v836_recovery" / "v836_reproduction_results.json").read_text(encoding="utf-8"))
    classification = reproduction.get("reproduction_classification")
    if classification not in REPRODUCTION_CLASSES:
        raise ValueError(f"invalid reproduction classification: {classification}")
    if reproduction.get("historical_status") != "PASS":
        raise ValueError("reproduction record rewrote historical V836 status")

    proposal = json.loads((ROOT / "experiments" / "v836_recovery" / "PROPOSED_NEXT_EXPERIMENT.json").read_text(encoding="utf-8"))
    if proposal.get("status") != "NOT_RUN_BLOCKED_BY_V836_REPRODUCTION":
        raise ValueError("post-V836 proposal must remain NOT_RUN while reproduction is blocked")
    assert_seed_ranges_disjoint(proposal["seed_ranges"])
    if proposal.get("fixed_gate_before_run", {}).get("gate_frozen_before_run") is not True:
        raise ValueError("proposed next gate must be frozen before any future run")


def validate_frontier_matrix() -> None:
    matrix = json.loads((ROOT / "experiments" / "post_v836_frontier.json").read_text(encoding="utf-8"))
    records = matrix.get("records", [])
    if len(records) != 428:
        raise ValueError(f"frontier matrix must contain 428 records, found {len(records)}")
    numbers = {int(record["number"]) for record in records}
    expected = set(range(450, 837))
    if numbers != expected:
        raise ValueError(f"frontier numeric coverage mismatch: missing={sorted(expected - numbers)} extra={sorted(numbers - expected)}")
    required_record_fields = {
        "version",
        "question",
        "mechanism",
        "human_supplied_scaffold",
        "what_was_learned",
        "what_was_fixed",
        "pass_fail",
        "fresh_audit_status",
        "main_lesson",
        "remaining_limitation",
    }
    for record in records:
        missing = required_record_fields - set(record)
        if missing:
            raise ValueError(f"{record.get('version')} missing matrix fields: {sorted(missing)}")


def validate_active_result_files() -> None:
    experiments = ROOT / "experiments"
    for result_path in experiments.glob("**/variants/**/results.json"):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        config_path = result_path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        validate_result_schema(data, config=config)


def main() -> int:
    validate_integrity_manifest()
    validate_v836_recovery()
    validate_frontier_matrix()
    validate_active_result_files()
    print("active research validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
