from __future__ import annotations

import hashlib
import json
import runpy
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

V837_FAILURE_CLASSES = FAILURE_CLASSES | {
    "MOTIF_DETECTION_FAILURE",
    "CAUSAL_VALIDATION_FAILURE",
    "COMPRESSION_FAILURE",
    "RETRIEVAL_FAILURE",
    "REUSE_FAILURE",
    "INPUT_ACCESS_FAILURE",
    "STATE_UPDATE_FAILURE",
    "INTERACTION_BASIS_FAILURE",
    "CAPACITY_WITHOUT_GENERALIZATION",
    "REGULARIZATION_ONLY_EFFECT",
    "MESSAGE_MEDIATION_FAILURE",
    "PARAMETER_COUNT_CONFOUND",
    "REFERENCE_MODEL_FAILURE",
    "OPTIMIZATION_BUDGET_FAILURE",
    "SAMPLE_EFFICIENCY_FAILURE",
    "CAPACITY_FAILURE",
    "REPRESENTATION_FAMILY_FAILURE_STRENGTHENED",
    "BENCHMARK_LEARNABILITY_UNRESOLVED",
    "REFERENCE_MECHANISM_LOCALIZATION_FAILURE",
    "ADAPTIVE_UPDATE_FAILURE",
    "VECTOR_UPDATE_FAILURE",
    "CANDIDATE_CONDITIONING_FAILURE",
    "MECHANISM_COUPLING_REQUIRED",
    "REPRESENTATION_ADEQUATE_SAMPLE_INEFFICIENT",
}
V837_GATE_SHA256 = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
V837_CAPACITY_CRITERION_SHA256 = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
V837_IMMUTABLE_HASHES = {
    "experiments/v837_primitive_invention/v837/results.json": "5fed69cc990be5c6f64a5229f59ff7f27af0c1fc26398bdfbe80ee46255eef14",
    "experiments/v837_primitive_invention/v837b/results.json": "f131110969e7700ec0cd9a82825e8554a51a9c05bb308d54625452db54e35cb0",
    "experiments/v837_primitive_invention/v837c/results.json": "994195fdd0e32e12ec44521ea782c1fc3561b8f596fd4a70e9d59f335fe7d009",
    "experiments/v837_primitive_invention/BLOCKER_ANALYSIS.md": "4eea85cbfb2fb9e379675765038527daf2ad6a49aa0721e861c9cc61b0155a20",
    "experiments/v837_primitive_invention/final_resource_accounting.json": "c712ea3c0771ebc398e4ccb80a4d0ffe0d8ead946d42fd460633577fbb3d9b37",
}
V837_RESULT_FIELDS = {
    "version", "parent", "research_question", "hypothesis", "single_change", "substrate_version",
    "task_families", "development_seeds", "validation_seeds", "fresh_audit_seeds", "baselines",
    "metrics", "resource_accounting", "motifs", "primitive_archive", "pass_gate", "pass",
    "failure_classification", "caveats", "next_question", "gate_file_sha256",
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
    allowed_proposal_status = {
        "NOT_RUN_BLOCKED_BY_V836_REPRODUCTION",
        "SUPERSEDED_BY_INDEPENDENT_V837_LINEAGE_AUTHORIZATION",
    }
    if proposal.get("status") not in allowed_proposal_status:
        raise ValueError(f"unexpected post-V836 proposal status: {proposal.get('status')}")
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


def validate_v837_lineage() -> None:
    base = ROOT / "experiments" / "v837_primitive_invention"
    gate_path = base / "frozen_gates.json"
    if sha256_file(gate_path) != V837_GATE_SHA256:
        raise ValueError("V837 frozen gate hash changed")
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    assert_seed_ranges_disjoint(gates["seed_ranges"])

    status = json.loads((base / "lineage_status.json").read_text(encoding="utf-8"))
    if status.get("historical_boundary", {}).get("v836_status") != "PASS":
        raise ValueError("V837 lineage rewrote historical V836 status")
    if status.get("historical_boundary", {}).get("v837_relation") != "independent_post_v836_lineage":
        raise ValueError("V837 must remain explicitly independent of V836 repair lineage")
    if status.get("outcome") not in {"A_MILESTONE_PASSED", "B_MILESTONE_FAILED_HONESTLY"}:
        raise ValueError("V837 lineage has no valid closed outcome")

    variants = ["v837", "v837b", "v837c"]
    completed = []
    for version in variants:
        result_path = base / version / "results.json"
        if not result_path.exists():
            continue
        completed.append(version)
        data = json.loads(result_path.read_text(encoding="utf-8"))
        missing = sorted(V837_RESULT_FIELDS - set(data))
        if missing:
            raise ValueError(f"{version} missing V837 result fields: {missing}")
        if data.get("gate_file_sha256") != V837_GATE_SHA256:
            raise ValueError(f"{version} references wrong frozen gate hash")
        if not isinstance(data.get("resource_accounting"), dict) or not data["resource_accounting"]:
            raise ValueError(f"{version} missing resource accounting")
        if "random" not in json.dumps(data.get("baselines", {})).lower():
            raise ValueError(f"{version} missing random matched control")
        if data.get("pass") is False:
            classes = data.get("failure_classification", [])
            if not classes:
                raise ValueError(f"{version} failure has no classification")
            invalid = sorted(set(classes) - V837_FAILURE_CLASSES)
            if invalid:
                raise ValueError(f"{version} has invalid failure classes: {invalid}")
    if not completed:
        raise ValueError("V837 lineage has no completed experimental result")

    audit = json.loads((base / "audit" / "audit_results.json").read_text(encoding="utf-8"))
    if status["outcome"] == "A_MILESTONE_PASSED":
        if audit.get("status") != "PASS":
            raise ValueError("successful V837 lineage requires completed fresh audit")
    else:
        if len(completed) < 3 or not status.get("stop_rule_triggered"):
            raise ValueError("Outcome B requires three controlled failures and stop-rule evidence")
        if status.get("primary_blocker") not in V837_FAILURE_CLASSES:
            raise ValueError("Outcome B requires a valid primary blocker classification")
        if audit.get("status") != "NOT_RUN_PREREQUISITE_FAILURE" or audit.get("episodes_consumed") != 0:
            raise ValueError("failed V837 lineage must preserve fresh-audit seeds")
        for required in (
            base / "BLOCKER_ANALYSIS.md",
            base / "failures" / "blocker_diagnostic_results.json",
            base / "failures" / "blocker_data_diagnostic_results.json",
            ROOT / "docs" / "V837_PRIMITIVE_INVENTION_REPORT.md",
        ):
            if not required.exists() or not required.read_text(encoding="utf-8", errors="ignore").strip():
                raise ValueError(f"required V837 failure evidence missing: {required.relative_to(ROOT)}")
        if status.get("primitives_promoted"):
            raise ValueError("Outcome B before motif validation cannot promote primitives")

    archive_source = (base / "common" / "primitive_archive.py").read_text(encoding="utf-8").lower()
    retrieve_block = archive_source.split("def retrieve", 1)[1].split("def increment_usage", 1)[0]
    for forbidden in ("task_family", "domain_label", "family_label"):
        if forbidden in retrieve_block:
            raise ValueError(f"primitive retrieval leaks forbidden label: {forbidden}")


def validate_v837_representation_recovery() -> None:
    base = ROOT / "experiments" / "v837_primitive_invention"
    recovery = base / "v837d"
    if not recovery.exists():
        return

    for relative, expected in V837_IMMUTABLE_HASHES.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"historical V837 artifact changed during representation recovery: {relative}")

    config = json.loads((recovery / "config.json").read_text(encoding="utf-8"))
    gate_reference = json.loads((recovery / "frozen_gate_reference.json").read_text(encoding="utf-8"))
    if config.get("historical_gate_hash") != V837_GATE_SHA256:
        raise ValueError("V837d config references the wrong historical gate hash")
    if config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837d capacity criterion changed")
    if gate_reference.get("historical_v837_gate_sha256") != V837_GATE_SHA256:
        raise ValueError("V837d gate reference changed")
    if gate_reference.get("capacity_criterion_sha256") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837d capacity criterion reference changed")
    if not str(config.get("single_change", "")).strip():
        raise ValueError("V837d requires an explicit single_change")
    if config.get("diagnostic_data", {}).get("fresh_audit_used") is not False:
        raise ValueError("V837d must not consume fresh-audit seeds")
    if config.get("primitive_mining_allowed") is not False:
        raise ValueError("primitive mining must remain blocked during V837d capacity recovery")

    audit = json.loads((base / "audit" / "audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise ValueError("representation recovery consumed fresh-audit episodes")
    status = json.loads((base / "lineage_status.json").read_text(encoding="utf-8"))
    if status.get("primitives_promoted"):
        raise ValueError("representation recovery promoted a primitive before competence recovery")

    broadcast_path = recovery / "diagnostics" / "broadcast_capacity.json"
    if broadcast_path.exists():
        broadcast = json.loads(broadcast_path.read_text(encoding="utf-8"))
        if broadcast.get("fresh_audit_consumed") is not False:
            raise ValueError("broadcast diagnostic consumed fresh-audit data")
        if broadcast.get("historical_gate_hash") != V837_GATE_SHA256:
            raise ValueError("broadcast diagnostic references wrong gate")

    sweep_path = recovery / "diagnostics" / "sparse_density_sweep.json"
    controls_path = recovery / "diagnostics" / "controls.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        if sweep.get("fresh_audit_consumed") is not False:
            raise ValueError("sparse sweep consumed fresh-audit data")
        if sweep.get("historical_gate_hash") != V837_GATE_SHA256:
            raise ValueError("sparse sweep references wrong gate")
    if controls_path.exists():
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
        if controls.get("fresh_audit_consumed") is not False:
            raise ValueError("V837d controls consumed fresh-audit data")
        rows = controls.get("rows", {})
        for condition in ("shuffled_sparse", "no_message"):
            pairs = {(row.get("family"), row.get("replicate")) for row in rows.get(condition, [])}
            if len(pairs) != 40:
                raise ValueError(f"{condition} does not contain the required paired 5x8 condition set")

    result_path = recovery / "results.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        required = {
            "version", "parent", "single_change", "representation_change", "historical_gate_hash",
            "fresh_audit_consumed", "conditions", "capacity_results", "representation_diagnostics",
            "resource_accounting", "pass_gate", "pass", "failure_classification", "interpretation",
            "next_experiment", "primitive_mining_allowed",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"V837d result missing fields: {missing}")
        if data.get("historical_gate_hash") != V837_GATE_SHA256:
            raise ValueError("V837d result references wrong historical gate")
        if data.get("fresh_audit_consumed") is not False:
            raise ValueError("V837d result consumed fresh-audit data")
        if data.get("primitive_mining_allowed") is not False:
            raise ValueError("V837d cannot reopen primitive mining from a capacity diagnostic")
        if not isinstance(data.get("resource_accounting"), dict) or not data["resource_accounting"]:
            raise ValueError("V837d missing resource accounting")
        for condition in ("broadcast", "fixed_sparse_selected_density", "degree_preserving_shuffled_sparse", "no_message"):
            if condition not in data.get("conditions", {}):
                raise ValueError(f"V837d missing required control: {condition}")
        if data.get("pass") is False:
            classes = data.get("failure_classification", [])
            if not classes:
                raise ValueError("failed V837d requires failure classification")
            invalid = sorted(set(classes) - V837_FAILURE_CLASSES)
            if invalid:
                raise ValueError(f"V837d has invalid failure classes: {invalid}")

    # Later representation variants are conditional and become mandatory once
    # their result files exist. All must preserve the original V837 gate,
    # fresh-audit lock, primitive-mining lock, single-variable declaration and
    # failure evidence.
    for version in ("v837g", "v837h"):
        variant = base / version
        result_file = variant / "results.json"
        if not result_file.exists():
            continue
        data = json.loads(result_file.read_text(encoding="utf-8"))
        required = {
            "version", "parent", "single_change", "representation_change", "historical_gate_hash",
            "fresh_audit_consumed", "conditions", "capacity_results", "representation_diagnostics",
            "resource_accounting", "pass_gate", "pass", "failure_classification", "interpretation",
            "next_experiment", "primitive_mining_allowed", "capacity_criterion_hash",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{version} result missing fields: {missing}")
        if data.get("historical_gate_hash") != V837_GATE_SHA256:
            raise ValueError(f"{version} references wrong historical V837 gate")
        if data.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
            raise ValueError(f"{version} changed the frozen capacity criterion")
        if not str(data.get("single_change", "")).strip():
            raise ValueError(f"{version} requires an explicit single_change")
        if data.get("fresh_audit_consumed") is not False:
            raise ValueError(f"{version} consumed fresh-audit data")
        if data.get("primitive_mining_allowed") is not False:
            raise ValueError(f"{version} reopened primitive mining before competence recovery")
        if not isinstance(data.get("resource_accounting"), dict) or not data["resource_accounting"]:
            raise ValueError(f"{version} missing resource accounting")
        if data.get("pass") is False:
            classes = data.get("failure_classification", [])
            if not classes:
                raise ValueError(f"failed {version} requires failure classification")
            invalid = sorted(set(classes) - V837_FAILURE_CLASSES)
            if invalid:
                raise ValueError(f"{version} has invalid failure classes: {invalid}")
            failure_doc = variant / "FAILURE.md"
            if not failure_doc.exists() or not failure_doc.read_text(encoding="utf-8").strip():
                raise ValueError(f"failed {version} missing FAILURE.md")

    recovery_status_path = base / "representation_recovery_status.json"
    if recovery_status_path.exists():
        recovery_status = json.loads(recovery_status_path.read_text(encoding="utf-8"))
        if recovery_status.get("outcome") != "C_REPRESENTATION_FAMILY_REMAINS_INADEQUATE":
            raise ValueError("unexpected representation recovery outcome")
        if recovery_status.get("scientifically_distinct_failed_variants") != ["V837d", "V837g", "V837h"]:
            raise ValueError("representation recovery stop rule must retain V837d/V837g/V837h sequence")
        if recovery_status.get("stop_rule_triggered") is not True:
            raise ValueError("representation recovery status must record stop-rule trigger")
        if recovery_status.get("neutral_substrate_competence") != "FAIL":
            raise ValueError("failed recovery must not mark neutral substrate competence PASS")
        if recovery_status.get("primitive_mining_allowed") is not False:
            raise ValueError("failed recovery must keep primitive mining blocked")
        if recovery_status.get("fresh_audit_episodes_consumed") != 0:
            raise ValueError("failed recovery consumed fresh-audit episodes")
        if recovery_status.get("primitives_promoted") != 0:
            raise ValueError("failed recovery promoted primitives")
        if recovery_status.get("v838_started") is not False:
            raise ValueError("failed representation recovery must not start V838")
        for required_path in (
            base / "INPUT_ACCESS_LINE_VERDICT.md",
            base / "REPRESENTATION_BLOCKER_ANALYSIS.md",
            base / "representation_recovery_resource_accounting.json",
            ROOT / "docs" / "V837_REPRESENTATION_RECOVERY_REPORT.md",
        ):
            if not required_path.exists() or not required_path.read_text(encoding="utf-8", errors="ignore").strip():
                raise ValueError(f"representation recovery evidence missing: {required_path.relative_to(ROOT)}")



def validate_v837_learned_reference_calibration() -> None:
    base = ROOT / "experiments" / "v837_primitive_invention"
    variant = base / "v837j"
    if not variant.exists():
        return
    config = json.loads((variant / "config.json").read_text(encoding="utf-8"))
    frozen = json.loads((variant / "frozen_reference_gate.json").read_text(encoding="utf-8"))
    if config.get("historical_gate_hash") != V837_GATE_SHA256:
        raise ValueError("V837j references wrong historical V837 gate")
    if config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837j changed the frozen capacity criterion")
    if frozen.get("historical_v837_gate_sha256") != V837_GATE_SHA256:
        raise ValueError("V837j frozen reference changed historical gate")
    if frozen.get("capacity_criterion_sha256") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837j frozen reference changed capacity criterion")
    if config.get("fresh_audit_allowed") is not False or config.get("reference_calibration_fresh_audit_consumed") is not False:
        raise ValueError("V837j must keep fresh audit locked")
    if config.get("primitive_mining_allowed") is not False:
        raise ValueError("V837j must keep primitive mining blocked")
    if config.get("task_family_label_allowed") is not False:
        raise ValueError("V837j learned references must not receive task labels")
    required_models = {"neutral_high_capacity", "gru_reference", "residual_rnn_reference"}
    if not required_models.issubset(set(config.get("models", []))):
        raise ValueError("V837j missing required learned-reference conditions")
    training = config.get("primary_training", {})
    for key in ("steps", "train_episodes", "validation_episodes", "development_seed_range", "validation_seed_range", "replicates"):
        if key not in training:
            raise ValueError(f"V837j training budget missing {key}")
    if int(training["steps"]) != 192 or int(training["train_episodes"]) != 128 or int(training["validation_episodes"]) != 128:
        raise ValueError("V837j primary matched budget drifted from blocker diagnostic")
    if training["development_seed_range"] != [10000, 10127] or training["validation_seed_range"] != [20000, 20127]:
        raise ValueError("V837j primary seed ranges drifted from frozen V837 development/validation regions")

    source = (base / "common" / "reference_models.py").read_text(encoding="utf-8")
    for forbidden in ("task_family_id", "family_label", "generator_class_id"):
        if forbidden in source:
            raise ValueError(f"V837j reference model source contains forbidden task-label input: {forbidden}")

    result_path = variant / "results.json"
    if not result_path.exists():
        return
    data = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "version", "parent", "single_change", "historical_gate_hash", "capacity_criterion_hash",
        "fresh_audit_consumed", "primitive_mining_allowed", "task_family_label_allowed",
        "baseline_compatibility", "matching_check", "models", "learning_curve_summary",
        "compute_accounting", "diagnosis", "pass", "failure_classification", "next_experiment",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"V837j result missing fields: {missing}")
    if data.get("version") != "V837j" or data.get("parent") != "V837h":
        raise ValueError("V837j version/parent mismatch")
    if data.get("historical_gate_hash") != V837_GATE_SHA256 or data.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837j result gate fingerprints mismatch")
    if data.get("fresh_audit_consumed") is not False or data.get("primitive_mining_allowed") is not False:
        raise ValueError("V837j result violated scientific locks")
    if data.get("task_family_label_allowed") is not False:
        raise ValueError("V837j result allowed task-family labels")
    if data.get("baseline_compatibility", {}).get("compatible") is not True:
        raise ValueError("V837j cannot be interpreted without compatible neutral baseline")
    matching = data.get("matching_check", {})
    for key in ("same_task_generators", "same_primary_training_seeds", "same_primary_validation_episodes", "same_optimizer_steps", "same_examples_processed_per_fit", "same_optimizer"):
        if matching.get(key) is not True:
            raise ValueError(f"V837j matching check failed: {key}")
    models = data.get("models", {})
    if not required_models.issubset(models):
        raise ValueError("V837j results missing required models")
    for model_name, record in models.items():
        if int(record.get("parameter_count", 0)) <= 0:
            raise ValueError(f"V837j {model_name} missing parameter count")
        family_results = record.get("family_results", {})
        if set(family_results) != {"conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"}:
            raise ValueError(f"V837j {model_name} family result coverage mismatch")
        if not record.get("resource_accounting"):
            raise ValueError(f"V837j {model_name} missing resource accounting")
    invalid = sorted(set(data.get("failure_classification", [])) - V837_FAILURE_CLASSES)
    if invalid:
        raise ValueError(f"V837j has invalid failure classes: {invalid}")
    if data.get("pass") is False:
        failure_doc = variant / "FAILURE.md"
        if not failure_doc.exists() or not failure_doc.read_text(encoding="utf-8").strip():
            raise ValueError("failed V837j missing FAILURE.md")
    else:
        pass_doc = variant / "PASS.md"
        if not pass_doc.exists() or not pass_doc.read_text(encoding="utf-8").strip():
            raise ValueError("successful diagnostic V837j missing PASS.md")
    raw_path = variant / "diagnostics" / "raw_runs.json"
    if not raw_path.exists():
        raise ValueError("V837j raw numerical runs missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("fresh_audit_consumed") is not False:
        raise ValueError("V837j raw runs consumed fresh audit")
    rows = raw.get("rows", [])
    expected_rows = len(config["models"]) * 5 * int(training["replicates"])
    if len(rows) != expected_rows:
        raise ValueError(f"V837j raw run count mismatch: {len(rows)} != {expected_rows}")
    for row in rows:
        if row.get("task_family_label_in_model_input") is not False:
            raise ValueError("V837j row leaked task-family label")
        if [row.get("train_seed_first"), row.get("train_seed_last")] != training["development_seed_range"]:
            raise ValueError("V837j row training seeds are not paired")
        if [row.get("validation_seed_first"), row.get("validation_seed_last")] != training["validation_seed_range"]:
            raise ValueError("V837j row validation seeds are not paired")
        resources = row.get("resources", {})
        if int(resources.get("optimizer_steps", -1)) != int(training["steps"]):
            raise ValueError("V837j row optimizer budget mismatch")
        if int(resources.get("examples_processed", -1)) != int(training["steps"]) * int(training["train_episodes"]):
            raise ValueError("V837j row example budget mismatch")

    audit = json.loads((base / "audit" / "audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise ValueError("V837j consumed fresh-audit episodes")
    recovery = json.loads((base / "representation_recovery_status.json").read_text(encoding="utf-8"))
    if recovery.get("primitives_promoted") != 0 or recovery.get("primitive_mining_allowed") is not False:
        raise ValueError("V837j must not alter primitive-mining status")

    # V837k: optimizer-step budget only. The 1x condition is V837j and the
    # 2x/4x conditions may not change architecture, data, optimizer, task
    # generators, validation episodes, or scientific locks.
    k_dir = base / "v837k"
    if (k_dir / "results.json").exists():
        k_config = json.loads((k_dir / "config.json").read_text(encoding="utf-8"))
        k_data = json.loads((k_dir / "results.json").read_text(encoding="utf-8"))
        if k_config.get("historical_gate_hash") != V837_GATE_SHA256 or k_config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
            raise ValueError("V837k gate fingerprints changed")
        if k_config.get("single_change", "").lower().find("optimizer step") < 0:
            raise ValueError("V837k must isolate optimizer-step budget")
        if k_config.get("fresh_audit_consumed") is not False or k_config.get("primitive_mining_allowed") is not False or k_config.get("task_family_label_allowed") is not False:
            raise ValueError("V837k violated scientific locks")
        base_training = k_config.get("base_training", {})
        if int(base_training.get("steps", -1)) != 192 or int(base_training.get("train_episodes", -1)) != 128 or int(base_training.get("validation_episodes", -1)) != 128:
            raise ValueError("V837k changed data budget instead of optimizer steps")
        if k_data.get("version") != "V837k" or k_data.get("parent") != "V837j":
            raise ValueError("V837k version/parent mismatch")
        if k_data.get("fresh_audit_consumed") is not False or k_data.get("primitive_mining_allowed") is not False:
            raise ValueError("V837k result violated scientific locks")
        if k_data.get("diagnosis") != "BENCHMARK_LEARNABILITY_UNRESOLVED":
            raise ValueError("V837k preserved result diagnosis changed")
        if k_data.get("executed_multipliers") != [2, 4]:
            raise ValueError("V837k must retain completed 2x/4x escalation")
        for multiplier, expected_steps in (("1x", 192), ("2x", 384), ("4x", 768)):
            condition = k_data.get("conditions", {}).get(multiplier)
            if condition is None:
                raise ValueError(f"V837k missing {multiplier} condition")
            for model_name in ("neutral_high_capacity", "gru_reference", "residual_rnn_reference"):
                record = condition.get("models", condition).get(model_name)
                if record is None:
                    raise ValueError(f"V837k {multiplier} missing {model_name}")
                resources = record.get("resource_accounting", {})
                model_fits = int(resources.get("model_fits", 0))
                if model_fits != 25 or int(resources.get("optimizer_steps", 0)) != expected_steps * model_fits:
                    raise ValueError(f"V837k {multiplier} optimizer budget mismatch for {model_name}")
        if not (k_dir / "FAILURE.md").exists():
            raise ValueError("V837k completed failure missing FAILURE.md")

    # V837l: unique development episodes only. It establishes benchmark
    # learnability because the matched GRU reaches 5/5 at 4x unique data,
    # while optimizer steps remain fixed at 192.
    l_dir = base / "v837l"
    if (l_dir / "results.json").exists():
        l_config = json.loads((l_dir / "config.json").read_text(encoding="utf-8"))
        l_data = json.loads((l_dir / "results.json").read_text(encoding="utf-8"))
        if l_config.get("historical_gate_hash") != V837_GATE_SHA256 or l_config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
            raise ValueError("V837l gate fingerprints changed")
        if "unique development episodes only" not in str(l_config.get("single_change", "")):
            raise ValueError("V837l must isolate unique development data")
        if l_config.get("fresh_audit_consumed") is not False or l_config.get("primitive_mining_allowed") is not False or l_config.get("task_family_label_allowed") is not False:
            raise ValueError("V837l violated scientific locks")
        if l_data.get("version") != "V837l" or l_data.get("parent") != "V837k":
            raise ValueError("V837l version/parent mismatch")
        if l_data.get("diagnosis") != "SAMPLE_EFFICIENCY_FAILURE" or l_data.get("pass") is not True:
            raise ValueError("V837l must preserve the resolved sample-efficiency diagnosis")
        if int(l_data.get("resolved_at_data_multiplier", 0)) != 4 or int(l_data.get("fixed_optimizer_steps", 0)) != 192:
            raise ValueError("V837l resolution/data-vs-step isolation changed")
        if l_data.get("fresh_audit_consumed") is not False or l_data.get("primitive_mining_allowed") is not False:
            raise ValueError("V837l result violated scientific locks")
        conditions = l_data.get("conditions", {})
        expected_families = {"1x": 2, "2x": 3, "4x": 5}
        for multiplier, expected in expected_families.items():
            condition = conditions.get(multiplier, {})
            gru = condition.get("gru_reference", {})
            if int(gru.get("families_passing", -1)) != expected:
                raise ValueError(f"V837l GRU {multiplier} family-pass count changed")
        if not (l_dir / "PASS.md").exists():
            raise ValueError("V837l diagnostic PASS missing PASS.md")

    # V837m: after learnability is calibrated, test one new cell-law
    # propertyâ€”stable general linear state transportâ€”against historical,
    # scalar-persistence and exact parameter-matched additive controls.
    m_dir = base / "v837m"
    if (m_dir / "results.json").exists():
        m_config = json.loads((m_dir / "config.json").read_text(encoding="utf-8"))
        m_data = json.loads((m_dir / "results.json").read_text(encoding="utf-8"))
        if m_config.get("historical_gate_hash") != V837_GATE_SHA256 or m_config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
            raise ValueError("V837m gate fingerprints changed")
        if m_config.get("fresh_audit_consumed") is not False or m_config.get("primitive_mining_allowed") is not False:
            raise ValueError("V837m config violated scientific locks")
        if m_data.get("version") != "V837m" or m_data.get("parent") != "V837l":
            raise ValueError("V837m version/parent mismatch")
        if m_data.get("diagnosis") != "LINEAR_STATE_TRANSPORT_INSUFFICIENT" or m_data.get("pass") is not False:
            raise ValueError("V837m preserved outcome changed")
        if m_data.get("fresh_audit_consumed") is not False or m_data.get("primitive_mining_allowed") is not False or m_data.get("full_structural_search_allowed") is not False:
            raise ValueError("V837m improperly reopened downstream research")
        matching = m_data.get("parameter_matching", {})
        if matching.get("exact_match") is not True or int(matching.get("linear_transport", 0)) != 1016 or int(matching.get("parameter_matched_additive", 0)) != 1016:
            raise ValueError("V837m parameter-matched control is not exact")
        conditions = m_data.get("conditions", {})
        expected_params = {"historical_direct": 856, "scalar_persistence": 866, "linear_transport": 1016, "parameter_matched_additive": 1016}
        expected_passes = {"historical_direct": 2, "scalar_persistence": 2, "linear_transport": 2, "parameter_matched_additive": 1}
        for condition_name, expected_params_count in expected_params.items():
            condition = conditions.get(condition_name)
            if condition is None:
                raise ValueError(f"V837m missing {condition_name}")
            if int(condition.get("parameter_count", -1)) != expected_params_count:
                raise ValueError(f"V837m {condition_name} parameter count changed")
            if int(condition.get("families_passing", -1)) != expected_passes[condition_name]:
                raise ValueError(f"V837m {condition_name} family-pass count changed")
        transport_diag = conditions["linear_transport"].get("stability_diagnostics", {})
        spectral = transport_diag.get("transport_spectral_norm_max", {}).get("mean")
        if spectral is None or float(spectral) > 0.951:
            raise ValueError("V837m transport stability bound not preserved")
        if not (m_dir / "FAILURE.md").exists():
            raise ValueError("V837m failed diagnostic missing FAILURE.md")



def validate_v837_gru_mechanism_localization() -> None:
    base = ROOT / "experiments" / "v837_primitive_invention"
    variant = base / "v837n"
    if not variant.exists():
        return
    config = json.loads((variant / "config.json").read_text(encoding="utf-8"))
    frozen = json.loads((variant / "frozen_mechanism_gate.json").read_text(encoding="utf-8"))
    if config.get("historical_gate_hash") != V837_GATE_SHA256:
        raise ValueError("V837n references wrong historical V837 gate")
    if config.get("capacity_criterion_hash") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837n changed the frozen capacity criterion")
    if frozen.get("historical_v837_gate_sha256") != V837_GATE_SHA256:
        raise ValueError("V837n frozen mechanism gate changed historical gate")
    if frozen.get("capacity_criterion_sha256") != V837_CAPACITY_CRITERION_SHA256:
        raise ValueError("V837n frozen mechanism gate changed capacity criterion")
    if config.get("reference_regime") != "4x_unique_data":
        raise ValueError("V837n mechanism localization must use the successful 4x reference regime")
    training = config.get("training", {})
    if int(training.get("train_episodes", -1)) != 512 or training.get("development_seed_range") != [10000, 10511]:
        raise ValueError("V837n development data is not the frozen V837l 4x regime")
    if int(training.get("validation_episodes", -1)) != 128 or training.get("validation_seed_range") != [20000, 20127]:
        raise ValueError("V837n validation episodes drifted from V837l")
    if int(training.get("steps", -1)) != 192 or training.get("optimizer") != "AdamW":
        raise ValueError("V837n optimizer-step regime drifted")
    if config.get("fresh_audit_consumed") is not False or config.get("fresh_audit_allowed") is not False:
        raise ValueError("V837n consumed or allowed fresh audit")
    if config.get("primitive_mining_allowed") is not False or config.get("structural_search_allowed") is not False:
        raise ValueError("V837n reopened downstream science")
    if config.get("task_family_label_allowed") is not False:
        raise ValueError("V837n allows task-family label leakage")

    positive_path = variant / "diagnostics" / "full_gru_positive_control.json"
    if not positive_path.exists():
        return
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    if positive.get("compatible") is not True or int(positive.get("families_passing", 0)) < 4:
        raise ValueError("V837n explicit full-GRU positive control is not compatible")
    if int(positive.get("parameter_count", 0)) != 875 or positive.get("parameter_count_match") is not True:
        raise ValueError("V837n explicit GRU does not match the 875-parameter reference")

    full_raw = json.loads((variant / "raw" / "full_gru.json").read_text(encoding="utf-8"))
    ablation_raw = json.loads((variant / "raw" / "ablations.json").read_text(encoding="utf-8"))
    rows = list(full_raw.get("rows", [])) + list(ablation_raw.get("rows", []))
    expected_conditions = set(config.get("conditions", []))
    if len(rows) != len(expected_conditions) * 5 * int(training.get("replicates", 0)):
        raise ValueError("V837n does not contain the complete paired condition/family/replicate set")
    seen = set()
    for row in rows:
        key = (row.get("condition"), row.get("family"), int(row.get("replicate", -1)))
        if key in seen:
            raise ValueError(f"duplicate V837n row: {key}")
        seen.add(key)
        if row.get("condition") not in expected_conditions:
            raise ValueError("V837n row has unknown condition")
        if row.get("task_family_label_in_model_input") is not False or row.get("fresh_audit_consumed") is not False:
            raise ValueError("V837n row violated task-label/fresh-audit locks")
        if [row.get("train_seed_first"), row.get("train_seed_last")] != training["development_seed_range"]:
            raise ValueError("V837n row development seeds are not paired")
        if [row.get("validation_seed_first"), row.get("validation_seed_last")] != training["validation_seed_range"]:
            raise ValueError("V837n row validation seeds are not paired")
        resources = row.get("resources", {})
        if int(resources.get("optimizer_steps", -1)) != int(training["steps"]):
            raise ValueError("V837n row optimizer-step budget mismatch")
        if int(resources.get("examples_processed", -1)) != int(training["steps"]) * int(training["train_episodes"]):
            raise ValueError("V837n row example budget mismatch")

    result_path = variant / "results.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "version", "parent", "question", "single_change", "reference_regime",
        "historical_gate_hash", "capacity_criterion_hash", "full_gru_reproduced",
        "full_gru_positive_control", "conditions", "families_passing", "paired_deltas",
        "gate_statistics", "mechanism_diagnosis", "diagnostic_pass", "resource_accounting",
        "fresh_audit_consumed", "primitive_mining_allowed", "structural_search_allowed", "next_experiment",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"V837n result missing fields: {missing}")
    if result.get("version") != "V837n" or result.get("parent") != "V837m":
        raise ValueError("V837n version/parent mismatch")
    if result.get("full_gru_reproduced") is not True or result.get("diagnostic_pass") is not True:
        raise ValueError("V837n must retain the reproduced positive control and diagnostic result")
    if result.get("mechanism_diagnosis") != "MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY":
        raise ValueError("V837n mechanism diagnosis changed unexpectedly")
    expected_counts = {
        "full_gru": 5, "static_update_vector": 4, "static_update_scalar": 4,
        "no_update": 5, "no_reset": 5, "static_reset_vector": 5, "no_update_no_reset": 3,
    }
    if result.get("families_passing") != expected_counts:
        raise ValueError("V837n recorded family-count outcome changed")
    if result.get("fresh_audit_consumed") is not False or result.get("primitive_mining_allowed") is not False or result.get("structural_search_allowed") is not False:
        raise ValueError("V837n result reopened locked science")
    invalid = sorted(set(result.get("failure_classification", [])) - V837_FAILURE_CLASSES)
    if invalid:
        raise ValueError(f"V837n has invalid failure classes: {invalid}")
    if not (variant / "PASS.md").exists():
        raise ValueError("V837n diagnostic PASS documentation missing")

    status_path = base / "gru_mechanism_localization_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("outcome") != "C_NO_INDIVIDUAL_GRU_MECHANISM_EXPLAINS_SUCCESS":
            raise ValueError("unexpected V837N+ program outcome")
        if status.get("full_structural_search_allowed") is not False or status.get("primitive_mining_allowed") is not False:
            raise ValueError("V837N+ status reopened downstream science")
        if int(status.get("fresh_audit_episodes_consumed", -1)) != 0 or int(status.get("primitives_promoted", -1)) != 0:
            raise ValueError("V837N+ status consumed audit/promoted primitives")
        if any((base / suffix).exists() for suffix in ("v837o", "v837p", "v837q", "v837r", "v837s", "v837t")) and not (base / "shared_state_path_localization_status.json").exists():
            raise ValueError("post-V837n variants exist without the shared-state-path localization authorization record")


def main() -> int:
    validate_integrity_manifest()
    validate_v836_recovery()
    validate_frontier_matrix()
    validate_active_result_files()
    validate_v837_lineage()
    validate_v837_representation_recovery()
    validate_v837_learned_reference_calibration()
    validate_v837_gru_mechanism_localization()
    runpy.run_path(str(ROOT / "scripts" / "validate_v837_shared_state_path.py"), run_name="__main__")
    print("active research validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
