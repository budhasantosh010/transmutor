from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
HERE = BASE / "v837r"
GATE_SHA = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
CAPACITY_SHA = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
PRIMARY = {"R0_local", "R1_rank1", "R2_rank2", "R3_rank4", "R4_rank8", "R5_dense_cross_block"}
CONTROLS = {"C1_rank1_local", "C2_rank2_local", "C3_rank4_local", "C4_rank8_local", "C5_dense_budget_local"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha(relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def _check_budget(row: dict, training: dict) -> None:
    if row.get("fresh_audit_consumed") is not False:
        raise ValueError("V837r raw row consumed fresh audit")
    if row.get("dynamic_modulation_enabled") is not False or row.get("shared_state_enabled") is not False:
        raise ValueError("V837r raw row changed dynamic/state-ownership variables")
    if row.get("interaction_mode") != "none" or row.get("state_layout") != "local_10x4" or int(row.get("total_state_dim", -1)) != 40:
        raise ValueError("V837r raw row changed frozen substrate semantics")
    if row.get("task_family_label_in_model_input") is not False:
        raise ValueError("V837r leaked task-family label")
    if [row.get("development_seed_first"), row.get("development_seed_last")] != training["development_seed_range"]:
        raise ValueError("V837r development seeds drifted")
    if [row.get("validation_seed_first"), row.get("validation_seed_last")] != training["validation_seed_range"]:
        raise ValueError("V837r validation seeds drifted")
    resources = row.get("resources", {})
    if int(resources.get("optimizer_steps", -1)) != int(training["steps"]):
        raise ValueError("V837r optimizer-step budget drifted")
    if int(resources.get("examples_processed", -1)) != int(training["steps"]) * int(training["train_episodes"]):
        raise ValueError("V837r examples-processed budget drifted")


def validate_v837_global_recurrent_coupling() -> None:
    if not HERE.exists():
        return
    config = load_json(HERE / "config.json")
    frozen = load_json(HERE / "frozen_global_coupling_gate.json")
    if config.get("experiment") != "V837r" or config.get("parent") != "V837q":
        raise ValueError("V837r version/parent mismatch")
    if config.get("single_change") != "degree of direct cross-dimensional recurrent coupling while recurrent state remains local 10x4":
        raise ValueError("V837r single variable changed")
    if config.get("data_regime") != "4x_unique" or config.get("state_layout") != "local_10x4":
        raise ValueError("V837r data/state layout changed")
    if int(config.get("total_state_dim", -1)) != 40 or int(config.get("local_state_dim", -1)) != 4 or int(config.get("num_cells", -1)) != 10:
        raise ValueError("V837r state capacity changed")
    if set(config.get("conditions", {})) != PRIMARY or set(config.get("matched_controls", {})) != CONTROLS:
        raise ValueError("V837r condition/control matrix changed")
    expected_ranks = {"R1_rank1": 1, "R2_rank2": 2, "R3_rank4": 4, "R4_rank8": 8}
    for name, rank in expected_ranks.items():
        row = config["conditions"][name]
        if row.get("coupling_mode") != "low_rank" or int(row.get("rank", -1)) != rank or row.get("cross_block_only") is not True:
            raise ValueError(f"V837r {name} changed")
    dense = config["conditions"]["R5_dense_cross_block"]
    if dense.get("coupling_mode") != "dense" or dense.get("cross_block_only") is not True:
        raise ValueError("V837r dense cross-block condition changed")
    if config["conditions"]["R0_local"].get("coupling_mode") != "none":
        raise ValueError("V837r R0 is not historical local recurrence")
    matching = {row["matches"]: name for name, row in config["matched_controls"].items()}
    if set(matching) != PRIMARY - {"R0_local"}:
        raise ValueError("V837r missing matched control")
    if int(config["matched_controls"]["C5_dense_budget_local"]["matched_local_rank"]) != 20:
        raise ValueError("V837r dense local control no longer matches 1600 parameters")
    training = config["training"]
    if training.get("optimizer") != "AdamW" or int(training.get("steps", -1)) != 192:
        raise ValueError("V837r optimizer regime changed")
    if int(training.get("train_episodes", -1)) != 512 or training.get("development_seed_range") != [10000, 10511]:
        raise ValueError("V837r 4x development regime changed")
    if int(training.get("validation_episodes", -1)) != 128 or training.get("validation_seed_range") != [20000, 20127]:
        raise ValueError("V837r validation regime changed")
    if int(training.get("replicates", -1)) != 5 or training.get("initialization_seed_namespace") != "v837j-primary-init":
        raise ValueError("V837r paired replicate policy changed")
    if config.get("historical_gate_hash") != GATE_SHA or config.get("capacity_criterion_hash") != CAPACITY_SHA:
        raise ValueError("V837r historical capacity gate changed")
    for key in ("dynamic_modulation_allowed", "shared_state_allowed", "structural_search_allowed", "primitive_mining_allowed", "fresh_audit_allowed", "fresh_audit_consumed"):
        if config.get(key) is not False:
            raise ValueError(f"V837r lock changed: {key}")
    if frozen.get("start_sha") != "7f623f99eadffc333a07a73f1155876e987356d3" or frozen.get("historical_gate_sha256") != GATE_SHA or frozen.get("capacity_criterion_sha256") != CAPACITY_SHA:
        raise ValueError("V837r frozen gate changed")
    if git_blob_sha("experiments/v837_primitive_invention/v837q/results.json") != frozen["parent_result_git_blob_sha256"]:
        raise ValueError("V837q parent result changed after V837r start")
    if git_blob_sha("experiments/v837_primitive_invention/v837p/results.json") != frozen["v837p_result_git_blob_sha256"]:
        raise ValueError("V837p result changed after V837r start")

    baseline = HERE / "raw" / "baseline_runs.json"
    if baseline.exists():
        rows = load_json(baseline).get("rows", [])
        if len(rows) != 25:
            raise ValueError(f"V837r baseline requires 25 fits, found {len(rows)}")
        for row in rows:
            if row.get("condition") != "R0_local" or int(row.get("parameter_count", -1)) != 856:
                raise ValueError("V837r R0 baseline changed")
            _check_budget(row, training)
        compatibility = load_json(HERE / "diagnostics" / "baseline_compatibility.json")
        if compatibility.get("compatible") is not True:
            raise ValueError("V837r baseline compatibility failed")

    screen = HERE / "raw" / "screen_runs.json"
    if screen.exists():
        rows = load_json(screen).get("rows", [])
        expected_conditions = set(config["screen_sequence"])
        if len(rows) != len(expected_conditions) * 25:
            raise ValueError("V837r screen row count mismatch")
        if {row["condition"] for row in rows} != expected_conditions:
            raise ValueError("V837r screen condition set mismatch")
        for row in rows:
            _check_budget(row, training)
        screen_decision = load_json(HERE / "diagnostics" / "screen_decision.json")
        localization = HERE / "raw" / "localization_runs.json"
        if screen_decision.get("localization_allowed") is not True and localization.exists():
            raise ValueError("V837r rank1/rank8 ran despite strict stop")

    localization = HERE / "raw" / "localization_runs.json"
    if localization.exists():
        rows = load_json(localization).get("rows", [])
        expected_conditions = set(config["localization_sequence"])
        if len(rows) != len(expected_conditions) * 25 or {row["condition"] for row in rows} != expected_conditions:
            raise ValueError("V837r localization row set mismatch")
        for row in rows:
            _check_budget(row, training)

    result_path = HERE / "results.json"
    if not result_path.exists():
        return
    result = load_json(result_path)
    if result.get("version") != "V837r" or result.get("parent") != "V837q":
        raise ValueError("V837r result version/parent changed")
    if result.get("state_layout") != "local_10x4" or int(result.get("total_state_dim", -1)) != 40:
        raise ValueError("V837r result state layout changed")
    if result.get("diagnosis") not in {
        "LOW_RANK_GLOBAL_COUPLING_SUFFICIENT",
        "HIGH_BANDWIDTH_GLOBAL_COUPLING_REQUIRED",
        "GLOBAL_COUPLING_PARTIAL_BENEFIT",
        "GLOBAL_COUPLING_SPECIFICITY_NOT_ESTABLISHED",
        "GLOBAL_RECURRENT_COUPLING_INSUFFICIENT",
    }:
        raise ValueError("V837r diagnosis outside frozen outcome set")
    if result.get("diagnostic_pass") is not True or result.get("fresh_audit_consumed") is not False:
        raise ValueError("V837r result diagnostic/audit lock changed")
    if result.get("structural_search_allowed") is not False or result.get("primitive_mining_allowed") is not False or result.get("v838_started") is not False:
        raise ValueError("V837r reopened locked science")
    if int(result.get("primitives_promoted", -1)) != 0:
        raise ValueError("V837r promoted primitives")
    conditions = result.get("conditions", {})
    if "R0_local" not in conditions or int(conditions["R0_local"].get("parameter_count", -1)) != 856:
        raise ValueError("V837r final result missing historical baseline")
    for name, summary in conditions.items():
        if name != "R0_local" and name not in PRIMARY:
            raise ValueError("unknown V837r primary condition in result")
        if summary["coupling_spec"].get("cross_block_only") is not True:
            raise ValueError("V837r primary coupling lost cross-block-only mask")
    controls = result.get("matched_controls", {})
    for primary_name in conditions:
        if primary_name == "R0_local":
            continue
        control_name = matching[primary_name]
        if control_name not in controls:
            raise ValueError(f"V837r result missing matched control for {primary_name}")
        if int(conditions[primary_name]["parameter_count"]) != int(controls[control_name]["parameter_count"]):
            raise ValueError(f"V837r parameter control mismatch for {primary_name}")
    decision = load_json(HERE / "diagnostics" / "decision_state.json")
    if decision.get("v837r_complete") is not True or decision.get("diagnosis") != result.get("diagnosis"):
        raise ValueError("V837r decision state inconsistent")
    if decision.get("interaction_followup_allowed") is not result.get("interaction_followup_allowed"):
        raise ValueError("V837r interaction guard inconsistent")
    if (BASE / "v837s").exists() and decision.get("interaction_followup_allowed") is not True:
        raise ValueError("V837s exists without V837r authorization")
    if (BASE / "v837t").exists() or (BASE / "v837u").exists():
        s_decision_path = BASE / "v837s" / "diagnostics" / "decision_state.json"
        if not s_decision_path.exists():
            raise ValueError("V837t/u exists without completed V837s")
        s_decision = load_json(s_decision_path)
        if s_decision.get("v837s_complete") is not True or s_decision.get("representation_adequacy_pass") is not False:
            raise ValueError("V837t/u continuation is inconsistent with V837s closure")
    for suffix in ("v837v", "v838"):
        if (BASE / suffix).exists():
            raise ValueError(f"V837r closure may not create downstream {suffix}")
    if not (HERE / "PASS.md").exists():
        raise ValueError("V837r diagnostic PASS document missing")
    report = ROOT / "docs" / "V837_GLOBAL_RECURRENT_COUPLING_REPORT.md"
    if report.exists() and not report.read_text(encoding="utf-8").strip():
        raise ValueError("V837r report is empty")
    program_resources = load_json(BASE / "global_recurrent_coupling_resource_accounting.json")
    unique = program_resources.get("unique_seed_defined_episodes", {})
    if int(unique.get("development_per_family", -1)) != 512 or int(unique.get("validation_per_family", -1)) != 128:
        raise ValueError("V837r/V837s unique seed-defined episode accounting changed")
    if int(unique.get("task_families", -1)) != 5 or int(unique.get("total_family_seed_pairs", -1)) != 3200:
        raise ValueError("V837r/V837s unique family/seed episode total changed")
    program_status = load_json(BASE / "global_recurrent_coupling_status.json")
    if int(program_status.get("unique_seed_defined_episodes", -1)) != 3200:
        raise ValueError("V837r/V837s status unique episode total changed")


if __name__ == "__main__":
    validate_v837_global_recurrent_coupling()
    print("V837 global recurrent coupling validation: PASS")
