from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
HERE = BASE / "v837q"
GATE_SHA = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
CAPACITY_SHA = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
EXPECTED_PRIMARY = {"Q0_local_10x4", "Q1_group5_5x8", "Q2_group2_2x20", "Q3_shared_1x40"}
EXPECTED_REFERENCES = {"QR1_dense_vanilla_rnn_40", "QR2_gru_reference"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha(relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def _check_row_budget(row: dict, training: dict) -> None:
    if row.get("fresh_audit_consumed") is not False or row.get("dynamic_modulation_enabled") is not False:
        raise ValueError("V837q raw row violated audit/dynamic-modulation locks")
    if row.get("task_family_label_in_model_input") is not False:
        raise ValueError("V837q raw row leaked task-family label")
    if [row.get("development_seed_first"), row.get("development_seed_last")] != training["development_seed_range"]:
        raise ValueError("V837q raw row development seed range drifted")
    if [row.get("validation_seed_first"), row.get("validation_seed_last")] != training["validation_seed_range"]:
        raise ValueError("V837q raw row validation seed range drifted")
    resources = row.get("resources", {})
    if int(resources.get("optimizer_steps", -1)) != int(training["steps"]):
        raise ValueError("V837q optimizer-step budget drifted")
    if int(resources.get("examples_processed", -1)) != int(training["steps"]) * int(training["train_episodes"]):
        raise ValueError("V837q examples-processed budget drifted")


def validate_v837_shared_state_organization() -> None:
    if not HERE.exists():
        return
    config = load_json(HERE / "config.json")
    frozen = load_json(HERE / "frozen_state_organization_gate.json")
    if config.get("experiment") != "V837q" or config.get("parent") != "V837p":
        raise ValueError("V837q version/parent mismatch")
    if config.get("single_change") != "recurrent state ownership from cell-local to progressively shared while preserving 40 total recurrent state dimensions":
        raise ValueError("V837q principal scientific variable changed")
    if config.get("data_regime") != "4x_unique" or int(config.get("total_state_dim", -1)) != 40 or int(config.get("local_view_dim", -1)) != 4:
        raise ValueError("V837q state-capacity/data regime changed")
    if set(config.get("conditions", {})) != EXPECTED_PRIMARY or set(config.get("references", {})) != EXPECTED_REFERENCES:
        raise ValueError("V837q primary/reference condition set changed")
    for name, expected_groups in {"Q0_local_10x4": 10, "Q1_group5_5x8": 5, "Q2_group2_2x20": 2, "Q3_shared_1x40": 1}.items():
        condition = config["conditions"][name]
        if int(condition.get("num_state_groups", -1)) != expected_groups or sum(condition.get("group_dims", [])) != 40:
            raise ValueError(f"V837q {name} does not preserve 40 total state dimensions")
    training = config.get("training", {})
    if training.get("optimizer") != "AdamW" or int(training.get("steps", -1)) != 192:
        raise ValueError("V837q optimizer regime changed")
    if int(training.get("train_episodes", -1)) != 512 or training.get("development_seed_range") != [10000, 10511]:
        raise ValueError("V837q 4x development regime changed")
    if int(training.get("validation_episodes", -1)) != 128 or training.get("validation_seed_range") != [20000, 20127]:
        raise ValueError("V837q validation regime changed")
    if int(training.get("replicates", -1)) != 5 or training.get("initialization_seed_namespace") != "v837j-primary-init":
        raise ValueError("V837q paired replicate policy changed")
    if config.get("historical_gate_hash") != GATE_SHA or config.get("capacity_criterion_hash") != CAPACITY_SHA:
        raise ValueError("V837q historical gate/capacity criterion changed")
    if config.get("dynamic_modulation_allowed") is not False or config.get("structural_search_allowed") is not False or config.get("primitive_mining_allowed") is not False or config.get("fresh_audit_consumed") is not False:
        raise ValueError("V837q science locks changed")
    if frozen.get("start_sha") != "f20316fd2aca8751b32226d653ac5b5f6976c7b3" or frozen.get("historical_gate_sha256") != GATE_SHA or frozen.get("capacity_criterion_sha256") != CAPACITY_SHA:
        raise ValueError("V837q frozen state-organization gate changed")
    if git_blob_sha("experiments/v837_primitive_invention/v837p/results.json") != frozen["parent_result_git_blob_sha256"]:
        raise ValueError("V837p parent result changed after V837q start")
    if git_blob_sha("experiments/v837_primitive_invention/v837l/results.json") != frozen["calibrated_reference_result_git_blob_sha256"]:
        raise ValueError("V837l calibrated reference result changed after V837q start")
    parent = load_json(BASE / "v837p" / "results.json")
    if parent.get("diagnosis") != "SHARED_PROPERTY_TRANSFER_FAILURE" or parent.get("representation_adequacy_pass") is not False:
        raise ValueError("V837q parent frontier changed")

    baseline_path = HERE / "raw" / "baseline_runs.json"
    if baseline_path.exists():
        rows = load_json(baseline_path).get("rows", [])
        if len(rows) != 25:
            raise ValueError(f"V837q baseline must contain 25 paired fits, found {len(rows)}")
        seen = set()
        for row in rows:
            if row.get("condition") != "Q0_local_10x4":
                raise ValueError("V837q baseline raw row has wrong condition")
            _check_row_budget(row, training)
            if int(row.get("parameter_count", -1)) != 856:
                raise ValueError("V837q Q0 parameter count drifted")
            key = (row["family"], int(row["replicate"]))
            if key in seen:
                raise ValueError(f"duplicate V837q baseline row: {key}")
            seen.add(key)
        compatibility = load_json(HERE / "diagnostics" / "baseline_compatibility.json")
        if compatibility.get("compatible") is not True:
            raise ValueError("V837q Q0 baseline compatibility failed")

    primary_path = HERE / "raw" / "primary_runs.json"
    if primary_path.exists():
        rows = load_json(primary_path).get("rows", [])
        expected_conditions = (EXPECTED_PRIMARY - {"Q0_local_10x4"}) | EXPECTED_REFERENCES
        if len(rows) != len(expected_conditions) * 5 * 5:
            raise ValueError(f"V837q primary batch size mismatch: {len(rows)}")
        seen = set()
        for row in rows:
            if row.get("condition") not in expected_conditions:
                raise ValueError("V837q primary raw row has unknown condition")
            _check_row_budget(row, training)
            key = (row["condition"], row["family"], int(row["replicate"]))
            if key in seen:
                raise ValueError(f"duplicate V837q primary row: {key}")
            seen.add(key)
            if row["condition"] in EXPECTED_PRIMARY and int(row.get("parameter_count", -1)) != 856:
                raise ValueError("V837q shared primary condition changed trainable parameter count")
            if row["condition"] == "QR1_dense_vanilla_rnn_40" and int(row.get("parameter_count", -1)) != 1921:
                raise ValueError("V837q dense vanilla RNN reference parameter count changed")
            if row["condition"] == "QR2_gru_reference" and int(row.get("parameter_count", -1)) != 875:
                raise ValueError("V837q GRU reference parameter count changed")

    result_path = HERE / "results.json"
    if not result_path.exists():
        return
    result = load_json(result_path)
    if result.get("version") != "V837q" or result.get("parent") != "V837p":
        raise ValueError("V837q result version/parent mismatch")
    if result.get("total_state_dim") != 40 or result.get("data_regime") != "4x_unique":
        raise ValueError("V837q result changed state/data regime")
    if set(result.get("conditions", {})) != EXPECTED_PRIMARY or set(result.get("references", {})) != EXPECTED_REFERENCES:
        raise ValueError("V837q result missing primary/reference condition")
    for name in EXPECTED_PRIMARY:
        record = result["conditions"][name]
        if int(record.get("parameter_count", -1)) != 856:
            raise ValueError(f"V837q {name} trainable parameter count is not matched")
        layout = record.get("layout", {})
        if sum(layout.get("group_dims", [])) != 40:
            raise ValueError(f"V837q {name} result does not preserve 40 recurrent dimensions")
    if int(result["references"]["QR1_dense_vanilla_rnn_40"].get("parameter_count", -1)) != 1921:
        raise ValueError("V837q QR1 parameter count changed")
    if int(result["references"]["QR2_gru_reference"].get("parameter_count", -1)) != 875:
        raise ValueError("V837q QR2 parameter count changed")
    if result.get("diagnosis") not in {"STATE_FRAGMENTATION_CRITICAL", "STATE_SHARING_PARTIAL_BENEFIT", "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED", "INTERMEDIATE_MODULARITY_OPTIMAL"}:
        raise ValueError("V837q diagnosis is not a frozen allowed outcome")
    expected_passes = {
        "Q0_local_10x4": 2,
        "Q1_group5_5x8": 2,
        "Q2_group2_2x20": 2,
        "Q3_shared_1x40": 2,
    }
    actual_passes = {name: int(row.get("families_passing", -1)) for name, row in result.get("conditions", {}).items()}
    if actual_passes != expected_passes or result.get("diagnosis") != "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED":
        raise ValueError("V837q frozen state-sharing outcome changed")
    expected_references = {"QR1_dense_vanilla_rnn_40": 2, "QR2_gru_reference": 5}
    actual_references = {name: int(row.get("families_passing", -1)) for name, row in result.get("references", {}).items()}
    if actual_references != expected_references:
        raise ValueError("V837q frozen reference-control outcome changed")
    if result.get("diagnostic_pass") is not True or result.get("representation_adequacy_pass") is not False or result.get("fresh_audit_consumed") is not False or result.get("primitive_mining_allowed") is not False or result.get("structural_search_allowed") is not False:
        raise ValueError("V837q result violated diagnostic/science locks")
    decision = load_json(HERE / "diagnostics" / "decision_state.json")
    if decision.get("v837q_complete") is not True or decision.get("diagnosis") != result.get("diagnosis"):
        raise ValueError("V837q decision state is inconsistent with result")
    q3_pass = int(result["conditions"]["Q3_shared_1x40"]["families_passing"]) >= 4
    if decision.get("q3_representation_adequacy_pass") is not q3_pass:
        raise ValueError("V837q Q3 conditional-control gate inconsistent")
    conditional_files = [HERE / "raw" / "q3_no_message_runs.json", HERE / "raw" / "q3_projection_sensitivity_runs.json"]
    if not q3_pass and any(path.exists() for path in conditional_files):
        raise ValueError("V837q conditional Q3 controls ran without Q3 representation adequacy")
    v837r_dir = BASE / "v837r"
    if v837r_dir.exists():
        r_config = load_json(v837r_dir / "config.json")
        if r_config.get("parent") != "V837q" or r_config.get("state_layout") != "local_10x4" or r_config.get("dynamic_modulation_allowed") is not False or r_config.get("shared_state_allowed") is not False:
            raise ValueError("V837r does not preserve the single-variable continuation authorized after V837q")
    v837s_dir = BASE / "v837s"
    if v837s_dir.exists():
        decision_path = v837r_dir / "diagnostics" / "decision_state.json"
        if not decision_path.exists() or load_json(decision_path).get("interaction_followup_allowed") is not True:
            raise ValueError("V837s exists without V837r interaction authorization")
    if (BASE / "v837t").exists() or (BASE / "v837u").exists():
        s_decision_path = v837s_dir / "diagnostics" / "decision_state.json"
        if not v837s_dir.exists() or not s_decision_path.exists():
            raise ValueError("V837t/u exists without the completed V837r->V837s lineage")
        s_decision = load_json(s_decision_path)
        if s_decision.get("v837s_complete") is not True or s_decision.get("representation_adequacy_pass") is not False:
            raise ValueError("V837t/u continuation is inconsistent with the closed V837s frontier")
    if not (HERE / "PASS.md").exists():
        raise ValueError("V837q diagnostic PASS documentation missing")

    status_path = BASE / "shared_state_organization_status.json"
    accounting_path = BASE / "shared_state_organization_resource_accounting.json"
    if not status_path.exists() or not accounting_path.exists():
        raise ValueError("V837q final program status/resource accounting missing")
    status = load_json(status_path)
    if status.get("outcome") != "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED" or status.get("representation_adequacy") != "FAIL":
        raise ValueError("V837q final program outcome changed")
    if status.get("full_structural_search_allowed") is not False or status.get("primitive_mining_allowed") is not False:
        raise ValueError("V837q final status reopened downstream science")
    if int(status.get("fresh_audit_episodes_consumed", -1)) != 0 or int(status.get("primitives_promoted", -1)) != 0 or status.get("v838_started") is not False:
        raise ValueError("V837q final status violated audit/primitive/V838 locks")
    accounting = load_json(accounting_path)
    totals = accounting.get("totals", {})
    if int(totals.get("model_fits", -1)) != 150 or int(totals.get("optimizer_steps", -1)) != 28800:
        raise ValueError("V837q resource accounting changed")
    if int(totals.get("fresh_audit_episodes", -1)) != 0 or int(totals.get("structural_search_runs", -1)) != 0 or int(totals.get("motif_mining_runs", -1)) != 0:
        raise ValueError("V837q resource accounting contains forbidden downstream work")

    report = ROOT / "docs" / "V837_SHARED_STATE_ORGANIZATION_REPORT.md"
    if not report.exists() or not report.read_text(encoding="utf-8").strip():
        raise ValueError("V837q final scientific report missing or empty")


if __name__ == "__main__":
    validate_v837_shared_state_organization()
    print("V837 shared-state-organization validation: PASS")
