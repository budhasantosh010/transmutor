from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
V837_GATE_SHA256 = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
CAPACITY_SHA256 = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v837_shared_state_path_localization() -> None:
    v837o = BASE / "v837o"
    if not v837o.exists():
        return

    config = load_json(v837o / "config.json")
    frozen = load_json(v837o / "frozen_factorial_gate.json")
    expected_conditions = {
        "G0_full_dynamic",
        "G1_dynamic_update_no_reset",
        "G2_no_update_dynamic_reset",
        "G3_static_update_vector_no_reset",
        "G4_no_update_static_reset_vector",
        "G5_static_update_vector_static_reset_vector",
        "G6_static_update_scalar_static_reset_vector",
        "G7_static_update_vector_static_reset_scalar",
        "G8_static_update_scalar_static_reset_scalar",
        "G9_no_update_no_reset",
    }
    if config.get("experiment") != "V837o" or config.get("parent") != "V837n":
        raise ValueError("V837o version/parent mismatch")
    if config.get("data_regime") != "4x_unique" or set(config.get("conditions", [])) != expected_conditions:
        raise ValueError("V837o factorial regime/condition matrix changed")
    training = config.get("training", {})
    if int(training.get("steps", -1)) != 192 or int(training.get("train_episodes", -1)) != 512:
        raise ValueError("V837o training budget drifted")
    if training.get("development_seed_range") != [10000, 10511] or training.get("validation_seed_range") != [20000, 20127]:
        raise ValueError("V837o paired seed ranges drifted")
    if int(training.get("validation_episodes", -1)) != 128 or int(training.get("replicates", -1)) != 5:
        raise ValueError("V837o validation/replication regime drifted")
    if config.get("fresh_audit_consumed") is not False or config.get("primitive_mining_allowed") is not False or config.get("structural_search_allowed") is not False:
        raise ValueError("V837o reopened locked science")
    if config.get("task_family_label_allowed") is not False:
        raise ValueError("V837o allows task-family label leakage")
    if frozen.get("historical_gate_sha256") != V837_GATE_SHA256 or frozen.get("capacity_criterion_sha256") != CAPACITY_SHA256:
        raise ValueError("V837o frozen factorial gate changed historical criteria")
    if frozen.get("positive_control") != "G0_full_dynamic" or int(frozen.get("families_required", -1)) != 4:
        raise ValueError("V837o positive-control gate changed")
    if frozen.get("fresh_audit_consumed") is not False or frozen.get("primitive_mining_allowed") is not False:
        raise ValueError("V837o frozen gate reopened locked science")

    positive = load_json(v837o / "diagnostics" / "full_gru_positive_control.json")
    if positive.get("compatible") is not True or int(positive.get("families_passing", -1)) != 5:
        raise ValueError("V837o full-GRU positive control failed")
    full_payload = load_json(v837o / "raw" / "full_gru.json")
    factorial_payload = load_json(v837o / "raw" / "factorial_runs.json")
    for payload in (full_payload, factorial_payload):
        paired = payload.get("paired_task_seeds", {})
        if paired.get("development_seeds") != list(range(10000, 10512)):
            raise ValueError("V837o raw payload development seeds are not the frozen paired set")
        if paired.get("validation_seeds") != list(range(20000, 20128)):
            raise ValueError("V837o raw payload validation seeds are not the frozen paired set")
    full_rows = full_payload.get("rows", [])
    factorial_rows = factorial_payload.get("rows", [])
    rows = list(full_rows) + list(factorial_rows)
    if len(rows) != 250:
        raise ValueError(f"V837o must contain 250 paired raw runs, found {len(rows)}")
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row.get("condition")), str(row.get("family")), int(row.get("replicate", -1)))
        if key in seen:
            raise ValueError(f"duplicate V837o row: {key}")
        seen.add(key)
        if row.get("condition") not in expected_conditions or row.get("fresh_audit_consumed") is not False:
            raise ValueError("V837o raw row violated condition/audit locks")
        resources = row.get("resources", {})
        if int(resources.get("optimizer_steps", -1)) != 192 or int(resources.get("examples_processed", -1)) != 192 * 512:
            raise ValueError("V837o raw row compute budget mismatch")

    result = load_json(v837o / "results.json")
    if result.get("version") != "V837o" or result.get("parent") != "V837n":
        raise ValueError("V837o result version/parent mismatch")
    if result.get("diagnostic_pass") is not True or result.get("full_gru_reproduced") is not True:
        raise ValueError("V837o diagnostic did not preserve its positive control")
    if result.get("mechanism_diagnosis") != "DYNAMIC_STATE_MODULATION_REQUIRED":
        raise ValueError("V837o mechanism diagnosis changed")
    expected_passes = {
        "G0_full_dynamic": 5,
        "G1_dynamic_update_no_reset": 5,
        "G2_no_update_dynamic_reset": 5,
        "G3_static_update_vector_no_reset": 3,
        "G4_no_update_static_reset_vector": 3,
        "G5_static_update_vector_static_reset_vector": 3,
        "G6_static_update_scalar_static_reset_vector": 3,
        "G7_static_update_vector_static_reset_scalar": 3,
        "G8_static_update_scalar_static_reset_scalar": 3,
        "G9_no_update_no_reset": 3,
    }
    actual_passes = {name: int(summary.get("families_passing", -1)) for name, summary in result.get("conditions", {}).items()}
    if actual_passes != expected_passes:
        raise ValueError("V837o recorded factorial family-pass counts changed")
    if result.get("neutral_followup_allowed") is not True or result.get("neutral_followup_type") != "single_dynamic_modulator":
        raise ValueError("V837o did not authorize exactly the localized dynamic-modulator follow-up")
    if result.get("fresh_audit_consumed") is not False or result.get("primitive_mining_allowed") is not False:
        raise ValueError("V837o result reopened locked science")
    if not (v837o / "PASS.md").exists():
        raise ValueError("V837o diagnostic PASS documentation missing")

    v837p = BASE / "v837p"
    if not v837p.exists():
        raise ValueError("V837o authorizes V837p but the completed transfer result is missing")
    p_config = load_json(v837p / "config.json")
    p_frozen = load_json(v837p / "frozen_transfer_gate.json")
    if p_config.get("experiment") != "V837p" or p_config.get("parent") != "V837o":
        raise ValueError("V837p version/parent mismatch")
    if p_config.get("selection_basis") != "V837o DYNAMIC_STATE_MODULATION_REQUIRED" or p_frozen.get("required_parent_diagnosis") != "DYNAMIC_STATE_MODULATION_REQUIRED":
        raise ValueError("V837p is not authorized by the V837o diagnosis")
    if p_config.get("data_regime") != "4x_unique":
        raise ValueError("V837p must use the calibrated 4x unique-data regime")
    p_training = p_config.get("training", {})
    if int(p_training.get("steps", -1)) != 192 or int(p_training.get("train_episodes", -1)) != 512:
        raise ValueError("V837p training budget drifted")
    if p_training.get("development_seed_range") != [10000, 10511] or p_training.get("validation_seed_range") != [20000, 20127]:
        raise ValueError("V837p paired seed ranges drifted")
    if int(p_training.get("replicates", -1)) != 5 or p_config.get("fresh_audit_consumed") is not False or p_config.get("primitive_mining_allowed") is not False:
        raise ValueError("V837p replication/audit/primitive locks changed")

    p_rows = load_json(v837p / "raw" / "runs.json").get("rows", [])
    if len(p_rows) != 100:
        raise ValueError(f"V837p must contain 100 paired raw runs, found {len(p_rows)}")
    p_conditions = set(p_config.get("conditions", {}))
    if p_conditions != {"historical_direct", "scalar_persistence", "dynamic_scalar_state_modulation", "parameter_matched_dynamic_additive"}:
        raise ValueError("V837p condition set changed")
    for row in p_rows:
        if row.get("condition") not in p_conditions or row.get("fresh_audit_consumed") is not False:
            raise ValueError("V837p raw row violated condition/audit locks")
        resources = row.get("resources", {})
        if int(resources.get("optimizer_steps", -1)) != 192 or int(resources.get("examples_processed", -1)) != 192 * 512:
            raise ValueError("V837p raw row compute budget mismatch")

    p_result = load_json(v837p / "results.json")
    if p_result.get("diagnosis") != "SHARED_PROPERTY_TRANSFER_FAILURE" or p_result.get("representation_adequacy_pass") is not False:
        raise ValueError("V837p transfer outcome changed")
    p_expected = {
        "historical_direct": 2,
        "scalar_persistence": 2,
        "dynamic_scalar_state_modulation": 3,
        "parameter_matched_dynamic_additive": 3,
    }
    p_actual = {name: int(summary.get("families_passing", -1)) for name, summary in p_result.get("conditions", {}).items()}
    if p_actual != p_expected:
        raise ValueError("V837p recorded family-pass counts changed")
    if int(p_result["conditions"]["dynamic_scalar_state_modulation"].get("parameter_count", -1)) != 1006:
        raise ValueError("V837p dynamic-modulator parameter count changed")
    if int(p_result["conditions"]["parameter_matched_dynamic_additive"].get("parameter_count", -1)) != 1006:
        raise ValueError("V837p matched-control parameter count changed")
    if p_result.get("sample_efficiency_tested") is not False or p_result.get("structural_search_allowed") is not False or p_result.get("primitive_mining_allowed") is not False:
        raise ValueError("V837p advanced past failed representation adequacy")
    if p_result.get("fresh_audit_consumed") is not False or int(p_result.get("primitives_promoted", -1)) != 0:
        raise ValueError("V837p consumed audit/promoted primitives")
    if not (v837p / "FAILURE.md").exists():
        raise ValueError("V837p transfer failure documentation missing")

    status = load_json(BASE / "shared_state_path_localization_status.json")
    if status.get("outcome") != "DYNAMIC_MODULATION_LOCALIZED_TRANSFER_INSUFFICIENT":
        raise ValueError("unexpected V837O+ program outcome")
    if status.get("full_structural_search_allowed") is not False or status.get("primitive_mining_allowed") is not False:
        raise ValueError("V837O+ program status reopened downstream science")
    if int(status.get("fresh_audit_episodes_consumed", -1)) != 0 or int(status.get("primitives_promoted", -1)) != 0:
        raise ValueError("V837O+ program status consumed audit/promoted primitives")
    accounting = load_json(BASE / "shared_state_path_localization_resource_accounting.json")
    if int(accounting.get("totals", {}).get("model_fits", -1)) != 350 or int(accounting.get("totals", {}).get("optimizer_steps", -1)) != 67200:
        raise ValueError("V837O+ program resource accounting changed")
    report = ROOT / "docs" / "V837_SHARED_STATE_PATH_LOCALIZATION_REPORT.md"
    if not report.exists() or not report.read_text(encoding="utf-8").strip():
        raise ValueError("V837O+ final scientific report missing")


if __name__ == "__main__":
    validate_v837_shared_state_path_localization()
    print("V837 shared-state-path validation: PASS")
