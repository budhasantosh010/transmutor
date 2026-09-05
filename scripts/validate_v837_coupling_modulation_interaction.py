from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
HERE = BASE / "v837s"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha256(relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def _check_budget(row: dict, training: dict) -> None:
    if int(row["development_seed_first"]) != int(training["development_seed_range"][0]) or int(row["development_seed_last"]) != int(training["development_seed_range"][1]):
        raise ValueError("V837s development seed mismatch")
    if int(row["validation_seed_first"]) != int(training["validation_seed_range"][0]) or int(row["validation_seed_last"]) != int(training["validation_seed_range"][1]):
        raise ValueError("V837s validation seed mismatch")
    resources = row["resources"]
    if int(resources["optimizer_steps"]) != int(training["steps"]):
        raise ValueError("V837s optimizer-step mismatch")
    if int(resources["examples_processed"]) != int(training["steps"]) * int(training["train_episodes"]):
        raise ValueError("V837s example-budget mismatch")
    if row.get("fresh_audit_consumed") is not False or row.get("task_family_label_in_model_input") is not False:
        raise ValueError("V837s raw run violated science locks")


def validate_v837_coupling_modulation_interaction() -> None:
    if not HERE.exists():
        return
    config = load_json(HERE / "config.json")
    if config.get("experiment") != "V837s" or config.get("parent") != "V837r":
        raise ValueError("V837s version/parent mismatch")
    if config.get("state_layout") != "local_10x4" or int(config.get("total_state_dim", -1)) != 40 or int(config.get("coupling_rank", -1)) != 4:
        raise ValueError("V837s state/coupling boundary changed")
    if config.get("data_regime") != "4x_unique":
        raise ValueError("V837s must remain in 4x representation regime")
    for key in ("fresh_audit_allowed", "structural_search_allowed", "primitive_mining_allowed", "task_family_label_allowed", "v838_allowed"):
        if config.get(key) is not False:
            raise ValueError(f"V837s science lock changed: {key}")
    expected_conditions = {
        "S0_local_no_modulation",
        "S1_local_dynamic_scalar",
        "S2_rank4_no_modulation",
        "S3_rank4_dynamic_scalar",
        "S3C_rank4_matched_dynamic_additive",
    }
    if set(config.get("conditions", {})) != expected_conditions:
        raise ValueError("V837s condition matrix changed")
    if config["conditions"]["S1_local_dynamic_scalar"]["state_modulation_mode"] != "dynamic_scalar_candidate":
        raise ValueError("V837s no longer reuses V837p dynamic scalar mechanism")
    if config["conditions"]["S3C_rank4_matched_dynamic_additive"]["state_modulation_mode"] != "dynamic_scalar_matched_additive":
        raise ValueError("V837s matched dynamic control changed")

    parent_decision = load_json(BASE / "v837r" / "diagnostics" / "decision_state.json")
    if parent_decision.get("v837r_complete") is not True or parent_decision.get("interaction_followup_allowed") is not True:
        raise ValueError("V837s lacks V837r machine authorization")
    if parent_decision.get("diagnosis") != config.get("required_v837r_diagnosis") or parent_decision.get("best_condition") != config.get("required_v837r_best_condition"):
        raise ValueError("V837s parent selection basis changed")
    if git_blob_sha256("experiments/v837_primitive_invention/v837r/results.json") != config["v837r_result_sha256"]:
        raise ValueError("V837r result changed after V837s authorization")
    if git_blob_sha256("experiments/v837_primitive_invention/v837p/results.json") != config["v837p_result_sha256"]:
        raise ValueError("V837p result changed")

    training = config["training"]
    if int(training["steps"]) != 192 or int(training["train_episodes"]) != 512 or int(training["validation_episodes"]) != 128 or int(training["replicates"]) != 5:
        raise ValueError("V837s training regime changed")
    if training["development_seed_range"] != [10000, 10511] or training["validation_seed_range"] != [20000, 20127]:
        raise ValueError("V837s seed regime changed")

    raw = HERE / "raw" / "runs.json"
    if raw.exists():
        rows = load_json(raw).get("rows", [])
        if len(rows) != 125:
            raise ValueError(f"V837s expected 125 paired runs, found {len(rows)}")
        if {row["condition"] for row in rows} != expected_conditions:
            raise ValueError("V837s raw condition set mismatch")
        keys = {(row["condition"], row["family"], int(row["replicate"])) for row in rows}
        if len(keys) != 125:
            raise ValueError("V837s raw paired-key uniqueness mismatch")
        for row in rows:
            _check_budget(row, training)
        params = {row["condition"]: int(row["parameter_count"]) for row in rows}
        if params["S0_local_no_modulation"] != 856 or params["S1_local_dynamic_scalar"] != 1006 or params["S2_rank4_no_modulation"] != 1176:
            raise ValueError("V837s primary parameter counts changed")
        if params["S3_rank4_dynamic_scalar"] != 1326 or params["S3C_rank4_matched_dynamic_additive"] != 1326:
            raise ValueError("V837s true/control interaction parameter counts do not match")

    result_path = HERE / "results.json"
    if not result_path.exists():
        return
    result = load_json(result_path)
    if result.get("version") != "V837s" or result.get("parent") != "V837r":
        raise ValueError("V837s result version/parent mismatch")
    if set(result.get("conditions", {})) != expected_conditions:
        raise ValueError("V837s result condition matrix incomplete")
    if result.get("diagnosis") not in {
        "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INTERACTION",
        "INTERACTION_RECOVERY_WITHOUT_MULTIPLICATIVE_SPECIFICITY",
        "REPRESENTATION_RECOVERY_WITHOUT_INTERACTION_NECESSITY",
        "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT",
    }:
        raise ValueError("V837s diagnosis outside frozen outcome set")
    if result.get("diagnostic_pass") is not True or result.get("fresh_audit_consumed") is not False:
        raise ValueError("V837s diagnostic/audit lock changed")
    if result.get("structural_search_allowed") is not False or result.get("primitive_mining_allowed") is not False or result.get("v838_started") is not False:
        raise ValueError("V837s reopened locked science")
    if int(result.get("primitives_promoted", -1)) != 0:
        raise ValueError("V837s promoted primitives")
    if int(result["conditions"]["S3_rank4_dynamic_scalar"]["parameter_count"]) != int(result["conditions"]["S3C_rank4_matched_dynamic_additive"]["parameter_count"]):
        raise ValueError("V837s dynamic specificity control not parameter matched")
    reproduction = result.get("parent_factor_reproduction", {})
    for condition in ("S0_local_no_modulation", "S2_rank4_no_modulation"):
        if reproduction.get(condition, {}).get("compatible") is not True:
            raise ValueError("V837s failed to reproduce a parent factorial corner")
    decision = load_json(HERE / "diagnostics" / "decision_state.json")
    if decision.get("v837s_complete") is not True or decision.get("diagnosis") != result.get("diagnosis"):
        raise ValueError("V837s decision state inconsistent")
    if decision.get("representation_adequacy_pass") is not result.get("representation_adequacy_pass"):
        raise ValueError("V837s representation gate inconsistent")
    if (BASE / "v837t").exists() and decision.get("coupling_compression_allowed") is not True:
        raise ValueError("V837t exists without V837s compression authorization")
    for suffix in ("v837u", "v837v", "v838"):
        if (BASE / suffix).exists():
            raise ValueError(f"V837s closure may not create downstream {suffix}")


if __name__ == "__main__":
    validate_v837_coupling_modulation_interaction()
    print("V837 coupling-modulation interaction validation: PASS")
