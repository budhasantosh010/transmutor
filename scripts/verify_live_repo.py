from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "verification" / "live_repo_manifest.json"
SHA_PATH = ROOT / "verification" / "active_research_sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_bytes(relative: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"unable to read committed Git blob: {relative}") from exc


def git_blob_sha256(relative: str) -> str:
    return hashlib.sha256(git_blob_bytes(relative)).hexdigest()


def load_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing required file: {relative}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {relative}: {exc}") from exc


def require_path(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise RuntimeError(f"missing required path: {relative}")
    return path


def verify_sha_manifest() -> int:
    if not SHA_PATH.is_file():
        raise RuntimeError("missing verification/active_research_sha256.txt")
    checked = 0
    for lineno, raw in enumerate(SHA_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "  " not in line:
            raise RuntimeError(f"malformed SHA line {lineno}")
        expected, relative = line.split("  ", 1)
        path = require_path(relative)
        if not path.is_file():
            raise RuntimeError(f"SHA entry is not a file: {relative}")
        actual = git_blob_sha256(relative)
        if actual != expected:
            raise RuntimeError(f"committed Git-blob SHA mismatch: {relative}: {actual} != {expected}")
        checked += 1
    if checked == 0:
        raise RuntimeError("active-research SHA manifest is empty")
    return checked


def main() -> int:
    manifest = load_json("verification/live_repo_manifest.json")
    if manifest.get("repository") != "budhasantosh010/transmutor":
        raise RuntimeError("unexpected repository identity")

    critical_paths = manifest.get("critical_paths", [])
    for relative in critical_paths:
        require_path(relative)

    historical_git = manifest["historical_git_blob_sha256"]
    historical_paths = {
        "v836_result_sha256": "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json",
        "v837_result_sha256": "experiments/v837_primitive_invention/v837/results.json",
        "v837b_result_sha256": "experiments/v837_primitive_invention/v837b/results.json",
        "v837c_result_sha256": "experiments/v837_primitive_invention/v837c/results.json",
        "frozen_gate_sha256": "experiments/v837_primitive_invention/frozen_gates.json",
    }
    for key, relative in historical_paths.items():
        require_path(relative)
        actual = git_blob_sha256(relative)
        expected = historical_git[key]
        if actual != expected:
            raise RuntimeError(f"historical committed-blob mismatch for {relative}: {actual} != {expected}")

    for variant_name in ("V837d", "V837g", "V837h"):
        record = manifest["current_variants"][variant_name]
        for relative in record.get("source", []):
            require_path(relative)
        require_path(record["config"])
        result = load_json(record["results"])
        if result.get("version") != variant_name:
            raise RuntimeError(f"{variant_name} result version mismatch")
        if result.get("pass") is not False:
            raise RuntimeError(f"{variant_name} is expected to be a preserved FAIL result")
        if result.get("fresh_audit_consumed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly consumed fresh-audit data")
        if result.get("primitive_mining_allowed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly reopened primitive mining")
        for relative in record.get("documentation", []):
            require_path(relative)
        for relative in record.get("plots", []):
            require_path(relative)

    v837d = load_json(manifest["current_variants"]["V837d"]["results"])
    if v837d["resource_accounting"].get("historical_parameter_count") != 856:
        raise RuntimeError("V837d historical parameter count is not 856")

    v837g = load_json(manifest["current_variants"]["V837g"]["results"])
    if v837g["resource_accounting"].get("parameter_count") != 866:
        raise RuntimeError("V837g parameter count is not 866")

    v837h = load_json(manifest["current_variants"]["V837h"]["results"])
    matching = v837h.get("parameter_matching", {})
    if matching.get("multiplicative_parameter_count") != 1096:
        raise RuntimeError("V837h multiplicative parameter count is not 1096")
    if matching.get("additive_control_parameter_count") != 1096:
        raise RuntimeError("V837h additive control parameter count is not 1096")
    if matching.get("additive_equals_multiplicative") is not True:
        raise RuntimeError("V837h parameter-matched control is not actually matched")

    # Learned-reference calibration and calibrated cell-law diagnostic.
    for variant_name in ("V837j", "V837k", "V837l", "V837m"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["results"])
        result = load_json(record["results"])
        if result.get("version") != variant_name:
            raise RuntimeError(f"{variant_name} result version mismatch")
        if result.get("fresh_audit_consumed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly consumed fresh-audit data")
        if result.get("primitive_mining_allowed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly reopened primitive mining")

    v837j = load_json(manifest["current_variants"]["V837j"]["results"])
    if v837j.get("diagnosis") != "BENCHMARK_LEARNABILITY_UNRESOLVED" or v837j.get("pass") is not False:
        raise RuntimeError("V837j matched-budget diagnosis changed")
    if int(v837j["models"]["gru_reference"]["parameter_count"]) != 875 or int(v837j["models"]["gru_reference"]["families_passing"]) != 2:
        raise RuntimeError("V837j GRU calibration record changed")

    v837k = load_json(manifest["current_variants"]["V837k"]["results"])
    if v837k.get("diagnosis") != "BENCHMARK_LEARNABILITY_UNRESOLVED" or v837k.get("pass") is not False:
        raise RuntimeError("V837k optimizer-budget diagnosis changed")
    for multiplier in ("1x", "2x", "4x"):
        if int(v837k["conditions"][multiplier]["models"]["gru_reference"]["families_passing"]) != 2:
            raise RuntimeError(f"V837k GRU {multiplier} pass count changed")

    v837l = load_json(manifest["current_variants"]["V837l"]["results"])
    if v837l.get("diagnosis") != "SAMPLE_EFFICIENCY_FAILURE" or v837l.get("pass") is not True:
        raise RuntimeError("V837l sample-efficiency diagnosis changed")
    if int(v837l.get("resolved_at_data_multiplier", 0)) != 4:
        raise RuntimeError("V837l learnability resolution multiplier changed")
    if int(v837l["conditions"]["4x"]["gru_reference"]["families_passing"]) != 5:
        raise RuntimeError("V837l 4x-data GRU no longer passes 5/5")
    if int(v837l["conditions"]["4x"]["neutral_high_capacity"]["families_passing"]) != 2:
        raise RuntimeError("V837l calibrated neutral pass count changed")

    v837m = load_json(manifest["current_variants"]["V837m"]["results"])
    if v837m.get("diagnosis") != "LINEAR_STATE_TRANSPORT_INSUFFICIENT" or v837m.get("pass") is not False:
        raise RuntimeError("V837m linear-transport diagnosis changed")
    m_matching = v837m.get("parameter_matching", {})
    if m_matching.get("exact_match") is not True or int(m_matching.get("linear_transport", 0)) != 1016 or int(m_matching.get("parameter_matched_additive", 0)) != 1016:
        raise RuntimeError("V837m parameter-matched control changed")
    if int(v837m["conditions"]["linear_transport"]["families_passing"]) != 2:
        raise RuntimeError("V837m linear-transport pass count changed")
    if v837m.get("full_structural_search_allowed") is not False:
        raise RuntimeError("V837m improperly reopened structural search")

    # Successful-reference mechanism localization.
    v837n_record = manifest["current_variants"].get("V837n")
    if not isinstance(v837n_record, dict):
        raise RuntimeError("verification manifest missing V837n")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in v837n_record.get(key, []):
            require_path(relative)
    require_path(v837n_record["config"])
    require_path(v837n_record["frozen_gate"])
    v837n = load_json(v837n_record["results"])
    if v837n.get("version") != "V837n" or v837n.get("diagnostic_pass") is not True:
        raise RuntimeError("V837n diagnostic result changed")
    if v837n.get("mechanism_diagnosis") != "MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY":
        raise RuntimeError("V837n mechanism diagnosis changed")
    expected_n_counts = {
        "full_gru": 5, "static_update_vector": 4, "static_update_scalar": 4,
        "no_update": 5, "no_reset": 5, "static_reset_vector": 5, "no_update_no_reset": 3,
    }
    if v837n.get("families_passing") != expected_n_counts:
        raise RuntimeError("V837n family-count outcome changed")
    positive = v837n.get("full_gru_positive_control", {})
    if positive.get("compatible") is not True or int(positive.get("parameter_count", 0)) != 875:
        raise RuntimeError("V837n explicit GRU positive control changed")
    if v837n.get("fresh_audit_consumed") is not False or v837n.get("primitive_mining_allowed") is not False or v837n.get("structural_search_allowed") is not False:
        raise RuntimeError("V837n reopened locked science")

    localization = load_json("experiments/v837_primitive_invention/gru_mechanism_localization_status.json")
    if localization.get("outcome") != "C_NO_INDIVIDUAL_GRU_MECHANISM_EXPLAINS_SUCCESS":
        raise RuntimeError("GRU mechanism-localization program outcome changed")
    if localization.get("full_structural_search_allowed") is not False or localization.get("primitive_mining_allowed") is not False:
        raise RuntimeError("GRU localization status reopened downstream science")
    if localization.get("fresh_audit_episodes_consumed") != 0 or localization.get("primitives_promoted") != 0:
        raise RuntimeError("GRU localization status violated audit/primitive locks")

    # Shared-property factorial localization and the single authorized neutral transfer.
    for variant_name in ("V837o", "V837p"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    v837o = load_json(manifest["current_variants"]["V837o"]["results"])
    if v837o.get("mechanism_diagnosis") != "DYNAMIC_STATE_MODULATION_REQUIRED" or v837o.get("diagnostic_pass") is not True:
        raise RuntimeError("V837o factorial diagnosis changed")
    expected_o = {
        "G0_full_dynamic": 5, "G1_dynamic_update_no_reset": 5, "G2_no_update_dynamic_reset": 5,
        "G3_static_update_vector_no_reset": 3, "G4_no_update_static_reset_vector": 3,
        "G5_static_update_vector_static_reset_vector": 3, "G6_static_update_scalar_static_reset_vector": 3,
        "G7_static_update_vector_static_reset_scalar": 3, "G8_static_update_scalar_static_reset_scalar": 3,
        "G9_no_update_no_reset": 3,
    }
    actual_o = {name: int(row.get("families_passing", -1)) for name, row in v837o.get("conditions", {}).items()}
    if actual_o != expected_o:
        raise RuntimeError("V837o family-count outcome changed")
    if v837o.get("fresh_audit_consumed") is not False or v837o.get("primitive_mining_allowed") is not False:
        raise RuntimeError("V837o reopened locked science")

    v837p = load_json(manifest["current_variants"]["V837p"]["results"])
    if v837p.get("diagnosis") != "SHARED_PROPERTY_TRANSFER_FAILURE" or v837p.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837p transfer outcome changed")
    expected_p = {"historical_direct": 2, "scalar_persistence": 2, "dynamic_scalar_state_modulation": 3, "parameter_matched_dynamic_additive": 3}
    actual_p = {name: int(row.get("families_passing", -1)) for name, row in v837p.get("conditions", {}).items()}
    if actual_p != expected_p:
        raise RuntimeError("V837p family-count outcome changed")
    if int(v837p["conditions"]["dynamic_scalar_state_modulation"].get("parameter_count", -1)) != 1006 or int(v837p["conditions"]["parameter_matched_dynamic_additive"].get("parameter_count", -1)) != 1006:
        raise RuntimeError("V837p parameter-matched control changed")
    if v837p.get("structural_search_allowed") is not False or v837p.get("primitive_mining_allowed") is not False or v837p.get("fresh_audit_consumed") is not False:
        raise RuntimeError("V837p reopened locked science")

    shared = load_json("experiments/v837_primitive_invention/shared_state_path_localization_status.json")
    if shared.get("outcome") != "DYNAMIC_MODULATION_LOCALIZED_TRANSFER_INSUFFICIENT":
        raise RuntimeError("shared-state-path program outcome changed")
    if shared.get("full_structural_search_allowed") is not False or shared.get("primitive_mining_allowed") is not False:
        raise RuntimeError("shared-state-path status reopened downstream science")
    if shared.get("fresh_audit_episodes_consumed") != 0 or shared.get("primitives_promoted") != 0:
        raise RuntimeError("shared-state-path status violated audit/primitive locks")

    # Shared-state organization localization.
    q_record = manifest["current_variants"].get("V837q")
    if not isinstance(q_record, dict):
        raise RuntimeError("verification manifest missing V837q")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in q_record.get(key, []):
            require_path(relative)
    require_path(q_record["config"])
    require_path(q_record["frozen_gate"])
    v837q = load_json(q_record["results"])
    if v837q.get("version") != "V837q" or v837q.get("diagnostic_pass") is not True:
        raise RuntimeError("V837q diagnostic result changed")
    if v837q.get("diagnosis") != "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED" or v837q.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837q state-organization diagnosis changed")
    expected_q = {"Q0_local_10x4": 2, "Q1_group5_5x8": 2, "Q2_group2_2x20": 2, "Q3_shared_1x40": 2}
    actual_q = {name: int(row.get("families_passing", -1)) for name, row in v837q.get("conditions", {}).items()}
    if actual_q != expected_q:
        raise RuntimeError("V837q primary family-count outcome changed")
    expected_q_refs = {"QR1_dense_vanilla_rnn_40": 2, "QR2_gru_reference": 5}
    actual_q_refs = {name: int(row.get("families_passing", -1)) for name, row in v837q.get("references", {}).items()}
    if actual_q_refs != expected_q_refs:
        raise RuntimeError("V837q reference-control outcome changed")
    for name in expected_q:
        record = v837q["conditions"][name]
        if int(record.get("parameter_count", -1)) != 856 or sum(record.get("layout", {}).get("group_dims", [])) != 40:
            raise RuntimeError(f"V837q state/parameter matching changed for {name}")
    if v837q.get("q3_no_message_control") is not None or v837q.get("projection_sensitivity") is not None:
        raise RuntimeError("V837q conditional controls ran despite failed Q3 gate")
    if v837q.get("fresh_audit_consumed") is not False or v837q.get("primitive_mining_allowed") is not False or v837q.get("structural_search_allowed") is not False:
        raise RuntimeError("V837q reopened locked science")
    q_status = load_json("experiments/v837_primitive_invention/shared_state_organization_status.json")
    if q_status.get("outcome") != "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED" or q_status.get("representation_adequacy") != "FAIL":
        raise RuntimeError("shared-state-organization status changed")
    if q_status.get("full_structural_search_allowed") is not False or q_status.get("primitive_mining_allowed") is not False:
        raise RuntimeError("shared-state-organization status reopened downstream science")
    if q_status.get("fresh_audit_episodes_consumed") != 0 or q_status.get("primitives_promoted") != 0 or q_status.get("v838_started") is not False:
        raise RuntimeError("shared-state-organization status violated audit/primitive/V838 locks")

    # Global recurrent coupling localization and the single authorized interaction.
    for variant_name in ("V837r", "V837s"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    v837r = load_json(manifest["current_variants"]["V837r"]["results"])
    if v837r.get("version") != "V837r" or v837r.get("diagnosis") != "GLOBAL_COUPLING_PARTIAL_BENEFIT":
        raise RuntimeError("V837r coupling diagnosis changed")
    expected_r = {"R0_local": 2, "R1_rank1": 2, "R2_rank2": 2, "R3_rank4": 3, "R4_rank8": 3, "R5_dense_cross_block": 2}
    actual_r = {name: int(row.get("families_passing", -1)) for name, row in v837r.get("conditions", {}).items()}
    if actual_r != expected_r:
        raise RuntimeError("V837r primary family-count outcome changed")
    expected_rc = {"C1_rank1_local": 2, "C2_rank2_local": 2, "C3_rank4_local": 2, "C4_rank8_local": 2, "C5_dense_budget_local": 2}
    actual_rc = {name: int(row.get("families_passing", -1)) for name, row in v837r.get("matched_controls", {}).items()}
    if actual_rc != expected_rc:
        raise RuntimeError("V837r matched-control outcome changed")
    for primary, control in (("R1_rank1", "C1_rank1_local"), ("R2_rank2", "C2_rank2_local"), ("R3_rank4", "C3_rank4_local"), ("R4_rank8", "C4_rank8_local"), ("R5_dense_cross_block", "C5_dense_budget_local")):
        if int(v837r["conditions"][primary]["parameter_count"]) != int(v837r["matched_controls"][control]["parameter_count"]):
            raise RuntimeError(f"V837r parameter matching changed for {primary}")
    if v837r.get("representation_adequacy_pass") is not False or v837r.get("interaction_followup_allowed") is not True:
        raise RuntimeError("V837r adequacy/interaction gate changed")
    if v837r.get("fresh_audit_consumed") is not False or v837r.get("primitive_mining_allowed") is not False or v837r.get("structural_search_allowed") is not False or v837r.get("v838_started") is not False:
        raise RuntimeError("V837r reopened locked science")

    v837s = load_json(manifest["current_variants"]["V837s"]["results"])
    if v837s.get("version") != "V837s" or v837s.get("diagnosis") != "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT":
        raise RuntimeError("V837s interaction diagnosis changed")
    expected_s = {"S0_local_no_modulation": 2, "S1_local_dynamic_scalar": 3, "S2_rank4_no_modulation": 3, "S3_rank4_dynamic_scalar": 3, "S3C_rank4_matched_dynamic_additive": 3}
    actual_s = {name: int(row.get("families_passing", -1)) for name, row in v837s.get("conditions", {}).items()}
    if actual_s != expected_s:
        raise RuntimeError("V837s family-count outcome changed")
    if int(v837s["conditions"]["S3_rank4_dynamic_scalar"]["parameter_count"]) != 1326 or int(v837s["conditions"]["S3C_rank4_matched_dynamic_additive"]["parameter_count"]) != 1326:
        raise RuntimeError("V837s matched dynamic control parameter count changed")
    if v837s.get("multiplicative_specificity_established") is not False or v837s.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837s specificity/adequacy outcome changed")
    if v837s.get("fresh_audit_consumed") is not False or v837s.get("primitive_mining_allowed") is not False or v837s.get("structural_search_allowed") is not False or v837s.get("v838_started") is not False:
        raise RuntimeError("V837s reopened locked science")

    coupling_status = load_json("experiments/v837_primitive_invention/global_recurrent_coupling_status.json")
    if coupling_status.get("v837r_outcome") != "GLOBAL_COUPLING_PARTIAL_BENEFIT" or coupling_status.get("v837s_outcome") != "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT" or coupling_status.get("representation_adequacy") != "FAIL":
        raise RuntimeError("global recurrent coupling program status changed")
    if coupling_status.get("fresh_audit_episodes_consumed") != 0 or coupling_status.get("primitives_promoted") != 0 or coupling_status.get("v838_started") is not False:
        raise RuntimeError("global recurrent coupling status violated audit/primitive/V838 locks")

    calibration = load_json("experiments/v837_primitive_invention/learned_reference_calibration_status.json")
    if calibration.get("benchmark_learnability") != "ESTABLISHED_UNDER_4X_UNIQUE_DEVELOPMENT_DATA":
        raise RuntimeError("learned-reference calibration status changed")
    if calibration.get("sample_efficiency_failure_supported") is not True:
        raise RuntimeError("sample-efficiency diagnosis not preserved")
    if calibration.get("primitive_mining_allowed") is not False or calibration.get("fresh_audit_episodes_consumed") != 0 or calibration.get("primitives_promoted") != 0:
        raise RuntimeError("learned-reference calibration violated downstream locks")

    audit = load_json("experiments/v837_primitive_invention/audit/audit_results.json")
    if audit.get("episodes_consumed") != 0:
        raise RuntimeError("fresh audit episodes have been consumed")

    lineage = load_json("experiments/v837_primitive_invention/lineage_status.json")
    promoted = lineage.get("primitives_promoted")
    if promoted not in (0, [], None):
        raise RuntimeError("primitive promotions must remain zero")

    recovery = load_json("experiments/v837_primitive_invention/representation_recovery_status.json")
    if recovery.get("primitive_mining_allowed") is not False:
        raise RuntimeError("representation recovery unexpectedly reopened primitive mining")
    if recovery.get("fresh_audit_episodes_consumed") != 0:
        raise RuntimeError("representation recovery consumed fresh-audit data")
    if recovery.get("primitives_promoted") != 0:
        raise RuntimeError("representation recovery promoted primitives")

    for relative in manifest.get("validators", []):
        require_path(relative)
    for relative in manifest.get("tests", []):
        require_path(relative)

    checked = verify_sha_manifest()
    print(f"live repository verification: PASS ({checked} SHA-256 entries checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
