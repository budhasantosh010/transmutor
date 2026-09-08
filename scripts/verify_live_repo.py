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

    # Dynamic-control granularity localization and exactly one machine-authorized neutral follow-up.
    for variant_name in ("V837t", "V837u"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    v837t = load_json(manifest["current_variants"]["V837t"]["results"])
    if v837t.get("version") != "V837t" or v837t.get("diagnosis") != "DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED":
        raise RuntimeError("V837t dynamic-granularity diagnosis changed")
    expected_t = {"T0_full_vector_gru": 5, "T1_vector_update_no_reset": 5, "T2_scalarized_update_no_reset": 4, "T3_no_update_vector_reset": 5, "T4_no_update_scalarized_reset": 3, "T5_dual_scalarized": 3}
    actual_t = {name: int(row.get("families_passing", -1)) for name, row in v837t.get("conditions", {}).items()}
    if actual_t != expected_t:
        raise RuntimeError("V837t family-count outcome changed")
    if v837t.get("positive_controls_pass") is not True or v837t.get("authorized_v837u_mode") != "DYNAMIC_SCALAR_CARRY":
        raise RuntimeError("V837t positive-control/authorization decision changed")
    if int(v837t.get("unique_seed_defined_episodes", -1)) != 3200 or v837t.get("fresh_audit_consumed") is not False or v837t.get("structural_search_allowed") is not False or v837t.get("primitive_mining_allowed") is not False or v837t.get("v838_started") is not False:
        raise RuntimeError("V837t data/science locks changed")

    v837u = load_json(manifest["current_variants"]["V837u"]["results"])
    if v837u.get("version") != "V837u" or v837u.get("authorized_mode") != "DYNAMIC_SCALAR_CARRY" or v837u.get("diagnosis") != "DYNAMIC_SCALAR_CARRY_INSUFFICIENT":
        raise RuntimeError("V837u scalar-carry diagnosis changed")
    expected_u = {"U0_historical_direct": 2, "U1_v837p_scalar_candidate": 3, "U2_dynamic_scalar_carry": 2, "U2C_scalar_scale_candidate_control": 3}
    actual_u = {name: int(row.get("families_passing", -1)) for name, row in v837u.get("conditions", {}).items()}
    if actual_u != expected_u:
        raise RuntimeError("V837u family-count outcome changed")
    if int(v837u["conditions"]["U2_dynamic_scalar_carry"].get("parameter_count", -1)) != 1006 or int(v837u["conditions"]["U2C_scalar_scale_candidate_control"].get("parameter_count", -1)) != 1006:
        raise RuntimeError("V837u scalar-carry matched-control parameter count changed")
    if v837u.get("representation_adequacy_pass") is not False or v837u.get("sample_efficiency_retest_allowed") is not False or v837u.get("fresh_audit_consumed") is not False or v837u.get("structural_search_allowed") is not False or v837u.get("primitive_mining_allowed") is not False or v837u.get("v838_started") is not False:
        raise RuntimeError("V837u adequacy/science locks changed")

    dynamic_status = load_json("experiments/v837_primitive_invention/dynamic_control_granularity_status.json")
    dynamic_resources = load_json("experiments/v837_primitive_invention/dynamic_control_granularity_resource_accounting.json")
    if dynamic_status.get("v837t", {}).get("diagnosis") != "DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED" or dynamic_status.get("v837u", {}).get("diagnosis") != "DYNAMIC_SCALAR_CARRY_INSUFFICIENT":
        raise RuntimeError("dynamic-control program status changed")
    if dynamic_status.get("v837u", {}).get("representation_adequacy") != "FAIL" or dynamic_status.get("fresh_audit_episodes_consumed") != 0 or dynamic_status.get("primitives_promoted") != 0 or dynamic_status.get("v838_started") is not False:
        raise RuntimeError("dynamic-control program lock state changed")
    combined = dynamic_resources.get("combined", {})
    if int(combined.get("model_fits", -1)) != 250 or int(combined.get("optimizer_steps", -1)) != 48000 or int(combined.get("processed_examples", -1)) != 24576000 or int(combined.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("dynamic-control combined resource accounting changed")

    # Control-scope, reference-information, and authorized global-scalar transfer.
    for variant_name in ("V837v", "V837w", "V837x"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    v837v = load_json(manifest["current_variants"]["V837v"]["results"])
    expected_v = {"V0_10_domains": 2, "V1_5_domains": 2, "V2_2_domains": 2, "V3_1_domain": 2}
    actual_v = {name: int(row.get("families_passing", -1)) for name, row in v837v.get("conditions", {}).items()}
    if v837v.get("diagnosis") != "CONTROL_SCOPE_ALONE_INSUFFICIENT" or actual_v != expected_v or v837v.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837v control-scope outcome changed")

    v837w = load_json(manifest["current_variants"]["V837w"]["results"])
    expected_w = {"W0_joint_input_state": 4, "W1_input_only": 3, "W2_state_only": 3, "W3_bias_only": 3}
    actual_w = {name: int(row.get("families_passing", -1)) for name, row in v837w.get("conditions", {}).items()}
    if v837w.get("diagnosis") != "JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED" or actual_w != expected_w or v837w.get("authorized_v837x_mode") != "JOINT_INPUT_STATE_GLOBAL_SCALAR":
        raise RuntimeError("V837w information-source outcome changed")

    v837x = load_json(manifest["current_variants"]["V837x"]["results"])
    expected_x = {"X0_historical_direct": 2, "X1_local_scalar_carry": 2, "X2_global_scalar_carry": 3, "X2C_global_scale_candidate_control": 3}
    actual_x = {name: int(row.get("families_passing", -1)) for name, row in v837x.get("conditions", {}).items()}
    if v837x.get("diagnosis") != "GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT" or actual_x != expected_x or v837x.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837x global-scalar outcome changed")
    if v837x.get("sample_efficiency_retest_allowed") is not False or v837x.get("structural_search_allowed") is not False or v837x.get("primitive_mining_allowed") is not False or v837x.get("fresh_audit_consumed") is not False or v837x.get("v838_started") is not False:
        raise RuntimeError("V837x science lock state changed")

    scope_status = load_json("experiments/v837_primitive_invention/control_scope_program_status.json")
    scope_resources = load_json("experiments/v837_primitive_invention/control_scope_program_resource_accounting.json")
    if scope_status.get("scalar_control_stop_rule_triggered") is not True or scope_status.get("next_single_variable") != "candidate_transformation_organization":
        raise RuntimeError("control-scope program stop-rule/next-variable state changed")
    scope_combined = scope_resources.get("combined", {})
    if int(scope_combined.get("model_fits", -1)) != 300 or int(scope_combined.get("optimizer_steps", -1)) != 57600 or int(scope_combined.get("processed_examples", -1)) != 29491200 or int(scope_combined.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("control-scope program resource accounting changed")

    # Candidate-interaction and machine-authorized candidate-stage localization.
    for variant_name in ("V837y", "V837z"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    v837y = load_json(manifest["current_variants"]["V837y"]["results"])
    expected_y = {"Y0_historical": 2, "Y1_global_control": 3, "Y2_rank4_candidate": 3, "Y3_global_control_rank4_candidate": 3, "Y3C_global_control_matched_local": 2}
    actual_y = {name: int(row.get("families_passing", -1)) for name, row in v837y.get("conditions", {}).items()}
    if v837y.get("diagnosis") != "GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT" or actual_y != expected_y or v837y.get("representation_adequacy_pass") is not False or v837y.get("v837z_allowed") is not True:
        raise RuntimeError("V837y candidate-interaction outcome changed")
    if v837y.get("selected_v837z_parent") != "Y3_global_control_rank4_candidate":
        raise RuntimeError("V837y selected V837z parent changed")

    v837z = load_json(manifest["current_variants"]["V837z"]["results"])
    expected_z = {"Z0_historical_candidate_stage": 3, "Z1_synchronous_candidate_stage": 2}
    actual_z = {name: int(row.get("families_passing", -1)) for name, row in v837z.get("conditions", {}).items()}
    if v837z.get("diagnosis") != "HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL" or actual_z != expected_z or v837z.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837z candidate-stage outcome changed")
    zdepth = v837z.get("candidate_stage_depth", {})
    if zdepth.get("Z1_synchronous_candidate_stage", {}).get("per_cell") != [1] * 10:
        raise RuntimeError("V837z synchronous stage depth changed")

    candidate_status = load_json("experiments/v837_primitive_invention/candidate_organization_program_status.json")
    candidate_resources = load_json("experiments/v837_primitive_invention/candidate_organization_program_resource_accounting.json")
    if candidate_status.get("v837y_diagnosis") != "GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT" or candidate_status.get("v837z_diagnosis") != "HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL" or candidate_status.get("representation_adequacy") != "FAIL":
        raise RuntimeError("candidate-organization program status changed")
    if candidate_status.get("fresh_audit_episodes_consumed") != 0 or candidate_status.get("primitives_promoted") != 0 or candidate_status.get("v838_started") is not False:
        raise RuntimeError("candidate-organization program lock state changed")
    candidate_combined = candidate_resources.get("combined", {})
    if int(candidate_combined.get("model_fits", -1)) != 175 or int(candidate_combined.get("optimizer_steps", -1)) != 33600 or int(candidate_combined.get("processed_examples", -1)) != 17203200 or int(candidate_combined.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("candidate-organization combined resource accounting changed")

    # V837aa diagnosis-only candidate-law alignment audit.
    aa_record = manifest["current_variants"].get("V837aa")
    if not isinstance(aa_record, dict):
        raise RuntimeError("verification manifest missing V837aa")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in aa_record.get(key, []):
            require_path(relative)
    require_path(aa_record["config"])
    require_path(aa_record["frozen_gate"])
    require_path(aa_record["results"])
    aa = load_json(aa_record["results"])
    if aa.get("candidate_law_diagnosis") != "GENUINELY_DIVERSE_CANDIDATE_LAWS" or aa.get("recommended_next_axis") != "NEXT_AXIS_SHARED_INPUT_REPRESENTATION":
        raise RuntimeError("V837aa diagnosis/recommendation changed")
    if aa.get("parent_reproduction_valid") is not True or aa.get("representation_adequacy") != "still_3_of_5_parent":
        raise RuntimeError("V837aa parent reproduction/representation state changed")
    if aa.get("direct_common_basis") is not False or aa.get("common_law_after_signed_permutation") is not False or aa.get("stable_small_type_vocabulary") is not False:
        raise RuntimeError("V837aa candidate-law classification changed")
    if aa.get("gradient_compatibility") != "mixed" or aa.get("fresh_audit_consumed") is not False or aa.get("primitive_count") != 0 or aa.get("v837ab_implemented") is not False or aa.get("v838_started") is not False:
        raise RuntimeError("V837aa lock/gradient state changed")
    aa_status = load_json("experiments/v837_primitive_invention/candidate_law_alignment_program_status.json")
    aa_resources = load_json("experiments/v837_primitive_invention/candidate_law_alignment_program_resource_accounting.json")
    if aa_status.get("candidate_law_diagnosis") != "GENUINELY_DIVERSE_CANDIDATE_LAWS" or aa_status.get("recommended_next_axis") != "NEXT_AXIS_SHARED_INPUT_REPRESENTATION":
        raise RuntimeError("candidate-law program status changed")
    if int(aa_resources.get("model_fits", -1)) != 25 or int(aa_resources.get("optimizer_steps", -1)) != 4800 or int(aa_resources.get("processed_training_examples", -1)) != 2457600 or int(aa_resources.get("unique_seed_defined_task_episodes", -1)) != 3200:
        raise RuntimeError("candidate-law program resource accounting changed")

    # V837ab reference-side input-factorization localization and V837ac minimal neutral transfer.
    for variant_name in ("V837ab", "V837ac"):
        record = manifest["current_variants"].get(variant_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"verification manifest missing {variant_name}")
        for key in ("source", "documentation", "plots", "diagnostics", "raw"):
            for relative in record.get(key, []):
                require_path(relative)
        require_path(record["config"])
        require_path(record["frozen_gate"])
        require_path(record["results"])

    ab = load_json(manifest["current_variants"]["V837ab"]["results"])
    expected_ab = {
        "AB0_exact_factorized_t2": 4,
        "AB1_fully_folded_equivalent": 3,
        "AB2_candidate_factorized_update_folded": 4,
        "AB3_candidate_folded_update_factorized": 4,
        "AB4_frozen_shared_projection": 2,
        "AB5_naive_projection_free": 3,
    }
    actual_ab = {name: int(row.get("families_passing", -1)) for name, row in ab.get("conditions", {}).items()}
    if actual_ab != expected_ab or ab.get("diagnosis") != "SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT" or ab.get("authorized_v837ac_mode") != "TRAINABLE_CONTROLLER_INPUT_FACTORIZATION":
        raise RuntimeError("V837ab input-factorization outcome changed")
    if ab.get("function_class_equivalence_proven") is not True or ab.get("step0_equivalence_proven") is not True or ab.get("fresh_audit_consumed") is not False or ab.get("v838_started") is not False:
        raise RuntimeError("V837ab equivalence/lock state changed")

    ac = load_json(manifest["current_variants"]["V837ac"]["results"])
    ac_decision = load_json("experiments/v837_primitive_invention/v837ac/diagnostics/decision_state.json")
    expected_ac = {"AC0_y3_parent": 3, "AC1_controller_input_factorization": 3, "AC1F_folded_control": 3}
    actual_ac = {name: int(row.get("families_passing", -1)) for name, row in ac.get("conditions", {}).items()}
    if actual_ac != expected_ac or ac.get("diagnosis") != "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT" or ac.get("representation_adequacy_pass") is not False:
        raise RuntimeError("V837ac input-transfer outcome changed")
    if ac.get("authorized_mode") != "TRAINABLE_CONTROLLER_INPUT_FACTORIZATION" or ac_decision.get("parent_reproduced") is not True or ac_decision.get("step0_equivalence_proven") is not True:
        raise RuntimeError("V837ac authorization/compatibility changed")
    if ac.get("sample_efficiency_retest_allowed") is not False or ac.get("structural_search_allowed") is not False or ac.get("primitive_mining_allowed") is not False or ac.get("fresh_audit_consumed") is not False or ac.get("v838_started") is not False:
        raise RuntimeError("V837ac science lock state changed")

    input_status = load_json("experiments/v837_primitive_invention/input_factorization_program_status.json")
    input_resources = load_json("experiments/v837_primitive_invention/input_factorization_program_resource_accounting.json")
    if input_status.get("v837ab_diagnosis") != "SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT" or input_status.get("v837ac_diagnosis") != "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT" or input_status.get("representation_adequacy") != "FAIL" or input_status.get("input_axis_closed") is not True:
        raise RuntimeError("input-factorization program status changed")
    combined_input = input_resources.get("combined", {})
    if int(combined_input.get("model_fits", -1)) != 225 or int(combined_input.get("optimizer_steps", -1)) != 43200 or int(combined_input.get("processed_training_examples", -1)) != 22118400 or int(combined_input.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("input-factorization program resource accounting changed")

    ad_record = manifest["current_variants"].get("V837ad")
    if not isinstance(ad_record, dict):
        raise RuntimeError("verification manifest missing V837ad")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in ad_record.get(key, []):
            require_path(relative)
    require_path(ad_record["config"])
    require_path(ad_record["frozen_gate"])
    require_path(ad_record["results"])
    ad = load_json(ad_record["results"])
    ad_decision = ad.get("decision", {})
    expected_ad = {
        "AD0_H13_dense": 4,
        "AD1_H40_dense": 4,
        "AD2_H40_2x20": 4,
        "AD3_H40_5x8": 4,
        "AD4_H40_10x4": 4,
        "AD4S_S0": 4,
    }
    actual_ad = {name: int(row.get("families_passing", -1)) for name, row in ad.get("conditions", {}).items()}
    if actual_ad != expected_ad:
        raise RuntimeError("V837ad candidate-geometry outcomes changed")
    if ad_decision.get("diagnosis") != "TEN_BY_FOUR_CANDIDATE_GEOMETRY_SUFFICIENT_IN_REFERENCE" or ad_decision.get("ad0_anchor_valid") is not True or int(ad_decision.get("ad1_dense_h40_families", -1)) != 4 or ad_decision.get("geometry_stage_run") is not True:
        raise RuntimeError("V837ad width/geometry decision changed")
    if ad_decision.get("authorized_v837ae_mode") is not None or ad_decision.get("ad4s_robustness_run") is not False or ad_decision.get("next_axis") != "GRAPH_MESSAGE_OUTPUT_INTERFACE_ORGANIZATION":
        raise RuntimeError("V837ad transfer/next-axis decision changed")
    if ad_decision.get("sample_efficiency_retest_allowed") is not False or ad_decision.get("structural_search_allowed") is not False or ad_decision.get("primitive_mining_allowed") is not False or ad_decision.get("fresh_audit_consumed") is not False or ad_decision.get("v838_started") is not False:
        raise RuntimeError("V837ad science lock state changed")
    ad4 = ad["conditions"]["AD4_H40_10x4"]
    ad4s = ad["conditions"]["AD4S_S0"]
    if int(ad4.get("active_candidate_recurrent_weights", -1)) != 160 or int(ad4s.get("active_candidate_recurrent_weights", -1)) != 160 or int(ad4.get("total_active_macs_per_timestep", -1)) != int(ad4s.get("total_active_macs_per_timestep", -2)):
        raise RuntimeError("V837ad degree-matched geometry control changed")
    if (ROOT / "experiments/v837_primitive_invention/v837ad/raw/robustness_runs.json").exists() or (ROOT / "experiments/v837_primitive_invention/v837ae").exists():
        raise RuntimeError("V837ad executed an unauthorized robustness/transfer stage")
    geometry_status = load_json("experiments/v837_primitive_invention/candidate_recurrent_geometry_program_status.json")
    geometry_resources = load_json("experiments/v837_primitive_invention/candidate_recurrent_geometry_program_resource_accounting.json")
    if geometry_status.get("v837ad_diagnosis") != "TEN_BY_FOUR_CANDIDATE_GEOMETRY_SUFFICIENT_IN_REFERENCE" or geometry_status.get("v837ae_run") is not False or geometry_status.get("next_single_variable") != "GRAPH_MESSAGE_OUTPUT_INTERFACE_ORGANIZATION":
        raise RuntimeError("candidate-recurrent-geometry program status changed")
    combined_geometry = geometry_resources.get("combined", {})
    if int(combined_geometry.get("model_fits", -1)) != 150 or int(combined_geometry.get("optimizer_steps", -1)) != 28800 or int(combined_geometry.get("processed_training_examples", -1)) != 14745600 or int(combined_geometry.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("candidate-recurrent-geometry program resource accounting changed")

    # V837af remaining candidate-input sibling closure. A neutral >=4/5 result
    # hard-stops the conditional V837ag/V837ah architecture-localization stages.
    af_record = manifest["current_variants"].get("V837af")
    if not isinstance(af_record, dict):
        raise RuntimeError("verification manifest missing V837af")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in af_record.get(key, []):
            require_path(relative)
    require_path(af_record["config"])
    require_path(af_record["frozen_gate"])
    require_path(af_record["results"])
    af = load_json(af_record["results"])
    af_decision = load_json("experiments/v837_primitive_invention/v837af/diagnostics/decision_state.json")
    expected_af = {
        "AF0_y3_parent": 3,
        "AF1_shared_candidate_input_factorization": 3,
        "AF1F_folded_candidate_input_control": 3,
        "AF1D_deshared_candidate_input_factorization": 4,
    }
    actual_af = {name: int(row.get("families_passing", -1)) for name, row in af.get("conditions", {}).items()}
    if actual_af != expected_af:
        raise RuntimeError("V837af candidate-input sibling outcomes changed")
    if af.get("diagnosis") != "DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT" or af.get("diagnosis_qualifiers") != ["SHARED_INPUT_BASIS_HARMFUL"]:
        raise RuntimeError("V837af diagnosis/qualifier changed")
    if af.get("best_passing_condition") != "AF1D_deshared_candidate_input_factorization" or af.get("representation_adequacy_pass") is not True or af.get("sample_efficiency_retest_allowed") is not True:
        raise RuntimeError("V837af adequacy/sample-efficiency gate changed")
    step0 = load_json("experiments/v837_primitive_invention/v837af/diagnostics/step0_equivalence.json")
    anchor = load_json("experiments/v837_primitive_invention/v837af/diagnostics/anchor_compatibility.json")
    if step0.get("step0_equivalence_proven") is not True or float(step0.get("maximum_error", 1.0)) > 1e-6 or anchor.get("parent_reproduced") is not True:
        raise RuntimeError("V837af equivalence/parent guard changed")
    if af_decision.get("v837ag_allowed") is not False or af.get("v837ag_allowed") is not False:
        raise RuntimeError("V837af improperly authorized V837ag after neutral success")
    if af.get("structural_search_allowed") is not False or af.get("primitive_mining_allowed") is not False or af.get("fresh_audit_consumed") is not False or af.get("v838_started") is not False:
        raise RuntimeError("V837af science lock state changed")
    for forbidden in ("v837ae", "v837ag", "v837ah", "v838"):
        if (ROOT / "experiments" / "v837_primitive_invention" / forbidden).exists():
            raise RuntimeError(f"unauthorized {forbidden} directory exists after V837af hard stop")
    sibling_status = load_json("experiments/v837_primitive_invention/input_sibling_controller_basis_program_status.json")
    sibling_resources = load_json("experiments/v837_primitive_invention/input_sibling_controller_basis_program_resource_accounting.json")
    if sibling_status.get("v837af_diagnosis") != "DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT" or sibling_status.get("architecture_localization_hard_stop_triggered") is not True or sibling_status.get("representation_adequacy") != "PASS_4_OF_5_AF1D":
        raise RuntimeError("input-sibling program status changed")
    if sibling_status.get("v837ag_run") is not False or sibling_status.get("v837ah_run") is not False or sibling_status.get("sample_efficiency_retest_allowed") is not True:
        raise RuntimeError("input-sibling program continuation gate changed")
    combined_sibling = sibling_resources.get("combined", {})
    if int(combined_sibling.get("model_fits", -1)) != 100 or int(combined_sibling.get("optimizer_steps", -1)) != 19200 or int(combined_sibling.get("processed_training_examples", -1)) != 9830400 or int(combined_sibling.get("unique_seed_defined_episodes", -1)) != 3200:
        raise RuntimeError("input-sibling program resource accounting changed")

    # V837ai freezes AF1D and characterizes only the historical 1x/2x/4x
    # unique-development regimes. AI4 must be reused, never silently retrained.
    ai_record = manifest["current_variants"].get("V837ai")
    if not isinstance(ai_record, dict):
        raise RuntimeError("verification manifest missing V837ai")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in ai_record.get(key, []):
            require_path(relative)
    require_path(ai_record["config"])
    require_path(ai_record["frozen_gate"])
    require_path(ai_record["results"])
    ai = load_json(ai_record["results"])
    ai_decision = load_json("experiments/v837_primitive_invention/v837ai/diagnostics/decision_state.json")
    expected_ai = {"1x": 2, "2x": 2, "4x": 4}
    actual_ai = {label: int(row.get("families_passing", -1)) for label, row in ai.get("conditions", {}).items()}
    if actual_ai != expected_ai:
        raise RuntimeError("V837ai data-frontier outcomes changed")
    if ai.get("architecture") != "AF1D_deshared_candidate_input_factorization" or ai.get("architecture_frozen") is not True:
        raise RuntimeError("V837ai frozen architecture changed")
    if ai.get("diagnosis") != "AF1D_REQUIRES_4X_UNIQUE_DATA" or ai.get("diagnosis_qualifiers") != ["SAMPLE_EFFICIENCY_THRESHOLD_NOT_IMPROVED_VS_HISTORICAL_GRU"]:
        raise RuntimeError("V837ai diagnosis changed")
    if ai.get("minimum_tested_sufficient_multiplier") != 4 or ai.get("pass_count_monotonic") is not True or ai.get("sample_efficiency_claim_allowed") is not True:
        raise RuntimeError("V837ai minimum/monotonic sample-efficiency decision changed")
    if ai.get("representation_adequacy_confirmed") is not True or ai.get("structural_search_recovery_allowed") is not True or ai.get("recommended_structural_search_multiplier") != 4:
        raise RuntimeError("V837ai structural-search recovery authorization changed")
    if ai.get("next_program") != "V837aj_STRUCTURAL_SEARCH_RECOVERY" or ai.get("primitive_mining_allowed") is not False or ai.get("fresh_audit_consumed") is not False or ai.get("v838_started") is not False:
        raise RuntimeError("V837ai downstream science locks changed")
    ai_anchor = load_json("experiments/v837_primitive_invention/v837ai/diagnostics/ai4_anchor_reuse.json")
    ai_nesting = load_json("experiments/v837_primitive_invention/v837ai/diagnostics/data_nesting.json")
    ai_pairing = load_json("experiments/v837_primitive_invention/v837ai/diagnostics/initialization_pairing.json")
    ai_historical = load_json("experiments/v837_primitive_invention/v837ai/diagnostics/historical_v837l_comparison.json")
    if ai_anchor.get("compatible") is not True or int(ai_anchor.get("rows_reused", -1)) != 25 or ai_anchor.get("reanalyzed", {}).get("families_passing") != 4:
        raise RuntimeError("V837ai reused 4x anchor changed")
    if ai_nesting.get("exact") is not True or ai_nesting.get("strict_nesting") is not True or ai_nesting.get("development_validation_overlap") != 0 or ai_nesting.get("total_unique_family_seed_episodes", {}).get("union") != 3200:
        raise RuntimeError("V837ai nested-data proof changed")
    if ai_pairing.get("pairing_exact") is not True or ai_pairing.get("parameter_count_constant") is not True or ai_pairing.get("macs_constant") is not True:
        raise RuntimeError("V837ai initialization/architecture pairing changed")
    if ai_historical.get("families_passing") != {"1x": {"gru_reference": 2, "neutral_high_capacity": 1, "residual_rnn_reference": 2}, "2x": {"gru_reference": 3, "neutral_high_capacity": 1, "residual_rnn_reference": 2}, "4x": {"gru_reference": 5, "neutral_high_capacity": 2, "residual_rnn_reference": 3}}:
        raise RuntimeError("V837ai historical V837l comparator changed")
    for raw_name, expected_rows in (("ai1_runs.json", 25), ("ai2_runs.json", 25), ("ai4_reused_anchor.json", 25)):
        raw = load_json(f"experiments/v837_primitive_invention/v837ai/raw/{raw_name}")
        if len(raw.get("rows", [])) != expected_rows:
            raise RuntimeError(f"V837ai {raw_name} row count changed")
    ai_resources = load_json("experiments/v837_primitive_invention/af1d_sample_efficiency_program_resource_accounting.json")
    new_resources = ai_resources.get("new_execution_resources", {})
    if int(ai_resources.get("new_model_fits", -1)) != 50 or int(ai_resources.get("reused_accepted_4x_fits", -1)) != 25 or int(ai_resources.get("union_unique_task_episodes", -1)) != 3200:
        raise RuntimeError("V837ai new-vs-reused accounting changed")
    if int(new_resources.get("optimizer_steps", -1)) != 9600 or int(new_resources.get("processed_training_examples", -1)) != 1843200 or int(ai_resources.get("active_parameters", -1)) != 1643 or int(ai_resources.get("af1d_macs_per_timestep", -1)) != 1206:
        raise RuntimeError("V837ai execution/architecture resource accounting changed")
    ai_status = load_json("experiments/v837_primitive_invention/af1d_sample_efficiency_program_status.json")
    if ai_status.get("v837ai_complete") is not True or ai_status.get("diagnosis") != "AF1D_REQUIRES_4X_UNIQUE_DATA" or ai_status.get("structural_search_recovery_allowed") is not True or ai_status.get("recommended_structural_search_multiplier") != 4:
        raise RuntimeError("V837ai program status changed")
    for forbidden in ("v837ae", "v837ag", "v837ah", "v838"):
        if (ROOT / "experiments" / "v837_primitive_invention" / forbidden).exists():
            raise RuntimeError(f"unauthorized {forbidden} directory exists after V837ai")

    # V837aj reopens only message-topology search on the exact AF1D substrate.
    # Stage B is legal only after the calibrated proxy passes the frozen Reality Gate.
    aj_record = manifest["current_variants"].get("V837aj")
    if not isinstance(aj_record, dict):
        raise RuntimeError("verification manifest missing V837aj")
    for key in ("source", "documentation", "plots", "diagnostics", "raw"):
        for relative in aj_record.get(key, []):
            require_path(relative)
    require_path(aj_record["config"])
    require_path(aj_record["frozen_gate"])
    require_path(aj_record["results"])
    aj = load_json(aj_record["results"])
    aj_decision = load_json("experiments/v837_primitive_invention/v837aj/diagnostics/decision_state.json")
    aj_anchor = load_json("experiments/v837_primitive_invention/v837aj/diagnostics/anchor_reproduction.json")
    aj_fidelity = load_json("experiments/v837_primitive_invention/v837aj/diagnostics/fidelity_decision.json")
    if aj_anchor.get("anchor_reproduced") is not True or int(aj_anchor.get("families_passing", -1)) != 4:
        raise RuntimeError("V837aj AF1D anchor reproduction changed")
    if float(aj_anchor.get("max_development_delta", 1.0)) != 0.0 or float(aj_anchor.get("max_validation_delta", 1.0)) != 0.0:
        raise RuntimeError("V837aj AF1D anchor drift changed")
    if aj_fidelity.get("calibration_complete") is not True or aj_fidelity.get("proxy_valid") is not True or aj_fidelity.get("selected_search_fidelity") != "F3":
        raise RuntimeError("V837aj frozen search-fidelity decision changed")
    f3 = aj_fidelity.get("metrics", {}).get("F3", {})
    if f3.get("passes_frozen_gate") is not True or float(f3.get("median_spearman_rho", 0.0)) < 0.75 or float(f3.get("median_kendall_tau", 0.0)) < 0.60 or float(f3.get("minimum_family_kendall_tau", 0.0)) < 0.35 or float(f3.get("median_pairwise_order_accuracy", 0.0)) < 0.72 or float(f3.get("minimum_family_top4_recall", 0.0)) < 0.50:
        raise RuntimeError("V837aj F3 Reality Gate no longer passes")
    if aj.get("stage_b_run") is not True or aj_decision.get("constructive_search_run") is not True or aj_decision.get("search_stage_allowed") is not True:
        raise RuntimeError("V837aj authorized Stage-B execution changed")
    budget = load_json("experiments/v837_primitive_invention/v837aj/diagnostics/search_random_budget_match.json")
    if budget.get("all_exact_64") is not True or budget.get("all_slot_matched") is not True:
        raise RuntimeError("V837aj directed/random budget pairing changed")
    allowed_aj_diagnoses = {
        "STRUCTURAL_SEARCH_RECOVERED",
        "RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT",
        "STRUCTURAL_SELECTION_OVERFIT",
        "STRUCTURAL_DISCOVERY_NOT_RECOVERED",
        "STRUCTURAL_DISCOVERY_SUPERIORITY_INCONCLUSIVE",
    }
    if aj.get("diagnosis") not in allowed_aj_diagnoses or aj_decision.get("diagnosis") != aj.get("diagnosis"):
        raise RuntimeError("V837aj final diagnosis changed or is invalid")
    mining_expected = aj.get("diagnosis") in {"STRUCTURAL_SEARCH_RECOVERED", "RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT"}
    if aj.get("primitive_mining_allowed_next") is not mining_expected or aj_decision.get("primitive_mining_allowed_next") is not mining_expected:
        raise RuntimeError("V837aj primitive-mining authorization disagrees with diagnosis")
    if aj.get("fresh_audit_consumed") is not False or aj.get("primitives_promoted") != 0 or aj.get("v838_started") is not False:
        raise RuntimeError("V837aj violated fresh-audit/primitive/V838 locks")
    aj_status = load_json("experiments/v837_primitive_invention/structural_search_recovery_program_status.json")
    if aj_status.get("diagnosis") != aj.get("diagnosis") or aj_status.get("fresh_audit_episodes_consumed") != 0 or aj_status.get("primitives_promoted") != 0 or aj_status.get("v838_started") is not False:
        raise RuntimeError("V837aj program status disagrees with results")

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
