from __future__ import annotations

from .utils import AQ, ROOT, read_json, git_blob_sha256, sha256_json

AQ_RESULTS_REL = "experiments/v837_primitive_invention/v837aq/results.json"
AQ_GATE_REL = "experiments/v837_primitive_invention/v837aq/frozen_program_level_causal_operator_gate.json"
AQ_OPERATOR_CONTRACT_REL = "experiments/v837_primitive_invention/v837aq/raw/frozen_operator_contracts.json"
AQ_PROGRAM_CONTRACT_REL = "experiments/v837_primitive_invention/v837aq/raw/frozen_program_operator_contracts.json"
AQ_FOLDS_REL = "experiments/v837_primitive_invention/v837aq/raw/frozen_source_folds.json"
EXPECTED_PROGRAM_CONTRACT_SHA = "461faadf9cb9cc1262c9a288236c08875ae5212685bb8dd06e6a4ed92768d9af"
CONFIRMED = ("conditional_routing", "delayed_recall", "iterative_state")
NEGATIVE = "variable_composition"


def load_aq_contracts() -> dict:
    results = read_json(ROOT / AQ_RESULTS_REL)
    gate = read_json(ROOT / AQ_GATE_REL)
    op = read_json(ROOT / AQ_OPERATOR_CONTRACT_REL)
    program = read_json(ROOT / AQ_PROGRAM_CONTRACT_REL)
    folds = read_json(ROOT / AQ_FOLDS_REL)
    if results.get("diagnosis") != "CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED":
        raise RuntimeError("V837AR_AQ_DIAGNOSIS_DRIFT")
    if results.get("coordinate_free_operator_pattern_established") is not True:
        raise RuntimeError("V837AR_AQ_OPERATOR_PATTERN_DRIFT")
    if tuple(results.get("heldout_confirmed_operator_families", ())) != CONFIRMED:
        raise RuntimeError("V837AR_AQ_CONFIRMED_FAMILY_DRIFT")
    if results.get("globally_compositionally_closed_families") != ["iterative_state"]:
        raise RuntimeError("V837AR_AQ_COMPOSITION_HISTORY_DRIFT")
    if results.get("second_order_required_families") != ["conditional_routing"]:
        raise RuntimeError("V837AR_AQ_ROUTING_ORDER_DRIFT")
    ranks = results.get("predictive_response_rank_99", {})
    if ranks != {"conditional_routing": 4, "delayed_recall": 1, "iterative_state": 3}:
        raise RuntimeError("V837AR_AQ_RANK_HISTORY_DRIFT")
    if program.get("contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA:
        raise RuntimeError("V837AR_AQ_PROGRAM_CONTRACT_HASH_DRIFT")
    if program.get("family_contracts", {}).get("variable_composition") is not None:
        raise RuntimeError("V837AR_VARIABLE_COMPOSITION_RESCUE_FORBIDDEN")
    for key in ("fresh_audit_consumed", "primitive_archive_population", "v838_started"):
        if results.get(key) is not False:
            raise RuntimeError(f"V837AR_PREDECESSOR_LOCK_DRIFT:{key}")
    if int(results.get("primitives_promoted", -1)) != 0:
        raise RuntimeError("V837AR_PREDECESSOR_PROMOTION_DRIFT")
    return {"results": results, "gate": gate, "operator_contracts": op, "program_contracts": program, "folds": folds}


def imported_thresholds() -> dict:
    c = load_aq_contracts()
    g = c["gate"]
    return {
        "operator_gate": dict(g["operator_gate"]),
        "reality_gate": dict(g["reality_gate"]),
        "composition": {"response_nrmse_max": 0.10, "direction_agreement_min": 0.85, "task_success_min": 0.80},
        "control_error_margin_min_aq": float(g.get("control_error_margin_min", 0.03)),
        "control_error_margin_min_ar": 0.05,
    }


def aq_import_snapshot() -> dict:
    c = load_aq_contracts()
    payload = {
        "version": "V837ar",
        "aq_diagnosis": c["results"]["diagnosis"],
        "confirmed_families": list(CONFIRMED),
        "negative_family": NEGATIVE,
        "operator_orders": {f: c["program_contracts"]["family_contracts"][f]["operator_order"] for f in CONFIRMED},
        "predictive_rank99": c["results"]["predictive_response_rank_99"],
        "aq_discovery_composition": {f: bool(c["program_contracts"]["family_contracts"][f]["compositionally_closed_in_discovery"]) for f in CONFIRMED},
        "aq_program_contract_hash": c["program_contracts"]["contract_sha256"],
        "aq_results_git_blob_sha256": git_blob_sha256(AQ_RESULTS_REL),
        "aq_gate_git_blob_sha256": git_blob_sha256(AQ_GATE_REL),
        "aq_operator_contract_git_blob_sha256": git_blob_sha256(AQ_OPERATOR_CONTRACT_REL),
        "aq_program_contract_git_blob_sha256": git_blob_sha256(AQ_PROGRAM_CONTRACT_REL),
        "aq_folds_git_blob_sha256": git_blob_sha256(AQ_FOLDS_REL),
        "aq_fold_sha256": c["folds"]["fold_sha256"],
        "thresholds": imported_thresholds(),
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "primitive_archive_population": False,
        "v838_started": False,
    }
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload
