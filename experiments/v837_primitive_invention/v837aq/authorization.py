from __future__ import annotations

import json

from .data_roles import PARTITIONS, assert_roles
from .source_folds import EXPECTED_FOLD_SHA
from .utils import HERE, ROOT, START_SHA, REQUIRED_BRANCH, branch_name, git_blob_sha256, head_sha, read_json, sha256_file, sha256_json, write_json

GATE = HERE / "frozen_program_level_causal_operator_gate.json"
POWERED_FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state", "variable_composition")
AP_DECISION = "experiments/v837_primitive_invention/v837ap/diagnostics/decision_state.json"
AP_RESULTS = "experiments/v837_primitive_invention/v837ap/results.json"
AP_REPORT = "docs/V837_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE_REPORT.md"
AP_FOLDS = "experiments/v837_primitive_invention/v837ap/raw/frozen_source_folds.json"
AK_RECON = "experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json"


def _source_state() -> dict:
    ap = read_json(ROOT / AP_DECISION)
    results = read_json(ROOT / AP_RESULTS)
    if ap.get("diagnosis") != "DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED":
        raise RuntimeError("V837AQ_BAD_V837AP_DIAGNOSIS")
    if ap.get("next_program") != "V837aq_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION":
        raise RuntimeError("V837AQ_BAD_V837AP_NEXT_PROGRAM")
    if int(ap.get("validated_family_count", -1)) != 0 or int(ap.get("primitives_promoted", -1)) != 0:
        raise RuntimeError("V837AQ_BAD_V837AP_CLOSURE")
    if ap.get("fresh_audit_consumed") is not False or ap.get("v838_started") is not False:
        raise RuntimeError("V837AQ_PREDECESSOR_LOCK_DRIFT")
    recon = read_json(ROOT / AK_RECON)
    rows = list(recon.get("rows", []))
    if len(rows) != 50 or sum(bool(r.get("competent")) for r in rows) != 40:
        raise RuntimeError("V837AQ_SOURCE_POPULATION_DRIFT")
    return {"ap": ap, "results": results, "population": rows}


def freeze_gate() -> dict:
    if GATE.is_file():
        return read_json(GATE)
    if head_sha() != START_SHA:
        raise RuntimeError(f"V837AQ_GATE_MUST_FREEZE_AT_START_SHA:{head_sha()}")
    if branch_name() != REQUIRED_BRANCH:
        raise RuntimeError(f"V837AQ_GATE_MUST_FREEZE_ON_REQUIRED_BRANCH:{branch_name()}")
    assert_roles()
    source = _source_state()
    checkpoint_hashes = {r["organism_id"]: sha256_file(ROOT / r["checkpoint"]) for r in source["population"]}
    gate = {
        "version": "V837aq",
        "program": "V837aq_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION",
        "start_sha": START_SHA,
        "required_branch": REQUIRED_BRANCH,
        "source_hashes": {
            "v837ap_decision": git_blob_sha256(AP_DECISION),
            "v837ap_results": git_blob_sha256(AP_RESULTS),
            "v837ap_report": git_blob_sha256(AP_REPORT),
            "v837ap_folds": git_blob_sha256(AP_FOLDS),
            "v837ak_reconstruction": git_blob_sha256(AK_RECON),
        },
        "source_checkpoint_hashes": checkpoint_hashes,
        "source_population": {"total": 50, "competent": 40, "incompetent": 10},
        "powered_families": list(POWERED_FAMILIES),
        "partial_observation_status": "UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY",
        "v837ap_statement": "V837ap found no powered family satisfying the complete valid causal SET/control gate; quotient, commutativity, and heldout stages received zero candidates. Engineering-invalid chart rows are not scientific negatives.",
        "ontology_lock": "primary invariant is a coordinate-free finite-horizon semantic intervention-response operator, not a hidden-state coordinate",
        "primary_intervention_lock": "task/environment semantic channels only; no arbitrary hidden-state SET is primary evidence",
        "operator_orders": [1, 2],
        "third_order_forbidden": True,
        "reality_gate": {
            "family": "iterative_state",
            "oracle_kernel": "0.35*(0.65**(h-1))*delta",
            "timings": ["EARLY", "MIDDLE", "LATE"],
            "requested_additive_magnitudes": [-1.0, -0.5, 0.5, 1.0],
            "horizons": [1, 2, 4, 8],
            "response_nrmse_max": 0.08,
            "pearson_min": 0.90,
            "direction_agreement_min": 0.90,
            "perturbed_task_success_min": 0.85,
            "gain_ratio": [0.75, 1.25],
            "family_pass": ">=60% discovery organisms and both structural-search engines",
            "kill_on_failure": True,
        },
        "operator_gate": {"response_nrmse_max": 0.10, "pearson_min": 0.85, "direction_min": 0.85, "task_success_min": 0.80, "zero_effect_median_abs_max": 0.10, "family_pass": ">=60% and both engines"},
        "controls": ["time_shuffled_labels", "magnitude_shuffled_labels", "wrong_phase", "wrong_family_operator_template", "endpoint_only_predictor", "matched_low_order_generic_regression", "historically_incompetent_organisms"],
        "control_error_margin_min": 0.03,
        "second_order_rule": "evaluate paired semantic interventions only when oracle interaction residual is non-negligible; no third order",
        "second_order_required_thresholds": {"median_abs_normalized": 0.05, "p90_abs_normalized": 0.10},
        "cross_organism_rule": "every accepted organism must match oracle; then compare semantic response tensors only, never neural coordinates",
        "localization_rule": "only after operator identity; paired-natural contribution/path patching over component x time, deterministic exhaustive singles plus frozen top-k prefixes; no evolutionary search",
        "localization_channels": ["cell_output", "aggregate_message", "global_term", "global_gate"],
        "localization_prefix_sizes": [1, 2, 4, 8],
        "composition_rule": "component operator contracts must predict compound/multi-step responses without hidden-state alignment",
        "predictive_state_rule": "fallback diagnostic only if accepted operator lacks simple composition/context closure; history x future-test response matrix; 99% SVD energy rank",
        "heldout_rule": "freeze operator family/order/control/localization/composition contract before reading heldout operator results; no refit or hyperparameter search",
        "episode_roles": PARTITIONS,
        "fold_sha256": EXPECTED_FOLD_SHA,
        "fresh_audit_consumed": False,
        "primitive_archive_population": False,
        "primitives_promoted": 0,
        "v838_started": False,
        "new_source_model_fits": 0,
        "source_optimizer_steps": 0,
        "source_training_examples": 0,
        "source_architecture_changes": 0,
        "no_cross_organism_state_alignment": True,
        "no_cross_organism_q_alignment": True,
    }
    gate["gate_sha256"] = sha256_json(gate)
    write_json(GATE, gate)
    return gate


def assert_authorized() -> dict:
    gate = read_json(GATE) if GATE.is_file() else freeze_gate()
    source = _source_state()
    if gate.get("start_sha") != START_SHA:
        raise RuntimeError("V837AQ_START_SHA_DRIFT")
    paths = {"v837ap_decision": AP_DECISION, "v837ap_results": AP_RESULTS, "v837ap_report": AP_REPORT, "v837ap_folds": AP_FOLDS, "v837ak_reconstruction": AK_RECON}
    for key, rel in paths.items():
        if git_blob_sha256(rel) != gate["source_hashes"][key]:
            raise RuntimeError(f"V837AQ_PROTECTED_BLOB_DRIFT:{rel}")
    for row in source["population"]:
        if sha256_file(ROOT / row["checkpoint"]) != gate["source_checkpoint_hashes"][row["organism_id"]]:
            raise RuntimeError(f"V837AQ_CHECKPOINT_DRIFT:{row['organism_id']}")
    payload = {
        "version": "V837aq", "authorized": True, "start_sha": START_SHA,
        "v837ap_diagnosis": source["ap"]["diagnosis"], "v837ap_next_program": source["ap"]["next_program"],
        "source_organisms": 50, "competent_organisms": 40, "incompetent_organisms": 10,
        "fresh_audit_consumed": False, "primitive_archive_population": False, "primitives_promoted": 0, "v838_started": False,
        "new_source_model_fits": 0, "source_optimizer_steps": 0, "hidden_state_set_primary_evidence": False,
    }
    write_json(HERE / "diagnostics/authorization.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(assert_authorized(), indent=2))
