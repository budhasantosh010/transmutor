from __future__ import annotations

import json

from .utils import HERE, ROOT, START_SHA, git_blob_sha256, head_sha, read_json, sha256_file, sha256_json, write_json

AM = ROOT / "experiments/v837_primitive_invention/v837am"
AK = ROOT / "experiments/v837_primitive_invention/v837ak"
GATE = HERE / "frozen_causal_primitive_gate.json"

SOURCE = {
    "v837am_decision": "experiments/v837_primitive_invention/v837am/diagnostics/decision_state.json",
    "v837am_results": "experiments/v837_primitive_invention/v837am/results.json",
    "v837am_am_a": "experiments/v837_primitive_invention/v837am/raw/am_a_selection.json",
    "v837am_am_c": "experiments/v837_primitive_invention/v837am/raw/am_c_selection.json",
    "v837am_failure_ledger": "experiments/v837_primitive_invention/v837am/diagnostics/failure_ledger.json",
    "v837ak_reconstruction": "experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json",
    "v837ak_source_population": "experiments/v837_primitive_invention/v837ak/raw/source_population.json",
}

PARTITIONS = {
    "AN_FIT": [10000, 10127],
    "AN_SELECT": [10128, 10255],
    "AN_ROUTING_FIT": [10256, 10319],
    "AN_ROUTING_SELECT": [10320, 10383],
    "AN_META_CONFIRM": [10384, 10447],
    "AN_FINAL_DEV": [10448, 10511],
    "AN_FINAL_VALIDATION": [20000, 20127],
    "FRESH_AUDIT": [90000, 90499],
}

PRIMARY_PROBES = {
    "conditional_routing": "ROUTING_CONTROL_STATE",
    "delayed_recall": "RECALL_MAINTAIN",
    "iterative_state": "ITERATIVE_RUNNING_STATE",
    "partial_observation": "PARTIAL_LATENT_Z",
    "variable_composition": "COMPOSITION_RUNNING_STATE",
}
SECONDARY_PROBES = {
    "conditional_routing": ["ROUTING_SELECTED_VALUE"],
    "delayed_recall": ["RECALL_WRITE", "RECALL_READ"],
    "partial_observation": ["PARTIAL_PREDICTIVE_MEAN"],
}
COUNTERFACTUALS = {
    "conditional_routing": "CONTROL_FLIP",
    "delayed_recall": "MEMORY_VALUE_FLIP",
    "iterative_state": "MID_INPUT_FLIP",
    "partial_observation": "INITIAL_LATENT_SIGN_FLIP",
    "variable_composition": "INITIAL_VALUE_SIGN_FLIP",
}
SUCCESS_TOLERANCE = {
    "conditional_routing": 0.25,
    "delayed_recall": 0.35,
    "iterative_state": 0.16,
    "partial_observation": 0.20,
    "variable_composition": 0.16,
}
MIN_ELIGIBLE = {
    "AN_FIT": 48,
    "AN_SELECT": 32,
    "AN_META_CONFIRM": 24,
    "AN_FINAL_DEV": 24,
    "AN_FINAL_VALIDATION": 32,
}
CARRIERS = ["STATE40", "OUTPUT40", "MESSAGE40", "GLOBAL40", "GATE1"]
DIAGNOSTIC_CARRIERS = ["COUPLING_FACTOR4"]
LATENT_DIMS = {
    "STATE40": [1, 2, 4, 8],
    "OUTPUT40": [1, 2, 4, 8],
    "MESSAGE40": [1, 2, 4, 8],
    "GLOBAL40": [1, 2, 4, 8],
    "GATE1": [1],
    "COUPLING_FACTOR4": [1, 2, 4],
}
SOURCE_SWAP_THRESHOLDS = {
    "eligible_pairs_min_select": 32,
    "median_recovery_min": 0.60,
    "direction_agreement_min": 0.75,
    "counterfactual_success_min": 0.70,
    "random_margin_min": 0.20,
    "paired_p_max": 0.01,
    "median_ood_ratio_max": 2.0,
}
FAMILY_SUPPORT = {"minimum_competent_organisms": 5, "minimum_pass_fraction": 0.60}
ROUTING_PREFIXES = {"message": [1, 2, 4, 8, 16, 32], "global": [1, 2, 4, 8, 10], "combined": [1, 2, 4, 8, 16, 32]}
ROUTING_THRESHOLDS = {"median_recovery_min": 0.60, "counterfactual_success_min": 0.70, "direction_agreement_min": 0.75, "random_margin_min": 0.20, "paired_p_max": 0.01}
SYNERGY_THRESHOLDS = {"median_recovery_min": 0.60, "counterfactual_success_min": 0.70, "direction_agreement_min": 0.75, "paired_p_max": 0.01, "median_ood_ratio_max": 2.0, "strong_best_member_max": 0.30, "strong_gain_min": 0.30, "strong_cardinality_max": 4}


def _derive_source_state() -> dict:
    decision = read_json(ROOT / SOURCE["v837am_decision"])
    results = read_json(ROOT / SOURCE["v837am_results"])
    if decision.get("diagnosis") != "DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE" or results.get("diagnosis") != "DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE":
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: V837am diagnosis")
    if decision.get("next_program") != "V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED":
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: V837am next program")
    expected = {"am_a_configs": 126, "am_a_pass_count": 0, "am_b_configs": 20, "am_b_pass_count": 0, "am_c_configs": 7, "am_c_pass_count": 0, "causal_recipient_count": 38, "new_model_fits": 0, "optimizer_steps": 0}
    for key, value in expected.items():
        if int(decision.get(key, -1)) != value:
            raise RuntimeError(f"V837AN_SOURCE_INTEGRITY_FAILURE: {key}")
    if decision.get("fresh_audit_consumed") is not False or decision.get("primitive_archive_allowed_next") is not False or decision.get("v838_started") is not False or int(decision.get("primitives_promoted", -1)) != 0:
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: V837am locks")

    am_a = read_json(ROOT / SOURCE["v837am_am_a"])
    invalid_count = sum(not bool(row.get("valid", True)) for row in am_a.get("results", []))
    if int(am_a.get("configs_evaluated", -1)) != 126 or invalid_count != 105:
        raise RuntimeError(f"V837AN_SOURCE_INTEGRITY_FAILURE: AM-A invalid count {invalid_count}")

    am_c = read_json(ROOT / SOURCE["v837am_am_c"])
    if int(am_c.get("configs_evaluated", -1)) != 7 or len(am_c.get("results", [])) != 7:
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: AM-C count")
    zero_rows = all(int(row.get("aggregate", {}).get("row_count", -1)) == 0 for row in am_c["results"])
    all_38_invalid = all(len(row.get("invalid_pairs", [])) == 38 for row in am_c["results"])
    if not zero_rows or not all_38_invalid:
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: AM-C validity history")

    reconstruction = read_json(ROOT / SOURCE["v837ak_reconstruction"])
    rows = reconstruction.get("rows", [])
    competent = sum(bool(r.get("competent")) for r in rows)
    if len(rows) != 50 or competent != 40 or len(rows) - competent != 10:
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: V837ak population")
    return {
        "decision": decision,
        "results": results,
        "am_a_invalid_count": invalid_count,
        "am_c_zero_rows": zero_rows,
        "am_c_all_38_invalid": all_38_invalid,
        "reconstruction": reconstruction,
        "population_rows": rows,
    }


def freeze_gate() -> dict:
    if GATE.is_file():
        return read_json(GATE)
    if head_sha() != START_SHA:
        raise RuntimeError(f"V837AN_GATE_MUST_FREEZE_AT_START_SHA: {head_sha()}")
    source = _derive_source_state()
    checkpoint_hashes = {r["organism_id"]: sha256_file(ROOT / r["checkpoint"]) for r in source["population_rows"]}
    gate = {
        "version": "V837an",
        "start_sha": START_SHA,
        "source_hashes": {key: git_blob_sha256(rel) for key, rel in SOURCE.items()},
        "v837am_decision_blob_hash": git_blob_sha256(SOURCE["v837am_decision"]),
        "v837am_results_blob_hash": git_blob_sha256(SOURCE["v837am_results"]),
        "v837ak_reconstruction_blob_hash": git_blob_sha256(SOURCE["v837ak_reconstruction"]),
        "source_checkpoint_hashes": checkpoint_hashes,
        "source_organisms_expected": 50,
        "competent_organisms_expected": 40,
        "incompetent_organisms_expected": 10,
        "historical_refinements": {"am_a_invalid_state_configs": 105, "am_c_zero_valid_rows": 7, "am_c_invalid_pairs_per_config": 38, "diagnosis_preserved": True},
        "data_partitions": PARTITIONS,
        "primary_semantic_probes": PRIMARY_PROBES,
        "secondary_semantic_probes": SECONDARY_PROBES,
        "counterfactual_definitions": COUNTERFACTUALS,
        "pair_eligibility_rule": "base solved AND counterfactual solved AND same length AND finite predictions AND target displacement >= historical success tolerance",
        "minimum_pair_counts": MIN_ELIGIBLE,
        "carrier_list": CARRIERS,
        "diagnostic_carriers": DIAGNOSTIC_CARRIERS,
        "latent_dimensions": LATENT_DIMS,
        "svd_rule": "deterministic SVD of paired counterfactual difference matrix; Q_k=V[0:k]^T; no invertibility requirement",
        "ridge_lambda": 1e-6,
        "semantic_compiler_zero_intercept": True,
        "random_subspace_controls": 32,
        "ood_threshold": 2.0,
        "source_swap_thresholds": SOURCE_SWAP_THRESHOLDS,
        "semantic_compiler_thresholds": SOURCE_SWAP_THRESHOLDS,
        "family_support_thresholds": FAMILY_SUPPORT,
        "routing_channels": ["all actual message edges", "10 global-source-cell contributions", "global gate"],
        "routing_prefix_sizes": ROUTING_PREFIXES,
        "routing_random_controls": 64,
        "routing_thresholds": ROUTING_THRESHOLDS,
        "cell_coalitions": {"nonempty_subsets": 1023, "cells": 10},
        "synergy_thresholds": SYNERGY_THRESHOLDS,
        "meta_confirm": {"partition": PARTITIONS["AN_META_CONFIRM"], "p_max": 0.05, "no_refit": True},
        "final_dev": {"partition": PARTITIONS["AN_FINAL_DEV"], "eligible_pairs_min": 24, **ROUTING_THRESHOLDS, "median_ood_ratio_max": 2.0},
        "validation_lock": "raw/frozen_family_abstractions.json must be written and hashed before any 20000-20127 evaluation; no second-best retry",
        "final_validation": {"partition": PARTITIONS["AN_FINAL_VALIDATION"], "eligible_pairs_min": 32, **ROUTING_THRESHOLDS, "median_ood_ratio_max": 2.0},
        "fresh_audit_consumed": False,
        "primitive_archive_allowed": False,
        "primitives_promoted": 0,
        "new_model_fits": 0,
        "optimizer_steps": 0,
        "adapter_gradient_steps": 0,
        "v838_started": False,
        "failure_ledger_required": True,
        "cross_organism_state_invertibility_required": False,
    }
    gate["gate_sha256"] = sha256_json(gate)
    write_json(GATE, gate)
    return gate


def assert_authorized() -> dict:
    gate = read_json(GATE) if GATE.is_file() else freeze_gate()
    source = _derive_source_state()
    if gate.get("start_sha") != START_SHA or gate.get("cross_organism_state_invertibility_required") is not False:
        raise RuntimeError("V837AN_SOURCE_INTEGRITY_FAILURE: gate")
    for key, rel in SOURCE.items():
        if git_blob_sha256(rel) != gate["source_hashes"][key]:
            raise RuntimeError(f"V837AN_SOURCE_INTEGRITY_FAILURE: {key} blob drift")
    for row in source["population_rows"]:
        if sha256_file(ROOT / row["checkpoint"]) != gate["source_checkpoint_hashes"][row["organism_id"]]:
            raise RuntimeError(f"V837AN_SOURCE_INTEGRITY_FAILURE: checkpoint {row['organism_id']}")
    payload = {
        "version": "V837an",
        "authorized": True,
        "start_sha": START_SHA,
        "source_organisms": 50,
        "competent_organisms": 40,
        "incompetent_organisms": 10,
        "historical_am_a_invalid_state_configs": 105,
        "historical_am_c_zero_valid_configs": 7,
        "cross_organism_state_invertibility_required": False,
        "new_model_fits": 0,
        "optimizer_steps": 0,
        "adapter_gradient_steps": 0,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics/authorization.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(assert_authorized(), indent=2))
