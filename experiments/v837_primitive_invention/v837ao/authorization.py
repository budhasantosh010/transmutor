from __future__ import annotations

import json

from .canonical_ir import all_ir_dicts
from .episode_partitions import PARTITIONS
from .source_contracts import POWERED_FAMILIES, SOURCE_PATHS, validate_v837an_source
from .utils import HERE, ROOT, START_SHA, git_blob_sha256, head_sha, read_json, sha256_file, sha256_json, write_json

GATE = HERE / "frozen_latent_canonicalization_gate.json"


def freeze_gate() -> dict:
    if GATE.is_file():
        return read_json(GATE)
    if head_sha() != START_SHA:
        raise RuntimeError(f"AO_GATE_MUST_FREEZE_AT_START_SHA:{head_sha()}")
    src = validate_v837an_source()
    checkpoints = {r["organism_id"]: sha256_file(ROOT / r["checkpoint"]) for r in src["population"]["rows"]}
    gate = {
        "version": "V837ao",
        "start_sha": START_SHA,
        "v837an_decision_state_hash": git_blob_sha256(SOURCE_PATHS["decision"]),
        "v837an_frozen_abstraction_hash": git_blob_sha256(SOURCE_PATHS["frozen_abstractions"]),
        "v837an_semantic_compiler_implementation_hash": git_blob_sha256(SOURCE_PATHS["semantic_compiler_impl"]),
        "v837an_source_population_hash": git_blob_sha256(SOURCE_PATHS["source_population"]),
        "source_checkpoint_hashes": checkpoints,
        "powered_families": list(POWERED_FAMILIES),
        "partial_observation_status": "UNRESOLVED_UNDERPOWERED",
        "organism_split_algorithm": "per powered family and V837aj engine: lexicographic competent organism IDs; last 2 holdout; remainder discovery; never performance-sort",
        "exact_holdout_count_rule": "2 per engine when available",
        "episode_partitions": PARTITIONS,
        "backend_variants": {
            "B0_GLOBAL_K1": "one q, one writer, one affine reader for all semantic-active phases",
            "B1_PHASE_GAUGE_K1": "same q; phase-specific affine reader scale/offset and writer gauge only",
            "B2_PHASE_K1": "phase-specific q, writer and reader at predeclared phases",
        },
        "backend_complexity_order": ["B0_GLOBAL_K1", "B1_PHASE_GAUGE_K1", "B2_PHASE_K1"],
        "reader_equation": "E(s)=a*(q^T s)+b",
        "full40_reader": "diagnostic-only affine reader",
        "ridge_lambda": 1e-6,
        "writer_raw": "w_raw=q*c where c is frozen zero-intercept semantic compiler coefficient",
        "writer_gauge_fix": "g=a*q; gamma=g^T w_raw; w=w_raw/gamma; g^T w=1",
        "set_operation": "SET(s,z*)=s+w*(z*-E(s))",
        "binary_target_grids": {"conditional_routing": [-1.0, 1.0], "delayed_recall": [-1.0, 1.0]},
        "continuous_target_grids": "pooled discovery AO_BACKEND_FIT Q10,Q30,Q50,Q70,Q90; semantic range P05/P95",
        "absolute_set_gate": {"median_recovery_min": 0.70, "direction_min": 0.80, "task_success_min": 0.75, "median_ood_ratio_max": 2.0, "target_fraction_recovery_060_min": 0.80},
        "random_direction_controls": 32,
        "random_control_margin_min": 0.20,
        "phase_definitions": {
            "conditional_routing": ["POST_CONTROL", "POST_PAYLOAD_A", "POST_PAYLOAD_B"],
            "delayed_recall": ["POST_WRITE", "MID_DELAY", "PRE_READ"],
            "iterative_state": ["EARLY", "MIDDLE", "LATE"],
            "variable_composition": ["EARLY", "MIDDLE", "LATE"],
        },
        "quotient_projector": "P=w*g^T with g^T w=1; residual=(I-P)s",
        "residual_donor_rule": "same organism/family/frozen phase; |za-zb|<=0.25Rz; choose maximum residual distance using AO_QUOTIENT_FIT only",
        "quotient_thresholds": {"median_RS_max": 0.20, "p90_RS_max": 0.40, "trajectory_disagreement_median_max": 0.10, "task_success_disagreement_max": 0.10, "median_ood_ratio_max": 2.0, "epsilon": 1e-8},
        "canonical_ir": all_ir_dicts(),
        "natural_commutativity": {"normalized_rmse_max": 0.10, "median_abs_normalized_error_max": 0.075, "direction_consistency_min": 0.90},
        "interventional_commutativity": {"one_step_normalized_error_max": 0.10, "multi_step_nrmse_max": 0.15, "final_recovery_min": 0.80, "task_success_min": 0.75, "random_margin_min": 0.20, "ood_max": 2.0},
        "rollout_horizons": [1, 2, 4, 8],
        "meta_gate": ">=60% discovery organisms pass full canonicalization, both engines, no refit, no variant fallback",
        "heldout_calibration_ladder": [1, 2, 4, 8, 16, 32, 64],
        "heldout_family_pass": ">=ceil(0.75*N_holdout), at least one pass per V837aj engine, exact discovery-selected variant",
        "cross_organism_agreement": {"median_pairwise_disagreement_max": 0.10, "p90_max": 0.25, "microstate_comparison": False, "q_alignment": False},
        "historical_validation_status": "REUSED_HISTORICAL_VALIDATION_DESCRIPTIVE_POST_FREEZE_ONLY",
        "fresh_audit_consumed": False,
        "primitive_archive_population": False,
        "primitives_promoted": 0,
        "new_model_fits": 0,
        "model_optimizer_steps": 0,
        "backend_gradient_steps": 0,
        "large_persistent_storage_tested": False,
        "v838_started": False,
        "failure_memory_required": True,
        "no_cross_organism_state_alignment": True,
        "no_cross_organism_q_alignment": True,
    }
    gate["gate_sha256"] = sha256_json(gate)
    write_json(GATE, gate)
    return gate


def assert_authorized() -> dict:
    gate = read_json(GATE) if GATE.is_file() else freeze_gate()
    src = validate_v837an_source()
    if gate.get("start_sha") != START_SHA:
        raise RuntimeError("AO_SOURCE_INTEGRITY_FAILURE:start SHA")
    for key in ("decision", "frozen_abstractions", "semantic_compiler_impl", "source_population"):
        expected = gate[{"decision":"v837an_decision_state_hash","frozen_abstractions":"v837an_frozen_abstraction_hash","semantic_compiler_impl":"v837an_semantic_compiler_implementation_hash","source_population":"v837an_source_population_hash"}[key]]
        if git_blob_sha256(SOURCE_PATHS[key]) != expected:
            raise RuntimeError(f"AO_SOURCE_INTEGRITY_FAILURE:V837an blob drift:{key}")
    for row in src["population"]["rows"]:
        if sha256_file(ROOT / row["checkpoint"]) != gate["source_checkpoint_hashes"][row["organism_id"]]:
            raise RuntimeError(f"AO_SOURCE_INTEGRITY_FAILURE:checkpoint:{row['organism_id']}")
    payload = {
        "version": "V837ao", "authorized": True, "start_sha": START_SHA,
        "v837an_diagnosis": src["decision"]["diagnosis"], "powered_families": 4,
        "causal_subspace_families": src["decision"]["causal_subspace_families"],
        "semantic_compiler_families": src["decision"]["semantic_compiler_families"],
        "validated_abstractions": src["decision"]["validated_family_abstractions"],
        "phase_conditional_representation": src["decision"]["phase_conditional_representation"],
        "partial_observation_status": "UNRESOLVED_UNDERPOWERED",
        "primitive_archive_allowed_at_start": False, "fresh_audit_consumed": False,
        "primitives_promoted": 0, "new_model_fits": 0, "model_optimizer_steps": 0,
        "backend_gradient_steps": 0, "v838_started": False,
        "no_cross_organism_state_alignment": True, "no_cross_organism_q_alignment": True,
    }
    write_json(HERE / "diagnostics/authorization.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(assert_authorized(), indent=2))
