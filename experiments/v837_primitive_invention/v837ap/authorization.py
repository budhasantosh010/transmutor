from __future__ import annotations

import json
from pathlib import Path

from .data_roles import PARTITIONS, assert_roles
from .source_folds import EXPECTED_AO_FOLD_SHA
from .utils import HERE, ROOT, START_SHA, git_blob_sha256, head_sha, read_json, sha256_json, write_json

GATE=HERE/"frozen_global_or_nonlinear_causal_state_gate.json"
POWERED_FAMILIES=("conditional_routing","delayed_recall","iterative_state","variable_composition")
PHASES={
 "conditional_routing":["POST_CONTROL","POST_PAYLOAD_A","POST_PAYLOAD_B"],
 "delayed_recall":["POST_WRITE","MID_DELAY","PRE_READ"],
 "iterative_state":["EARLY","MIDDLE","LATE"],
 "variable_composition":["EARLY","MIDDLE","LATE"],
}
V837AO_PATHS={
 "decision":"experiments/v837_primitive_invention/v837ao/diagnostics/decision_state.json",
 "results":"experiments/v837_primitive_invention/v837ao/results.json",
 "report":"docs/V837_LATENT_PRIMITIVE_CANONICALIZATION_REPORT.md",
 "folds":"experiments/v837_primitive_invention/v837ao/raw/frozen_organism_folds.json",
 "frozen_specs":"experiments/v837_primitive_invention/v837ao/raw/frozen_canonical_family_specs.json",
}
V837AN_FROZEN="experiments/v837_primitive_invention/v837an/raw/frozen_family_abstractions.json"

def _source_state()->dict:
    ao=read_json(ROOT/V837AO_PATHS["decision"]); an=read_json(ROOT/"experiments/v837_primitive_invention/v837an/diagnostics/decision_state.json")
    if ao.get("diagnosis")!="CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE": raise RuntimeError("V837AP_BAD_V837AO_DIAGNOSIS")
    if ao.get("next_program")!="V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE": raise RuntimeError("V837AP_BAD_V837AO_NEXT_PROGRAM")
    if an.get("diagnosis")!="GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN": raise RuntimeError("V837AP_V837AN_STEERING_SOURCE_INVALID")
    if ao.get("fresh_audit_consumed") is not False or ao.get("v838_started") is not False or ao.get("primitives_promoted")!=0: raise RuntimeError("V837AP_PREDECESSOR_LOCK_CHANGED")
    return {"v837ao":ao,"v837an":an}

def freeze_gate()->dict:
    if GATE.is_file(): return read_json(GATE)
    if head_sha()!=START_SHA: raise RuntimeError(f"V837AP_GATE_MUST_FREEZE_AT_START_SHA:{head_sha()}")
    _source_state(); assert_roles()
    gate={
      "version":"V837ap","start_sha":START_SHA,
      "v837ao_decision_hash":git_blob_sha256(V837AO_PATHS["decision"]),
      "v837ao_report_hash":git_blob_sha256(V837AO_PATHS["report"]),
      "v837ao_results_hash":git_blob_sha256(V837AO_PATHS["results"]),
      "v837ao_frozen_geometry_hash":git_blob_sha256(V837AO_PATHS["frozen_specs"]),
      "v837an_frozen_abstraction_hash":git_blob_sha256(V837AN_FROZEN),
      "v837ao_organism_fold_hash":EXPECTED_AO_FOLD_SHA,
      "powered_families":list(POWERED_FAMILIES),"partial_observation_status":"UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY",
      "episode_roles":PARTITIONS,
      "heldout_organism_protocol":"reuse V837ao engine-stratified folds; freeze geometry family before reading heldout chart metrics",
      "projected_dimensions":[1,2,4,8],"semantic_dimension":1,
      "continuous_k1_chart_families":["AFFINE","POLYNOMIAL_2","POLYNOMIAL_3","MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6"],
      "binary_k1_chart_families":["AFFINE","LOGISTIC_MONOTONE","ISOTONIC_4","ISOTONIC_8"],
      "projected_chart_families":["LINEAR","QUADRATIC","CUBIC"],"max_polynomial_degree":3,"ridge_lambda":1e-6,
      "newton_max_iterations":4,"gradient_epsilon":1e-8,"trust_radius_rule":"each projected Newton/tangent step clipped to 2*D90, D90=FIT-only P90 norm natural paired delta-h",
      "tangent_field_families":["CONSTANT","AFFINE_STATE_FIELD","QUADRATIC_STATE_FIELD"],
      "phase_definitions":PHASES,
      "binary_targets":{"conditional_routing":[-1.0,1.0],"delayed_recall":[-1.0,1.0]},
      "continuous_targets":"AP_CHART_FIT pooled Q10,Q25,Q40,Q50,Q60,Q75,Q90; Rz=Q95-Q05",
      "reader_thresholds":{"continuous_nrmse_max":.10,"continuous_abs_pearson_min":.95,"continuous_spearman_min":.95,"binary_accuracy_min":.95,"binary_balanced_accuracy_min":.95,"binary_nrmse_max":.15},
      "set_thresholds":{"continuous_read_after_set_nrmse_max":.05,"binary_read_after_set_nrmse_max":.10,"median_recovery_min":.70,"direction_min":.80,"task_success_min":.75,"trajectory_nrmse_max":.15,"median_ood_max":2.0},
      "control_thresholds":{"shuffled_margin_min":.20,"random_subspace_margin_min":.20,"norm_random_margin_min":.20,"paired_one_sided_permutation_p_max":.01},
      "random_subspaces":32,"norm_random_setters":32,
      "quotient_rule":"same organism/family/phase; |E(sa)-E(sb)|<=0.10Rz; choose max projected residual distance without outcome effects; align semantics before common future",
      "quotient_thresholds":{"median_RS_max":.20,"p90_RS_max":.40,"trajectory_disagreement_nrmse_max":.10,"final_prediction_disagreement_normalized_max":.10,"task_success_disagreement_max":.10,"median_ood_max":2.0,"epsilon":1e-8},
      "dynamics_thresholds":{"one_step_nrmse_max":.10,"multi_step_nrmse_max":.15,"final_recovery_min":.80,"task_success_min":.75,"control_margin_min":.20,"median_ood_max":2.0},
      "rollout_horizons":[1,2,4,8],
      "complexity_order":["K1_AFFINE_ANCHOR","K1_SIMPLE_NONLINEAR","K1_NONLINEAR_PLUS_TANGENT","K2_PROJECTED","K4_PROJECTED","K8_PROJECTED","PHASE_ATLAS"],
      "complexity_ceiling_k":8,"family_pass":">=60% discovery organisms and both V837aj engines; first passing candidate wins, never score-maximize",
      "meta_rule":"no refit; failure -> family null; no next-best fallback","discovery_final_rule":"10448-10511 no refit, identical gates",
      "heldout_calibration_ladder":[2,4,8,16,32,64],"heldout_min_samples_rule":"effective independent fit samples >=2*deployed continuous coefficients unless exact lower-dimensional closed form needs fewer",
      "heldout_family_pass":">=ceil(.75*N_holdout) and >=1 directed + >=1 random; same reader/SET/control/quotient/dynamics/OOD gates",
      "historical_validation_label":"HELDOUT_ORGANISM / REUSED_HISTORICAL_EPISODE_VALIDATION",
      "fresh_audit_consumed":False,"primitive_archive_population":False,"primitives_promoted":0,"v838_started":False,"failure_ledger_required":True,
      "new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0,
      "no_cross_organism_state_alignment":True,"no_cross_organism_q_alignment":True,"large_persistent_storage_tested":False,
    }
    gate["gate_sha256"]=sha256_json(gate); write_json(GATE,gate); return gate

def assert_authorized()->dict:
    gate=read_json(GATE) if GATE.is_file() else freeze_gate(); state=_source_state()
    if gate.get("start_sha")!=START_SHA: raise RuntimeError("V837AP_START_SHA_DRIFT")
    for key,path in [("v837ao_decision_hash",V837AO_PATHS["decision"]),("v837ao_report_hash",V837AO_PATHS["report"]),("v837ao_results_hash",V837AO_PATHS["results"]),("v837an_frozen_abstraction_hash",V837AN_FROZEN)]:
        if git_blob_sha256(path)!=gate[key]: raise RuntimeError(f"V837AP_PROTECTED_BLOB_DRIFT:{path}")
    payload={"version":"V837ap","authorized":True,"start_sha":START_SHA,"v837ao_diagnosis":state["v837ao"]["diagnosis"],"v837ao_next_program":state["v837ao"]["next_program"],"v837an_causal_steering_preserved":True,"powered_families":4,"partial_observation_status":"UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY","primitive_archive_allowed":False,"primitives_promoted":0,"fresh_audit_consumed":False,"v838_started":False,"new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0}
    write_json(HERE/"diagnostics/authorization.json",payload); return payload

if __name__=="__main__": print(json.dumps(assert_authorized(),indent=2))
