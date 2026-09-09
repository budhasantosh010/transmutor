from __future__ import annotations

import json
from pathlib import Path

from .utils import HERE, ROOT, START_SHA, git_blob_sha256, head_sha, read_json, sha256_file, sha256_json, write_json

AL=ROOT/"experiments/v837_primitive_invention/v837al"
AK=ROOT/"experiments/v837_primitive_invention/v837ak"
GATE=HERE/"frozen_context_or_redefinition_gate.json"
SOURCE={
 "v837al_decision":"experiments/v837_primitive_invention/v837al/diagnostics/decision_state.json",
 "v837al_results":"experiments/v837_primitive_invention/v837al/results.json",
 "v837al_pairs":"experiments/v837_primitive_invention/v837al/raw/frozen_pairs.json",
 "v837al_power":"experiments/v837_primitive_invention/v837al/diagnostics/statistical_power.json",
 "v837ak_causal":"experiments/v837_primitive_invention/v837ak/diagnostics/causal_specificity.json",
 "v837ak_results":"experiments/v837_primitive_invention/v837ak/results.json",
 "v837ak_reconstruction":"experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json",
}

CONTEXT_SETS={
 "C1_CONTROL":["TIME","GATE"],
 "C2_INPUT":["TIME","GATE","PROJECTED_INPUT"],
 "C3_LOCAL_STATE":["TIME","GATE","PREV_STATE"],
 "C4_COMMUNICATION":["TIME","GATE","EXTERNAL_MESSAGE","GLOBAL_TERM"],
 "C5_FULL_LOCAL_CONTEXT":["TIME","GATE","PREV_STATE","EXTERNAL_MESSAGE","GLOBAL_TERM","PROJECTED_INPUT"],
}
PORT_BUNDLES={
 "P1_OUTPUT":["output"],
 "P2_STATE_OUTPUT":["state","output"],
 "P3_COMMUNICATION":["state","external_messages","output"],
 "P4_GLOBAL_CONTROL":["state","global_term","gate","output"],
 "P5_INPUT_LOCAL":["state","projected_input","output"],
 "P6_ALL":["state","external_messages","global_term","projected_input","gate","output"],
}
AM_A_FAMILIES=["STATIC_FULL_AFFINE","CONTEXT_ADDITIVE","CONTEXT_MOD_R1","CONTEXT_MOD_R2","CONTEXT_MOD_R4"]
AM_B_BOUNDARIES=["MESSAGE+1","MESSAGE+2","MESSAGE+4","GLOBAL+1","GLOBAL+2","GLOBAL+4","COMBINED+1","COMBINED+2","COMBINED+4","WHOLE_SYSTEM"]
AM_C=["TEMPORAL_H2","TEMPORAL_H4","TEMPORAL_H8","INTERACTION_COMBINED+1","INTERACTION_COMBINED+2","INTERACTION_COMBINED+4","WHOLE_SYSTEM_INTERACTION"]
PARTITIONS={"AM_A_FIT":[10000,10063],"AM_A_SELECT":[10064,10127],"AM_B_FIT":[10128,10191],"AM_B_SELECT":[10192,10255],"AM_C_FIT":[10256,10319],"AM_C_SELECT":[10320,10383],"META_CONFIRM":[10384,10447],"FINAL_DEV_CONFIRM":[10448,10511],"FINAL_VALIDATION":[20000,20127],"FRESH_AUDIT":[90000,90499]}


def _derive():
    al=read_json(ROOT/SOURCE["v837al_decision"]); alr=read_json(ROOT/SOURCE["v837al_results"])
    if al.get("diagnosis")!="LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT" or alr.get("diagnosis")!="LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT": raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: V837al diagnosis")
    if int(al.get("configs_evaluated",-1))!=253 or al.get("global_selected_config") is not None or al.get("causal_selected_config") is not None: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: V837al selection state")
    if al.get("next_program")!="V837am_CONTEXT_CONDITIONED_INTERFACE_OR_PRIMITIVE_REDEFINITION" or al.get("fresh_audit_consumed") is not False or al.get("v838_started") is not False: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: V837al downstream locks")
    causal=read_json(ROOT/SOURCE["v837ak_causal"]); passed=[c for c in causal["classes"] if c.get("causal_specificity_pass")]
    if len(passed)!=1: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: causal class count")
    cid=passed[0]["class_id"]
    power=read_json(ROOT/SOURCE["v837al_power"])["primary_causal"]
    if power["class_id"]!=cid or int(power["independent_recipients"])!=38 or not power["strong_claim_powered"]: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: causal N=38")
    pairs=read_json(ROOT/SOURCE["v837al_pairs"]); cp=[p for p in pairs["pairs"] if p.get("primary_causal")]
    if len(cp)!=38 or any(p["class_id"]!=cid for p in cp): raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: frozen causal pairs")
    return al,cid,power,pairs


def freeze_gate()->dict:
    if GATE.is_file(): return read_json(GATE)
    if head_sha()!=START_SHA: raise RuntimeError(f"V837AM_GATE_MUST_FREEZE_AT_START_SHA: {head_sha()}")
    al,cid,power,pairs=_derive(); rec=read_json(ROOT/SOURCE["v837ak_reconstruction"])
    checkpoint_hashes={r["organism_id"]:sha256_file(ROOT/r["checkpoint"]) for r in rec["rows"]}
    gate={
      "version":"V837am","start_sha":START_SHA,"source_hashes":{k:git_blob_sha256(v) for k,v in SOURCE.items()},
      "v837al_decision_hash":git_blob_sha256(SOURCE["v837al_decision"]),"v837ak_causal_artifact_hash":git_blob_sha256(SOURCE["v837ak_causal"]),
      "primary_causal_class_derivation":"sole class with causal_specificity_pass=true in v837ak/diagnostics/causal_specificity.json",
      "primary_causal_class_id":cid,"causal_recipient_count":38,"minimum_possible_p":2.0**-38,
      "checkpoint_hashes":checkpoint_hashes,"v837al_frozen_pairs_sha256":pairs["frozen_pairs_sha256"],
      "data_partitions":PARTITIONS,"context_sets":CONTEXT_SETS,"port_bundles":PORT_BUNDLES,"am_a_transform_families":AM_A_FAMILIES,
      "am_a_config_count":126,"am_b_boundaries":AM_B_BOUNDARIES,"am_b_config_count":20,"am_c_definitions":AM_C,"am_c_config_count":7,
      "ridge_lambda":1e-6,"selection_thresholds":{"output_ratio":0.75,"state_ratio":0.85,"beat_both_fraction":0.60,"random_refit_ratio_min":1.20},
      "state_conditioning_limits":{"minimum_singular_value":0.05,"condition_number_max":1000.0},"meta_confirmation":{"p_max":0.05,"no_refit":True},
      "final_dev_thresholds":{"output_ratio":0.75,"state_ratio":0.85,"beat_both_fraction":0.60,"random_refit_ratio_min":1.20,"p_max":0.01},
      "final_selection_cost_rule":["new_primitive_macs+adapter_macs","continuous_parameter_count","additional_cells","history_length","branch_priority_AM-B_AM-A_AM-C"],
      "final_validation_lock":"selected_final_hypothesis.json must be written+hashed before 20000-20127 access; no second-best retry",
      "closed_loop_thresholds":{"median_success_drop_max":0.05,"retain_fraction":0.60,"retain_success_threshold":0.85,"control_margin":0.10,"p_max":0.01},
      "failure_document_requirements":{"failure_analysis_required":True,"machine_ledger_required":True,"central_append_only_ledger_required":True,"every_failed_config_recorded":True,"engineering_failures_recorded":True},
      "new_model_fits":0,"organism_optimizer_steps":0,"adapter_gradient_steps":0,"fresh_audit_consumed":False,"primitive_archive_allowed":False,"primitives_promoted":0,"v838_started":False,
    }
    gate["gate_sha256"]=sha256_json(gate); write_json(GATE,gate); return gate


def assert_authorized()->dict:
    gate=read_json(GATE) if GATE.is_file() else freeze_gate(); al,cid,power,pairs=_derive()
    if gate["primary_causal_class_id"]!=cid or gate["causal_recipient_count"]!=38: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: gate causal drift")
    for k,rel in SOURCE.items():
        if git_blob_sha256(rel)!=gate["source_hashes"][k]: raise RuntimeError(f"V837AM_SOURCE_INTEGRITY_FAILURE: {k} hash drift")
    rec=read_json(ROOT/SOURCE["v837ak_reconstruction"])
    for r in rec["rows"]:
        if sha256_file(ROOT/r["checkpoint"])!=gate["checkpoint_hashes"][r["organism_id"]]: raise RuntimeError("V837AM_SOURCE_INTEGRITY_FAILURE: checkpoint drift")
    payload={"version":"V837am","authorized":True,"start_sha":START_SHA,"primary_causal_class":cid,"causal_recipient_count":38,"minimum_possible_p":2.0**-38,"new_model_fits":0,"optimizer_steps":0,"adapter_gradient_steps":0,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False}
    write_json(HERE/"diagnostics/authorization.json",payload); return payload

if __name__=="__main__": print(json.dumps(assert_authorized(),indent=2))
