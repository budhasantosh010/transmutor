from __future__ import annotations

import json, subprocess
from pathlib import Path

from .utils import HERE, ROOT, read_json, sha256_file, sha256_json, write_json

START_SHA="802478151a5ab0d4cbbe099e3c18f056e8ce65bd"
AK=ROOT/"experiments/v837_primitive_invention/v837ak"
GATE=HERE/"frozen_interface_alignment_gate.json"

SOURCE_FILES={
 "decision":AK/"diagnostics/decision_state.json",
 "results":AK/"results.json",
 "confirmed":AK/"raw/confirmed_candidate_classes.json",
 "causal":AK/"raw/causal_results.json",
 "boundary":AK/"raw/boundary_substitution_results.json",
 "reconstruction":AK/"raw/reconstruction_results.json",
}


def _derive_source():
    d=read_json(SOURCE_FILES["decision"]); r=read_json(SOURCE_FILES["results"])
    if d.get("diagnosis")!="CONTEXT_BOUND_COMPUTATIONAL_MOTIFS" or r.get("diagnosis")!="CONTEXT_BOUND_COMPUTATIONAL_MOTIFS": raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: diagnosis")
    if d.get("next_program")!="V837al_PRIMITIVE_INTERFACE_ALIGNMENT" or d.get("fresh_audit_consumed") is not False or d.get("v838_started") is not False: raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: downstream locks")
    conf=read_json(SOURCE_FILES["confirmed"]); confirmed=[c for c in conf["classes"] if c.get("confirmed")]
    causal=read_json(SOURCE_FILES["causal"]); causal_classes=[c for c in causal["classes"] if c.get("causal_specificity_pass")]
    boundary=read_json(SOURCE_FILES["boundary"])
    if len(confirmed)!=6 or len(causal_classes)!=1 or int(boundary.get("boundary_interchangeable_count",-1))!=0: raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: frozen class counts")
    rec=read_json(SOURCE_FILES["reconstruction"])
    if rec.get("complete") is not True or rec.get("organisms_reconstructed")!=50: raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: checkpoints")
    return d, confirmed, causal_classes[0], rec


def freeze_gate():
    d,confirmed,causal,rec=_derive_source()
    if GATE.exists(): return read_json(GATE)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if head!=START_SHA: raise RuntimeError(f"V837AL_SOURCE_INTEGRITY_FAILURE: gate must freeze at {START_SHA}, got {head}")
    checkpoint_hashes={}
    for row in rec["rows"]:
        p=ROOT/row["checkpoint"]
        if not p.is_file(): raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: missing checkpoint")
        checkpoint_hashes[row["organism_id"]]=sha256_file(p)
    gate={
      "version":"V837al","starting_sha":START_SHA,
      "source_hashes":{k:sha256_file(v) for k,v in SOURCE_FILES.items()},
      "v837ak_decision_sha256":sha256_file(SOURCE_FILES["decision"]),
      "confirmed_class_ids":[c["class_id"] for c in confirmed],
      "primary_causal_class_id":causal["class_id"],
      "checkpoint_hashes":checkpoint_hashes,
      "align_fit_seeds":[10000,10063],"align_select_seeds":[10064,10127],"align_test_seeds":[20000,20127],
      "pairing_rule":{"recipient":"one compatible occurrence per organism; min class distance then occurrence ID","donor":"same family; different organism; opposite engine preferred; min class distance; organism ID","different_control":"different frozen class; same size/family/other organism; fallback frozen census"},
      "strong_power_rule":{"min_independent_recipients":7,"p_min_formula":"2^-N","p_max":0.01},
      "ports":["state","external_messages","global_term","projected_input","gate","output"],
      "transform_families":["SIGNED_PERMUTATION","DIAGONAL_AFFINE","RIGID_AFFINE","FULL_AFFINE"],
      "config_count":253,"ridge_lambda":1e-6,"state_condition_number_cap":1000.0,
      "selection_thresholds":{"output_ratio":0.75,"state_ratio":0.85,"beat_both_fraction":0.60},
      "test_thresholds":{"output_ratio":0.75,"state_ratio":0.85,"beat_both_fraction":0.60,"p_max":0.01},
      "closed_loop_thresholds":{"median_success_drop_max":0.05,"retain_success_threshold":0.85,"retain_fraction":0.60,"control_margin":0.10,"p_max":0.01},
      "canonicalization_required_for_archive":True,
      "fresh_audit_consumed":False,"archive_promotion":False,"primitives_promoted":0,"v838_started":False,
    }
    write_json(GATE,gate); return gate


def assert_authorized():
    gate=read_json(GATE) if GATE.exists() else freeze_gate()
    d,confirmed,causal,rec=_derive_source()
    if gate["starting_sha"]!=START_SHA or gate["confirmed_class_ids"]!=[c["class_id"] for c in confirmed] or gate["primary_causal_class_id"]!=causal["class_id"]: raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: gate drift")
    for k,p in SOURCE_FILES.items():
        if sha256_file(p)!=gate["source_hashes"][k]: raise RuntimeError(f"V837AL_SOURCE_INTEGRITY_FAILURE: {k} drift")
    for row in rec["rows"]:
        if sha256_file(ROOT/row["checkpoint"])!=gate["checkpoint_hashes"][row["organism_id"]]: raise RuntimeError("V837AL_SOURCE_INTEGRITY_FAILURE: checkpoint drift")
    payload={"version":"V837al","authorized":True,"starting_sha":START_SHA,"confirmed_classes":6,"causal_classes":1,"primary_causal_class_id":gate["primary_causal_class_id"],"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False}
    write_json(HERE/"diagnostics/authorization.json",payload); return payload


def main()->int:
    g=freeze_gate(); a=assert_authorized(); print(json.dumps({"authorized":True,"confirmed":len(g["confirmed_class_ids"]),"causal":g["primary_causal_class_id"]},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
