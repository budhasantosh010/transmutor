from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/"experiments/v837_primitive_invention/v837ap"
START="c47649227baa0800d311ff41128d8f169ddf4f0d"
FAMILIES=("conditional_routing","delayed_recall","iterative_state","variable_composition")
PLOTS={
 "canonicalization_ladder.png","v837ao_vs_v837ap_setpoint_recovery.png","global_reader_failure_map.png","nonlinear_k1_chart_family_comparison.png","binary_class_manifold_geometry.png","k2_k4_k8_reader_curve.png","chart_complexity_vs_validation.png","writer_field_conditioning.png","gradient_newton_convergence.png","random_subspace_control_distribution.png","shuffled_semantic_control.png","set_magnitude_generalization.png","projected_dimension_vs_set_recovery.png","quotient_residual_sensitivity.png","natural_commutativity.png","interventional_commutativity.png","phase_atlas_transitions.png","calibration_frontier.png","cross_organism_semantic_agreement.png","historical_robustness.png","law_audit_comparison.png","failure_map_v837ao_to_v837ap.png"
}

def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))
def req(x,msg):
    if not x:raise AssertionError(msg)

def validate():
    required=[
      "RESEARCH_SPEC.md","FAILURE_ANALYSIS.md","frozen_global_or_nonlinear_causal_state_gate.json",
      "raw/source_state.json","raw/frozen_source_folds.json","raw/data_role_lock.json","raw/v837ao_baseline_reproduction.json",
      "raw/projected_subspace_hashes.json","raw/discovery_family_geometry_winners.json","raw/model_complexity_adjudication.json",
      "raw/meta_confirmation.json","raw/discovery_final.json","raw/frozen_v837ap_family_geometries.json",
      "raw/heldout_backend_results.json","raw/heldout_calibration_frontier.json","raw/cross_organism_agreement.json",
      "raw/historical_validation_robustness.json","raw/law_recovery.json","raw/failure_ledger.json",
      "diagnostics/decision_state.json","diagnostics/resource_accounting.json","results.json"
    ]
    for rel in required:req((HERE/rel).is_file(),f"missing {rel}")
    gate=j("frozen_global_or_nonlinear_causal_state_gate.json")
    req(gate["start_sha"]==START,"start sha")
    req(gate["fresh_audit_consumed"] is False and gate["primitives_promoted"]==0 and gate["v838_started"] is False,"science locks")
    req(gate["heldout_calibration_ladder"]==[2,4,8,16,32,64],"calibration ladder")
    source=j("diagnostics/source_integrity.json")
    req(source["pass"] is True and source["v837ao_diagnosis"]=="CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE","source")
    req(source["v837an_causal_steering_preserved"] is True,"V837an preservation")
    anchor=j("raw/v837ao_baseline_reproduction.json")
    req(anchor["metric_tolerance"]==1e-9 and all(r["max_abs_metric_diff"]<=1e-9 for r in anchor["rows"]),"V837ao anchor")
    folds=j("raw/frozen_source_folds.json")
    req(folds["same_as_v837ao"] is True and folds["performance_sorting"] is False,"folds")
    for f in FAMILIES:req(len(folds["families"][f]["holdout"])==4 and not(set(folds["families"][f]["holdout"])&set(folds["families"][f]["discovery"])),f"fold {f}")
    nest=j("diagnostics/subspace_nesting.json");req(nest["pass"] is True,"nested subspaces")
    selection=j("raw/discovery_family_geometry_winners.json");req(set(selection["family_winners"])==set(FAMILIES),"family winner schema")
    meta=j("raw/meta_confirmation.json");req(meta["no_refit"] is True and meta["no_geometry_fallback"] is True,"meta")
    df=j("raw/discovery_final.json");req(df["no_refit"] is True and df["no_fallback"] is True,"discovery final")
    frozen=j("raw/frozen_v837ap_family_geometries.json");req(frozen["frozen_before_heldout_open"] is True and frozen["heldout_organism_results_read_before_freeze"] is False,"freeze")
    held=j("raw/heldout_calibration_frontier.json");req(held["calibration_ladder"]==[2,4,8,16,32,64],"calibration")
    req(held["geometry_search_on_holdout"] is False and held["degree_search_on_holdout"] is False and held["k_search_on_holdout"] is False,"holdout search")
    req(held["evaluation_label"]=="HELDOUT_ORGANISM / REUSED_HISTORICAL_EPISODE_VALIDATION" and held["fresh_audit_consumed"] is False,"heldout labeling")
    agree=j("raw/cross_organism_agreement.json")
    req(agree["microstate_comparison_used"] is False and agree["q_alignment_used"] is False and agree["state_alignment_used"] is False,"agreement")
    robust=j("raw/historical_validation_robustness.json");req(robust["descriptive_post_freeze_only"] is True and robust["fresh_audit_consumed"] is False,"robustness")
    law=j("raw/law_recovery.json");req(law["cannot_rescue_family"] is True,"law diagnostic")
    ledger=j("raw/failure_ledger.json");req(ledger==j("diagnostics/failure_ledger.json"),"failure mirrors");req(all(e["failure_type"] in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"} for e in ledger["entries"]),"failure typing")
    decision=j("diagnostics/decision_state.json")
    req(decision["primitive_archive_allowed_next"] is False and decision["primitives_promoted"]==0,"archive")
    req(decision["fresh_audit_consumed"] is False and decision["v838_started"] is False,"audit")
    req(decision["new_source_model_fits"]==0 and decision["source_optimizer_steps"]==0,"training")
    req({p.name for p in (HERE/"plots").glob("*.png")}==PLOTS,"plots")
    req((ROOT/"docs/V837_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE_REPORT.md").is_file(),"report")
    protected=subprocess.check_output(["git","diff","--name-only",START,"--","experiments/v837_primitive_invention/v837ao","experiments/v837_primitive_invention/v837an","experiments/v837_primitive_invention/v837am","experiments/v837_primitive_invention/v837al","experiments/v837_primitive_invention/v837ak"],cwd=ROOT,text=True).strip()
    req(protected=="","protected history")
    req(not (ROOT/"experiments/v837_primitive_invention/v838").exists(),"V838")
    print("V837ap global coordinate or nonlinear causal state validation: PASS")

if __name__=="__main__":validate()
