from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/"experiments/v837_primitive_invention/v837aq"
START="213f4f06ecaf4d5c767e06389ba5b6e6484a3122"
FAMILIES=("conditional_routing","delayed_recall","iterative_state","variable_composition")
PLOTS={"aq2_reality_nrmse.png","operator_pass_fraction.png","operator_response_nrmse.png","control_margin_by_family.png","cross_organism_disagreement.png","operator_order.png","compact_localization_pass_fraction.png","localization_support_size.png","composition_pass_fraction.png","predictive_hankel_rank.png","meta_operator_vs_program.png","heldout_operator_pass.png","historical_robustness.png","evidence_ladder.png"}

def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))
def req(x,msg):
    if not x:raise AssertionError(msg)

def validate():
    required=["RESEARCH_SPEC.md","FAILURE_ANALYSIS.md","PASS.md","config.json","frozen_program_level_causal_operator_gate.json","frozen_localization_policy.json","raw/source_state.json","raw/frozen_source_folds.json","raw/natural_intervention_library.json","raw/operator_reality_gate.json","raw/operator_discovery.json","raw/interaction_results.json","raw/localization_results.json","raw/composition_results.json","raw/predictive_state_diagnostic.json","raw/meta_confirmation.json","raw/frozen_program_operator_contracts.json","raw/heldout_confirmation.json","raw/historical_validation_robustness.json","raw/response_tensor_manifest.json","raw/failure_ledger.json","diagnostics/decision_state.json","diagnostics/resource_accounting.json","results.json","v837aq_resource_accounting.json"]
    for rel in required:req((HERE/rel).is_file(),f"missing {rel}")
    gate=j("frozen_program_level_causal_operator_gate.json");req(gate["start_sha"]==START,"start");req(gate["third_order_forbidden"] is True,"order ceiling");req(gate["primary_intervention_lock"].startswith("task/environment"),"natural interventions")
    req(gate["fresh_audit_consumed"] is False and gate["primitives_promoted"]==0 and gate["v838_started"] is False,"locks")
    source=j("raw/source_state.json");req(source["source_population"]=={"total":50,"competent":40,"incompetent":10},"population");req(source["v837ap_quotient_rows"]==source["v837ap_commutativity_rows"]==source["v837ap_heldout_rows"]==0,"V837ap forensic boundary")
    folds=j("raw/frozen_source_folds.json");req(folds["fold_sha256"]=="b96eed0f053ec48b5cb183929296a371857e549ed05bbe9650df2ef32dee1c95","fold hash")
    reality=j("raw/operator_reality_gate.json");req(reality["pass"] is True and reality["family_gate"]["passing"]==6 and reality["family_gate"]["both_engines"] is True,"reality")
    disc=j("raw/operator_discovery.json");req(disc["accepted_families"]==["conditional_routing","delayed_recall","iterative_state"],"operator families");req(disc["families"]["conditional_routing"]["operator_order"]==2,"routing order");req(disc["families"]["variable_composition"]["pass"] is False,"composition null")
    for f in disc["accepted_families"]:req(disc["families"][f]["cross_organism"]["pass"] is True,f"cross {f}")
    req(len(disc["incompetent_controls"])==3 and all(not r["pass"] for r in disc["incompetent_controls"]),"incompetent controls")
    loc=j("raw/localization_results.json");req(loc["hidden_state_set_used"] is False,"hidden set");req(loc["families"]["delayed_recall"]["compact_support_gate"]["passing"]==4,"recall localization");req(loc["families"]["conditional_routing"]["compact_support_gate"]["passing"]==0,"routing distributed");req(loc["families"]["iterative_state"]["compact_support_gate"]["passing"]==0,"iterative distributed")
    comp=j("raw/composition_results.json");req(comp["families"]["iterative_state"]["pass"] is True,"iterative composition");req(comp["families"]["conditional_routing"]["pass"] is False and comp["families"]["delayed_recall"]["pass"] is False,"composition boundary")
    psr=j("raw/predictive_state_diagnostic.json");req({f:psr["families"][f]["hankel"]["oracle_rank_99"] for f in psr["families"]}=={"conditional_routing":4,"delayed_recall":1,"iterative_state":3},"predictive ranks");req(psr["non_gating"] is True,"psr non gating")
    meta=j("raw/meta_confirmation.json");req(meta["confirmed_family_count"]==3 and meta["program_closed_families"]==["iterative_state"],"meta")
    frozen=j("raw/frozen_program_operator_contracts.json");req(frozen["heldout_results_read"] is False,"freeze before holdout");req(frozen["family_contracts"]["variable_composition"] is None,"frozen null")
    held=j("raw/heldout_confirmation.json");req(held["heldout_confirmed_family_count"]==3,"heldout operators");req(held["operator_refit"] is False and held["hyperparameter_search"] is False,"heldout no refit")
    hist=j("raw/historical_validation_robustness.json");req(hist["descriptive_only"] is True and hist["can_rescue_failed_discovery_or_meta"] is False,"history non rescue")
    result=j("results.json");req(result["diagnosis"]=="CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED","diagnosis");req(result["globally_compositionally_closed_families"]==["iterative_state"],"global program");req(result["coordinate_free_operator_pattern_established"] is True and result["broad_program_composition_established"] is False,"claim boundary");req(result["next_program"]=="V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR","next")
    req(result["fresh_audit_consumed"] is False and result["primitive_archive_allowed_next"] is False and result["primitives_promoted"]==0 and result["v838_started"] is False,"final locks")
    ledger=j("raw/failure_ledger.json");req(ledger==j("diagnostics/failure_ledger.json"),"ledger mirror");req(all(e["failure_type"] in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"} for e in ledger["entries"]),"typing")
    req({p.name for p in (HERE/"plots").glob("*.png")}==PLOTS,"plots");req((ROOT/"docs/V837_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION_REPORT.md").is_file(),"report")
    protected=subprocess.check_output(["git","diff","--name-only",START,"--","experiments/v837_primitive_invention/v837ap","experiments/v837_primitive_invention/v837ao","experiments/v837_primitive_invention/v837an","experiments/v837_primitive_invention/v837ak"],cwd=ROOT,text=True).strip();req(protected=="","protected history")
    req(not (ROOT/"experiments/v837_primitive_invention/v838").exists(),"V838")
    print("V837aq program-level causal operator localization validation: PASS")
    return 0

if __name__=="__main__":raise SystemExit(validate())
