from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from experiments.v837_primitive_invention.v837aq.data_roles import PARTITIONS
from experiments.v837_primitive_invention.v837aq.natural_interventions import build_first_order_cases
from experiments.v837_primitive_invention.v837aq.operator_ir import program_ir
from experiments.v837_primitive_invention.v837aq.operator_metrics import family_gate, response_metrics
from experiments.v837_primitive_invention.v837aq.predictive_state import _rank_at_energy

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/"experiments/v837_primitive_invention/v837aq"
def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))

def test_start_sha():assert j("frozen_program_level_causal_operator_gate.json")["start_sha"]=="213f4f06ecaf4d5c767e06389ba5b6e6484a3122"
def test_required_branch():assert j("frozen_program_level_causal_operator_gate.json")["required_branch"]=="research/v837-program-level-causal-operator-localization"
def test_predecessor_diagnosis():assert j("raw/source_state.json")["authorization"]["v837ap_diagnosis"]=="DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED"
def test_predecessor_unreached_stages():
    s=j("raw/source_state.json");assert [s["v837ap_quotient_rows"],s["v837ap_commutativity_rows"],s["v837ap_heldout_rows"]]==[0,0,0]
def test_failure_types_preserved():
    s=j("raw/source_state.json");assert s["v837ap_engineering_failures_preserved"]==11 and s["v837ap_scientific_failures_preserved"]==297
def test_population():assert j("raw/source_state.json")["source_population"]=={"total":50,"competent":40,"incompetent":10}
def test_fold_hash():assert j("raw/frozen_source_folds.json")["fold_sha256"]=="b96eed0f053ec48b5cb183929296a371857e549ed05bbe9650df2ef32dee1c95"
@pytest.mark.parametrize("family,n_d,n_h",[("conditional_routing",5,4),("delayed_recall",6,4),("iterative_state",6,4),("variable_composition",4,4)])
def test_folds(family,n_d,n_h):
    f=j("raw/frozen_source_folds.json")["families"][family];assert len(f["discovery"])==n_d and len(f["holdout"])==n_h and not(set(f["discovery"])&set(f["holdout"]))
def test_roles_disjoint():
    names=[k for k in PARTITIONS if k not in {"REUSED_HISTORICAL_VALIDATION","FRESH_AUDIT"}];sets=[set(range(PARTITIONS[k][0],PARTITIONS[k][1]+1)) for k in names];assert all(not (sets[i]&sets[k]) for i in range(len(sets)) for k in range(i+1,len(sets)))
def test_fresh_audit_separate():assert tuple(PARTITIONS["FRESH_AUDIT"])==(90000,90499)
def test_natural_primary_lock():assert "no arbitrary hidden-state SET" in j("frozen_program_level_causal_operator_gate.json")["primary_intervention_lock"]
def test_third_order_forbidden():assert j("frozen_program_level_causal_operator_gate.json")["third_order_forbidden"] is True
@pytest.mark.parametrize("family",["conditional_routing","delayed_recall","iterative_state","variable_composition"])
def test_operator_ir_family(family):
    ir=program_ir(family);assert ir["name"] and ir["causal_response_law"] and ir["temporal_contract"] and ir["composition_rule"]
@pytest.mark.parametrize("seed",[11000,11001,11011,11031,11063])
def test_iterative_exact_kernel(seed):
    for c in build_first_order_cases("iterative_state",[seed],reality=True):
        if c.intervention=="INPUT_PERTURB":assert abs(c.oracle_response-0.35*(0.65**(c.horizon-1))*c.actual_delta)<1e-10
def test_reality_pass():
    r=j("raw/operator_reality_gate.json");assert r["pass"] and r["family_gate"]["passing"]==6 and r["family_gate"]["both_engines"]
@pytest.mark.parametrize("family,passing,order",[("conditional_routing",5,2),("delayed_recall",6,1),("iterative_state",6,1),("variable_composition",0,1)])
def test_discovery_family_outcome(family,passing,order):
    d=j("raw/operator_discovery.json")["families"][family];assert d["first_order_gate"]["passing"]==passing;assert d.get("operator_order")==order
@pytest.mark.parametrize("family",["conditional_routing","delayed_recall","iterative_state"])
def test_cross_organism_pass(family):assert j("raw/operator_discovery.json")["families"][family]["cross_organism"]["pass"]
def test_no_state_alignment():
    for f in ["conditional_routing","delayed_recall","iterative_state"]:
        c=j("raw/operator_discovery.json")["families"][f]["cross_organism"];assert not c["state_alignment_used"] and not c["q_alignment_used"]
def test_incompetent_controls():
    rows=j("raw/operator_discovery.json")["incompetent_controls"];assert len(rows)==3 and all(not r["pass"] for r in rows)
def test_routing_second_order_required():assert j("raw/interaction_results.json")["families"]["conditional_routing"]["second_order_required"]
@pytest.mark.parametrize("family",["delayed_recall","iterative_state"])
def test_no_second_order_needed(family):assert not j("raw/interaction_results.json")["families"][family]["second_order_required"]
@pytest.mark.parametrize("family,passing",[("conditional_routing",0),("delayed_recall",4),("iterative_state",0)])
def test_localization_discovery(family,passing):assert j("raw/localization_results.json")["families"][family]["compact_support_gate"]["passing"]==passing
def test_localization_no_hidden_set():assert j("raw/localization_results.json")["hidden_state_set_used"] is False
def test_recall_median_support_four():assert j("raw/localization_results.json")["families"]["delayed_recall"]["median_winner_units"]==4.0
@pytest.mark.parametrize("family,pass_expected",[("conditional_routing",False),("delayed_recall",False),("iterative_state",True)])
def test_discovery_composition(family,pass_expected):assert j("raw/composition_results.json")["families"][family]["pass"] is pass_expected
@pytest.mark.parametrize("family,rank",[("conditional_routing",4),("delayed_recall",1),("iterative_state",3)])
def test_predictive_rank(family,rank):assert j("raw/predictive_state_diagnostic.json")["families"][family]["hankel"]["oracle_rank_99"]==rank
def test_predictive_non_gating():assert j("raw/predictive_state_diagnostic.json")["non_gating"] is True
def test_rank_helper():
    rank,_,_=_rank_at_energy(np.eye(3),.99);assert rank==3
def test_meta_operator_three():assert j("raw/meta_confirmation.json")["confirmed_family_count"]==3
def test_meta_program_only_iterative():assert j("raw/meta_confirmation.json")["program_closed_families"]==["iterative_state"]
def test_recall_localization_does_not_meta_confirm():assert j("raw/meta_confirmation.json")["families"]["delayed_recall"]["compact_localization_meta_gate"]["passing"]==3
def test_freeze_precedes_heldout():assert j("raw/frozen_program_operator_contracts.json")["heldout_results_read"] is False
def test_frozen_composition_null():assert j("raw/frozen_program_operator_contracts.json")["family_contracts"]["variable_composition"] is None
@pytest.mark.parametrize("family",["conditional_routing","delayed_recall","iterative_state"])
def test_heldout_operator(family):
    h=j("raw/heldout_confirmation.json")["families"][family];assert h["operator_gate"]["passing"]==4 and h["operator_gate"]["pass"]
def test_heldout_no_refit():
    h=j("raw/heldout_confirmation.json");assert h["operator_refit"] is False and h["hyperparameter_search"] is False
def test_historical_nonrescue():
    h=j("raw/historical_validation_robustness.json");assert h["descriptive_only"] and not h["can_rescue_failed_discovery_or_meta"]
@pytest.mark.parametrize("family",["conditional_routing","delayed_recall","iterative_state"])
def test_historical_family_pass(family):assert j("raw/historical_validation_robustness.json")["families"][family]["pass"]
def test_response_tensor_total():assert sum(v["rows"] for v in j("raw/response_tensor_manifest.json")["families"].values())==15638
def test_metrics_perfect():
    m=response_metrics(np.array([1.,-1.]),np.array([1.,-1.]),np.ones(2));assert m["response_nrmse"]==0 and m["direction_agreement"]==1
def test_family_gate_both_engines():
    rows=[{"pass":True,"engine":"DIRECTED_STRUCTURAL_SEARCH"},{"pass":True,"engine":"RANDOM_STRUCTURAL_SAMPLER"},{"pass":True,"engine":"DIRECTED_STRUCTURAL_SEARCH"}];assert family_gate(rows)["pass"]
def test_final_diagnosis():assert j("results.json")["diagnosis"]=="CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED"
def test_final_operator_pattern():assert j("results.json")["coordinate_free_operator_pattern_established"] is True
def test_final_program_not_broad():assert j("results.json")["broad_program_composition_established"] is False
def test_global_program_iterative_only():assert j("results.json")["globally_compositionally_closed_families"]==["iterative_state"]
def test_next_program():assert j("results.json")["next_program"]=="V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR"
@pytest.mark.parametrize("key,value",[("fresh_audit_consumed",False),("primitive_archive_allowed_next",False),("primitives_promoted",0),("v838_started",False),("new_source_model_fits",0),("source_optimizer_steps",0),("hidden_state_set_primary_evidence",False),("cross_organism_state_alignment",False),("cross_organism_q_alignment",False)])
def test_final_locks(key,value):assert j("results.json")[key]==value
def test_failure_ledger_mirror():assert j("raw/failure_ledger.json")==j("diagnostics/failure_ledger.json")
def test_failure_types():assert all(e["failure_type"] in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"} for e in j("raw/failure_ledger.json")["entries"])
def test_predictive_engineering_failure_recorded():assert any(e["failure_id"]=="V837aq-ENG-PREDICTIVE-TEST-CANONICALIZATION" for e in j("raw/failure_ledger.json")["entries"])
def test_analyzer_engineering_failure_recorded():assert any(e["failure_id"]=="V837aq-ENG-AQ11-REALITY-ARTIFACT-PATH" for e in j("raw/failure_ledger.json")["entries"])
def test_plot_count():assert len(list((HERE/"plots").glob("*.png")))==14
def test_report_exists():assert (ROOT/"docs/V837_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION_REPORT.md").is_file()
def test_v838_absent():assert not (ROOT/"experiments/v837_primitive_invention/v838").exists()
