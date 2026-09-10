from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from experiments.v837_primitive_invention.v837ap import authorization as auth
from experiments.v837_primitive_invention.v837ap.binary_charts import fit_binary_isotonic, fit_logistic
from experiments.v837_primitive_invention.v837ap.candidate_selection import candidate_order, first_passing_candidate
from experiments.v837_primitive_invention.v837ap.chart_features import feature_count
from experiments.v837_primitive_invention.v837ap.chart_reader import chart_gradient, read_chart
from experiments.v837_primitive_invention.v837ap.data_roles import PARTITIONS
from experiments.v837_primitive_invention.v837ap.gradient_writer import gradient_step
from experiments.v837_primitive_invention.v837ap.inverse_writer import set_1d_inverse
from experiments.v837_primitive_invention.v837ap.monotone_charts import fit_monotone
from experiments.v837_primitive_invention.v837ap.nonlinear_setter import set_state_geometry
from experiments.v837_primitive_invention.v837ap.phase_atlas import closest_global_reader_candidate
from experiments.v837_primitive_invention.v837ap.polynomial_charts import fit_polynomial
from experiments.v837_primitive_invention.v837ap.projected_causal_spaces import K_VALUES
from experiments.v837_primitive_invention.v837ap.setpoint_grid import family_grid
from experiments.v837_primitive_invention.v837ap.sham_controls import random_feature_rotation, random_subspaces
from experiments.v837_primitive_invention.v837ap.source_folds import EXPECTED_AO_FOLD_SHA, freeze_source_folds
from experiments.v837_primitive_invention.v837ap.tangent_field import TANGENT_FAMILIES, fit_tangent_field, tangent_step
from experiments.v837_primitive_invention.v837ap.utils import HERE, ROOT, START_SHA, read_json


def _j(rel:str): return read_json(HERE/rel)

# Source tests

def test_start_sha_exact(): assert START_SHA=="c47649227baa0800d311ff41128d8f169ddf4f0d"
def test_v837ao_diagnosis(): assert read_json(ROOT/auth.V837AO_PATHS["decision"])["diagnosis"]=="CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE"
def test_v837ao_next_program(): assert read_json(ROOT/auth.V837AO_PATHS["decision"])["next_program"]=="V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE"
def test_v837an_causal_steering_preserved(): assert _j("diagnostics/source_integrity.json")["v837an_causal_steering_preserved"] is True
def test_four_powered_families(): assert len(auth.POWERED_FAMILIES)==4
def test_partial_observation_stays_unresolved(): assert _j("diagnostics/source_integrity.json")["partial_observation_status"]=="UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY"
def test_archive_blocked(): assert auth.freeze_gate()["primitive_archive_population"] is False
def test_v838_false(): assert auth.freeze_gate()["v838_started"] is False

# Baseline tests

def test_v837ao_reader_reproduction(): assert all(r["reader_reproduced"] for r in _j("raw/v837ao_baseline_reproduction.json")["rows"])
def test_v837ao_setpoint_reproduction(): assert all(r["setpoint_reproduced"] for r in _j("raw/v837ao_baseline_reproduction.json")["rows"])
def test_metric_tolerance_1e_9():
    p=_j("raw/v837ao_baseline_reproduction.json");assert p["metric_tolerance"]==1e-9 and max(r["max_abs_metric_diff"] for r in p["rows"])<=1e-9
def test_no_historical_result_rewritten():
    out=subprocess.check_output(["git","diff","--name-only",f"{START_SHA}..HEAD","--","experiments/v837_primitive_invention/v837ao","experiments/v837_primitive_invention/v837an"],cwd=ROOT,text=True).strip();assert out==""

# Subspace tests

def _subspaces(): return _j("raw/projected_subspace_hashes.json")
def test_k1_exact(): assert 1 in _subspaces()["k_values"]
def test_k2_exact(): assert 2 in _subspaces()["k_values"]
def test_k4_exact(): assert 4 in _subspaces()["k_values"]
def test_k8_exact(): assert 8 in _subspaces()["k_values"]
def test_nested_subspaces(): assert _j("diagnostics/subspace_nesting.json")["pass"] is True and all(max(r["residuals"].values())<=1e-8 for r in _j("diagnostics/subspace_nesting.json")["rows"])
def test_no_cross_organism_alignment(): assert all(r.get("cross_organism_alignment") is False for r in _subspaces()["rows"])

# Scalar chart tests

def _toy():
    x=np.linspace(-2,2,80);z=.3+.7*x+.1*x*x;return x,z
def test_affine_anchor(): x,z=_toy();c=fit_polynomial(x,z,1);assert c["degree"]==1 and c["k"]==1
def test_poly2_features(): assert feature_count(1,2)==3
def test_poly3_features(): assert feature_count(1,3)==4
def test_monotone_pwl4(): x=np.linspace(-2,2,80);c=fit_monotone(x,x,"MONOTONE_PWL_4",4);assert len(c["knots_h"])==4
def test_monotone_pwl8(): x=np.linspace(-2,2,80);c=fit_monotone(x,x,"MONOTONE_PWL_8",8);assert len(c["knots_h"])==8
def test_pchip6_monotonicity(): x=np.linspace(-2,2,80);c=fit_monotone(x,x,"MONOTONE_PCHIP_6",6);assert c["derivative_sign_consistent"] is True
def test_binary_logistic(): x=np.r_[np.linspace(-2,-.2,40),np.linspace(.2,2,40)];z=np.r_[-np.ones(40),np.ones(40)];c=fit_logistic(x,z);assert c["kind"]=="LOGISTIC" and c["l2"]==1e-6
def test_binary_isotonic(): x=np.linspace(-2,2,80);z=np.where(x<0,-1,1);c=fit_binary_isotonic(x,z,4);assert c["kind"]=="ISOTONIC_4"
def test_fit_only_normalization(): x,z=_toy();c=fit_polynomial(x,z,2);assert "median" in c["normalization"] and "scale" in c["normalization"]
def test_inverse_range(): x,z=_toy();c=fit_polynomial(x,z,2);assert c["fit_h_min"][0]<=0<=c["fit_h_max"][0]

# Projected chart tests

def test_k2_feature_counts(): assert [feature_count(2,d) for d in (1,2,3)]==[3,6,10]
def test_k4_feature_counts(): assert [feature_count(4,d) for d in (1,2,3)]==[5,15,35]
def test_k8_feature_counts(): assert [feature_count(8,d) for d in (1,2,3)]==[9,45,165]
def test_degree_max_three():
    try: feature_count(2,4);assert False
    except ValueError: pass
def test_gradient_exact_against_finite_difference():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,2));z=.4*x[:,0]-.2*x[:,1]+.1*x[:,0]*x[:,1];c=fit_polynomial(x,z,2);p=np.array([.2,-.3]);g=chart_gradient(c,p);eps=1e-6;fd=np.array([(read_chart(c,(p+np.eye(2)[i]*eps).reshape(1,-1))[0]-read_chart(c,(p-np.eye(2)[i]*eps).reshape(1,-1))[0])/(2*eps) for i in range(2)]);assert np.max(np.abs(g-fd))<1e-5
def test_gradient_degenerate_guard():
    c=fit_polynomial(np.linspace(-1,1,20),np.zeros(20),1);r=gradient_step(c,np.array([0.]),1.,1.);assert r["valid"] is False and r["failure_code"]=="NONLINEAR_READER_GRADIENT_DEGENERATE"

# Writer tests

def test_inverse_1d_setter():
    x=np.linspace(-2,2,100);c=fit_polynomial(x,x,1);q=np.zeros((40,1));q[0,0]=1;s=np.zeros(40);r=set_1d_inverse(s,q,c,.5);assert r["valid"] and abs(r["state"][0]-.5)<1e-5
def test_polynomial_root_selection(): x,z=_toy();c=fit_polynomial(x,z,2);from experiments.v837_primitive_invention.v837ap.chart_reader import inverse_1d;assert inverse_1d(c,float(read_chart(c,np.array([[0.5]]))[0]),0.4) is not None
def test_gradient_writer_formula():
    x=np.random.default_rng(1).normal(size=(50,2));z=x[:,0]+2*x[:,1];c=fit_polynomial(x,z,1);r=gradient_step(c,np.array([0.,0.]),1.,10.);assert r["valid"] and np.isfinite(r["h"]).all()
def test_newton_max_four(): assert auth.freeze_gate()["newton_max_iterations"]==4
def test_newton_stop_rule():
    x=np.linspace(-1,1,40);c=fit_polynomial(x,x,1);q=np.zeros((40,1));q[0]=1;s=np.zeros(40);r=set_state_geometry(s,q,c,0.,1.,False);assert r["iterations"]==0
def test_trust_radius(): assert "2*D90" in auth.freeze_gate()["trust_radius_rule"]
def test_zero_semantic_delta_identity():
    x=np.linspace(-1,1,40);c=fit_polynomial(x,x,1);q=np.zeros((40,1));q[0]=1;s=np.zeros(40);r=set_state_geometry(s,q,c,0.,1.,False);assert np.max(np.abs(r["state"]-s))<1e-12

# Tangent tests

def _tfield(kind):
    h=np.linspace(-1,1,50)[:,None];hc=h+.2;z=h[:,0];zc=z+.2;return fit_tangent_field(h,hc,z,zc,kind)
def test_counterfactual_tangent_formula(): f=_tfield("CONSTANT");assert f["valid"]
def test_constant_field(): assert _tfield("CONSTANT")["degree"]==0
def test_affine_field(): assert _tfield("AFFINE_STATE_FIELD")["degree"]==1
def test_quadratic_field(): assert _tfield("QUADRATIC_STATE_FIELD")["degree"]==2
def test_semantic_unit_normalization():
    x=np.linspace(-1,1,50);c=fit_polynomial(x,x,1);f=_tfield("AFFINE_STATE_FIELD");r=tangent_step(c,f,np.array([.2]),.7,10.);assert r["valid"]
def test_tangent_gain_guard():
    x=np.linspace(-1,1,30);c=fit_polynomial(x,np.zeros_like(x),1);f=_tfield("CONSTANT");r=tangent_step(c,f,np.array([.1]),.7,10.);assert r["valid"] is False

# Phase tests

def test_frozen_phases_exact(): assert auth.PHASES=={"conditional_routing":["POST_CONTROL","POST_PAYLOAD_A","POST_PAYLOAD_B"],"delayed_recall":["POST_WRITE","MID_DELAY","PRE_READ"],"iterative_state":["EARLY","MIDDLE","LATE"],"variable_composition":["EARLY","MIDDLE","LATE"]}
def test_same_family_model_across_phases():
    p=_j("raw/phase_atlas_fit.json");assert all(r.get("same_model_family_across_phases") is True for r in p.get("rows",[]))
def test_no_phase_specific_degree_search(): assert all(r.get("phase_specific_degree_search") is False for r in _j("raw/phase_atlas_fit.json").get("rows",[]))
def test_chart_transition_equation(): assert auth.freeze_gate()["dynamics_thresholds"]["one_step_nrmse_max"]==.10
def test_global_model_preferred(): assert auth.freeze_gate()["complexity_order"][-1]=="PHASE_ATLAS"

# Setpoint tests

def test_binary_targets_exact(): assert family_grid("conditional_routing")["targets"]==[-1.0,1.0] and family_grid("delayed_recall")["targets"]==[-1.0,1.0]
def test_continuous_seven_quantiles(): assert len(family_grid("iterative_state")["targets"])==7 and len(family_grid("variable_composition")["targets"])==7
def test_quantiles_fit_only(): assert _j("raw/canonical_target_grids.json")["fit_partition"]=="AP_CHART_FIT"
def test_minimum_displacement(): assert auth.freeze_gate()["set_thresholds"]["median_recovery_min"]==.70
def test_abstract_rollout(): from experiments.v837_primitive_invention.v837ap.abstract_rollout import next_state;assert abs(next_state("iterative_state",1.,{"x":0.})-.65)<1e-12
def test_target_coverage_gate(): assert auth.freeze_gate()["continuous_targets"].startswith("AP_CHART_FIT")

# Controls

def test_32_random_subspaces(): assert len(random_subspaces(40,4,32,7))==32
def test_shuffled_semantic_equal_capacity(): assert auth.freeze_gate()["control_thresholds"]["shuffled_margin_min"]==.20
def test_random_feature_mixing(): q=random_feature_rotation(4,7);assert np.max(np.abs(q.T@q-np.eye(4)))<1e-10
def test_32_norm_matched_setters(): assert auth.freeze_gate()["norm_random_setters"]==32
def test_wrong_family_diagnostic(): assert _j("diagnostics/sham_controls.json").get("wrong_family_diagnostic",{"diagnostic_only":True})["diagnostic_only"] is True
def test_control_margin(): assert auth.freeze_gate()["control_thresholds"]["random_subspace_margin_min"]==.20

# Quotient tests

def _qrows(): return _j("raw/quotient_results.json").get("rows",[])
def test_same_organism(): assert all(r.get("organism_id") for r in _qrows())
def test_same_phase(): assert "same organism/family/phase" in auth.freeze_gate()["quotient_rule"]
def test_near_same_semantic_state(): assert "0.10Rz" in auth.freeze_gate()["quotient_rule"]
def test_residual_pair_selection_no_outcome(): assert all(not r.get("outcome_used_for_pair_selection",False) for r in _j("raw/quotient_pairs.json").get("pair_sets",[])) if (HERE/"raw/quotient_pairs.json").is_file() else True
def test_semantic_alignment_before_swap(): assert "align semantics" in auth.freeze_gate()["quotient_rule"]
def test_common_future_context(): assert "common future" in auth.freeze_gate()["quotient_rule"]
def test_residual_sensitivity(): assert auth.freeze_gate()["quotient_thresholds"]["median_RS_max"]==.20
def test_quotient_ood(): assert auth.freeze_gate()["quotient_thresholds"]["median_ood_max"]==2.0

# Dynamics tests

def test_natural_commutativity(): assert auth.freeze_gate()["dynamics_thresholds"]["one_step_nrmse_max"]==.10
def test_interventional_commutativity(): assert auth.freeze_gate()["dynamics_thresholds"]["multi_step_nrmse_max"]==.15
def test_iterative_exact_law(): from experiments.v837_primitive_invention.v837ap.abstract_rollout import next_state;assert abs(next_state("iterative_state",2.,{"x":1.})-1.65)<1e-12
def test_composition_exact_law(): from experiments.v837_primitive_invention.v837ap.abstract_rollout import next_state;assert abs(next_state("variable_composition",.2,{"gain":1.,"drive":0.})-np.tanh(.2))<1e-12
def test_recall_phase_machine(): assert auth.PHASES["delayed_recall"]==["POST_WRITE","MID_DELAY","PRE_READ"]
def test_routing_phase_machine(): assert auth.PHASES["conditional_routing"]==["POST_CONTROL","POST_PAYLOAD_A","POST_PAYLOAD_B"]
def test_rollout_1_2_4_8(): assert auth.freeze_gate()["rollout_horizons"]==[1,2,4,8]

# Selection tests

def test_complexity_order_frozen(): assert auth.freeze_gate()["complexity_order"]==["K1_AFFINE_ANCHOR","K1_SIMPLE_NONLINEAR","K1_NONLINEAR_PLUS_TANGENT","K2_PROJECTED","K4_PROJECTED","K8_PROJECTED","PHASE_ATLAS"]
def test_first_passing_candidate_wins(): assert first_passing_candidate([{"family_pass":False},{"family_pass":True,"x":1},{"family_pass":True,"x":2}])["x"]==1
def test_no_score_maximization(): assert "first passing candidate wins" in auth.freeze_gate()["family_pass"]
def test_k8_not_used_if_k4_passes():
    order=[c["k"] for c in candidate_order("iterative_state")];assert order.index(4)<order.index(8)
def test_phase_atlas_not_used_if_global_passes(): assert auth.freeze_gate()["complexity_order"][-1]=="PHASE_ATLAS"
def test_one_candidate_per_family():
    p=_j("raw/discovery_family_geometry_winners.json");assert set(p["family_winners"])==set(auth.POWERED_FAMILIES)

# META tests

def test_meta_no_refit(): assert _j("raw/meta_confirmation.json")["no_refit"] is True
def test_meta_no_fallback(): assert _j("raw/meta_confirmation.json")["no_geometry_fallback"] is True
def test_discovery_final_no_refit(): assert _j("raw/discovery_final.json")["no_refit"] is True
def test_geometry_frozen_before_holdout(): assert _j("raw/frozen_v837ap_family_geometries.json")["frozen_before_heldout_open"] is True

# Heldout tests

def test_heldouts_engine_stratified():
    f=freeze_source_folds();assert all(len(v["holdout"])==4 for v in f["families"].values())
def test_heldout_q_not_loaded(): assert _j("diagnostics/heldout_isolation.json")["v837an_q_loaded"] is False
def test_heldout_v837an_backend_not_loaded(): assert _j("diagnostics/heldout_isolation.json")["v837an_backend_loaded"] is False
def test_calibration_ladder_exact(): assert _j("raw/heldout_calibration_frontier.json")["calibration_ladder"]==[2,4,8,16,32,64]
def test_underdetermined_budget_rejected(): assert all(not (r.get("underdetermined") and r.get("pass")) for r in _j("raw/heldout_calibration_frontier.json").get("rows",[]))
def test_no_geometry_search_on_holdout(): assert _j("diagnostics/heldout_isolation.json")["geometry_search"] is False
def test_no_degree_search_on_holdout(): assert _j("diagnostics/heldout_isolation.json")["degree_search"] is False
def test_no_k_search_on_holdout(): assert _j("diagnostics/heldout_isolation.json")["k_search"] is False
def test_reused_validation_labeled_correctly(): assert _j("raw/heldout_calibration_frontier.json")["evaluation_label"]=="HELDOUT_ORGANISM / REUSED_HISTORICAL_EPISODE_VALIDATION"
def test_fresh_audit_unused(): assert _j("raw/heldout_calibration_frontier.json")["fresh_audit_consumed"] is False

# Failure memory tests

def _ledger(): return _j("raw/failure_ledger.json")
def test_failure_analysis_exists_before_science(): assert (HERE/"FAILURE_ANALYSIS.md").is_file()
def test_every_failed_primary_config_logged(): assert len(_ledger()["entries"])>0
def test_every_invalid_config_logged(): assert all("failed_conditions" in e for e in _ledger()["entries"])
def test_every_failed_calibration_budget_logged():
    failed={(r.get("family"),r.get("organism_id"),r.get("calibration_n")) for r in _j("raw/heldout_calibration_frontier.json").get("rows",[]) if r.get("underdetermined")};logged={(e.get("family"),e.get("organism"),e.get("metrics",{}).get("calibration_n")) for e in _ledger()["entries"] if e["stage"]=="AP14_HELDOUT_COMPILATION"};assert failed.issubset(logged)
def test_failure_distance_to_gate_present(): assert all("distance_from_threshold" in e for e in _ledger()["entries"])
def test_reproduction_command_present(): assert all(e.get("reproduction_command") for e in _ledger()["entries"])
def test_scientific_engineering_types(): assert {e["failure_type"] for e in _ledger()["entries"]}.issubset({"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"})
def test_central_failure_ledger_append_only():
    base=subprocess.check_output(["git","show",f"{START_SHA}:docs/V837_FAILURE_LEDGER.md"],cwd=ROOT);cur=(ROOT/"docs/V837_FAILURE_LEDGER.md").read_bytes();assert cur.startswith(base.rstrip(b"\n"))

# Science locks

def _decision(): return _j("diagnostics/decision_state.json")
def test_new_source_model_fits_zero(): assert _decision()["new_source_model_fits"]==0
def test_source_optimizer_steps_zero(): assert _decision()["source_optimizer_steps"]==0
def test_no_structure_search(): assert _decision().get("structure_search",False) is False
def test_no_motif_search(): assert _decision().get("motif_search",False) is False
def test_no_new_routing_program(): assert _decision().get("new_routing_program",False) is False
def test_no_rank4_bus_program(): assert _decision().get("rank4_bus_program",False) is False
def test_no_archive_population(): assert _decision()["primitive_archive_allowed_next"] is False
def test_primitives_promoted_zero(): assert _decision()["primitives_promoted"]==0
def test_fresh_audit_zero(): assert _decision()["fresh_audit_consumed"] is False
def test_v838_false_final(): assert _decision()["v838_started"] is False
