from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from experiments.v837_primitive_invention.v837ap.authorization import POWERED_FAMILIES, START_SHA, assert_authorized
from experiments.v837_primitive_invention.v837ap.canonical_ir import canonical_ir
from experiments.v837_primitive_invention.v837ap.candidate_selection import candidate_order
from experiments.v837_primitive_invention.v837ap.chart_reader import read_chart, chart_gradient, inverse_1d
from experiments.v837_primitive_invention.v837ap.data_roles import PARTITIONS, assert_roles
from experiments.v837_primitive_invention.v837ap.gradient_writer import gradient_step
from experiments.v837_primitive_invention.v837ap.monotone_charts import fit_monotone
from experiments.v837_primitive_invention.v837ap.nonlinear_setter import set_state_geometry
from experiments.v837_primitive_invention.v837ap.polynomial_charts import fit_polynomial
from experiments.v837_primitive_invention.v837ap.projected_charts import fit_projected_chart, expected_feature_count
from experiments.v837_primitive_invention.v837ap.scalar_charts import fit_scalar_chart
from experiments.v837_primitive_invention.v837ap.source_folds import freeze_source_folds
from experiments.v837_primitive_invention.v837ap.tangent_field import fit_tangent_field, tangent_vector
from experiments.v837_primitive_invention.v837ap.utils import HERE

ROOT=HERE.parents[2]
def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))

# Source contracts
def test_start_sha_exact():assert START_SHA=="c47649227baa0800d311ff41128d8f169ddf4f0d"
def test_authorized_predecessor():assert assert_authorized()["v837ao_diagnosis"]=="CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE"
def test_predecessor_next_program_exact():assert assert_authorized()["v837ao_next_program"]=="V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE"
def test_powered_families_exact():assert len(POWERED_FAMILIES)==4 and "partial_observation" not in POWERED_FAMILIES
def test_archive_blocked_at_start():assert assert_authorized()["primitive_archive_allowed"] is False
def test_v838_false_at_start():assert assert_authorized()["v838_started"] is False

# Organism fold discipline
def test_fold_reuse_hash():assert freeze_source_folds()["same_as_v837ao"] is True
def test_fold_discovery_holdout_disjoint():assert all(not(set(x["discovery"])&set(x["holdout"])) for x in freeze_source_folds()["families"].values())
def test_fold_exact_counts():
    f=freeze_source_folds()["families"];assert (len(f["conditional_routing"]["discovery"]),len(f["conditional_routing"]["holdout"]))==(5,4);assert (len(f["delayed_recall"]["discovery"]),len(f["delayed_recall"]["holdout"]))==(6,4);assert (len(f["iterative_state"]["discovery"]),len(f["iterative_state"]["holdout"]))==(6,4);assert (len(f["variable_composition"]["discovery"]),len(f["variable_composition"]["holdout"]))==(4,4)
def test_no_performance_sorting():assert freeze_source_folds()["performance_sorting"] is False

# Data-role discipline
def test_data_roles_exact():assert PARTITIONS["AP_CHART_FIT"]==(10000,10063) and PARTITIONS["AP_DISCOVERY_FINAL"]==(10448,10511)
def test_historical_validation_exact():assert PARTITIONS["REUSED_HISTORICAL_VALIDATION"]==(20000,20127)
def test_fresh_audit_exact():assert PARTITIONS["FRESH_AUDIT"]==(90000,90499)
def test_data_roles_disjoint():assert_roles()

# IR exactness
def test_routing_ir_exact():assert canonical_ir("conditional_routing").output_law.startswith("SELECT")
def test_recall_ir_exact():assert canonical_ir("delayed_recall").semantic_variable=="remembered_value"
def test_iterative_ir_exact():assert canonical_ir("iterative_state").transition_law=="z_next=0.65*z+0.35*x"
def test_composition_ir_exact():assert canonical_ir("variable_composition").transition_law=="z_next=tanh(g*z+d)"
def test_all_ir_dimension_one():assert all(canonical_ir(f).semantic_dimension==1 for f in POWERED_FAMILIES)

# Reader behavior
def test_polynomial_reader_affine():
    h=np.linspace(-2,2,25);z=2*h+1;c=fit_polynomial(h,z,1);assert np.max(np.abs(read_chart(c,h)-z))<1e-5
def test_polynomial_reader_quadratic():
    h=np.linspace(-1,1,41);z=.2+h+.5*h*h;c=fit_polynomial(h,z,2);assert np.max(np.abs(read_chart(c,h)-z))<1e-4
def test_monotone_reader():
    h=np.linspace(-2,2,50);z=np.tanh(h);c=fit_monotone(h,z,"MONOTONE_PWL_8",8);pred=read_chart(c,h);assert np.corrcoef(z,pred)[0,1]>.99
def test_scalar_chart_degree_three():
    h=np.linspace(-1,1,31);z=h+h**3;c=fit_scalar_chart(h,z,"POLYNOMIAL_3",False);assert c["kind"]=="POLYNOMIAL"
def test_binary_logistic_monotone():
    h=np.linspace(-3,3,80);z=np.where(h>0,1.,-1.);c=fit_scalar_chart(h,z,"LOGISTIC_MONOTONE",True);assert np.mean((read_chart(c,h)>0)==(z>0))>.9
def test_reader_gradient_finite():
    h=np.linspace(-1,1,20);c=fit_scalar_chart(h,h**3+h,"POLYNOMIAL_3",False);assert np.all(np.isfinite(chart_gradient(c,np.array([.2]))))
def test_inverse_reader():
    h=np.linspace(-1,1,30);c=fit_scalar_chart(h,2*h+.3,"AFFINE",False);x=inverse_1d(c,.7,0.);assert abs(float(read_chart(c,np.array([x]))[0])-.7)<1e-6

# Projected charts
def test_projected_feature_count_linear():assert expected_feature_count(4,"LINEAR")==5
def test_projected_feature_count_quadratic():assert expected_feature_count(2,"QUADRATIC")==6
def test_projected_chart_linear():
    rng=np.random.default_rng(2);h=rng.normal(size=(100,2));z=2*h[:,0]-h[:,1];c=fit_projected_chart(h,z,"LINEAR",False);assert np.sqrt(np.mean((read_chart(c,h)-z)**2))<1e-4
def test_projected_chart_quadratic():
    rng=np.random.default_rng(3);h=rng.normal(size=(150,2));z=h[:,0]**2+.3*h[:,1];c=fit_projected_chart(h,z,"QUADRATIC",False);assert np.sqrt(np.mean((read_chart(c,h)-z)**2))<1e-3

# Writer behavior
def test_nonlinear_setter_affine_exact():
    q=np.zeros((40,1));q[0,0]=1;h=np.linspace(-1,1,50);c=fit_scalar_chart(h,2*h,"AFFINE",False);s=np.zeros(40);r=set_state_geometry(s,q,c,1.0,2.0,False);assert r["valid"] and abs(r["read_after"]-1)<1e-6
def test_setter_zero_change():
    q=np.zeros((40,1));q[0,0]=1;h=np.linspace(-1,1,50);c=fit_scalar_chart(h,h,"AFFINE",False);s=np.zeros(40);r=set_state_geometry(s,q,c,0.0,2.0,False);assert r["valid"] and np.linalg.norm(r["state"]-s)<1e-10
def test_gradient_step_bounded():
    rng=np.random.default_rng(5);h=rng.normal(size=(100,2));z=h[:,0]+h[:,1];c=fit_projected_chart(h,z,"LINEAR",False);r=gradient_step(c,np.array([0.,0.]),2.,.5);assert r["valid"] and np.linalg.norm(np.asarray(r["h"]))<=1.000001
def test_tangent_fit_constant():
    h=np.linspace(-1,1,50)[:,None];hc=h+.25;z=np.linspace(-1,1,50);zc=z+.5;f=fit_tangent_field(h,hc,z,zc,"CONSTANT");assert f["valid"] and np.all(np.isfinite(tangent_vector(f,np.array([0.]))))

# Frozen hierarchy
def test_candidate_k_order():
    for f in POWERED_FAMILIES:
        ks=[c["k"] for c in candidate_order(f)];assert ks==sorted(ks)
def test_projected_order_exact():
    rows=[c for c in candidate_order("iterative_state") if c["k"]==2];assert [c["chart_family"] for c in rows]==["LINEAR","QUADRATIC","CUBIC"]
def test_no_mlp_gp_kernel_families():
    text=" ".join(c["chart_family"] for f in POWERED_FAMILIES for c in candidate_order(f)).lower();assert "mlp" not in text and "gaussian" not in text and "kernel" not in text

# Artifact science locks (available after pipeline completion)
def test_fresh_audit_unused():assert j("diagnostics/decision_state.json")["fresh_audit_consumed"] is False
def test_primitives_promoted_zero():assert j("diagnostics/decision_state.json")["primitives_promoted"]==0
def test_v838_not_started():assert j("diagnostics/decision_state.json")["v838_started"] is False and not (ROOT/"experiments/v837_primitive_invention/v838").exists()
def test_new_model_fits_zero():assert j("diagnostics/decision_state.json")["new_source_model_fits"]==0
def test_source_optimizer_steps_zero():assert j("diagnostics/decision_state.json")["source_optimizer_steps"]==0
def test_failure_types_explicit():assert all(e["failure_type"] in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"} for e in j("raw/failure_ledger.json")["entries"])
def test_meta_no_refit():assert j("raw/meta_confirmation.json")["no_refit"] is True
def test_discovery_final_no_refit():assert j("raw/discovery_final_confirmation.json")["no_refit"] is True
def test_freeze_before_holdout():assert j("raw/frozen_family_geometry.json")["frozen_before_any_heldout_backend_read"] is True
def test_holdout_no_geometry_search():
    h=j("raw/calibration_frontier.json");assert not h["branch_search_on_holdout"] and not h["k_search_on_holdout"] and not h["chart_search_on_holdout"] and not h["degree_search_on_holdout"] and not h["phase_atlas_search_on_holdout"] and not h["writer_search_on_holdout"]
def test_calibration_ladder_exact():assert j("raw/calibration_frontier.json")["calibration_ladder"]==[2,4,8,16,32,64]
def test_cross_organism_no_alignment():
    x=j("raw/cross_organism_agreement.json");assert x["microstate_comparison_used"] is False and x["q_alignment_used"] is False and x["state_alignment_used"] is False
