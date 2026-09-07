from __future__ import annotations

import json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text(encoding="utf-8")); FAMILIES=[t.name for t in all_tasks()]; CONDITIONS=CONFIG["conditions"]

def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows():
    out=[]
    for f in ("ac0_runs.json","transfer_runs.json"): out+=load(HERE/"raw"/f)["rows"]
    return sorted(out,key=lambda r:(r["condition"],r["family"],r["replicate_id"]))
def summary(rs,c):
    out={"families_passing":0,"family_results":{},"mean_family_validation_median":0.0}; vals=[]
    for f in FAMILIES:
        rr=[r for r in rs if r["condition"]==c and r["family"]==f]; dev=float(np.median([r["development_success"] for r in rr])); val=float(np.median([r["validation_success"] for r in rr])); passed=capacity_demonstrated(dev,val); out["families_passing"]+=int(passed); vals.append(val); out["family_results"][f]={"development":{"median":dev},"validation":{"median":val},"capacity_demonstrated":bool(passed)}
    first=[r for r in rs if r["condition"]==c][0]; out["mean_family_validation_median"]=float(np.mean(vals)); out["parameter_count"]=int(first["parameter_count"]); out["projection_specific_macs"]=int(first["projection_specific_macs"]); out["recurrent_controller_projection_macs"]=int(first["recurrent_controller_projection_macs"]); return out

def projection_dynamics(rs):
    out={}
    for c in CONDITIONS:
        by=defaultdict(list)
        for r in [x for x in rs if x["condition"]==c]:
            for p in r["projection_trajectory"]: by[int(p["step"])].append(p["projection"])
        out[c]={}
        for step,ps in sorted(by.items()):
            active=[p for p in ps if p.get("active")]; row={"fits":len(ps),"active_projection_fits":len(active),"effective_controller_input_norm_median":float(np.median([p["effective_controller_input_norm"] for p in ps]))}
            if active: row.update({"projection_frobenius_median":float(np.median([p["frobenius_norm"] for p in active])),"projection_condition_number_median":float(np.median([p["condition_number"] for p in active])),"projection_drift_median":float(np.median([p["projection_drift"] for p in active])),"projection_bias_drift_median":float(np.median([p["projection_bias_drift"] for p in active]))})
            out[c][str(step)]=row
    return out

def diagnostics(rs):
    out={}
    for c in CONDITIONS:
        rr=[r for r in rs if r["condition"]==c]; out[c]={"message_success_drop_median":float(np.median([r["diagnostics"]["message_dependency"]["success_drop"] for r in rr])),"input_term_norm_median":float(np.median([r["diagnostics"]["input_term"]["mean_norm"] for r in rr])),"input_term_temporal_variance_median":float(np.median([r["diagnostics"]["input_term"]["mean_temporal_variance"] for r in rr])),"candidate_jacobian_norm_approx_median":float(np.median([r["diagnostics"]["input_term"]["mean_candidate_jacobian_norm_approx"] for r in rr])),"candidate_input_map_pairwise_cosine_median":float(np.median([r["diagnostics"]["input_term"]["pairwise_effective_candidate_input_cosine_median"] for r in rr])),"controller_effective_input_norm_median":float(np.median([r["diagnostics"]["controller_input"]["effective_controller_input_norm"] for r in rr]))}
    return out

def resources(rs):
    out={"version":"V837ac","model_fits":len(rs),"optimizer_steps":sum(int(r["resources"]["optimizer_steps"]) for r in rs),"processed_training_examples":sum(int(r["processed_examples"]) for r in rs),"unique_seed_defined_episodes":3200,"environment_interactions":sum(int(r["resources"]["environment_steps"]) for r in rs),"forward_calls":sum(int(r["resources"]["forward_calls"]) for r in rs),"backward_calls":sum(int(r["resources"]["optimizer_steps"]) for r in rs),"cpu_seconds_worker_sum":sum(float(r["resources"]["cpu_seconds"]) for r in rs),"wall_seconds_worker_sum":sum(float(r["resources"]["wall_seconds"]) for r in rs),"gpu_seconds":0.0,"fresh_audit_consumed":False}; return out

def decision(s):
    ac0=s["AC0_y3_parent"]["families_passing"]; ac1=s["AC1_controller_input_factorization"]["families_passing"]; acf=s["AC1F_folded_control"]["families_passing"]
    if ac1>=4 and acf>=4: return "INPUT_EFFECTIVE_MAPPING_SUFFICIENT",True,"AC1F_folded_control"
    if ac1>=4: return "SHARED_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT",True,"AC1_controller_input_factorization"
    if ac1<ac0: return "INPUT_ORGANIZATION_TRANSFER_HARMFUL",False,None
    return "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT",False,None

def plots(s,dyn,deltas):
    p=HERE/"plots"; p.mkdir(exist_ok=True); labels=["AC0","AC1","AC1F"]
    fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(FAMILIES)); w=.25
    for j,c in enumerate(CONDITIONS): ax.bar(x+(j-1)*w,[s[c]["family_results"][f]["validation"]["median"] for f in FAMILIES],w,label=labels[j])
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES,rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(p/"neutral_input_transfer_family_scores.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); cats=["conditional_routing","partial_observation","variable_composition"]; ax.bar(cats,[deltas["AC1_controller_input_factorization"][k] for k in cats]); ax.set_ylabel("AC1 validation median delta vs Y3"); ax.tick_params(axis='x',rotation=20); fig.tight_layout(); fig.savefig(p/"routing_partial_composition_deltas.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["AC1","AC1F"],[s[CONDITIONS[1]]["families_passing"],s[CONDITIONS[2]]["families_passing"]]); ax.axhline(4,linestyle="--"); ax.set_ylim(0,5.3); fig.tight_layout(); fig.savefig(p/"shared_vs_folded_control.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3)); ax.axis('off'); ax.text(.5,.5,"AC1D not applicable\n(controller-only single consumer)",ha='center',va='center'); fig.tight_layout(); fig.savefig(p/"shared_vs_deshared_projection.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); steps=sorted(int(x) for x in dyn[CONDITIONS[1]]); ax.plot(steps,[dyn[CONDITIONS[1]][str(x)].get("projection_drift_median",0) for x in steps],marker='o'); ax.set_xlabel("optimizer step"); ax.set_ylabel("projection drift"); fig.tight_layout(); fig.savefig(p/"projection_dynamics.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter([s[c]["recurrent_controller_projection_macs"] for c in CONDITIONS],[s[c]["families_passing"] for c in CONDITIONS]); [ax.annotate(labels[i],(s[c]["recurrent_controller_projection_macs"],s[c]["families_passing"])) for i,c in enumerate(CONDITIONS)]; ax.set_xlabel("active recurrent/controller/projection MACs"); ax.set_ylabel("families passing"); fig.tight_layout(); fig.savefig(p/"capability_vs_active_macs.png",dpi=150); plt.close(fig)
def main():
    rs=rows();
    if len(rs)!=75: raise SystemExit(f"expected 75 fits, found {len(rs)}")
    parent=load(HERE/"diagnostics/parent_compatibility.json"); step0=load(HERE/"diagnostics/step0_equivalence.json"); s={c:summary(rs,c) for c in CONDITIONS}; diag=diagnostics(rs); dyn=projection_dynamics(rs); res=resources(rs); diagnosis,rep_pass,winner=decision(s)
    deltas={c:{f:s[c]["family_results"][f]["validation"]["median"]-s["AC0_y3_parent"]["family_results"][f]["validation"]["median"] for f in FAMILIES} for c in CONDITIONS}
    write_json(HERE/"diagnostics/input_projection_dynamics.json",dyn); write_json(HERE/"diagnostics/input_path_diagnostics.json",diag); write_json(HERE/"diagnostics/compute_efficiency.json",res)
    state={"v837ac_complete":True,"authorized_mode":CONFIG["authorized_mode"],"parent_reproduced":bool(parent["parent_reproduced"]),"step0_equivalence_proven":bool(step0["step0_equivalence_proven"]),"families_passing":{c:s[c]["families_passing"] for c in CONDITIONS},"diagnosis":diagnosis,"best_passing_condition":winner,"representation_adequacy_pass":rep_pass,"sample_efficiency_retest_allowed":rep_pass,"structural_search_allowed":rep_pass,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False}
    write_json(HERE/"diagnostics/decision_state.json",state)
    result={"version":"V837ac","parent":"V837ab","neutral_parent":"Y3_global_control_rank4_candidate","authorized_mode":CONFIG["authorized_mode"],"conditions":s,"family_deltas_vs_y3":deltas,"input_path_diagnostics":diag,"projection_dynamics":dyn,"diagnosis":diagnosis,"best_passing_condition":winner,"representation_adequacy_pass":rep_pass,"sample_efficiency_retest_allowed":rep_pass,"structural_search_allowed":rep_pass,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"resource_accounting":res}
    write_json(HERE/"results.json",result); write_json(ROOT/"experiments/v837_primitive_invention/v837ac_resource_accounting.json",res); plots(s,dyn,deltas)
    text=f"# V837ac — {diagnosis}\n\nAC0 Y3 parent: {s['AC0_y3_parent']['families_passing']}/5.\nAC1 controller factorization: {s['AC1_controller_input_factorization']['families_passing']}/5.\nAC1F folded control: {s['AC1F_folded_control']['families_passing']}/5.\n\nRepresentation adequacy: {'PASS' if rep_pass else 'FAIL'}.\n"
    (HERE/("PASS.md" if rep_pass else "FAILURE.md")).write_text(text,encoding="utf-8"); print(json.dumps(state,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
