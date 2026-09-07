from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]


def _load_rows(name: str) -> list[dict]:
    path = HERE / "raw" / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _all_rows() -> list[dict]:
    rows=[]
    for name in ("ad0_runs.json","ad1_runs.json","geometry_runs.json","robustness_runs.json"):
        rows.extend(_load_rows(name))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    result={"families_passing":0,"family_results":{}}
    for family in FAMILIES:
        fr=[r for r in rows if r["condition"]==condition and r["family"]==family]
        if not fr: continue
        dev=np.array([r["development_success"] for r in fr],dtype=float); val=np.array([r["validation_success"] for r in fr],dtype=float)
        d=float(np.median(dev)); v=float(np.median(val)); passed=capacity_demonstrated(d,v); result["families_passing"] += int(passed)
        result["family_results"][family]={"development":{"median":d,"values":[float(x) for x in dev]},"validation":{"median":v,"values":[float(x) for x in val]},"capacity_demonstrated":bool(passed)}
    if result["family_results"]:
        result["mean_family_validation_median"]=float(np.mean([x["validation"]["median"] for x in result["family_results"].values()]))
    return result


def _median(rows: list[dict], path: tuple[str,...]) -> float:
    vals=[]
    for row in rows:
        cur=row
        try:
            for key in path: cur=cur[key]
            vals.append(float(cur))
        except (KeyError,TypeError,ValueError): pass
    return float(np.median(vals)) if vals else 0.0


def _aggregate_diagnostics(rows: list[dict], conditions: list[str]) -> None:
    matrix={}; ranks={}; gates={}; grads={}; trajectories={}; efficiency={}
    for condition in conditions:
        cr=[r for r in rows if r["condition"]==condition]
        if not cr: continue
        geometry=cr[0]["candidate_matrix_geometry"]
        matrix[condition]={
            "active_weight_count":int(geometry["active_weights"]), "density":float(geometry["density"]),
            "fan_in_distribution":geometry["fan_in"], "fan_out_distribution":geometry["fan_out"], "weak_components":geometry["weak_components"],
            "strongly_connected":geometry["strongly_connected"], "directed_diameter":geometry["directed_diameter"],
            "spectral_radius_median":_median(cr,("candidate_matrix_geometry","spectral_radius")),
            "spectral_norm_median":_median(cr,("candidate_matrix_geometry","spectral_norm")),
            "frobenius_norm_median":_median(cr,("candidate_matrix_geometry","frobenius_norm")),
            "numerical_rank_median":_median(cr,("candidate_matrix_geometry","numerical_rank")),
            "singular_values_median":[float(x) for x in np.median(np.array([r["candidate_matrix_geometry"]["singular_values"] for r in cr],dtype=float),axis=0).tolist()],
            "within_4d_block_energy_median":_median(cr,("candidate_matrix_geometry","within_4d_block_energy")),
            "cross_4d_block_energy_median":_median(cr,("candidate_matrix_geometry","cross_4d_block_energy")),
        }
        ranks[condition]={
            "state_effective_rank_median":_median(cr,("trace_diagnostics","state","effective_rank")),
            "state_participation_ratio_median":_median(cr,("trace_diagnostics","state","participation_ratio")),
            "state_norm_mean_median":_median(cr,("trace_diagnostics","state","norm_mean")),
            "state_correlation_median":_median(cr,("trace_diagnostics","state","mean_abs_offdiag_correlation")),
            "candidate_effective_rank_median":_median(cr,("trace_diagnostics","candidate","effective_rank")),
            "candidate_participation_ratio_median":_median(cr,("trace_diagnostics","candidate","participation_ratio")),
            "candidate_norm_mean_median":_median(cr,("trace_diagnostics","candidate","norm_mean")),
            "candidate_correlation_median":_median(cr,("trace_diagnostics","candidate","mean_abs_offdiag_correlation")),
            "candidate_jacobian_spectral_norm_median":_median(cr,("trace_diagnostics","candidate_jacobian","spectral_norm_median")),
            "candidate_jacobian_frobenius_norm_median":_median(cr,("trace_diagnostics","candidate_jacobian","frobenius_norm_median")),
            "candidate_jacobian_effective_rank_median":_median(cr,("trace_diagnostics","candidate_jacobian","effective_rank_median")),
            "hidden_candidate_norm_median":_median(cr,("trace_diagnostics","candidate_recurrent_influence","hidden_candidate_norm_mean")),
            "hidden_candidate_temporal_variance_median":_median(cr,("trace_diagnostics","candidate_recurrent_influence","hidden_candidate_temporal_variance")),
            "candidate_input_hidden_ratio_median":_median(cr,("trace_diagnostics","candidate_recurrent_influence","candidate_input_to_hidden_norm_ratio")),
        }
        gates[condition]={k:_median(cr,("trace_diagnostics","gate",k)) for k in ("carry_fraction_mean","rewrite_fraction_mean","temporal_variance","p10","p90","near_zero_fraction","near_one_fraction")}
        gates[condition]["update_hidden_norm_mean_median"]=_median(cr,("trace_diagnostics","update_hidden_norm_mean"))
        grads[condition]={k:_median(cr,("gradient_diagnostics",k)) for k in ("candidate_input_slice_norm","candidate_recurrent_active_grad_norm","candidate_recurrent_masked_grad_norm","candidate_recurrent_masked_grad_max_abs","update_input_slice_norm","update_recurrent_slice_norm","input_projection_grad_norm","readout_grad_norm")}
        trajectories[condition]={}
        for step in CONFIG["training"]["curve_steps"]:
            sr=[]
            for r in cr:
                sr.extend([x for x in r["learning_trajectory"] if int(x["step"])==int(step)])
            if sr:
                trajectories[condition][str(step)]={
                    "development_loss_median":float(np.median([x["development_loss"] for x in sr])),
                    "development_success_median":float(np.median([x["development_success"] for x in sr])),
                    "candidate_matrix_norm_median":float(np.median([x["candidate_matrix_norm"] for x in sr])),
                    "state_effective_rank_median":float(np.median([x["state_effective_rank"] for x in sr])),
                    "candidate_effective_rank_median":float(np.median([x["candidate_effective_rank"] for x in sr])),
                    "gate_carry_median":float(np.median([x["gate"]["carry_fraction_mean"] for x in sr])),
                }
        efficiency[condition]={
            "nominal_parameters":int(cr[0]["nominal_parameters"]), "trainable_raw_parameters":int(cr[0]["trainable_raw_parameters"]), "active_parameters":int(cr[0]["active_parameters"]),
            "active_candidate_recurrent_weights":int(cr[0]["active_candidate_recurrent_weights"]), "masked_candidate_recurrent_weights":int(cr[0]["masked_candidate_recurrent_weights"]),
            "candidate_recurrent_macs":int(cr[0]["candidate_recurrent_macs"]), "total_active_macs_per_timestep":int(cr[0]["total_active_macs_per_timestep"]), "parameter_bytes":int(cr[0]["parameter_bytes"]),
        }
    write_json(HERE/"diagnostics/candidate_matrix_geometry.json",matrix)
    write_json(HERE/"diagnostics/state_candidate_rank.json",ranks)
    write_json(HERE/"diagnostics/gate_dynamics.json",gates)
    write_json(HERE/"diagnostics/gradient_diagnostics.json",grads)
    write_json(HERE/"diagnostics/learning_trajectories.json",trajectories)
    write_json(HERE/"diagnostics/compute_efficiency.json",efficiency)


def _decision(summaries: dict[str,dict]) -> dict:
    anchor=json.loads((HERE/"diagnostics/anchor_compatibility.json").read_text(encoding="utf-8"))
    width=json.loads((HERE/"diagnostics/width_gate.json").read_text(encoding="utf-8")) if (HERE/"diagnostics/width_gate.json").exists() else {}
    base={
        "v837ad_complete":True,"ad0_anchor_valid":bool(anchor.get("ad0_anchor_valid")),"ad1_dense_h40_families":None,"geometry_stage_run":False,"families_passing":{k:int(v.get("families_passing",0)) for k,v in summaries.items()},
        "ad4s_robustness_run":False,"ad4s_topology_pass_count":None,"diagnosis":"","authorized_v837ae_mode":None,"representation_recovery_in_reference":False,
        "fresh_audit_consumed":False,"structural_search_allowed":False,"primitive_mining_allowed":False,"sample_efficiency_retest_allowed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,
    }
    if not base["ad0_anchor_valid"]:
        base["diagnosis"]="REFERENCE_GEOMETRY_BASELINE_DRIFT"; base["v837ad_complete"]=False; return base
    ad1=int(summaries.get("AD1_H40_dense",{}).get("families_passing",-1)); base["ad1_dense_h40_families"]=ad1
    if ad1 < 4:
        base["diagnosis"]="DENSE_H40_REFERENCE_INADEQUATE"; base["next_axis"]="HIDDEN_WIDTH_LOCALIZATION"; return base
    base["geometry_stage_run"]=all(k in summaries for k in CONFIG["stage_b_conditions"])
    if not base["geometry_stage_run"]:
        base["v837ad_complete"]=False; base["diagnosis"]="GEOMETRY_STAGE_REQUIRED"; return base
    ad2=int(summaries["AD2_H40_2x20"]["families_passing"]); ad3=int(summaries["AD3_H40_5x8"]["families_passing"]); ad4=int(summaries["AD4_H40_10x4"]["families_passing"]); s0=int(summaries["AD4S_S0"]["families_passing"])
    if ad4 >= 4:
        base["diagnosis"]="TEN_BY_FOUR_CANDIDATE_GEOMETRY_SUFFICIENT_IN_REFERENCE"; base["representation_recovery_in_reference"]=True; base["next_axis"]="GRAPH_MESSAGE_OUTPUT_INTERFACE_ORGANIZATION"; return base
    if ad4 < 4 and s0 >= 4:
        robust_keys=[f"AD4S_S{i}" for i in range(5)]; present=all(k in summaries for k in robust_keys)
        if not present:
            base["v837ad_complete"]=False; base["diagnosis"]="SPARSE_GEOMETRY_ROBUSTNESS_REQUIRED"; return base
        passes=sum(int(summaries[k]["families_passing"]>=4) for k in robust_keys); base["ad4s_robustness_run"]=True; base["ad4s_topology_pass_count"]=passes
        if passes >= 4:
            base["diagnosis"]="GLOBAL_SPARSE_CANDIDATE_GEOMETRY_SUFFICIENT"; base["authorized_v837ae_mode"]="DIRECT_GLOBAL_SPARSE_CANDIDATE_RECURRENCE"; base["representation_recovery_in_reference"]=True; return base
        base["diagnosis"]="SPARSE_GEOMETRY_TOPOLOGY_SENSITIVE"; base["next_axis"]="SPARSE_TOPOLOGY_LOCALIZATION"; return base
    # Nonmonotonic partition containment: a narrower block passes after a wider block failed.
    if (ad2 < 4 and ad3 >= 4) or (ad3 < 4 and ad4 >= 4):
        base["diagnosis"]="NONMONOTONIC_CANDIDATE_GEOMETRY_RESPONSE"; base["next_axis"]="TARGETED_GEOMETRY_ANALYSIS"; return base
    if ad3 >= 4 and ad4 < 4 and s0 < 4:
        base["diagnosis"]="CANDIDATE_RECURRENT_BANDWIDTH_BELOW_EIGHT_INSUFFICIENT"; base["next_axis"]="CANDIDATE_RECURRENT_FANIN_LOCALIZATION"; return base
    if ad2 >= 4 and ad3 < 4:
        base["diagnosis"]="CANDIDATE_RECURRENT_BANDWIDTH_BELOW_TWENTY_INSUFFICIENT"; base["next_axis"]="CANDIDATE_RECURRENT_FANIN_LOCALIZATION"; return base
    base["diagnosis"]="CANDIDATE_GEOMETRY_BROADLY_INSUFFICIENT_OR_INCONCLUSIVE"; base["next_axis"]="CANDIDATE_GEOMETRY_BLOCKER_ANALYSIS"; return base


def _resources(rows: list[dict]) -> dict:
    return {
        "version":"V837ad","model_fits":len(rows),"optimizer_steps":sum(int(r["resources"]["optimizer_steps"]) for r in rows),
        "processed_training_examples":sum(int(r["resources"]["examples_processed"]) for r in rows),"unique_seed_defined_episodes":3200,
        "environment_interactions":sum(int(r["resources"]["environment_steps"]) for r in rows),"forward_calls":sum(int(r["resources"]["forward_calls"]) for r in rows),
        "backward_calls":sum(int(r["resources"]["optimizer_steps"]) for r in rows),"cpu_seconds_worker_sum":sum(float(r["resources"]["cpu_seconds"]) for r in rows),
        "wall_seconds_worker_sum":sum(float(r["resources"]["wall_seconds"]) for r in rows),"gpu_seconds":0.0,"fresh_audit_consumed":False,
    }


def _plots(summaries: dict[str,dict]) -> None:
    import matplotlib.pyplot as plt
    names=list(summaries); labels=[n.replace("_H40_", "\n").replace("AD0_H13_","AD0\n") for n in names]
    values=[summaries[n]["families_passing"] for n in names]
    plot_specs=[
        ("families_passing_vs_candidate_geometry.png", values, "Families passing", "Capability vs candidate geometry"),
        ("candidate_active_weights_vs_capability.png", [summaries[n].get("active_candidate_recurrent_weights",0) for n in names], "Active candidate weights", "Candidate active weights"),
        ("candidate_macs_vs_capability.png", [summaries[n].get("candidate_recurrent_macs",0) for n in names], "Candidate recurrent MACs", "Candidate recurrent MACs"),
    ]
    for filename, vals, ylabel, title in plot_specs:
        fig=plt.figure(figsize=(9,4)); ax=fig.add_subplot(111); ax.bar(range(len(names)),vals); ax.set_xticks(range(len(names)),labels,rotation=25,ha="right"); ax.set_ylabel(ylabel); ax.set_title(title); fig.tight_layout(); fig.savefig(HERE/"plots"/filename,dpi=120); plt.close(fig)
    # Family scores vs block width.
    fig=plt.figure(figsize=(9,4)); ax=fig.add_subplot(111)
    for family in FAMILIES:
        ax.plot(range(len(names)),[summaries[n].get("family_results",{}).get(family,{}).get("validation",{}).get("median",np.nan) for n in names],marker="o",label=family)
    ax.set_xticks(range(len(names)),labels,rotation=25,ha="right"); ax.set_ylabel("Validation median"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(HERE/"plots"/"family_scores_vs_block_width.png",dpi=120); plt.close(fig)
    diag=json.loads((HERE/"diagnostics/state_candidate_rank.json").read_text(encoding="utf-8")); gate=json.loads((HERE/"diagnostics/gate_dynamics.json").read_text(encoding="utf-8"))
    for key,filename,title in (("state_effective_rank_median","state_effective_rank_vs_geometry.png","State effective rank"),("candidate_effective_rank_median","candidate_effective_rank_vs_geometry.png","Candidate effective rank"),("candidate_jacobian_effective_rank_median","candidate_jacobian_rank.png","Candidate Jacobian rank")):
        fig=plt.figure(figsize=(9,4)); ax=fig.add_subplot(111); ax.bar(range(len(names)),[diag.get(n,{}).get(key,0) for n in names]); ax.set_xticks(range(len(names)),labels,rotation=25,ha="right"); ax.set_title(title); fig.tight_layout(); fig.savefig(HERE/"plots"/filename,dpi=120); plt.close(fig)
    fig=plt.figure(figsize=(9,4)); ax=fig.add_subplot(111); ax.bar(range(len(names)),[gate.get(n,{}).get("carry_fraction_mean",0) for n in names]); ax.set_xticks(range(len(names)),labels,rotation=25,ha="right"); ax.set_title("Gate compensation vs geometry"); fig.tight_layout(); fig.savefig(HERE/"plots"/"gate_compensation_vs_geometry.png",dpi=120); plt.close(fig)
    if any(n.startswith("AD4S_S1") for n in names):
        sk=[f"AD4S_S{i}" for i in range(5) if f"AD4S_S{i}" in summaries]; fig=plt.figure(figsize=(6,4)); ax=fig.add_subplot(111); ax.bar(sk,[summaries[k]["families_passing"] for k in sk]); ax.set_ylim(0,5); ax.set_title("Sparse topology robustness"); fig.tight_layout(); fig.savefig(HERE/"plots"/"sparse_topology_robustness.png",dpi=120); plt.close(fig)


def main() -> int:
    for n in ("diagnostics","plots"): (HERE/n).mkdir(exist_ok=True)
    rows=_all_rows(); conditions=sorted({r["condition"] for r in rows},key=lambda n:list(CONFIG["stage_a_conditions"]+CONFIG["stage_b_conditions"]+CONFIG["robustness_conditions"]).index(n))
    summaries={c:_summary(rows,c) for c in conditions}
    for c in conditions:
        cr=[r for r in rows if r["condition"]==c]
        if cr:
            summaries[c].update({"nominal_parameters":cr[0]["nominal_parameters"],"active_parameters":cr[0]["active_parameters"],"active_candidate_recurrent_weights":cr[0]["active_candidate_recurrent_weights"],"masked_candidate_recurrent_weights":cr[0]["masked_candidate_recurrent_weights"],"candidate_recurrent_macs":cr[0]["candidate_recurrent_macs"],"total_active_macs_per_timestep":cr[0]["total_active_macs_per_timestep"],"parameter_bytes":cr[0]["parameter_bytes"]})
    _aggregate_diagnostics(rows,conditions); decision=_decision(summaries); resources=_resources(rows)
    results={"version":"V837ad","question":CONFIG["question"],"conditions":summaries,"width_gate":json.loads((HERE/"diagnostics/width_gate.json").read_text(encoding="utf-8")) if (HERE/"diagnostics/width_gate.json").exists() else {},"decision":decision,"resource_accounting":resources,"unique_seed_defined_episodes":3200,"fresh_audit_consumed":False,"large_persistent_storage_tested":False,"v838_started":False}
    write_json(HERE/"results.json",results); write_json(HERE/"diagnostics/decision_state.json",decision); write_json(HERE.parent/"v837ad_resource_accounting.json",resources)
    _plots(summaries)
    if decision.get("v837ad_complete"):
        (HERE/"PASS.md").write_text(f"# V837ad COMPLETE\n\nDiagnosis: `{decision['diagnosis']}`.\n\nFresh audit consumed: 0. V838 not started.\n",encoding="utf-8")
        failure=HERE/"FAILURE.md"
        if failure.exists(): failure.unlink()
    else:
        (HERE/"FAILURE.md").write_text(f"# V837ad INCOMPLETE\n\nDiagnosis: `{decision['diagnosis']}`.\n",encoding="utf-8")
    print(json.dumps(decision,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
