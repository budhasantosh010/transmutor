from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])
LABELS = {
    "Y0_historical": "Y0 historical",
    "Y1_global_control": "Y1 global control",
    "Y2_rank4_candidate": "Y2 rank4 candidate",
    "Y3_global_control_rank4_candidate": "Y3 interaction",
    "Y3C_global_control_matched_local": "Y3C matched local",
}


def _load_rows() -> list[dict]:
    rows = []
    for name in ("anchor_runs.json", "interaction_runs.json"):
        payload = json.loads((HERE / "raw" / name).read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate_id"]))
    write_json(HERE / "raw" / "runs.json", {"rows": rows, "unique_seed_defined_episodes": 3200, "reuse_policy": "same family/seed episodes reused across conditions and replicates"})
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    out = {"families_passing": 0, "family_results": {}, "mean_family_validation_median": 0.0}
    for family in FAMILIES:
        fr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        devs = [r["development_success"] for r in fr]; vals = [r["validation_success"] for r in fr]
        dev = float(np.median(devs)); val = float(np.median(vals)); passing = bool(capacity_demonstrated(dev, val))
        out["families_passing"] += int(passing)
        out["family_results"][family] = {
            "development": {"median": dev, "min": float(np.min(devs)), "max": float(np.max(devs))},
            "validation": {"median": val, "min": float(np.min(vals)), "max": float(np.max(vals)), "std": float(np.std(vals))},
            "capacity_demonstrated": passing,
        }
    out["mean_family_validation_median"] = float(np.mean([v["validation"]["median"] for v in out["family_results"].values()]))
    sample = next(r for r in rows if r["condition"] == condition)
    for key in ("parameter_count", "active_parameter_count", "controller_param_count", "candidate_branch_param_count", "local_recurrent_macs", "candidate_branch_macs", "controller_macs", "total_recurrent_controller_macs"):
        out[key] = int(sample[key])
    return out


def _distribution(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"median": float(np.median(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr)), "p10": float(np.quantile(arr, .10)), "p90": float(np.quantile(arr, .90)), "n": int(arr.size)}


def _factorial(rows: list[dict]) -> dict:
    lookup = {(r["condition"], r["family"], r["replicate_id"]): r["validation_success"] for r in rows}
    per_family = {}; all_ab=[]; all_a=[]; all_b=[]
    for family in FAMILIES:
        a=[]; b=[]; ab=[]
        for rep in range(CONFIG["training"]["replicates"]):
            y0=lookup[("Y0_historical",family,rep)]; y1=lookup[("Y1_global_control",family,rep)]; y2=lookup[("Y2_rank4_candidate",family,rep)]; y3=lookup[("Y3_global_control_rank4_candidate",family,rep)]
            av=((y1-y0)+(y3-y2))/2.0; bv=((y2-y0)+(y3-y1))/2.0; iv=y3-y2-y1+y0
            a.append(av); b.append(bv); ab.append(iv); all_a.append(av); all_b.append(bv); all_ab.append(iv)
        per_family[family]={"A_global_control":_distribution(a),"B_rank4_candidate":_distribution(b),"AB_interaction":_distribution(ab)}
    return {"definition":"AB = Y3 - Y2 - Y1 + Y0; A/B are average factorial main effects","per_family":per_family,"overall":{"A_global_control":_distribution(all_a),"B_rank4_candidate":_distribution(all_b),"AB_interaction":_distribution(all_ab)}}


def _complementarity(summaries: dict) -> dict:
    output={}
    for family in FAMILIES:
        vals={c: summaries[c]["family_results"][family]["validation"]["median"] for c in CONDITIONS}
        y0=vals["Y0_historical"]; d1=vals["Y1_global_control"]-y0; d2=vals["Y2_rank4_candidate"]-y0
        source="global_control" if d1>d2 else ("rank4_candidate" if d2>d1 else "tie")
        best_parent=max(vals["Y1_global_control"],vals["Y2_rank4_candidate"])
        output[family]={"validation_medians":vals,"global_control_delta_vs_y0":d1,"rank4_candidate_delta_vs_y0":d2,"larger_parent_improvement":source,"y3_retains_best_parent":bool(vals["Y3_global_control_rank4_candidate"]>=best_parent-0.02),"y3_delta_vs_best_parent":vals["Y3_global_control_rank4_candidate"]-best_parent,"y3c_delta_vs_best_parent":vals["Y3C_global_control_matched_local"]-best_parent}
    return output


def _scalar_diag(rows, condition, path):
    values=[r["diagnostics"] for r in rows if r["condition"]==condition]
    def collect(keys):
        cur=[]
        for v in values:
            x=v
            try:
                for k in keys: x=x[k]
                if isinstance(x,(int,float)) and math.isfinite(float(x)): cur.append(float(x))
            except (KeyError,TypeError): pass
        return _distribution(cur) if cur else None
    return collect(path)


def _diagnostic_summaries(rows):
    coupling={}; controller={}; message={}; causal={}; gradients={}
    for c in CONDITIONS:
        coupling[c]={k:_scalar_diag(rows,c,["coupling_matrix",k]) for k in ("effective_rank","spectral_norm","frobenius_norm","cross_block_energy","offdiag_fraction")}
        coupling[c].update({k:_scalar_diag(rows,c,["candidate_contributions",k]) for k in ("local_recurrent_norm","rank4_global_norm","matched_local_extra_norm","message_norm","input_norm","bias_norm","global_to_local_ratio","global_to_message_ratio","global_to_input_ratio","added_to_local_ratio")})
        controller[c]={k:_scalar_diag(rows,c,["controller",k]) for k in ("mean","median","std","temporal_variance","p10","p90","near_zero_fraction","near_one_fraction","carry_fraction","rewrite_fraction")}
        message[c]={k:_scalar_diag(rows,c,["message_dependency",k]) for k in ("baseline_success","no_message_success","success_drop","mean_abs_prediction_delta")}
        causal[c]={k:_scalar_diag(rows,c,["cross_cell_causal_influence",k]) for k in ("mean_abs_other_candidate_delta","mean_abs_other_next_state_delta","mean_abs_final_output_delta")}
        gradients[c]={k:_scalar_diag(rows,c,["gradient",k]) for k in ("base_cell_gradient_norm","controller_gradient_norm","candidate_branch_gradient_norm","cell_gradient_norm_variance","all_parameter_gradient_norm")}
    return coupling,controller,message,causal,gradients


def _resources(rows, summaries):
    result={}
    for c in CONDITIONS:
        rr=[r for r in rows if r["condition"]==c]
        def total(k): return float(sum(float(r["resources"].get(k,0.0)) for r in rr))
        result[c]={
            "families_passing":summaries[c]["families_passing"],
            "mean_family_validation_median":summaries[c]["mean_family_validation_median"],
            "trainable_params":summaries[c]["parameter_count"],
            "active_params":summaries[c]["active_parameter_count"],
            "controller_params":summaries[c]["controller_param_count"],
            "rank4_or_matched_params":summaries[c]["candidate_branch_param_count"],
            "recurrent_macs":summaries[c]["local_recurrent_macs"]+summaries[c]["candidate_branch_macs"],
            "controller_macs":summaries[c]["controller_macs"],
            "total_recurrent_controller_macs":summaries[c]["total_recurrent_controller_macs"],
            "model_fits":int(total("model_fits")),"optimizer_steps":int(total("optimizer_steps")),"processed_examples":int(total("examples_processed")),"environment_interactions":int(total("environment_steps")),"forward_calls":int(total("forward_calls")),"cpu_seconds":total("cpu_seconds"),"wall_seconds":total("wall_seconds"),"gpu_seconds":0.0,"unique_seed_defined_episodes":3200,
        }
    combined={k:sum(v[k] for v in result.values()) for k in ("model_fits","optimizer_steps","processed_examples","environment_interactions","forward_calls","cpu_seconds","wall_seconds","gpu_seconds")}
    combined["unique_seed_defined_episodes"]=3200
    return {"conditions":result,"combined":combined}


def _select_parent(summaries: dict) -> str:
    candidates=["Y0_historical","Y1_global_control","Y2_rank4_candidate","Y3_global_control_rank4_candidate"]
    # Frozen deterministic order: family count, mean family validation, then lower active compute/params.
    return max(candidates,key=lambda c:(summaries[c]["families_passing"],summaries[c]["mean_family_validation_median"],-summaries[c]["total_recurrent_controller_macs"],-summaries[c]["active_parameter_count"]))


def _decision(summaries, anchors_ok: bool) -> dict:
    counts={c:summaries[c]["families_passing"] for c in CONDITIONS}
    means={c:summaries[c]["mean_family_validation_median"] for c in CONDITIONS}
    if not anchors_ok:
        return {"diagnosis":"CANDIDATE_INTERACTION_BASELINE_DRIFT","representation_adequacy_pass":False,"v837z_allowed":False,"sample_efficiency_retest_allowed":False,"selected_v837z_parent":None}
    y3=counts["Y3_global_control_rank4_candidate"]; y3c=counts["Y3C_global_control_matched_local"]
    if y3>=4 and y3c<4:
        diagnosis="GLOBAL_CONTROL_X_CROSS_CELL_CANDIDATE_INTERACTION_SUFFICIENT"
    elif y3>=4 and y3c>=4:
        diagnosis="CANDIDATE_CAPACITY_INTERACTION_SUFFICIENT"
    elif y3c>=4:
        diagnosis="CANDIDATE_CAPACITY_INTERACTION_SUFFICIENT"
    elif y3<max(counts["Y1_global_control"],counts["Y2_rank4_candidate"]):
        diagnosis="GLOBAL_CONTROL_X_CANDIDATE_MIXING_INTERFERENCE"
    else:
        parent_mean=max(means["Y1_global_control"],means["Y2_rank4_candidate"])
        equally_improved=(y3==y3c and means["Y3_global_control_rank4_candidate"]>parent_mean+0.01 and means["Y3C_global_control_matched_local"]>parent_mean+0.01 and abs(means["Y3_global_control_rank4_candidate"]-means["Y3C_global_control_matched_local"])<=0.02)
        diagnosis="INTERACTION_SPECIFICITY_NOT_ESTABLISHED" if equally_improved else "GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT"
    representation=max(counts.values())>=CONFIG["representation_family_gate"]
    return {"diagnosis":diagnosis,"representation_adequacy_pass":representation,"v837z_allowed":bool(not representation),"sample_efficiency_retest_allowed":bool(representation),"selected_v837z_parent":None if representation else _select_parent(summaries)}


def _plot(summaries,factorial,controller,message):
    (HERE/"plots").mkdir(exist_ok=True)
    xs=np.arange(len(CONDITIONS)); labels=[LABELS[c] for c in CONDITIONS]
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[summaries[c]["families_passing"] for c in CONDITIONS]); plt.axhline(4,linestyle="--"); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("families passing"); plt.tight_layout(); fig.savefig(HERE/"plots/factorial_families_passing.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(9,5));
    for family in FAMILIES: plt.plot(xs,[summaries[c]["family_results"][family]["validation"]["median"] for c in CONDITIONS],marker="o",label=family)
    plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("validation median"); plt.legend(fontsize=7); plt.tight_layout(); fig.savefig(HERE/"plots/factorial_family_scores.png",dpi=140); plt.close(fig)
    focus=["conditional_routing","variable_composition","partial_observation"]; fig=plt.figure(figsize=(9,5)); w=.15
    for j,f in enumerate(focus): plt.bar(xs+(j-1)*w,[summaries[c]["family_results"][f]["validation"]["median"] for c in CONDITIONS],width=w,label=f)
    plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("validation median"); plt.legend(); plt.tight_layout(); fig.savefig(HERE/"plots/routing_composition_complementarity.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(7,4)); plt.bar(np.arange(len(FAMILIES)),[factorial["per_family"][f]["AB_interaction"]["median"] for f in FAMILIES]); plt.axhline(0); plt.xticks(np.arange(len(FAMILIES)),FAMILIES,rotation=25,ha="right"); plt.ylabel("AB interaction"); plt.tight_layout(); fig.savefig(HERE/"plots/interaction_effect_per_family.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[summaries[c]["total_recurrent_controller_macs"] for c in CONDITIONS]); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("recurrent/controller MACs"); plt.tight_layout(); fig.savefig(HERE/"plots/capability_vs_active_macs.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[summaries[c]["active_parameter_count"] for c in CONDITIONS]); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("active parameters"); plt.tight_layout(); fig.savefig(HERE/"plots/capability_vs_parameters.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[message[c]["success_drop"]["median"] if message[c]["success_drop"] else 0 for c in CONDITIONS]); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("message ablation success drop"); plt.tight_layout(); fig.savefig(HERE/"plots/message_dependence.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[controller[c]["temporal_variance"]["median"] if controller[c]["temporal_variance"] else 0 for c in CONDITIONS]); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("gate temporal variance"); plt.tight_layout(); fig.savefig(HERE/"plots/controller_gate_dynamics.png",dpi=140); plt.close(fig)
    fig=plt.figure(figsize=(8,4)); plt.bar(xs,[summaries[c]["candidate_branch_macs"] for c in CONDITIONS]); plt.xticks(xs,labels,rotation=25,ha="right"); plt.ylabel("candidate branch MACs"); plt.tight_layout(); fig.savefig(HERE/"plots/candidate_branch_utilization.png",dpi=140); plt.close(fig)


def main() -> int:
    rows=_load_rows(); summaries={c:_summary(rows,c) for c in CONDITIONS}
    compat=json.loads((HERE/"diagnostics/baseline_compatibility.json").read_text(encoding="utf-8")); anchors_ok=bool(compat.get("compatible"))
    factorial=_factorial(rows); complementarity=_complementarity(summaries)
    coupling,controller,message,causal,gradients=_diagnostic_summaries(rows)
    resources=_resources(rows,summaries); decision=_decision(summaries,anchors_ok)
    write_json(HERE/"diagnostics/factorial_effects.json",factorial)
    write_json(HERE/"diagnostics/complementarity.json",complementarity)
    write_json(HERE/"diagnostics/coupling_diagnostics.json",coupling)
    write_json(HERE/"diagnostics/controller_diagnostics.json",controller)
    write_json(HERE/"diagnostics/message_diagnostics.json",message)
    write_json(HERE/"diagnostics/cross_cell_causal_influence.json",causal)
    write_json(HERE/"diagnostics/gradient_diagnostics.json",gradients)
    write_json(HERE/"diagnostics/compute_efficiency.json",resources)
    decision_state={"v837y_complete":True,"anchors_reproduced":anchors_ok,"families_passing":{c:summaries[c]["families_passing"] for c in CONDITIONS},"best_passing_condition":max(CONDITIONS,key=lambda c:(summaries[c]["families_passing"],summaries[c]["mean_family_validation_median"],-summaries[c]["total_recurrent_controller_macs"])),**decision,"structural_search_allowed":False,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False}
    write_json(HERE/"diagnostics/decision_state.json",decision_state)
    result={"version":"V837y","parent":"V837x","single_question":"global temporal control x candidate integration","conditions":summaries,"baseline_compatibility":compat,"factorial_effects":factorial,"complementarity":complementarity,"coupling_diagnostics":coupling,"controller_diagnostics":controller,"message_diagnostics":message,"cross_cell_causal_influence":causal,"gradient_diagnostics":gradients,"compute_efficiency":resources,"unique_seed_defined_episodes":3200,"representation_adequacy_pass":decision["representation_adequacy_pass"],"diagnosis":decision["diagnosis"],"v837z_allowed":decision["v837z_allowed"],"selected_v837z_parent":decision["selected_v837z_parent"],"sample_efficiency_retest_allowed":decision["sample_efficiency_retest_allowed"],"structural_search_allowed":False,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"resource_accounting":resources["combined"]}
    write_json(HERE/"results.json",result)
    _plot(summaries,factorial,controller,message)
    if decision["representation_adequacy_pass"]:
        text="# V837y PASS\n\nDiagnosis: `%s`. Representation adequacy is restored at the frozen >=4/5 gate. Architecture modification stops and sample-efficiency characterization is authorized.\n"%decision["diagnosis"]
        (HERE/"PASS.md").write_text(text,encoding="utf-8")
        if (HERE/"FAILURE.md").exists(): (HERE/"FAILURE.md").unlink()
    else:
        text="# V837y FAILURE\n\nDiagnosis: `%s`. Representation adequacy remains below 4/5. V837z is authorized only through the committed decision state and only for the selected parent `%s`.\n"%(decision["diagnosis"],decision["selected_v837z_parent"])
        (HERE/"FAILURE.md").write_text(text,encoding="utf-8")
        if (HERE/"PASS.md").exists(): (HERE/"PASS.md").unlink()
    print(json.dumps(decision_state,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
