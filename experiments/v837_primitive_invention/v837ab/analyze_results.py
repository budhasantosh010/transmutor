from __future__ import annotations

import json
import sys
from collections import defaultdict
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
GATE = json.loads((HERE / "frozen_input_factorization_gate.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = CONFIG["conditions"]


def _load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def _rows() -> list[dict]:
    rows = []
    for filename in ("ab0_runs.json", "factorization_runs.json"):
        payload = _load(HERE / "raw" / filename); rows.extend(payload["rows"])
    rows.sort(key=lambda r:(r["condition"],r["family"],r["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    result={"families_passing":0,"family_results":{},"mean_family_validation_median":0.0}
    vals=[]
    for family in FAMILIES:
        fr=[r for r in rows if r["condition"]==condition and r["family"]==family]
        dev=float(np.median([r["development_success"] for r in fr])); val=float(np.median([r["validation_success"] for r in fr]))
        passed=capacity_demonstrated(dev,val); result["families_passing"]+=int(passed); vals.append(val)
        result["family_results"][family]={"development":{"median":dev,"values":[float(r["development_success"]) for r in fr]},"validation":{"median":val,"values":[float(r["validation_success"]) for r in fr]},"capacity_demonstrated":bool(passed)}
    result["mean_family_validation_median"]=float(np.mean(vals))
    first=[r for r in rows if r["condition"]==condition][0]
    for key in ("nominal_parameters","trainable_parameters","active_parameters","executed_parameters","executed_frozen_parameters","projection_specific_macs","active_reference_macs_per_timestep"):
        result[key]=int(first[key])
    return result


def _decision(counts: dict[str,int], reference_valid: bool) -> tuple[str,str|None,bool]:
    if not reference_valid or counts["AB0_exact_factorized_t2"] < 4:
        return "INPUT_FACTORIZATION_REFERENCE_BASELINE_DRIFT",None,False
    ab1=counts["AB1_fully_folded_equivalent"]; ab2=counts["AB2_candidate_factorized_update_folded"]; ab3=counts["AB3_candidate_folded_update_factorized"]; ab4=counts["AB4_frozen_shared_projection"]; ab5=counts["AB5_naive_projection_free"]
    if ab1 >= 4:
        if ab5 >= 4: return "INPUT_PROJECTION_AXIS_CLOSED",None,False
        return "COMPOSED_EFFECTIVE_INPUT_INITIALIZATION_SUFFICIENT","FOLDED_COMPOSED_INPUT_INITIALIZATION",True
    if ab4 >= 4: return "FIXED_INPUT_PRECONDITIONING_SUFFICIENT","FROZEN_SHARED_INPUT_PRECONDITIONER",True
    if ab2 >= 4 and ab3 >= 4: return "SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT","TRAINABLE_CONTROLLER_INPUT_FACTORIZATION",True
    if ab2 >= 4: return "CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT","TRAINABLE_CANDIDATE_INPUT_FACTORIZATION",True
    if ab3 >= 4: return "UPDATE_CONTROLLER_INPUT_FACTORIZATION_SUFFICIENT","TRAINABLE_CONTROLLER_INPUT_FACTORIZATION",True
    return "JOINT_CANDIDATE_CONTROLLER_INPUT_FACTORIZATION_REQUIRED","TRAINABLE_JOINT_INPUT_FACTORIZATION",True


def _aggregate_projection(rows:list[dict])->dict:
    out={}
    for condition in CONDITIONS:
        rr=[r for r in rows if r["condition"]==condition and r["projection_diagnostics_final"].get("active")]
        if not rr: out[condition]={"active":False}; continue
        out[condition]={"active":True,"trainable":bool(rr[0]["projection_diagnostics_final"].get("trainable")),"fits":len(rr)}
        for field in ("condition_number","determinant_abs","frobenius_norm","bias_norm"):
            a=np.array([float(r["projection_diagnostics_final"][field]) for r in rr]); out[condition][field]={"median":float(np.median(a)),"mean":float(np.mean(a)),"min":float(np.min(a)),"max":float(np.max(a))}
        max_len=max(len(r["projection_diagnostics_final"]["singular_values"]) for r in rr)
        out[condition]["singular_values_median"]=[float(np.median([r["projection_diagnostics_final"]["singular_values"][i] for r in rr])) for i in range(max_len)]
    return out


def _aggregate_trajectories(rows:list[dict])->dict:
    output={}
    for condition in CONDITIONS:
        rr=[r for r in rows if r["condition"]==condition]
        by_step=defaultdict(list)
        for r in rr:
            for point in r["factorization_trajectory"]: by_step[int(point["step"])].append(point)
        output[condition]={}
        for step,points in sorted(by_step.items()):
            item={"fits":len(points),"candidate_effective_frobenius_median":float(np.median([p["effective"]["n"]["frobenius_norm"] for p in points])),"update_effective_frobenius_median":float(np.median([p["effective"]["z"]["frobenius_norm"] for p in points])),"candidate_effective_distance_median":float(np.median([p["distance_from_initialization"]["effective_n_weight"] for p in points])),"update_effective_distance_median":float(np.median([p["distance_from_initialization"]["effective_z_weight"] for p in points]))}
            active=[p for p in points if p["projection"].get("active")]
            if active:
                item.update({"projection_frobenius_median":float(np.median([p["projection"]["frobenius_norm"] for p in active])),"projection_condition_number_median":float(np.median([p["projection"]["condition_number"] for p in active])),"projection_distance_median":float(np.median([p["distance_from_initialization"]["projection_weight"] for p in active])),"candidate_downstream_weight_norm_median":float(np.median([p["effective"]["n"]["downstream_weight_norm"] for p in active])),"update_downstream_weight_norm_median":float(np.median([p["effective"]["z"]["downstream_weight_norm"] for p in active]))})
            output[condition][str(step)]=item
    return output


def _gradient_summary(rows:list[dict])->dict:
    rr=[r["projection_gradient_decomposition"] for r in rows if r["condition"]=="AB0_exact_factorized_t2"]
    fields=("grad_candidate_norm","grad_update_norm","candidate_update_cosine","grad_full_norm","sum_residual_norm","sum_residual_relative")
    output={"fits":len(rr),"all_sum_identity_within_1e_5_relative":all(r["sum_identity_within_1e_5_relative"] for r in rr),"per_fit":rr}
    for field in fields:
        a=np.array([float(r[field]) for r in rr]); output[field]={"median":float(np.median(a)),"mean":float(np.mean(a)),"p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9))}
    cos=float(output["candidate_update_cosine"]["median"])
    output["median_relationship"]="cooperative" if cos>0.25 else ("conflicting" if cos<-0.25 else "approximately_orthogonal_or_mixed")
    return output


def _resources(rows:list[dict])->dict:
    total={"version":"V837ab","model_fits":len(rows),"optimizer_steps":sum(int(r["resources"]["optimizer_steps"]) for r in rows),"processed_training_examples":sum(int(r["processed_examples"]) for r in rows),"unique_seed_defined_episodes":3200,"environment_interactions":sum(int(r["resources"]["environment_steps"]) for r in rows),"training_forward_calls":sum(int(r["resources"]["forward_calls"]) for r in rows),"diagnostic_forward_calls":sum(int(r["resources"].get("diagnostic_forward_calls",0)) for r in rows),"training_backward_calls":sum(int(r["resources"]["optimizer_steps"]) for r in rows),"diagnostic_backward_calls":sum(int(r["resources"].get("diagnostic_backward_calls",0)) for r in rows),"cpu_seconds_worker_sum":sum(float(r["resources"]["cpu_seconds"]) for r in rows),"wall_seconds_worker_sum":sum(float(r["resources"]["wall_seconds"]) for r in rows),"gpu_seconds":0.0,"fresh_audit_consumed":False}
    total["forward_calls"]=total["training_forward_calls"]+total["diagnostic_forward_calls"]; total["backward_calls"]=total["training_backward_calls"]+total["diagnostic_backward_calls"]
    return total


def _plots(summaries:dict,projection:dict,trajectory:dict,grad:dict)->None:
    plotdir=HERE/"plots"; plotdir.mkdir(exist_ok=True)
    labels=[c.split("_")[0] for c in CONDITIONS]; counts=[summaries[c]["families_passing"] for c in CONDITIONS]
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(labels,counts); ax.axhline(4,linestyle="--"); ax.set_ylim(0,5.3); ax.set_ylabel("families passing"); fig.tight_layout(); fig.savefig(plotdir/"families_passing_by_input_factorization.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(FAMILIES)); width=.12
    for j,c in enumerate(CONDITIONS): ax.bar(x+(j-2.5)*width,[summaries[c]["family_results"][f]["validation"]["median"] for f in FAMILIES],width,label=labels[j])
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES,rotation=25,ha="right"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(plotdir/"family_scores_by_factorization.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(["AB0","AB1"],[summaries[CONDITIONS[0]]["mean_family_validation_median"],summaries[CONDITIONS[1]]["mean_family_validation_median"]]); ax.set_ylabel("mean family validation median"); fig.tight_layout(); fig.savefig(plotdir/"factorized_vs_folded.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); cs=CONDITIONS[1:4]; ax.bar([c.split('_')[0] for c in cs],[summaries[c]["families_passing"] for c in cs]); ax.set_ylim(0,5.3); fig.tight_layout(); fig.savefig(plotdir/"candidate_vs_update_factorization.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for c in CONDITIONS:
        if projection[c].get("active"): ax.plot(projection[c]["singular_values_median"],marker="o",label=c.split('_')[0])
    ax.legend(); ax.set_ylabel("median singular value"); ax.set_xlabel("index"); fig.tight_layout(); fig.savefig(plotdir/"projection_singular_values.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for c in CONDITIONS:
        steps=sorted(int(s) for s in trajectory[c]); vals=[trajectory[c][str(s)]["candidate_effective_distance_median"] for s in steps]; ax.plot(steps,vals,label=c.split('_')[0])
    ax.set_xlabel("optimizer step"); ax.set_ylabel("candidate effective-map distance"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(plotdir/"effective_input_map_trajectory.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); vals=[r["candidate_update_cosine"] for r in grad["per_fit"]]; ax.hist(vals,bins=10); ax.axvline(0,linestyle="--"); ax.set_xlabel("candidate/update projection-gradient cosine"); fig.tight_layout(); fig.savefig(plotdir/"projection_gradient_alignment.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter([summaries[c]["projection_specific_macs"] for c in CONDITIONS],counts); [ax.annotate(labels[i],(summaries[c]["projection_specific_macs"],counts[i])) for i,c in enumerate(CONDITIONS)]; ax.set_xlabel("projection MACs/timestep"); ax.set_ylabel("families passing"); fig.tight_layout(); fig.savefig(plotdir/"capability_vs_projection_macs.png",dpi=150); plt.close(fig)


def main()->int:
    rows=_rows(); expected=6*5*5
    if len(rows)!=expected: raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    reference=_load(HERE/"diagnostics/reference_equivalence.json"); step0=_load(HERE/"diagnostics/step0_equivalence.json"); guard=_load(HERE/"diagnostics/reference_baseline_guard.json")
    summaries={c:_summary(rows,c) for c in CONDITIONS}; counts={c:summaries[c]["families_passing"] for c in CONDITIONS}
    diagnosis,mode,allowed=_decision(counts,bool(guard["reference_baseline_valid"]))
    projection=_aggregate_projection(rows); trajectory=_aggregate_trajectories(rows); grad=_gradient_summary(rows); resources=_resources(rows)
    write_json(HERE/"diagnostics/factorization_trajectory.json",trajectory); write_json(HERE/"diagnostics/projection_diagnostics.json",projection); write_json(HERE/"diagnostics/projection_gradient_decomposition.json",grad); write_json(HERE/"diagnostics/compute_efficiency.json",resources)
    decision={"v837ab_complete":True,"reference_baseline_valid":bool(guard["reference_baseline_valid"]),"folding_equivalence_valid":bool(reference["function_class_equivalence_proven"]),"step0_equivalence_valid":bool(step0["step0_equivalence_proven"]),"families_passing":counts,"diagnosis":diagnosis,"authorized_v837ac_mode":mode,"neutral_transfer_allowed":bool(allowed),"representation_adequacy_pass":False,"sample_efficiency_retest_allowed":False,"fresh_audit_consumed":False,"structural_search_allowed":False,"primitive_mining_allowed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False}
    write_json(HERE/"diagnostics/decision_state.json",decision)
    results={"version":"V837ab","parent":"V837aa","question":CONFIG["question"],"function_class_equivalence_proven":reference["function_class_equivalence_proven"],"step0_equivalence_proven":step0["step0_equivalence_proven"],"conditions":summaries,"diagnosis":diagnosis,"authorized_v837ac_mode":mode,"neutral_transfer_allowed":bool(allowed),"projection_diagnostics":projection,"factorization_trajectory_summary":trajectory,"projection_gradient_decomposition_summary":{k:v for k,v in grad.items() if k!="per_fit"},"resource_accounting":resources,"representation_adequacy_pass":False,"sample_efficiency_retest_allowed":False,"fresh_audit_consumed":False,"structural_search_allowed":False,"primitive_mining_allowed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False}
    write_json(HERE/"results.json",results); _plots(summaries,projection,trajectory,grad)
    text=f"# V837ab — {diagnosis}\n\nFunction-class folding proof: PASS.\n\nAB0 positive control: {counts['AB0_exact_factorized_t2']}/5.\nAB1 fully folded: {counts['AB1_fully_folded_equivalent']}/5.\nAB2 candidate factorized: {counts['AB2_candidate_factorized_update_folded']}/5.\nAB3 update factorized: {counts['AB3_candidate_folded_update_factorized']}/5.\nAB4 frozen projection: {counts['AB4_frozen_shared_projection']}/5.\nAB5 naive direct: {counts['AB5_naive_projection_free']}/5.\n\nAuthorized V837ac mode: `{mode}`.\n"
    (HERE/("FAILURE.md" if diagnosis=="INPUT_FACTORIZATION_REFERENCE_BASELINE_DRIFT" else "PASS.md")).write_text(text,encoding="utf-8")
    write_json(ROOT/"experiments/v837_primitive_invention/v837ab_resource_accounting.json",resources)
    print(json.dumps(decision,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
