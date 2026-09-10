from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.v837an.instrumented_af1d import PatchPlan, TensorPatch, load_model_by_id, run_instrumented

from .data_roles import seeds
from .failure_ledger import add, make_entry
from .natural_interventions import OperatorCase, build_first_order_cases
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, deterministic_seed, read_json, write_json

CHANNELS=("cell_output","aggregate_message","global_term","global_gate")
PREFIXES=(1,2,4,8)


def _primary_case(family:str, seed:int)->OperatorCase:
    rows=build_first_order_cases(family,[seed])
    if family=="conditional_routing": return next(r for r in rows if r.intervention=="CONTROL_FLIP")
    if family=="delayed_recall": return next(r for r in rows if r.intervention=="WRITE_FLIP")
    if family=="iterative_state":
        cand=[r for r in rows if r.intervention=="INPUT_PERTURB" and r.phase=="MIDDLE" and r.horizon==2]
        return max(cand,key=lambda r:abs(float(r.actual_delta or 0.0)))
    if family=="variable_composition":
        cand=[r for r in rows if r.intervention=="INITIAL_VALUE_PERTURB"]
        return max(cand,key=lambda r:abs(float(r.actual_delta or 0.0)))
    raise KeyError(family)


def _phase_names(family:str)->tuple[str,...]:
    if family=="conditional_routing":return ("CONTROL","PAYLOAD_A","PAYLOAD_B","POST")
    if family=="delayed_recall":return ("WRITE","EARLY_DELAY","MID_DELAY","LATE_DELAY","READ")
    return ("EARLY","MIDDLE","LATE")


def _phase_for(family:str,t:int,T:int)->str:
    if family=="conditional_routing":
        if t==1:return "CONTROL"
        if t==2:return "PAYLOAD_A"
        if t==3:return "PAYLOAD_B"
        return "POST"
    if family=="delayed_recall":
        if t==1:return "WRITE"
        if t>=T-1:return "READ"
        pos=t-2; n=max(T-3,1); third=max(int(np.ceil(n/3)),1)
        if pos<third:return "EARLY_DELAY"
        if pos<2*third:return "MID_DELAY"
        return "LATE_DELAY"
    # Ignore prelude; divide active computational timesteps into thirds.
    pos=max(t-1,0);n=max(T-1,1);third=max(int(np.ceil(n/3)),1)
    if pos<third:return "EARLY"
    if pos<2*third:return "MIDDLE"
    return "LATE"


def _units(family:str)->list[dict]:
    rows=[]
    for phase in _phase_names(family):
        for channel in ("cell_output","aggregate_message","global_term"):
            for cell in range(10):rows.append({"channel":channel,"cell":cell,"phase":phase,"unit_id":f"{channel}:c{cell}:{phase}"})
        rows.append({"channel":"global_gate","cell":None,"phase":phase,"unit_id":f"global_gate:{phase}"})
    return rows


def _trace_source(trace,channel:str):
    if channel=="cell_output":return trace.outputs
    if channel=="aggregate_message":return trace.messages
    if channel=="global_term":return trace.global_terms
    if channel=="global_gate":return trace.gate
    raise KeyError(channel)


def _build_plan(unit_sets:list[list[dict]], cases:list[OperatorCase], cf_trace, lengths:torch.Tensor, *, time_shift:int=0)->PatchPlan:
    C=len(unit_sets);B=len(cases);T=int(cf_trace.outputs.shape[1]);plan=PatchPlan();lens=lengths.detach().cpu().numpy().astype(int)
    for t in range(T):
        for channel in CHANNELS:
            if channel=="global_gate":
                vals=torch.zeros((C*B,1),dtype=cf_trace.gate.dtype);mask=torch.zeros((C*B,1),dtype=torch.bool)
            else:
                src=_trace_source(cf_trace,channel);vals=torch.zeros((C*B,10,4),dtype=src.dtype);mask=torch.zeros((C*B,10,4),dtype=torch.bool)
            any_mask=False
            for ci,uset in enumerate(unit_sets):
                for bi,case in enumerate(cases):
                    L=int(lens[bi]); donor_t=t
                    if time_shift and L>1:donor_t=(t+time_shift)%L
                    for unit in uset:
                        if unit["channel"]!=channel or t>=L or _phase_for(case.family,t,L)!=unit["phase"]:continue
                        row=ci*B+bi
                        if channel=="global_gate":
                            vals[row]=cf_trace.gate[bi,donor_t];mask[row]=True
                        else:
                            src=_trace_source(cf_trace,channel);vals[row,unit["cell"],:]=src[bi,donor_t,unit["cell"],:];mask[row,unit["cell"],:]=True
                        any_mask=True
            if any_mask:
                patch=TensorPatch(vals,mask)
                if channel=="cell_output":plan.output[t]=patch
                elif channel=="aggregate_message":plan.message[t]=patch
                elif channel=="global_term":plan.global_term[t]=patch
                else:plan.gate[t]=patch
    return plan


def _run_sets(model,cases:list[OperatorCase],unit_sets:list[list[dict]],cf_trace,lengths,time_shift:int=0)->np.ndarray:
    B=len(cases);C=len(unit_sets);obs,base_lengths,_=episodes_to_batch([c.base_episode for c in cases]);obs=obs.repeat((C,1,1));lens=base_lengths.repeat(C);plan=_build_plan(unit_sets,cases,cf_trace,base_lengths,time_shift=time_shift)
    with torch.no_grad():pred=run_instrumented(model,obs,lens,patch_plan=plan,return_trace=False)
    return pred.detach().cpu().numpy().astype(np.float64).reshape(C,B)


def _preservation(full:np.ndarray,patched:np.ndarray)->dict:
    full=np.asarray(full,dtype=float);patched=np.asarray(patched,dtype=float);effect=np.abs(full)>=.05
    if not np.any(effect):effect=np.ones(len(full),dtype=bool)
    f=full[effect];p=patched[effect];den=float(np.sqrt(np.mean(f*f))+1e-8);score=float(1.0-np.sqrt(np.mean((p-f)**2))/den);per=1.0-np.abs(p-f)/(np.abs(f)+1e-8)
    return {"preservation":score,"p25_case_preservation":float(np.quantile(per,.25)),"median_case_preservation":float(np.median(per)),"effect_cases":int(effect.sum()),"all_cases":int(len(full))}


def _evaluate_organism(oid:str,family:str,rank_seeds:list[int],select_seeds:list[int])->dict:
    model,row,_,_=load_model_by_id(oid)
    def traces(seed_values):
        cases=[_primary_case(family,s) for s in seed_values];bo,bl,_=episodes_to_batch([c.base_episode for c in cases]);co,cl,_=episodes_to_batch([c.intervened_episode for c in cases])
        with torch.no_grad():bp,bt=run_instrumented(model,bo,bl,return_trace=True);cp,ct=run_instrumented(model,co,cl,return_trace=True)
        return cases,bl,bp.detach().cpu().numpy().astype(np.float64),cp.detach().cpu().numpy().astype(np.float64),ct
    rcases,rlens,rb,rc,rct=traces(rank_seeds);full_r=rc-rb;units=_units(family)
    rank_rows=[]
    for channel in CHANNELS:
        cu=[u for u in units if u["channel"]==channel]
        preds=_run_sets(model,rcases,[[u] for u in cu],rct,rlens)
        for unit,p in zip(cu,preds):rank_rows.append({**unit,**_preservation(full_r,p-rb)})
    rank_rows.sort(key=lambda r:(-r["preservation"],r["unit_id"]));ordered=[{k:v for k,v in r.items() if k in {"channel","cell","phase","unit_id"}} for r in rank_rows]
    scases,slens,sb,sc,sct=traces(select_seeds);full_s=sc-sb;prefix_rows=[];selected=None
    for n in PREFIXES:
        uset=ordered[:n];pred=_run_sets(model,scases,[uset],sct,slens)[0];met=_preservation(full_s,pred-sb);entry={"n":n,"units":uset,**met,"pass":bool(met["preservation"]>=.80 and met["p25_case_preservation"]>=.60)};prefix_rows.append(entry)
        if selected is None and entry["pass"]:selected=entry
    if selected is not None:
        n=selected["n"];rng=np.random.default_rng(deterministic_seed("v837aq-localization-random",oid,family));random_sets=[]
        for _ in range(32):idx=rng.choice(len(units),size=n,replace=False);random_sets.append([units[int(i)] for i in idx])
        rpred=_run_sets(model,scases,random_sets,sct,slens);rpres=np.asarray([_preservation(full_s,p-sb)["preservation"] for p in rpred],dtype=float)
        tsh=_run_sets(model,scases,[selected["units"]],sct,slens,time_shift=1)[0];tmet=_preservation(full_s,tsh-sb);rmed=float(np.median(rpres));controls={"random_mask_count":32,"random_mask_median_preservation":rmed,"time_shuffled_preservation":tmet["preservation"],"random_margin":selected["preservation"]-rmed,"time_shuffle_margin":selected["preservation"]-tmet["preservation"],"pass":bool(selected["preservation"]-rmed>=.20 and selected["preservation"]-tmet["preservation"]>=.20)}
    else:controls={"random_mask_count":0,"random_mask_median_preservation":None,"time_shuffled_preservation":None,"random_margin":None,"time_shuffle_margin":None,"pass":False}
    passed=bool(selected is not None and controls["pass"]);phase_span=len({u["phase"] for u in selected["units"]}) if selected else None;scalar_slots=sum((1 if u["channel"]=="global_gate" else 4) for u in selected["units"]) if selected else None
    return {"organism_id":oid,"family":family,"engine":row["engine"],"rank_rows":rank_rows,"prefixes":prefix_rows,"selected":selected,"controls":controls,"active_component_phase_units":selected["n"] if selected else None,"active_scalar_phase_slots":scalar_slots,"phase_span":phase_span,"modeled_active_macs":None,"modeled_active_macs_exact":False,"pass":passed}


def run_localization()->dict:
    discovery=read_json(HERE/"raw/operator_discovery.json");families=discovery.get("accepted_families",[]);allseeds=seeds("AQ_LOCALIZATION");rank=allseeds[:32];select=allseeds[32:];payload={"version":"V837aq","stage":"AQ6_AQ7_SPATIOTEMPORAL_LOCALIZATION","rank_seeds":[rank[0],rank[-1]],"select_seeds":[select[0],select[-1]],"families":{},"rows":[],"paired_natural_only":True,"hidden_state_set_used":False}
    for family in families:
        rows=[]
        for i,oid in enumerate(discovery_ids(family),1):
            r=_evaluate_organism(oid,family,rank,select);rows.append(r);payload["rows"].append(r);print(f"V837aq localization {family} {i}/{len(discovery_ids(family))} pass={r['pass']} n={r['active_component_phase_units']} pres={(r['selected'] or {}).get('preservation')}",flush=True)
            if not r["pass"]:add(make_entry(failure_id=f"V837aq-AQ6-{family}-{oid[:14]}",stage="AQ6_AQ7_LOCALIZATION",branch="PAIRED_NATURAL_COMPONENT_PHASE",family=family,organism=oid,intervention="PRIMARY_OPERATOR_COUNTERFACTUAL",operator_order=discovery["families"][family]["operator_order"],metrics={"prefixes":r["prefixes"],"controls":r["controls"]},acceptance_gate={"preservation_min":.80,"p25_min":.60,"random_margin_min":.20,"time_shuffle_margin_min":.20,"max_units":8},failed_conditions=["no compact controlled support"] if r["selected"] is None else ["localization controls"],scientific_interpretation="The accepted external operator was not reproduced by a compact, specificity-controlled paired-natural component×phase support under the frozen localization search.",confounds_ruled_out=["arbitrary hidden-state SET","evolutionary structure search","outcome-based ranking"],confounds_remaining=["broader organism-scale distributed realization"],next_justified_experiment="Retain the external operator result; do not promote this localization as compact."))
        gate=family_gate(rows);compact=bool(gate["pass"]);payload["families"][family]={"gate":gate,"compact_spatiotemporal_support":compact,"median_selected_units":float(np.median([r["selected"]["n"] for r in rows if r["selected"]])) if any(r["selected"] for r in rows) else None}
    write_json(HERE/"raw/spatiotemporal_localization.json",payload);write_json(HERE/"diagnostics/spatiotemporal_localization.json",payload);write_json(HERE/"diagnostics/active_support_minimality.json",{"version":"V837aq","families":payload["families"],"storage_not_equal_active_computation":True,"modeled_active_macs_not_fabricated":True})
    return payload


if __name__=="__main__":print(json.dumps(run_localization(),indent=2))
