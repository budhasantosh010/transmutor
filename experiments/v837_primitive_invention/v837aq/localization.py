from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.v837an.instrumented_af1d import PatchPlan, TensorPatch, load_model_by_id, run_instrumented

from .data_roles import seeds
from .natural_interventions import OperatorCase, build_first_order_cases
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, deterministic_seed, read_json, write_json

CONFIG=read_json(HERE/"config.json")
POLICY=HERE/"frozen_localization_policy.json"

PHASES={
    "conditional_routing":["CONTROL","PAYLOAD_A","PAYLOAD_B","POST_PAYLOAD"],
    "delayed_recall":["WRITE","HOLD_EARLY","HOLD_MIDDLE","HOLD_LATE","READ"],
    "iterative_state":["EARLY","MIDDLE","LATE"],
}
CHANNELS=("cell_output","aggregate_message","global_term","global_gate")


def freeze_localization_policy()->dict:
    if POLICY.is_file():return read_json(POLICY)
    payload={
        "version":"V837aq","stage":"AQ6_AQ7","selection_partition":"AQ_LOCALIZATION",
        "ranking_seeds":[11256,11287],"verification_seeds":[11288,11319],
        "primary_counterfactuals":{"conditional_routing":"CONTROL_FLIP","delayed_recall":"WRITE_FLIP","iterative_state":"MIDDLE_INPUT_PERTURB_MAX_HORIZON"},
        "oracle_effect_eligibility_min":0.05,"phase_groups":PHASES,"channels":list(CHANNELS),
        "candidate_unit":"channel x destination-cell(if applicable) x phase",
        "patch_values":"paired-natural counterfactual contributions only; no fabricated hidden state",
        "ranking":"descending median response preservation on ranking half only; deterministic lexical tie break",
        "prefix_sizes":CONFIG["localization_gate"]["prefix_sizes"],"random_masks":16,
        "gate":CONFIG["localization_gate"],"evolutionary_search":False,"hidden_state_set":False,
    }
    write_json(POLICY,payload);return payload


def _primary_case(family:str,seed:int)->OperatorCase|None:
    rows=build_first_order_cases(family,[seed])
    if family=="conditional_routing":cand=[r for r in rows if r.intervention=="CONTROL_FLIP"]
    elif family=="delayed_recall":cand=[r for r in rows if r.intervention=="WRITE_FLIP"]
    elif family=="iterative_state":
        cand=[r for r in rows if r.intervention=="INPUT_PERTURB" and r.phase=="MIDDLE"]
        if not cand:cand=[r for r in rows if r.intervention=="INPUT_PERTURB"]
        cand=sorted(cand,key=lambda r:(r.horizon,abs(float(r.actual_delta or 0.0))),reverse=True)
    else:raise KeyError(family)
    if not cand:return None
    row=cand[0]
    if abs(float(row.oracle_response))<float(freeze_localization_policy()["oracle_effect_eligibility_min"]):return None
    return row


def _phase_labels(family:str,length:int)->list[str|None]:
    labels=[None]*length
    if family=="conditional_routing":
        if length>1:labels[1]="CONTROL"
        if length>2:labels[2]="PAYLOAD_A"
        if length>3:labels[3]="PAYLOAD_B"
        for t in range(4,length):labels[t]="POST_PAYLOAD"
    elif family=="delayed_recall":
        if length>1:labels[1]="WRITE"
        if length>2:labels[-1]="READ"
        hold=list(range(2,max(2,length-1)))
        chunks=np.array_split(np.asarray(hold,dtype=int),3) if hold else []
        for phase,chunk in zip(("HOLD_EARLY","HOLD_MIDDLE","HOLD_LATE"),chunks):
            for t in chunk.tolist():labels[int(t)]=phase
    elif family=="iterative_state":
        steps=list(range(1,length));chunks=np.array_split(np.asarray(steps,dtype=int),3) if steps else []
        for phase,chunk in zip(("EARLY","MIDDLE","LATE"),chunks):
            for t in chunk.tolist():labels[int(t)]=phase
    else:raise KeyError(family)
    return labels


def _candidate_units(family:str)->list[dict]:
    units=[]
    for phase in PHASES[family]:
        for kind in CHANNELS:
            if kind=="global_gate":units.append({"kind":kind,"cell":None,"phase":phase})
            else:
                for cell in range(10):units.append({"kind":kind,"cell":cell,"phase":phase})
    return units


def _prepare_batch(organism_id:str,family:str,seed_values:list[int])->dict:
    cases=[c for s in seed_values if (c:=_primary_case(family,int(s))) is not None]
    if not cases:raise RuntimeError(f"V837AQ_NO_LOCALIZATION_CASES:{organism_id}:{family}")
    model,row,_,_=load_model_by_id(organism_id)
    base=[c.base_episode for c in cases];cf=[c.intervened_episode for c in cases]
    bo,bl,_=episodes_to_batch(base);co,cl,_=episodes_to_batch(cf)
    if not torch.equal(bl,cl):raise RuntimeError("V837AQ_LOCALIZATION_LENGTH_DRIFT")
    with torch.no_grad():
        bp,btr=run_instrumented(model,bo,bl,return_trace=True);cp,ctr=run_instrumented(model,co,cl,return_trace=True)
    B,T=bo.shape[:2];members={p:torch.zeros(B,T,dtype=torch.bool) for p in PHASES[family]}
    for i,c in enumerate(cases):
        for t,p in enumerate(_phase_labels(family,int(bl[i].item()))):
            if p in members:members[p][i,t]=True
    return {"model":model,"source":row,"cases":cases,"obs":bo,"lengths":bl,"base_pred":bp.detach(),"cf_pred":cp.detach(),"base_trace":btr,"cf_trace":ctr,"members":members}


def _build_plan(batch:dict,units:list[dict])->PatchPlan:
    ctr=batch["cf_trace"];B,T=ctr.states.shape[:2];plan=PatchPlan()
    by_kind={k:{} for k in CHANNELS}
    for u in units:
        phase=u["phase"];member=batch["members"][phase]
        for t in range(T):
            active=member[:,t]
            if not bool(active.any()):continue
            kind=u["kind"]
            if kind=="global_gate":
                mask=by_kind[kind].setdefault(t,torch.zeros(B,1,dtype=torch.bool));mask[:,0]|=active
            else:
                mask=by_kind[kind].setdefault(t,torch.zeros(B,10,4,dtype=torch.bool));mask[:,int(u["cell"]),:]|=active[:,None]
    for kind,items in by_kind.items():
        for t,mask in items.items():
            if kind=="cell_output":plan.output[t]=TensorPatch(ctr.outputs[:,t,:,:].detach(),mask)
            elif kind=="aggregate_message":plan.message[t]=TensorPatch(ctr.messages[:,t,:,:].detach(),mask)
            elif kind=="global_term":plan.global_term[t]=TensorPatch(ctr.global_terms[:,t,:,:].detach(),mask)
            elif kind=="global_gate":plan.gate[t]=TensorPatch(ctr.gate[:,t,:].detach(),mask)
    return plan


def _preservation(batch:dict,units:list[dict])->dict:
    with torch.no_grad():pred=run_instrumented(batch["model"],batch["obs"],batch["lengths"],patch_plan=_build_plan(batch,units),return_trace=False)
    full=(batch["cf_pred"]-batch["base_pred"]).detach().cpu().numpy().astype(float);got=(pred-batch["base_pred"]).detach().cpu().numpy().astype(float)
    per=1.0-np.abs(got-full)/(np.abs(full)+1e-8)
    vector=1.0-float(np.linalg.norm(got-full))/(float(np.linalg.norm(full))+1e-8)
    return {"n":len(per),"vector_preservation":vector,"median_preservation":float(np.median(per)),"p25_preservation":float(np.quantile(per,.25)),"mean_abs_response_error":float(np.mean(np.abs(got-full))),"full_response_rms":float(np.sqrt(np.mean(full*full))),"patched_response_rms":float(np.sqrt(np.mean(got*got)))}


def _shift_units(family:str,units:list[dict])->list[dict]:
    phases=PHASES[family];out=[]
    for u in units:
        j=phases.index(u["phase"]);out.append({**u,"phase":phases[(j+1)%len(phases)]})
    return out


def _random_masks(family:str,k:int,selected:set[tuple],batch:dict,organism_id:str)->list[dict]:
    pool=_candidate_units(family);rng=np.random.default_rng(deterministic_seed("v837aq-localization-random",family,organism_id,k));rows=[]
    for i in range(16):
        chosen=[]
        available=[u for u in pool if (u["kind"],u["cell"],u["phase"]) not in selected]
        if len(available)<k:available=pool
        idx=rng.choice(len(available),size=k,replace=False);chosen=[available[int(j)] for j in idx]
        rows.append({"index":i,"units":chosen,"metrics":_preservation(batch,chosen)})
    return rows


def verify_frozen_units(organism_id:str,family:str,units:list[dict],seed_values:list[int])->dict:
    batch=_prepare_batch(organism_id,family,seed_values);m=_preservation(batch,units);gate=freeze_localization_policy()["gate"]
    m["pass"]=bool(m["median_preservation"]>=gate["response_preservation_median_min"] and m["p25_preservation"]>=gate["response_preservation_p25_min"])
    return m


def localize_organism(organism_id:str,family:str)->dict:
    policy=freeze_localization_policy();rank_batch=_prepare_batch(organism_id,family,list(range(11256,11288)));verify_batch=_prepare_batch(organism_id,family,list(range(11288,11320)))
    singles=[]
    for u in _candidate_units(family):
        m=_preservation(rank_batch,[u]);singles.append({"unit":u,"metrics":m})
    singles.sort(key=lambda r:(-r["metrics"]["median_preservation"],r["unit"]["kind"],-1 if r["unit"]["cell"] is None else r["unit"]["cell"],r["unit"]["phase"]))
    prefixes=[];winner=None
    for k in policy["prefix_sizes"]:
        units=[r["unit"] for r in singles[:int(k)]];m=_preservation(verify_batch,units);selected={(u["kind"],u["cell"],u["phase"]) for u in units};rnd=_random_masks(family,int(k),selected,verify_batch,organism_id);random_best=max(r["metrics"]["median_preservation"] for r in rnd);shift=_preservation(verify_batch,_shift_units(family,units));margins={"random_mask":m["median_preservation"]-random_best,"time_shuffle":m["median_preservation"]-shift["median_preservation"]}
        passed=bool(m["median_preservation"]>=policy["gate"]["response_preservation_median_min"] and m["p25_preservation"]>=policy["gate"]["response_preservation_p25_min"] and margins["random_mask"]>=policy["gate"]["random_mask_margin_min"] and margins["time_shuffle"]>=policy["gate"]["time_shuffle_margin_min"])
        row={"k":int(k),"units":units,"metrics":m,"random_control_best_median":random_best,"time_shuffled_metrics":shift,"margins":margins,"pass":passed};prefixes.append(row)
        if passed and winner is None:winner=row;break
    return {"version":"V837aq","organism_id":organism_id,"family":family,"engine":rank_batch["source"]["engine"],"candidate_units":len(singles),"ranking_top16":singles[:16],"prefixes":prefixes,"winner":winner,"compact_support_pass":winner is not None,"hidden_state_set_used":False,"paired_natural_patch_used":True}


def run_localization()->dict:
    from .operator_discovery import run_operator_discovery
    discovery=read_json(HERE/"raw/operator_discovery.json") if (HERE/"raw/operator_discovery.json").is_file() else run_operator_discovery();families=list(discovery.get("accepted_families",[]));payload={"version":"V837aq","stage":"AQ6_AQ7_SPATIOTEMPORAL_LOCALIZATION","families":{},"rows":[],"accepted_operator_families":families,"hidden_state_set_used":False}
    for family in families:
        rows=[]
        for i,oid in enumerate(discovery_ids(family),1):
            r=localize_organism(oid,family);rows.append(r);payload["rows"].append({k:v for k,v in r.items() if k!="ranking_top16"});print(f"V837aq localize {family} {i}/{len(discovery_ids(family))} compact={r['compact_support_pass']} k={None if r['winner'] is None else r['winner']['k']}",flush=True)
        gate=family_gate([{"pass":r["compact_support_pass"],"engine":r["engine"]} for r in rows]);payload["families"][family]={"compact_support_gate":gate,"pass":gate["pass"],"median_winner_units":float(np.median([r["winner"]["k"] for r in rows if r["winner"]])) if any(r["winner"] for r in rows) else None,"organism_rows":rows}
    write_json(HERE/"raw/localization_results.json",payload);write_json(HERE/"diagnostics/spatiotemporal_localization.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_localization(),indent=2))
