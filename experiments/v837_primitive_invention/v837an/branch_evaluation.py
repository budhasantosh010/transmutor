from __future__ import annotations

import math
import numpy as np

from experiments.v837_primitive_invention.tasks import task_by_name

from .an_a_causal_macrovariables import _carrier_values, _eligibility, _semantic_values, _metrics
from .an_b_causal_routing import _paired_metrics, _random_sets
from .authorization import CARRIERS, LATENT_DIMS, MIN_ELIGIBLE, PARTITIONS, PRIMARY_PROBES, SOURCE_SWAP_THRESHOLDS
from .carrier_spaces import CARRIER_DIM
from .causal_interventions import coalition_state_predictions, many_patched_predictions, patched_predictions
from .causal_metrics import direction_agreement, paired_sign_flip_p, recovery, success_vector
from .counterfactual_subspace import projected_delta
from .ood_diagnostics import compare as ood_compare, fit_reference
from .random_controls import random_subspaces
from .routing_interventions import channel_universe, routing_forward_reverse
from .semantic_compiler import compile_delta
from .trace_cache import pair_traces
from .utils import HERE, deterministic_seed, range_list, read_json


def _fit_record(organism_id:str,config_id:str)->dict:
    rows=read_json(HERE/"raw/an_a_fit.json")["records"]
    return next(r for r in rows if r["organism_id"]==organism_id and r["config_id"]==config_id)


def _partition_min(partition:str)->int:
    if partition=="AN_META_CONFIRM":return 24
    if partition=="AN_FINAL_DEV":return 24
    if partition=="AN_FINAL_VALIDATION":return 32
    return MIN_ELIGIBLE.get(partition,24)


def evaluate_a(organism:dict,config_id:str,partition:str,*,use_compiler:bool)->dict:
    record=_fit_record(organism["organism_id"],config_id);carrier=record["carrier"];q=np.asarray(record["q"],dtype=np.float64);compiler={"q":q,"coef":np.asarray(record["semantic_compiler"]["coef"],dtype=np.float64),"ridge_lambda":1e-6,"intercept":0.0}
    seeds=range_list(PARTITIONS[partition]);data=pair_traces(organism["organism_id"],organism["family"],seeds);mask,power=_eligibility(data,organism["family"]);idx=np.flatnonzero(mask)
    if len(idx)<_partition_min(partition):return {"eligible":len(idx),"powered":False,"pass":False,"reason":"ORGANISM_COUNTERFACTUAL_POWER_INSUFFICIENT"}
    times=np.asarray([p.primary_phase for p in data["pairs"]],dtype=np.int64);sbase=_carrier_values(data["base_trace"],carrier,times)[idx];scf=_carrier_values(data["cf_trace"],carrier,times)[idx];delta=scf-sbase
    if use_compiler:
        semantic=_semantic_values(data["pairs"],times,True)[idx]-_semantic_values(data["pairs"],times,False)[idx];d=compile_delta(semantic,compiler)
    else:d=projected_delta(delta,q)
    fv=sbase+d;rv=scf-d;bo=data["base_obs"][idx];bl=data["base_lengths"][idx];co=data["cf_obs"][idx];cl=data["cf_lengths"][idx];t=times[idx];model=data["model"]
    pf=patched_predictions(model,bo,bl,carrier,fv,t);pr=patched_predictions(model,co,cl,carrier,rv,t)
    controls=random_subspaces(CARRIER_DIM[carrier],q.shape[1],32,deterministic_seed("v837an-frozen-a-controls",organism["organism_id"],config_id,partition));fsets=np.stack([sbase+projected_delta(delta,rq) for rq in controls]);rsets=np.stack([scf+projected_delta(-delta,rq) for rq in controls]);rf=many_patched_predictions(model,bo,bl,carrier,fsets,t);rr=many_patched_predictions(model,co,cl,carrier,rsets,t)
    fit=pair_traces(organism["organism_id"],organism["family"],range_list(PARTITIONS["AN_FIT"]));fm,_=_eligibility(fit,organism["family"]);fi=np.flatnonzero(fm);ft=np.asarray([p.primary_phase for p in fit["pairs"]],dtype=np.int64);fb=_carrier_values(fit["base_trace"],carrier,ft)[fi];fc=_carrier_values(fit["cf_trace"],carrier,ft)[fi];ref=fit_reference(np.concatenate([fb,fc]));oodf=ood_compare(fv,scf,ref);oodr=ood_compare(rv,sbase,ref)
    bp=data["base_prediction"].detach().cpu().numpy()[idx];cp=data["cf_prediction"].detach().cpu().numpy()[idx];bt=data["base_targets"].detach().cpu().numpy()[idx];ct=data["cf_targets"].detach().cpu().numpy()[idx];m=_metrics(task_by_name(organism["family"]),bp,cp,pf,pr,bt,ct,rf,rr,oodf,oodr,deterministic_seed("v837an-frozen-a-p",organism["organism_id"],config_id,partition,"compiler" if use_compiler else "source"));m={kk:vv for kk,vv in m.items() if kk not in {"recovery_values","random_pair_median"}};m.update({"eligible":len(idx),"powered":True,"config_id":config_id,"carrier":carrier,"k":q.shape[1],"semantic_compiler":use_compiler});return m


def evaluate_b(organism:dict,config_id:str,partition:str)->dict:
    source=read_json(HERE/"raw/routing_selection.json");org=next(r for r in source["organism_results"] if r["organism_id"]==organism["organism_id"]);channels=tuple(org["configs"][config_id]["channels"])
    data=pair_traces(organism["organism_id"],organism["family"],range_list(PARTITIONS[partition]));mask,_=_eligibility(data,organism["family"]);idx=np.flatnonzero(mask)
    if len(idx)<_partition_min(partition):return {"eligible":len(idx),"powered":False,"pass":False,"reason":"ORGANISM_COUNTERFACTUAL_POWER_INSUFFICIENT"}
    f,r,_,_=routing_forward_reverse(data,[channels],idx);universe=channel_universe(data["model"])
    if config_id.startswith("MESSAGE_"):cu=[x for x in universe if x.startswith("M:")]
    elif config_id.startswith("GLOBAL_"):cu=[x for x in universe if x.startswith("G:")]
    else:cu=universe
    controls=_random_sets(cu,len(channels),64,deterministic_seed("v837an-frozen-b-controls",organism["organism_id"],config_id,partition));rf,rr,_,_=routing_forward_reverse(data,controls,idx);m=_paired_metrics(data,idx,f[0],r[0],rf,rr,organism["family"],deterministic_seed("v837an-frozen-b-p",organism["organism_id"],config_id,partition));m.update({"eligible":len(idx),"powered":True,"config_id":config_id,"channels":list(channels),"channel_count":len(channels),"intervention_dof":org["configs"][config_id]["intervention_dof"]});return m


def _state_values(data,idx):
    times=np.asarray([p.primary_phase for p in data["pairs"]],dtype=np.int64);x=data["base_trace"].states.flatten(2);y=data["cf_trace"].states.flatten(2);rows=np.asarray(idx,dtype=np.int64);return x[rows,times[rows]].detach().cpu().numpy(),y[rows,times[rows]].detach().cpu().numpy(),times[rows]


def evaluate_c(organism:dict,nodes:list[int],partition:str)->dict:
    data=pair_traces(organism["organism_id"],organism["family"],range_list(PARTITIONS[partition]));mask,_=_eligibility(data,organism["family"]);idx=np.flatnonzero(mask)
    if len(idx)<_partition_min(partition):return {"eligible":len(idx),"powered":False,"pass":False,"reason":"ORGANISM_COUNTERFACTUAL_POWER_INSUFFICIENT"}
    b,c,t=_state_values(data,idx);bo=data["base_obs"][idx];bl=data["base_lengths"][idx];co=data["cf_obs"][idx];cl=data["cf_lengths"][idx];subset=tuple(int(x) for x in nodes);f=coalition_state_predictions(data["model"],bo,bl,b,c,t,[subset])[0];r=coalition_state_predictions(data["model"],co,cl,c,b,t,[subset])[0]
    bp=data["base_prediction"].detach().cpu().numpy()[idx];cp=data["cf_prediction"].detach().cpu().numpy()[idx];bt=data["base_targets"].detach().cpu().numpy()[idx];ct=data["cf_targets"].detach().cpu().numpy()[idx];task=task_by_name(organism["family"]);rec=np.concatenate([recovery(bp,cp,f),recovery(cp,bp,r)]);direction=np.concatenate([direction_agreement(bp,cp,f),direction_agreement(cp,bp,r)]);success=np.concatenate([success_vector(task,f,ct),success_vector(task,r,bt)])
    # Matched control is the strongest singleton member of the frozen coalition.
    singleton_sets=[(int(cell),) for cell in subset];sf=coalition_state_predictions(data["model"],bo,bl,b,c,t,singleton_sets);sr=coalition_state_predictions(data["model"],co,cl,c,b,t,singleton_sets);controls=[]
    for fi,ri in zip(sf,sr):controls.append(np.concatenate([recovery(bp,cp,fi),recovery(cp,bp,ri)]))
    control=np.max(np.stack(controls),axis=0) if controls else np.zeros_like(rec);margin=float(np.median(rec)-np.median(control));p=paired_sign_flip_p(rec-control,deterministic_seed("v837an-frozen-c-p",organism["organism_id"],subset,partition))
    fit=pair_traces(organism["organism_id"],organism["family"],range_list(PARTITIONS["AN_FIT"]));fm,_=_eligibility(fit,organism["family"]);fi=np.flatnonzero(fm);fb,fc,_=_state_values(fit,fi);ref=fit_reference(np.concatenate([fb,fc]));pf=b.copy();pr=c.copy()
    for cell in subset:pf[:,4*cell:4*cell+4]=c[:,4*cell:4*cell+4];pr[:,4*cell:4*cell+4]=b[:,4*cell:4*cell+4]
    ood=np.concatenate([ood_compare(pf,c,ref)["ood_ratio"],ood_compare(pr,b,ref)["ood_ratio"]])
    return {"eligible":len(idx),"powered":True,"nodes":list(subset),"cardinality":len(subset),"intervention_dof":4*len(subset),"median_recovery":float(np.median(rec)),"direction_agreement":float(np.mean(direction)),"counterfactual_success":float(np.mean(success)),"control_median_recovery":float(np.median(control)),"control_margin":margin,"paired_p":float(p),"median_ood_ratio":float(np.median(ood))}


def gate_result(metrics:dict,*,p_max:float,require_control_margin:bool,require_ood:bool=True)->bool:
    if not metrics.get("powered"):return False
    passed=metrics.get("median_recovery",-999)>=.60 and metrics.get("direction_agreement",-999)>=.75 and metrics.get("counterfactual_success",-999)>=.70 and metrics.get("paired_p",999)<=p_max
    margin=metrics.get("random_margin",metrics.get("control_margin",-999))
    if require_control_margin:passed=passed and margin>=.20
    if require_ood and "median_ood_ratio" in metrics:passed=passed and metrics["median_ood_ratio"]<=2.0
    return bool(passed)


def aggregate_family(organisms:list[dict],rows:list[dict],*,p_max:float,require_control_margin:bool)->dict:
    powered=[r for r in rows if r.get("powered")];passed=[r for r in powered if gate_result(r,p_max=p_max,require_control_margin=require_control_margin)];source_engines={o["engine"] for o in organisms};pass_engines={r["engine"] for r in passed if "engine" in r};n=len(powered);required=max(1,math.ceil(.60*n));return {"powered_organisms":n,"organisms_passing":len(passed),"required_passes":required,"pass_fraction":0 if n==0 else len(passed)/n,"family_pass":n>=5 and len(passed)>=required and (len(source_engines)<2 or len(pass_engines)>=2),"pass_engines":sorted(pass_engines)}
