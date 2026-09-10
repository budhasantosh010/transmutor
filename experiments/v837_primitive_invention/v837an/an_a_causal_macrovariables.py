from __future__ import annotations

from collections import defaultdict
import json
import math
import time

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name

from .authorization import CARRIERS, FAMILY_SUPPORT, LATENT_DIMS, MIN_ELIGIBLE, PARTITIONS, PRIMARY_PROBES, SOURCE_SWAP_THRESHOLDS, SUCCESS_TOLERANCE
from .carrier_spaces import CARRIER_DEPLOYMENT_COST, CARRIER_DIM
from .causal_interventions import many_patched_predictions, patched_predictions
from .causal_metrics import direction_agreement, paired_sign_flip_p, recovery, success_vector
from .counterfactual_subspace import fit_difference_subspace, projected_delta, shuffled_pair_subspace
from .failure_ledger import add, make_entry
from .instrumented_af1d import load_population_rows
from .ood_diagnostics import compare as ood_compare, fit_reference
from .random_controls import orthogonal_residual, random_subspaces, same_norm_noise
from .semantic_compiler import compile_delta, fit_semantic_compiler
from .trace_cache import pair_traces
from .utils import HERE, deterministic_seed, range_list, write_json


def _seeds(name:str)->list[int]:return range_list(PARTITIONS[name])


def _carrier_values(trace,carrier:str,times:np.ndarray)->np.ndarray:
    x=trace.carrier(carrier);idx=torch.as_tensor(times,dtype=torch.long,device=x.device);rows=torch.arange(x.shape[0],device=x.device);return x[rows,idx].detach().cpu().numpy().astype(np.float64)


def _semantic_values(pairs,times:np.ndarray,cf:bool)->np.ndarray:
    out=[]
    for p,t in zip(pairs,times):
        ep=p.counterfactual_episode if cf else p.base_episode;out.append(float(ep.macrostate_traces[PRIMARY_PROBES[p.family]][int(t)]))
    return np.asarray(out,dtype=np.float64)


def _eligibility(data:dict,family:str)->tuple[np.ndarray,dict]:
    task=task_by_name(family);bp=data["base_prediction"].detach().cpu().numpy();cp=data["cf_prediction"].detach().cpu().numpy();pairs=data["pairs"]
    mask=[];reasons=defaultdict(int)
    for i,p in enumerate(pairs):
        base_ok=task.success(float(bp[i]),float(p.base_episode.target));cf_ok=task.success(float(cp[i]),float(p.counterfactual_episode.target));same_len=len(p.base_episode.observations)==len(p.counterfactual_episode.observations);finite=bool(np.isfinite(bp[i]) and np.isfinite(cp[i]));meaningful=abs(float(p.base_episode.target)-float(p.counterfactual_episode.target))>=SUCCESS_TOLERANCE[family]
        ok=base_ok and cf_ok and same_len and finite and meaningful;mask.append(ok)
        if not base_ok:reasons["base_unsolved"]+=1
        if not cf_ok:reasons["counterfactual_unsolved"]+=1
        if not meaningful:reasons["target_displacement_below_tolerance"]+=1
    return np.asarray(mask,dtype=bool),{"eligible":int(np.sum(mask)),"total":len(mask),"reasons":dict(reasons)}


def _decoder_diag(fit_x,fit_z,select_x,select_z)->dict:
    x=np.asarray(fit_x,dtype=np.float64);z=np.asarray(fit_z,dtype=np.float64);xs=np.asarray(select_x,dtype=np.float64);zs=np.asarray(select_z,dtype=np.float64)
    design=np.concatenate([x,np.ones((len(x),1))],axis=1);ridge=1e-6*np.eye(design.shape[1]);ridge[-1,-1]=0.0;coef=np.linalg.solve(design.T@design+ridge,design.T@z)
    pred=np.concatenate([xs,np.ones((len(xs),1))],axis=1)@coef;ss_res=float(np.sum((zs-pred)**2));ss_tot=float(np.sum((zs-zs.mean())**2));r2=1.0-ss_res/max(ss_tot,1e-12);corr=float(np.corrcoef(zs,pred)[0,1]) if np.std(zs)>1e-12 and np.std(pred)>1e-12 else 0.0
    sign_acc=float(np.mean(np.sign(zs)==np.sign(pred)))
    return {"r2":r2,"correlation":corr,"sign_accuracy":sign_acc,"decodability_is_pass_gate":False}


def _metrics(task,bp,cp,pf,pr,base_targets,cf_targets,random_f,random_r,ood_f,ood_r,seed:int)->dict:
    rec_f=recovery(bp,cp,pf);rec_r=recovery(cp,bp,pr);rec=np.concatenate([rec_f,rec_r]);direction=np.concatenate([direction_agreement(bp,cp,pf),direction_agreement(cp,bp,pr)])
    success=np.concatenate([success_vector(task,pf,cf_targets),success_vector(task,pr,base_targets)])
    rand_rec=[]
    for rf,rr in zip(random_f,random_r):rand_rec.append(np.concatenate([recovery(bp,cp,rf),recovery(cp,bp,rr)]))
    rand=np.stack(rand_rec);rand_pair_median=np.median(rand,axis=0);margin=float(np.median(rec)-np.median(rand));p=paired_sign_flip_p(rec-rand_pair_median,seed)
    ood=np.concatenate([ood_f["ood_ratio"],ood_r["ood_ratio"]]);thresholds=SOURCE_SWAP_THRESHOLDS
    passed=(float(np.median(rec))>=thresholds["median_recovery_min"] and float(np.mean(direction))>=thresholds["direction_agreement_min"] and float(np.mean(success))>=thresholds["counterfactual_success_min"] and margin>=thresholds["random_margin_min"] and p<=thresholds["paired_p_max"] and float(np.median(ood))<=thresholds["median_ood_ratio_max"])
    return {"eligible_pairs_per_direction":len(bp),"median_recovery":float(np.median(rec)),"direction_agreement":float(np.mean(direction)),"counterfactual_success":float(np.mean(success)),"random_subspace_median_recovery":float(np.median(rand)),"random_margin":margin,"paired_p":float(p),"median_ood_ratio":float(np.median(ood)),"pass":bool(passed),"recovery_values":rec.tolist(),"random_pair_median":rand_pair_median.tolist()}


def _evaluate_one(organism:dict,carrier:str,k:int,fit_data:dict,select_data:dict,fit_mask:np.ndarray,select_mask:np.ndarray)->tuple[dict,dict]:
    family=organism["family"];task=task_by_name(family);config_id=f"{carrier}-K{k}";fit_idx=np.flatnonzero(fit_mask);sel_idx=np.flatnonzero(select_mask)
    ft=np.asarray([p.primary_phase for p in fit_data["pairs"]]);st=np.asarray([p.primary_phase for p in select_data["pairs"]])
    fbase=_carrier_values(fit_data["base_trace"],carrier,ft)[fit_idx];fcf=_carrier_values(fit_data["cf_trace"],carrier,ft)[fit_idx]
    sub=fit_difference_subspace(fbase,fcf,k);q=sub["q"];fd=fcf-fbase;fz=_semantic_values(fit_data["pairs"],ft,False)[fit_idx];fzc=_semantic_values(fit_data["pairs"],ft,True)[fit_idx];compiler=fit_semantic_compiler(fzc-fz,fd,q)
    ood_ref=fit_reference(np.concatenate([fbase,fcf],axis=0))
    sbase=_carrier_values(select_data["base_trace"],carrier,st)[sel_idx];scf=_carrier_values(select_data["cf_trace"],carrier,st)[sel_idx];sd=scf-sbase;times=st[sel_idx]
    bp=select_data["base_prediction"].detach().cpu().numpy()[sel_idx].astype(np.float64);cp=select_data["cf_prediction"].detach().cpu().numpy()[sel_idx].astype(np.float64);base_targets=select_data["base_targets"].detach().cpu().numpy()[sel_idx].astype(np.float64);cf_targets=select_data["cf_targets"].detach().cpu().numpy()[sel_idx].astype(np.float64)
    bo=select_data["base_obs"][sel_idx];bl=select_data["base_lengths"][sel_idx];co=select_data["cf_obs"][sel_idx];cl=select_data["cf_lengths"][sel_idx];model=select_data["model"]
    natural_delta=projected_delta(sd,q);fv=sbase+natural_delta;rv=scf-natural_delta
    pf=patched_predictions(model,bo,bl,carrier,fv,times);pr=patched_predictions(model,co,cl,carrier,rv,times)
    controls=random_subspaces(CARRIER_DIM[carrier],k,32,deterministic_seed("v837an-random-subspaces",organism["organism_id"],config_id));fsets=np.stack([sbase+projected_delta(sd,rq) for rq in controls]);rsets=np.stack([scf+projected_delta(-sd,rq) for rq in controls]);random_f=many_patched_predictions(model,bo,bl,carrier,fsets,times);random_r=many_patched_predictions(model,co,cl,carrier,rsets,times)
    ood_f=ood_compare(fv,scf,ood_ref);ood_r=ood_compare(rv,sbase,ood_ref);source_metrics=_metrics(task,bp,cp,pf,pr,base_targets,cf_targets,random_f,random_r,ood_f,ood_r,deterministic_seed("v837an-p",organism["organism_id"],config_id,"source"))
    sem_delta=_semantic_values(select_data["pairs"],st,True)[sel_idx]-_semantic_values(select_data["pairs"],st,False)[sel_idx];compiled=compile_delta(sem_delta,compiler);cfv=sbase+compiled;crv=scf+compile_delta(-sem_delta,compiler)
    cpf=patched_predictions(model,bo,bl,carrier,cfv,times);cpr=patched_predictions(model,co,cl,carrier,crv,times);coodf=ood_compare(cfv,scf,ood_ref);coodr=ood_compare(crv,sbase,ood_ref);compiler_metrics=_metrics(task,bp,cp,cpf,cpr,base_targets,cf_targets,random_f,random_r,coodf,coodr,deterministic_seed("v837an-p",organism["organism_id"],config_id,"compiler"))
    # Additional sham diagnostics, deliberately not used as a selection gate.
    shuffled=shuffled_pair_subspace(fbase,fcf,k,deterministic_seed("v837an-shuffle",organism["organism_id"],config_id))["q"];shuf_f=sbase+projected_delta(sd,shuffled);shuf_pred=patched_predictions(model,bo,bl,carrier,shuf_f,times);shuf_rec=float(np.median(recovery(bp,cp,shuf_pred)))
    noise= same_norm_noise(natural_delta,deterministic_seed("v837an-noise",organism["organism_id"],config_id));noise_pred=patched_predictions(model,bo,bl,carrier,sbase+noise,times);noise_rec=float(np.median(recovery(bp,cp,noise_pred)))
    residual=orthogonal_residual(sd,q);res_pred=patched_predictions(model,bo,bl,carrier,sbase+residual,times);res_rec=float(np.median(recovery(bp,cp,res_pred)))
    fit_z=np.concatenate([fz,fzc]);select_z=np.concatenate([_semantic_values(select_data["pairs"],st,False)[sel_idx],_semantic_values(select_data["pairs"],st,True)[sel_idx]]);decoder=_decoder_diag(np.concatenate([fbase,fcf]),fit_z,np.concatenate([sbase,scf]),select_z)
    fit_record={"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"config_id":config_id,"carrier":carrier,"k":k,"q":q.tolist(),"singular_values":sub["singular_values"].tolist(),"semantic_compiler":{"coef":np.asarray(compiler["coef"]).tolist(),"ridge_lambda":1e-6,"intercept":0.0},"fit_eligible":len(fit_idx),"gradient_steps":0,"invertibility_required":False}
    source_summary={kk:vv for kk,vv in source_metrics.items() if kk not in {"recovery_values","random_pair_median"}}
    compiler_summary={kk:vv for kk,vv in compiler_metrics.items() if kk not in {"recovery_values","random_pair_median"}}
    selection={"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"config_id":config_id,"carrier":carrier,"k":k,"fit_eligible":len(fit_idx),"select_eligible":len(sel_idx),"source_swap":source_summary,"semantic_compiler":compiler_summary,"controls":{"shuffled_semantic_median_recovery":shuf_rec,"same_norm_noise_median_recovery":noise_rec,"orthogonal_residual_median_recovery":res_rec,"random_subspaces":32},"decodability":decoder}
    return fit_record,selection


def _family_config_summary(family:str,config_id:str,rows:list[dict],population:list[dict])->dict:
    powered=[r for r in rows if r["fit_eligible"]>=MIN_ELIGIBLE["AN_FIT"] and r["select_eligible"]>=MIN_ELIGIBLE["AN_SELECT"]];source_pass=[r for r in powered if r["source_swap"]["pass"]];compiler_pass=[r for r in powered if r["semantic_compiler"]["pass"]]
    source_engines={r["engine"] for r in population if r["family"]==family and bool(r["competent"])};pass_engines={r["engine"] for r in source_pass};compiler_engines={r["engine"] for r in compiler_pass};n=len(powered);required=max(1,math.ceil(FAMILY_SUPPORT["minimum_pass_fraction"]*n));powered_family=n>=FAMILY_SUPPORT["minimum_competent_organisms"]
    engine_source_ok=len(source_engines)<2 or len(pass_engines)>=2;engine_compiler_ok=len(source_engines)<2 or len(compiler_engines)>=2
    family_pass=powered_family and len(source_pass)>=required and engine_source_ok;compiler_family_pass=powered_family and len(compiler_pass)>=required and engine_compiler_ok
    return {"family":family,"config_id":config_id,"powered_organisms":n,"required_passes":required,"source_swap_organisms_passing":len(source_pass),"semantic_compiler_organisms_passing":len(compiler_pass),"source_swap_pass_fraction":0.0 if n==0 else len(source_pass)/n,"semantic_compiler_pass_fraction":0.0 if n==0 else len(compiler_pass)/n,"source_swap_engines":sorted(pass_engines),"semantic_compiler_engines":sorted(compiler_engines),"source_swap_support_organisms":[r["organism_id"] for r in source_pass],"semantic_compiler_support_organisms":[r["organism_id"] for r in compiler_pass],"family_powered":powered_family,"family_pass":family_pass,"semantic_compiler_family_pass":compiler_family_pass}


def run_an_a()->dict:
    start=time.perf_counter();population=load_population_rows();competent=[r for r in population if bool(r["competent"])];fit_records=[];selection_rows=[];power=[]
    fit_seeds=_seeds("AN_FIT");select_seeds=_seeds("AN_SELECT")
    for oi,organism in enumerate(competent,1):
        family=organism["family"];fit_data=pair_traces(organism["organism_id"],family,fit_seeds);select_data=pair_traces(organism["organism_id"],family,select_seeds);fit_mask,fp=_eligibility(fit_data,family);sel_mask,sp=_eligibility(select_data,family);power.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"AN_FIT":fp,"AN_SELECT":sp,"adequately_powered":fp["eligible"]>=MIN_ELIGIBLE["AN_FIT"] and sp["eligible"]>=MIN_ELIGIBLE["AN_SELECT"]})
        if fp["eligible"]<MIN_ELIGIBLE["AN_FIT"] or sp["eligible"]<MIN_ELIGIBLE["AN_SELECT"]:continue
        for carrier in CARRIERS:
            for k in LATENT_DIMS[carrier]:
                fr,sr=_evaluate_one(organism,carrier,k,fit_data,select_data,fit_mask,sel_mask);fit_records.append(fr);selection_rows.append(sr)
        print(f"V837an AN-A {oi}/{len(competent)} {organism['organism_id']} complete",flush=True)
    by=defaultdict(list)
    for row in selection_rows:by[(row["family"],row["config_id"])].append(row)
    summaries=[];winners={};compiler_winners={}
    for family in sorted({r["family"] for r in competent}):
        configs=[]
        for carrier in CARRIERS:
            for k in LATENT_DIMS[carrier]:
                cid=f"{carrier}-K{k}";summary=_family_config_summary(family,cid,by.get((family,cid),[]),competent);summary.update({"carrier":carrier,"k":k,"carrier_cost":CARRIER_DEPLOYMENT_COST[carrier],"carrier_dimension":CARRIER_DIM[carrier]});configs.append(summary);summaries.append(summary)
                if not summary["family_pass"]:
                    failed=[]
                    if not summary["family_powered"]:failed.append("FAMILY_CAUSAL_EVIDENCE_UNDERPOWERED")
                    else:
                        if summary["source_swap_pass_fraction"]<.60:failed.append("<60% powered competent organisms pass source-swap gate")
                        if len(summary["source_swap_engines"])<2:failed.append("both discovery engines not represented among passes")
                    add(make_entry(failure_id=f"V837an-AN-A-{family}-{cid}",stage="AN-A",hypothesis=f"{family} uses a {k}D causal macrovariable in {carrier}.",why="Test lowest-dimensional many-to-one causal carrier without cross-organism state invertibility.",implementation={"family":family,"carrier":carrier,"k":k,"method":"counterfactual-difference SVD + bidirectional source-swap"},data={"fit":PARTITIONS["AN_FIT"],"select":PARTITIONS["AN_SELECT"]},fit_seeds=PARTITIONS["AN_FIT"],selection_seeds=PARTITIONS["AN_SELECT"],controls=["32 Haar random subspaces","shuffled semantic","same-norm noise","orthogonal residual"],parameter_count=0,mac_cost=k,metrics={k:v for k,v in summary.items() if k!="organism_rows"},gate=SOURCE_SWAP_THRESHOLDS,failed=failed,distance={"pass_fraction_shortfall":max(0.0,.60-summary["source_swap_pass_fraction"])},result_status="underpowered" if not summary["family_powered"] else "definitive within tested development scope",failure_type="UNDERPOWERED" if not summary["family_powered"] else "SCIENTIFIC_FAILURE",confounds_ruled_out=["full-state invertibility assumption","random subspace dimensionality","one-way perturbation artifact"],confounds_remaining=["other causal carrier granularity","routing","synergy"],meaning="This exact low-dimensional carrier configuration did not establish the frozen family-level causal abstraction gate.",uncertainty="Other V837an carriers or routing/coalition abstractions may still pass.",next_experiment="Continue frozen V837an grid; do not repeat this configuration unchanged.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage macrovariables --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/an_a_selection.json"]))
        passing=[c for c in configs if c["family_pass"]]
        cp=[c for c in configs if c["semantic_compiler_family_pass"]]
        key=lambda c:(c["k"],c["carrier_cost"],c["carrier_dimension"],c["config_id"])
        winners[family]=min(passing,key=key) if passing else None;compiler_winners[family]=min(cp,key=key) if cp else None
    fit_payload={"version":"V837an","stage":"AN-A-FIT","fit_seeds":PARTITIONS["AN_FIT"],"records":fit_records,"gradient_steps":0,"cross_organism_state_invertibility_required":False}
    sel_payload={"version":"V837an","stage":"AN-A-SELECT","select_seeds":PARTITIONS["AN_SELECT"],"organism_results":selection_rows,"family_config_summaries":summaries,"family_winners":winners,"semantic_compiler_winners":compiler_winners,"causal_subspace_families":sum(v is not None for v in winners.values()),"semantic_compiler_families":sum(v is not None for v in compiler_winners.values()),"wall_seconds":time.perf_counter()-start}
    write_json(HERE/"raw/an_a_fit.json",fit_payload);write_json(HERE/"raw/an_a_selection.json",sel_payload);write_json(HERE/"raw/an_a_semantic_compilers.json",{"version":"V837an","winners":compiler_winners});write_json(HERE/"raw/counterfactual_power.json",{"version":"V837an","rows":power});write_json(HERE/"diagnostics/eligible_pair_power.json",{"version":"V837an","rows":power});write_json(HERE/"diagnostics/causal_subspace_stability.json",{"version":"V837an","family_winners":winners});write_json(HERE/"diagnostics/semantic_compiler.json",{"version":"V837an","family_winners":compiler_winners});write_json(HERE/"diagnostics/decodability_vs_causality.json",{"version":"V837an","rows":[{"family":r["family"],"organism_id":r["organism_id"],"config_id":r["config_id"],"decodability":r["decodability"],"causal_pass":r["source_swap"]["pass"]} for r in selection_rows]});write_json(HERE/"diagnostics/random_subspace_controls.json",{"version":"V837an","controls_per_candidate":32,"candidate_count":len(selection_rows)});write_json(HERE/"diagnostics/intervention_ood.json",{"version":"V837an","threshold":2.0,"rows":[{"organism_id":r["organism_id"],"config_id":r["config_id"],"source_swap_median_ood":r["source_swap"]["median_ood_ratio"],"compiler_median_ood":r["semantic_compiler"]["median_ood_ratio"]} for r in selection_rows]})
    return sel_payload


if __name__=="__main__":print(json.dumps(run_an_a(),indent=2))
