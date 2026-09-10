from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan, many_patched_predictions
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .abstract_rollout import rollout_from_setpoint
from .canonical_reader import read_k1
from .canonical_writer import set_state
from .k1_backend import _phase_arrays, component_for_phase
from .metrics import causal_recovery, direction_agreement, safe_median
from .ood_diagnostics import compare as ood_compare, fit_reference
from .phase_backends import PHASES, representative_phase_timesteps
from .random_controls import random_directions
from .setpoint_grid import family_grid
from .trajectory_eval import trajectory_nrmse


def _fit_reference_by_phase(backend:dict)->dict:
    family=backend["family"];data=pair_traces(backend["organism_id"],family,list(range(10000,10064)));mask,_=_eligibility(data,family);out={}
    for phase in PHASES[family]:
        b,c,z0,z1=_phase_arrays(data,mask,family,phase)
        states=np.concatenate([b,c]) if len(b) else np.empty((0,40));semantic=np.concatenate([z0,z1]) if len(b) else np.empty(0)
        if len(states):out[phase]={"reference":fit_reference(states),"states":states,"semantic":semantic}
    return out


def _shuffled_state(state:np.ndarray,target:float,component:dict)->np.ndarray|None:
    sh=component.get("shuffled_control",{})
    if not sh.get("valid") or sh.get("writer") is None:return None
    return set_state(state,target,sh["reader"],np.asarray(sh["writer"],dtype=np.float64))


def evaluate_setpoints(backend:dict,eval_seeds:list[int],*,partition_name:str="AO_BACKEND_SELECT",random_controls:bool=True)->dict:
    oid=backend["organism_id"];family=backend["family"];grid=family_grid(family);R=float(grid["semantic_range"]);task=task_by_name(family)
    data=pair_traces(oid,family,eval_seeds);mask,_=_eligibility(data,family);eligible=np.flatnonzero(mask);refs=_fit_reference_by_phase(backend)
    trace_np=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40)
    samples=[]
    for i in eligible:
        p=data["pairs"][i];ep=p.base_episode;phase_times=representative_phase_timesteps(ep)
        for phase,t in phase_times.items():
            if phase not in PHASES[family] or phase not in refs:continue
            comp=component_for_phase(backend,phase)
            if not comp.get("valid"):continue
            state=trace_np[i,t].astype(np.float64);z=float(read_k1(state[None,:],comp["reader"])[0])
            for target in grid["targets"]:
                target=float(target)
                if abs(target-z)<0.10*R:continue
                patched=set_state(state,target,comp["reader"],np.asarray(comp["writer"],dtype=np.float64))
                expected=rollout_from_setpoint(family,ep,t,target)["final_target"]
                corpus=refs[phase];j=int(np.argmin(np.abs(corpus["semantic"]-target)));natural_target=corpus["states"][j]
                ood=float(ood_compare(patched[None,:],natural_target[None,:],corpus["reference"])["median_ood_ratio"])
                samples.append({"episode_index":int(i),"seed":int(p.base_seed),"phase":phase,"timestep":int(t),"target":target,"state":state,"patched":patched,"z_before":z,"expected_target":float(expected),"base_prediction":float(data["base_prediction"][i].item()),"ood_ratio":ood})
    if not samples:
        return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"sample_count":0,"pass":False,"failure_code":"ABSOLUTE_SETPOINT_FAIL"}
    idx=np.asarray([s["episode_index"] for s in samples],dtype=np.int64);times=np.asarray([s["timestep"] for s in samples],dtype=np.int64);values=np.stack([s["patched"] for s in samples])
    obs=data["base_obs"][torch.as_tensor(idx,dtype=torch.long)];lengths=data["base_lengths"][torch.as_tensor(idx,dtype=torch.long)]
    plan=build_patch_plan("STATE40",values,times)
    with torch.no_grad():pred,trace=run_instrumented(data["model"],obs,lengths,patch_plan=plan,return_trace=True)
    pred_np=pred.detach().cpu().numpy().astype(np.float64);trace_np_p=trace.states.detach().cpu().numpy().reshape(len(samples),trace.states.shape[1],40)
    base=np.asarray([s["base_prediction"] for s in samples]);expected=np.asarray([s["expected_target"] for s in samples]);rec=causal_recovery(base,expected,pred_np);direction=direction_agreement(base,expected,pred_np);success=np.asarray([task.success(float(p),float(t)) for p,t in zip(pred_np,expected)],dtype=np.float64)
    trajectory=np.asarray([trajectory_nrmse(backend,data["pairs"][s["episode_index"]].base_episode,s["timestep"],s["target"],trace_np_p[j],R) for j,s in enumerate(samples)],dtype=np.float64)
    read_err=[];intervention_norm=[]
    for s in samples:
        comp=component_for_phase(backend,s["phase"]);read_err.append(abs(float(read_k1(s["patched"][None,:],comp["reader"])[0])-s["target"])/max(R,1e-12));intervention_norm.append(float(np.linalg.norm(s["patched"]-s["state"])))
    rand_median=float("nan")
    if random_controls:
        canonical_updates=np.stack([s["patched"]-s["state"] for s in samples]);random_preds=[]
        # Every direction is organism/backend deterministic and norm matched per sample.
        dirs=random_directions(oid,backend["variant"])
        value_sets=[]
        for q in dirs:
            q=q/max(float(np.linalg.norm(q)),1e-12);norm=np.linalg.norm(canonical_updates,axis=1);value_sets.append(np.stack([s["state"] for s in samples])+norm[:,None]*q[None,:])
        random_preds=many_patched_predictions(data["model"],obs,lengths,"STATE40",np.stack(value_sets),times)
        rand_rec=np.stack([causal_recovery(base,expected,row) for row in random_preds]);rand_median=float(np.median(rand_rec))
    shuffled_predictions=[]
    for j,s in enumerate(samples):
        comp=component_for_phase(backend,s["phase"]);ss=_shuffled_state(s["state"],s["target"],comp)
        shuffled_predictions.append(ss)
    shuffled_median=float("nan")
    if all(x is not None for x in shuffled_predictions):
        sh_values=np.stack(shuffled_predictions);sh_plan=build_patch_plan("STATE40",sh_values,times)
        with torch.no_grad():sh_pred=run_instrumented(data["model"],obs,lengths,patch_plan=sh_plan,return_trace=False).detach().cpu().numpy().astype(np.float64)
        shuffled_median=float(np.median(causal_recovery(base,expected,sh_pred)))
    control_reference=max(x for x in [rand_median,shuffled_median] if np.isfinite(x)) if any(np.isfinite(x) for x in [rand_median,shuffled_median]) else -np.inf
    margin=float(np.median(rec)-control_reference)
    by_target=[]
    for target in sorted(set(s["target"] for s in samples)):
        m=np.asarray([abs(s["target"]-target)<1e-12 for s in samples]);by_target.append({"target":float(target),"n":int(m.sum()),"median_recovery":float(np.median(rec[m])),"task_success":float(np.mean(success[m])),"pass_recovery_060":bool(np.median(rec[m])>=.60)})
    target_fraction=float(np.mean([r["pass_recovery_060"] for r in by_target])) if by_target else 0.0
    metrics={"median_counterfactual_recovery":float(np.median(rec)),"direction_agreement":float(np.mean(direction)),"counterfactual_task_success":float(np.mean(success)),"median_ood_ratio":float(np.median([s["ood_ratio"] for s in samples])),"target_fraction_recovery_060":target_fraction,"random_median_recovery":rand_median,"shuffled_median_recovery":shuffled_median,"control_margin":margin,"median_semantic_read_after_set_error":float(np.median(read_err)),"p99_semantic_read_after_set_error":float(np.quantile(read_err,.99)),"median_trajectory_nrmse":float(np.median(trajectory)),"median_intervention_norm":float(np.median(intervention_norm))}
    passed=(metrics["median_counterfactual_recovery"]>=.70 and metrics["direction_agreement"]>=.80 and metrics["counterfactual_task_success"]>=.75 and metrics["median_ood_ratio"]<=2.0 and target_fraction>=.80 and margin>=.20 and metrics["median_semantic_read_after_set_error"]<=1e-6 and metrics["p99_semantic_read_after_set_error"]<=1e-5)
    return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"sample_count":len(samples),"skipped_small_delta":int(len(eligible)*len(PHASES[family])*len(grid["targets"])-len(samples)),"metrics":metrics,"target_results":by_target,"pass":bool(passed),"failure_code":None if passed else "ABSOLUTE_SETPOINT_FAIL"}
