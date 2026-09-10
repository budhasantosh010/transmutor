from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .abstract_rollout import rollout_from_setpoint
from .canonical_reader import read_k1
from .canonical_writer import set_state
from .k1_backend import component_for_phase
from .metrics import safe_median
from .ood_diagnostics import compare as ood_compare
from .quotient_pairs import select_quotient_pairs
from .setpoint_eval import _fit_reference_by_phase
from .setpoint_grid import family_grid
from .trajectory_eval import phase_at_timestep, read_state_at, semantic_timesteps


def _trajectory_disagreement(backend,ep,start_t,ta,tb,R):
    vals=[]
    for t in semantic_timesteps(ep):
        if t<=start_t:continue
        phase=phase_at_timestep(ep,t)
        if phase is None:continue
        za=read_state_at(backend,ta[t],phase);zb=read_state_at(backend,tb[t],phase);vals.append(abs(za-zb)/max(R,1e-12))
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0


def evaluate_quotient(backend:dict,eval_seeds:list[int],*,partition_name:str="AO_QUOTIENT_SELECT")->dict:
    oid=backend["organism_id"];family=backend["family"];grid=family_grid(family);R=float(grid["semantic_range"]);pairs=select_quotient_pairs(backend,eval_seeds,partition_name);data=pair_traces(oid,family,eval_seeds);refs=_fit_reference_by_phase(backend);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);task=task_by_name(family)
    samples=[]
    for row in pairs["rows"]:
        phase=row["phase"];comp=component_for_phase(backend,phase);ai=row["anchor_episode_index"];di=row["donor_episode_index"];at=row["anchor_timestep"];dt=row["donor_timestep"];anchor=states[ai,at].astype(np.float64);donor=states[di,dt].astype(np.float64);za=float(read_k1(anchor[None,:],comp["reader"])[0]);donor_aligned=set_state(donor,za,comp["reader"],np.asarray(comp["writer"]));target=max(grid["targets"],key=lambda x:abs(float(x)-za));target=float(target);aset=set_state(anchor,target,comp["reader"],np.asarray(comp["writer"]));dset=set_state(donor_aligned,target,comp["reader"],np.asarray(comp["writer"]));ep=data["pairs"][ai].base_episode;abstract_star=rollout_from_setpoint(family,ep,at,target)["final_target"];abstract_anchor=rollout_from_setpoint(family,ep,at,za)["final_target"]
        corpus=refs[phase];ja=int(np.argmin(np.abs(corpus["semantic"]-za)));jt=int(np.argmin(np.abs(corpus["semantic"]-target)));ood1=float(ood_compare(donor_aligned[None,:],corpus["states"][ja][None,:],corpus["reference"])["median_ood_ratio"]);ood2=float(ood_compare(aset[None,:],corpus["states"][jt][None,:],corpus["reference"])["median_ood_ratio"]);ood3=float(ood_compare(dset[None,:],corpus["states"][jt][None,:],corpus["reference"])["median_ood_ratio"])
        samples.append({**row,"anchor_state":anchor,"donor_aligned":donor_aligned,"anchor_set":aset,"donor_set":dset,"target":target,"abstract_target":float(abstract_star),"abstract_anchor":float(abstract_anchor),"ood_ratio":max(ood1,ood2,ood3)})
    if not samples:return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"pair_count":0,"pass":False,"failure_code":"QUOTIENT_RESIDUAL_SENSITIVITY"}
    idx=np.asarray([s["anchor_episode_index"] for s in samples],dtype=np.int64);times=np.asarray([s["anchor_timestep"] for s in samples]);obs=data["base_obs"][torch.as_tensor(idx,dtype=torch.long)];lens=data["base_lengths"][torch.as_tensor(idx,dtype=torch.long)]
    pa=build_patch_plan("STATE40",np.stack([s["anchor_set"] for s in samples]),times);pb=build_patch_plan("STATE40",np.stack([s["donor_set"] for s in samples]),times)
    with torch.no_grad():ya,ta=run_instrumented(data["model"],obs,lens,patch_plan=pa,return_trace=True);yb,tb=run_instrumented(data["model"],obs,lens,patch_plan=pb,return_trace=True)
    ya=ya.detach().cpu().numpy().astype(np.float64);yb=yb.detach().cpu().numpy().astype(np.float64);ta_np=ta.states.detach().cpu().numpy().reshape(len(samples),ta.states.shape[1],40);tb_np=tb.states.detach().cpu().numpy().reshape(len(samples),tb.states.shape[1],40)
    rs=[];traj=[];succdiff=[]
    for j,s in enumerate(samples):
        denom=abs(s["abstract_target"]-s["abstract_anchor"])+1e-8;rs.append(abs(float(ya[j])-float(yb[j]))/denom);ep=data["pairs"][s["anchor_episode_index"]].base_episode;traj.append(_trajectory_disagreement(backend,ep,s["anchor_timestep"],ta_np[j],tb_np[j],R));sa=task.success(float(ya[j]),s["abstract_target"]);sb=task.success(float(yb[j]),s["abstract_target"]);succdiff.append(float(sa!=sb))
    metrics={"median_residual_sensitivity":float(np.median(rs)),"p90_residual_sensitivity":float(np.quantile(rs,.90)),"median_trajectory_disagreement":float(np.median(traj)),"task_success_disagreement":float(np.mean(succdiff)),"median_ood_ratio":float(np.median([s["ood_ratio"] for s in samples])),"pair_count":len(samples),"median_residual_distance":float(np.median([s["residual_distance"] for s in samples]))}
    passed=metrics["median_residual_sensitivity"]<=.20 and metrics["p90_residual_sensitivity"]<=.40 and metrics["median_trajectory_disagreement"]<=.10 and metrics["task_success_disagreement"]<=.10 and metrics["median_ood_ratio"]<=2.0
    return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"metrics":metrics,"pass":bool(passed),"failure_code":None if passed else ("RESIDUAL_SWAP_OOD" if metrics["median_ood_ratio"]>2 else "QUOTIENT_RESIDUAL_SENSITIVITY"),"pair_rows":[{k:v for k,v in s.items() if k not in {"anchor_state","donor_aligned","anchor_set","donor_set"}} for s in samples]}
