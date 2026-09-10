from __future__ import annotations

from itertools import combinations
import numpy as np

from experiments.v837_primitive_invention.tasks import task_by_name

from .causal_interventions import coalition_state_predictions
from .causal_metrics import direction_agreement, paired_sign_flip_p, recovery, success_vector
from .ood_diagnostics import compare as ood_compare, fit_reference
from .utils import deterministic_seed


def all_cell_subsets()->list[tuple[int,...]]:
    rows=[]
    for n in range(1,11):rows.extend(combinations(range(10),n))
    if len(rows)!=1023:raise RuntimeError("V837AN_COALITION_UNIVERSE_INVALID")
    return rows


def _state_at(trace,times,indices):
    x=trace.states.flatten(2);return x[np.asarray(indices),np.asarray(times)[np.asarray(indices)]].detach().cpu().numpy().astype(np.float64)


def scan_organism_coalitions(data:dict,indices:np.ndarray,fit_reference_points:np.ndarray,*,chunk_size:int=32)->list[dict]:
    idx=np.asarray(indices,dtype=np.int64);pairs=[data["pairs"][i] for i in idx];times=np.asarray([p.primary_phase for p in data["pairs"]],dtype=np.int64);bt=times[idx]
    bstate=_state_at(data["base_trace"],times,idx);cstate=_state_at(data["cf_trace"],times,idx);bp=data["base_prediction"].detach().cpu().numpy()[idx];cp=data["cf_prediction"].detach().cpu().numpy()[idx];base_targets=data["base_targets"].detach().cpu().numpy()[idx];cf_targets=data["cf_targets"].detach().cpu().numpy()[idx];bo=data["base_obs"][idx];bl=data["base_lengths"][idx];co=data["cf_obs"][idx];cl=data["cf_lengths"][idx];task=task_by_name(pairs[0].family);reference=fit_reference(fit_reference_points)
    subsets=all_cell_subsets();rows=[]
    for start in range(0,len(subsets),chunk_size):
        batch=subsets[start:start+chunk_size];f=coalition_state_predictions(data["model"],bo,bl,bstate,cstate,bt,batch);r=coalition_state_predictions(data["model"],co,cl,cstate,bstate,bt,batch)
        for bi,subset in enumerate(batch):
            frec=recovery(bp,cp,f[bi]);rrec=recovery(cp,bp,r[bi]);rec=np.concatenate([frec,rrec]);direction=np.concatenate([direction_agreement(bp,cp,f[bi]),direction_agreement(cp,bp,r[bi])]);success=np.concatenate([success_vector(task,f[bi],cf_targets),success_vector(task,r[bi],base_targets)])
            pf=bstate.copy();pr=cstate.copy()
            for cell in subset:
                lo=cell*4;hi=lo+4;pf[:,lo:hi]=cstate[:,lo:hi];pr[:,lo:hi]=bstate[:,lo:hi]
            ood=np.concatenate([ood_compare(pf,cstate,reference)["ood_ratio"],ood_compare(pr,bstate,reference)["ood_ratio"]]);p=paired_sign_flip_p(rec,deterministic_seed("v837an-coalition-p",data["row"]["organism_id"],subset));passed=float(np.median(rec))>=.60 and float(np.mean(success))>=.70 and float(np.mean(direction))>=.75 and p<=.01 and float(np.median(ood))<=2.0
            rows.append({"nodes":list(subset),"cardinality":len(subset),"median_recovery":float(np.median(rec)),"direction_agreement":float(np.mean(direction)),"counterfactual_success":float(np.mean(success)),"paired_p":float(p),"median_ood_ratio":float(np.median(ood)),"pass":bool(passed)})
    return rows


def minimum_sufficient(rows:list[dict])->dict|None:
    passing=[r for r in rows if r["pass"]]
    if not passing:return None
    return min(passing,key=lambda r:(r["cardinality"],-r["median_recovery"],r["median_ood_ratio"],tuple(r["nodes"])))
