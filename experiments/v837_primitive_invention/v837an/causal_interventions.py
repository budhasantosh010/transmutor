from __future__ import annotations

import numpy as np
import torch

from .instrumented_af1d import PatchPlan, TensorPatch, run_instrumented


def _full_values(carrier:str,values:np.ndarray)->torch.Tensor:
    v=torch.as_tensor(np.asarray(values),dtype=torch.float32)
    if carrier in {"STATE40","OUTPUT40","MESSAGE40","GLOBAL40"}:return v.reshape(v.shape[0],10,4)
    if carrier=="GATE1":return v.reshape(v.shape[0],1)
    raise ValueError(f"carrier {carrier} is diagnostic-only or unsupported for causal patching")


def build_patch_plan(carrier:str,values:np.ndarray,timesteps:list[int]|np.ndarray)->PatchPlan:
    values_t=_full_values(carrier,values);times=np.asarray(timesteps,dtype=np.int64);n=len(times);plan=PatchPlan()
    target={"STATE40":plan.state,"OUTPUT40":plan.output,"MESSAGE40":plan.message,"GLOBAL40":plan.global_term,"GATE1":plan.gate}[carrier]
    for t in sorted(set(int(x) for x in times.tolist())):
        rowmask=torch.as_tensor(times==t,dtype=torch.bool)
        if carrier=="GATE1":mask=rowmask[:,None]
        else:mask=rowmask[:,None,None].expand(n,10,4)
        target[int(t)]=TensorPatch(values_t,mask)
    return plan


def patched_predictions(model,observations:torch.Tensor,lengths:torch.Tensor,carrier:str,values:np.ndarray,timesteps:list[int]|np.ndarray)->np.ndarray:
    plan=build_patch_plan(carrier,values,timesteps)
    with torch.no_grad():pred=run_instrumented(model,observations,lengths,patch_plan=plan,return_trace=False)
    return pred.detach().cpu().numpy().astype(np.float64)


def many_patched_predictions(model,observations:torch.Tensor,lengths:torch.Tensor,carrier:str,value_sets:np.ndarray,timesteps:list[int]|np.ndarray)->np.ndarray:
    vals=np.asarray(value_sets,dtype=np.float64)
    if vals.ndim!=3:raise ValueError("value_sets must be [C,N,D]")
    c,n,_=vals.shape
    obs=observations.repeat((c,1,1));lens=lengths.repeat(c);flat=vals.reshape(c*n,vals.shape[-1]);times=np.tile(np.asarray(timesteps,dtype=np.int64),c)
    pred=patched_predictions(model,obs,lens,carrier,flat,times)
    return pred.reshape(c,n)


def coalition_state_predictions(model,observations:torch.Tensor,lengths:torch.Tensor,base_state:np.ndarray,cf_state:np.ndarray,timesteps:list[int]|np.ndarray,subsets:list[tuple[int,...]])->np.ndarray:
    b=np.asarray(base_state,dtype=np.float64).reshape(len(base_state),10,4);c=np.asarray(cf_state,dtype=np.float64).reshape(len(cf_state),10,4);sets=[]
    for subset in subsets:
        x=b.copy();x[:,list(subset),:]=c[:,list(subset),:];sets.append(x.reshape(len(x),40))
    return many_patched_predictions(model,observations,lengths,"STATE40",np.stack(sets),timesteps)
