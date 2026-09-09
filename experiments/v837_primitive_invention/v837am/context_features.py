from __future__ import annotations

import numpy as np
import torch

from .authorization import CONTEXT_SETS
from .interface_data import mask

NAME_TO_PORT={"GATE":"gate","PREV_STATE":"state","EXTERNAL_MESSAGE":"external_messages","GLOBAL_TERM":"global_term","PROJECTED_INPUT":"projected_input"}

def _prev_state(trace,node):
    x=trace.states[:,:,node,:];out=torch.zeros_like(x);out[:,1:]=x[:,:-1];return out

def context_tensor(trace,nodes,li:int,context_set:str,target_port:str,history:int=1):
    B,T=trace.states.shape[:2];parts=[];names=CONTEXT_SETS[context_set]
    for name in names:
        if NAME_TO_PORT.get(name)==target_port:continue
        if name=="TIME":
            t=torch.arange(T,dtype=trace.states.dtype).view(1,T,1).expand(B,-1,-1);den=torch.clamp(trace.lengths.to(trace.states.dtype)-1,min=1).view(B,1,1);parts.append(t/den)
        elif name=="GATE":parts.append(trace.gates)
        elif name=="PREV_STATE":parts.append(_prev_state(trace,nodes[li]))
        elif name=="EXTERNAL_MESSAGE":parts.append(trace.external_messages[:,:,li,:])
        elif name=="GLOBAL_TERM":parts.append(trace.global_terms[:,:,li,:])
        elif name=="PROJECTED_INPUT":parts.append(trace.projected_inputs[:,:,li,:])
    if history>1:
        valid=[]
        for lag in range(1,history):
            ok=(torch.arange(T).view(1,T)>=lag).to(trace.states.dtype).expand(B,-1).unsqueeze(-1);valid.append(ok)
            for src in (trace.gates,trace.external_messages[:,:,li,:],trace.global_terms[:,:,li,:],trace.projected_inputs[:,:,li,:]):
                z=torch.zeros_like(src);z[:,lag:]=src[:,:-lag];parts.append(z)
        parts.append(sum(valid)/max(history-1,1))
    if not parts:return torch.zeros(B,T,0,dtype=trace.states.dtype)
    return torch.cat(parts,dim=-1)
def gate_context_tensor(trace,nodes,context_set:str,history:int=1):
    """One position-invariant context for the single AF1D global scalar gate."""
    per=[context_tensor(trace,nodes,li,context_set,"gate",history) for li in range(len(nodes))]
    if not per:return torch.zeros(trace.states.shape[0],trace.states.shape[1],0,dtype=trace.states.dtype)
    return torch.stack(per,dim=0).mean(dim=0)

def active_context(trace,nodes,li,context_set,target_port,history=1):return context_tensor(trace,nodes,li,context_set,target_port,history)[mask(trace)].numpy()
def active_gate_context(trace,nodes,context_set,history=1):return gate_context_tensor(trace,nodes,context_set,history)[mask(trace)].numpy()
def fit_normalizer(c):
    c=np.asarray(c,dtype=np.float64)
    if c.shape[1]==0:return {"mean":[],"scale":[]}
    mean=c.mean(0);scale=c.std(0);scale=np.where(scale<1e-8,1.0,scale);return {"mean":mean.tolist(),"scale":scale.tolist()}
def normalize(c,norm):
    c=np.asarray(c,dtype=np.float64)
    if c.shape[-1]==0:return c
    return (c-np.asarray(norm["mean"]))/np.asarray(norm["scale"])
def normalize_tensor(c,norm):
    if c.shape[-1]==0:return c
    return (c-torch.tensor(norm["mean"],dtype=c.dtype))/torch.tensor(norm["scale"],dtype=c.dtype)
