from __future__ import annotations

from dataclasses import dataclass
import torch

from .instrumented_af1d import AF1DTrace
from .utils import HERE, write_json


@dataclass
class MessageContributionTrace:
    contributions: torch.Tensor  # [B,T,E,4]
    edge_ids: list[str]
    edge_meta: list[dict]


def edge_id(index:int,edge)->str:
    return f"M:e{int(index)}:{int(edge.src)}->{int(edge.dst)}:R{int(bool(edge.recurrent))}"


def decompose_message_contributions(model,trace:AF1DTrace,lengths:torch.Tensor|None=None)->MessageContributionTrace:
    B,T,N,D=trace.outputs.shape
    prev=torch.zeros(B,N,D,dtype=trace.outputs.dtype,device=trace.outputs.device)
    rows=[];ids=[];meta=[]
    for ei,edge in enumerate(model.graph.edges):
        ids.append(edge_id(ei,edge));meta.append({"edge_index":ei,"src":int(edge.src),"dst":int(edge.dst),"recurrent":bool(edge.recurrent)})
    for t in range(T):
        per=[]
        for ei,edge in enumerate(model.graph.edges):
            # Historical mixed schedule: at dst cell evaluation, non-recurrent
            # edges may consume a same-step source only when that source cell
            # has already been evaluated (src < dst). All other edges consume
            # previous-timestep output.
            source=trace.outputs[:,t,int(edge.src),:] if (not edge.recurrent and int(edge.src)<int(edge.dst)) else prev[:,int(edge.src),:]
            per.append(model.base.edge_weights[ei]*source)
        step=torch.stack(per,dim=1) if per else torch.zeros(B,0,D,dtype=trace.outputs.dtype,device=trace.outputs.device)
        if lengths is not None:
            active=(t<lengths.to(device=step.device)).to(step.dtype)[:,None,None]
            step=step*active
        rows.append(step)
        prev=trace.outputs[:,t]
    return MessageContributionTrace(torch.stack(rows,dim=1),ids,meta)


def verify_message_decomposition(model,trace:AF1DTrace,lengths:torch.Tensor|None=None,tolerance:float=1e-6)->dict:
    dec=decompose_message_contributions(model,trace,lengths=lengths);recon=torch.zeros_like(trace.messages)
    for ei,edge in enumerate(model.graph.edges):recon[:,:,int(edge.dst),:]+=dec.contributions[:,:,ei,:]
    delta=float(torch.max(torch.abs(recon-trace.messages)).item())
    if delta>tolerance:raise RuntimeError(f"MESSAGE_CONTRIBUTION_DECOMPOSITION_INVALID:{delta}")
    return {"pass":True,"edge_count":len(dec.edge_ids),"max_abs_error":delta,"tolerance":tolerance,"edge_ids":dec.edge_ids}
