from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive


@dataclass
class InterventionTrace:
    states: torch.Tensor
    outputs: torch.Tensor
    messages: torch.Tensor
    candidates: torch.Tensor


def _randomize_tensor_norm(t:torch.Tensor,g:torch.Generator)->torch.Tensor:
    if not torch.is_floating_point(t):return t.clone()
    norm=torch.linalg.vector_norm(t)
    if float(norm)==0.0:return torch.zeros_like(t)
    r=torch.randn(t.shape,generator=g,dtype=t.dtype);rnorm=torch.linalg.vector_norm(r)
    return r*(norm/max(float(rnorm),1e-12))


def randomized_primitive(primitive:PortedPrimitive,seed:int)->PortedPrimitive:
    p=copy.deepcopy(primitive);g=torch.Generator(device="cpu").manual_seed(int(seed))
    for attr in ("ws","wm","wx","b","wo","proj_w","proj_b"):
        setattr(p,attr,[_randomize_tensor_norm(t,g) for t in getattr(p,attr)])
    for edge in p.internal_edges:edge["weight"]=_randomize_tensor_norm(edge["weight"],g)
    return p


def run_intervened(model,observations:torch.Tensor,lengths:torch.Tensor,*,freeze_nodes:Sequence[int]=(),recipient_nodes:Sequence[int]|None=None,donor:PortedPrimitive|None=None,disabled_edges:set[tuple[int,int,bool]]|None=None,return_trace:bool=False):
    if observations.ndim!=3:raise ValueError("observations must be [B,T,D]")
    B,T,_=observations.shape;dtype=observations.dtype;freeze=set(int(x) for x in freeze_nodes)
    recipient_nodes=tuple(sorted(int(x) for x in (recipient_nodes or ())))
    selected=set(recipient_nodes);rec_local={n:i for i,n in enumerate(recipient_nodes)}
    if donor is not None and len(donor.nodes)!=len(recipient_nodes):raise ValueError("donor/recipient motif size mismatch")
    # Preserve every recipient edge except edges internal to a transplanted motif.
    recipient_edges=[];disabled_edges=disabled_edges or set()
    for idx,e in enumerate(model.graph.edges):
        key=(int(e.src),int(e.dst),bool(e.recurrent))
        if key in disabled_edges:continue
        if donor is not None and int(e.src) in selected and int(e.dst) in selected:continue
        recipient_edges.append((int(e.src),int(e.dst),bool(e.recurrent),model.base.edge_weights[idx].detach()))
    donor_edges=[]
    if donor is not None:
        donor_pos={n:i for i,n in enumerate(donor.nodes)}
        for e in donor.internal_edges:
            src=recipient_nodes[donor_pos[e["src"]]];dst=recipient_nodes[donor_pos[e["dst"]]]
            donor_edges.append((src,dst,bool(e["recurrent"]),e["weight"]))
    edges=recipient_edges+donor_edges
    prev_states=[torch.zeros(B,4,dtype=dtype) for _ in range(10)];prev_outputs=[torch.zeros(B,4,dtype=dtype) for _ in range(10)]
    trace_states=[];trace_outputs=[];trace_messages=[];trace_candidates=[]
    for t in range(T):
        x_t=observations[:,t,:];snapshot=prev_states;global_terms=model._global_terms(snapshot);matched_terms=model._matched_local_terms(snapshot);gate=model._global_gate(snapshot,x_t)
        current_states=[];current_outputs=[];current_messages=[];current_candidates=[]
        active=(t<lengths).to(dtype).unsqueeze(1);inactive=1.0-active
        for cell in range(10):
            message=torch.zeros(B,4,dtype=dtype)
            for src,dst,rec,w in edges:
                if dst!=cell:continue
                source=prev_outputs[src] if rec or src>=len(current_outputs) else current_outputs[src]
                message=message+w*source
            if donor is not None and cell in selected:
                li=rec_local[cell];visible=F.linear(x_t,donor.proj_w[li],donor.proj_b[li]);ws,wm,wx,b,wo=donor.ws[li],donor.wm[li],donor.wx[li],donor.b[li],donor.wo[li]
            else:
                visible=model._visible_input(x_t,cell);ws,wm,wx,b,wo=model.base.cell_ws[cell],model.base.cell_wm[cell],model.base.cell_wx[cell],model.base.cell_b[cell],model.base.cell_wo[cell]
            local=snapshot[cell]@ws.T;msg=message@wm.T;inp=visible@wx.T;candidate=torch.tanh(local+global_terms[cell]+matched_terms[cell]+msg+inp+b)
            proposed=gate*snapshot[cell]+(1.0-gate)*candidate;proposed_out=proposed@wo.T
            state=active*proposed+inactive*snapshot[cell];output=active*proposed_out+inactive*prev_outputs[cell]
            if cell in freeze:
                state=active*snapshot[cell]+inactive*snapshot[cell];output=active*prev_outputs[cell]+inactive*prev_outputs[cell]
            current_states.append(state);current_outputs.append(output);current_messages.append(active*message);current_candidates.append(candidate)
        prev_states=current_states;prev_outputs=current_outputs
        if return_trace:
            trace_states.append(torch.stack(current_states,1));trace_outputs.append(torch.stack(current_outputs,1));trace_messages.append(torch.stack(current_messages,1));trace_candidates.append(torch.stack(current_candidates,1))
    prediction=torch.tanh(model.base.readout(torch.cat(prev_states,1))).squeeze(-1)
    if not return_trace:return prediction
    return prediction,InterventionTrace(states=torch.stack(trace_states,1),outputs=torch.stack(trace_outputs,1),messages=torch.stack(trace_messages,1),candidates=torch.stack(trace_candidates,1))
