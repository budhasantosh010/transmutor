from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from .adapter_families import apply_gate_tensor, apply_tensor, inverse_tensor
from .aligned_primitive import enabled_ports


@dataclass
class AlignedTransplantTrace:
    states:torch.Tensor
    outputs:torch.Tensor
    messages:torch.Tensor
    candidates:torch.Tensor
    gates:torch.Tensor
    donor_native_states:torch.Tensor


def _map(bundle,port,li=0):
    maps=bundle['ports'][port];return maps[li] if isinstance(maps,list) else maps


def run_aligned_transplant(model,observations:torch.Tensor,lengths:torch.Tensor,*,recipient_nodes:Sequence[int],donor:PortedPrimitive,bundle:dict|None=None,scope:str='000000',return_trace:bool=False):
    recipient_nodes=tuple(sorted(int(x) for x in recipient_nodes));selected=set(recipient_nodes);rec_local={n:i for i,n in enumerate(recipient_nodes)};enabled=enabled_ports(scope);bundle=bundle or {'ports':{}}
    if len(donor.nodes)!=len(recipient_nodes):raise ValueError('donor/recipient motif size mismatch')
    B,T,_=observations.shape;dtype=observations.dtype
    # Recipient external graph is authoritative; only selected->selected internal edges are replaced by donor internals.
    ext_edges=[]
    for idx,e in enumerate(model.graph.edges):
        src,dst,rec=int(e.src),int(e.dst),bool(e.recurrent)
        if src in selected and dst in selected:continue
        ext_edges.append((src,dst,rec,model.base.edge_weights[idx].detach()))
    donor_pos={n:i for i,n in enumerate(donor.nodes)}
    donor_edges=[(donor_pos[e['src']],donor_pos[e['dst']],bool(e['recurrent']),e['weight']) for e in donor.internal_edges]
    prev_states=[torch.zeros(B,4,dtype=dtype) for _ in range(10)];prev_outputs=[torch.zeros(B,4,dtype=dtype) for _ in range(10)]
    donor_prev=[]
    for li in range(len(recipient_nodes)):
        z=torch.zeros(B,4,dtype=dtype);donor_prev.append(apply_tensor(z,_map(bundle,'state',li)) if 'state' in enabled else z)
    trace_s=[];trace_o=[];trace_m=[];trace_c=[];trace_g=[];trace_dn=[]
    for t in range(T):
        x_t=observations[:,t,:];snapshot=prev_states;global_terms=model._global_terms(snapshot);matched_terms=model._matched_local_terms(snapshot);recipient_gate=model._global_gate(snapshot,x_t);active=(t<lengths).to(dtype).unsqueeze(1);inactive=1.0-active
        cur_s=[];cur_o=[];cur_m=[];cur_c=[];cur_dn=[];donor_cur_native_o=[];donor_cur_native_s=[]
        for cell in range(10):
            if cell in selected:
                li=rec_local[cell];external=torch.zeros(B,4,dtype=dtype)
                for src,dst,rec,w in ext_edges:
                    if dst!=cell:continue
                    source=prev_outputs[src] if rec or src>=len(cur_o) else cur_o[src]
                    external=external+w*source
                ext_native=apply_tensor(external,_map(bundle,'external_messages',li)) if 'external_messages' in enabled else external
                internal=torch.zeros(B,4,dtype=dtype)
                for src_li,dst_li,rec,w in donor_edges:
                    if dst_li!=li:continue
                    source=donor_prev[src_li]@donor.wo[src_li].T if rec or src_li>=len(donor_cur_native_o) else donor_cur_native_o[src_li]
                    internal=internal+w*source
                glob=global_terms[cell]+matched_terms[cell]
                if 'global_term' in enabled:glob=apply_tensor(glob,_map(bundle,'global_term',li))
                if 'projected_input' in enabled:
                    p_r=model._visible_input(x_t,cell);projected=apply_tensor(p_r,_map(bundle,'projected_input',li))
                else:projected=F.linear(x_t,donor.proj_w[li],donor.proj_b[li])
                gate=apply_gate_tensor(recipient_gate,_map(bundle,'gate')) if 'gate' in enabled else recipient_gate
                native_prev=donor_prev[li];candidate_native=torch.tanh(native_prev@donor.ws[li].T+(ext_native+internal)@donor.wm[li].T+projected@donor.wx[li].T+glob+donor.b[li]);proposed_native=gate*native_prev+(1.0-gate)*candidate_native;native_out=proposed_native@donor.wo[li].T
                mirror= inverse_tensor(proposed_native,_map(bundle,'state',li)) if 'state' in enabled else proposed_native
                candidate_r=inverse_tensor(candidate_native,_map(bundle,'state',li)) if 'state' in enabled else candidate_native
                out_r=apply_tensor(native_out,_map(bundle,'output',li)) if 'output' in enabled else native_out
                state=active*mirror+inactive*snapshot[cell];out=active*out_r+inactive*prev_outputs[cell];native_state=active*proposed_native+inactive*native_prev
                # Report selected-cell message in recipient coordinates when a message transform exists.
                native_message=ext_native+internal
                msg_r=inverse_tensor(native_message,_map(bundle,'external_messages',li)) if 'external_messages' in enabled else native_message
                cur_s.append(state);cur_o.append(out);cur_m.append(active*msg_r);cur_c.append(candidate_r);donor_cur_native_s.append(native_state);donor_cur_native_o.append(native_out);cur_dn.append(native_state)
            else:
                message=torch.zeros(B,4,dtype=dtype)
                for src,dst,rec,w in ext_edges:
                    if dst!=cell:continue
                    source=prev_outputs[src] if rec or src>=len(cur_o) else cur_o[src]
                    message=message+w*source
                visible=model._visible_input(x_t,cell);candidate=torch.tanh(snapshot[cell]@model.base.cell_ws[cell].T+global_terms[cell]+matched_terms[cell]+message@model.base.cell_wm[cell].T+visible@model.base.cell_wx[cell].T+model.base.cell_b[cell]);proposed=recipient_gate*snapshot[cell]+(1.0-recipient_gate)*candidate;proposed_out=proposed@model.base.cell_wo[cell].T;state=active*proposed+inactive*snapshot[cell];out=active*proposed_out+inactive*prev_outputs[cell]
                cur_s.append(state);cur_o.append(out);cur_m.append(active*message);cur_c.append(candidate);cur_dn.append(torch.full_like(state,float('nan')))
        donor_prev=donor_cur_native_s;prev_states=cur_s;prev_outputs=cur_o
        if return_trace:
            trace_s.append(torch.stack(cur_s,1));trace_o.append(torch.stack(cur_o,1));trace_m.append(torch.stack(cur_m,1));trace_c.append(torch.stack(cur_c,1));trace_g.append(recipient_gate);trace_dn.append(torch.stack(cur_dn,1))
    prediction=torch.tanh(model.base.readout(torch.cat(prev_states,1))).squeeze(-1)
    if not return_trace:return prediction
    return prediction,AlignedTransplantTrace(states=torch.stack(trace_s,1),outputs=torch.stack(trace_o,1),messages=torch.stack(trace_m,1),candidates=torch.stack(trace_c,1),gates=torch.stack(trace_g,1),donor_native_states=torch.stack(trace_dn,1))


class AlignedPrimitiveTransplant:
    def __init__(self,recipient_model,recipient_nodes,donor,bundle=None,scope='000000'):
        self.recipient_model=recipient_model;self.recipient_nodes=tuple(recipient_nodes);self.donor=donor;self.bundle=bundle or {'ports':{}};self.scope=scope
    def __call__(self,observations,lengths,return_trace=False):
        return run_aligned_transplant(self.recipient_model,observations,lengths,recipient_nodes=self.recipient_nodes,donor=self.donor,bundle=self.bundle,scope=self.scope,return_trace=return_trace)
