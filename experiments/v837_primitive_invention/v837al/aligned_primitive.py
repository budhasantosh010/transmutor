from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from .adapter_families import apply_tensor, inverse_tensor, apply_gate_tensor

PORT_ORDER=("state","external_messages","global_term","projected_input","gate","output")


def enabled_ports(scope:str)->set[str]:
    if len(scope)!=6 or any(c not in "01" for c in scope): raise ValueError(scope)
    return {p for p,b in zip(PORT_ORDER,scope) if b=="1"}

@dataclass
class AlignedReplay:
    states:torch.Tensor
    candidates:torch.Tensor
    outputs:torch.Tensor
    donor_native_states:torch.Tensor
    donor_native_outputs:torch.Tensor
    internal_messages:torch.Tensor


class AlignedPortedPrimitive:
    """Experiment-local AF1D primitive with six explicit low-complexity boundary ports."""
    def __init__(self, donor:PortedPrimitive, recipient:PortedPrimitive):
        if len(donor.nodes)!=len(recipient.nodes): raise ValueError("motif size mismatch")
        self.donor=donor;self.recipient=recipient;self.k=len(donor.nodes)

    @staticmethod
    def _m(bundle,port,li=0):
        maps=bundle["ports"][port];return maps[li] if isinstance(maps,list) else maps

    def replay_teacher_forced(self,trace,recipient_nodes:Sequence[int],bundle:dict,scope:str)->AlignedReplay:
        enabled=enabled_ports(scope);obs=trace.observations.detach().cpu();lengths=trace.lengths.detach().cpu();B,T=obs.shape[:2];dtype=obs.dtype
        rnodes=list(recipient_nodes);rstates=trace.states[:,:,rnodes,:].detach().cpu();prev_r=torch.zeros_like(rstates);prev_r[:,1:]=rstates[:,:-1]
        native_states=[];native_outputs=[];states=[];cands=[];outs=[];internals=[]
        for t in range(T):
            x_t=obs[:,t,:];cur_native_s=[];cur_native_o=[];cur_s=[];cur_c=[];cur_o=[];cur_i=[]
            for li,orig in enumerate(self.donor.nodes):
                rs=prev_r[:,t,li,:]
                ds=apply_tensor(rs,self._m(bundle,"state",li)) if "state" in enabled else rs
                msg=torch.zeros(B,4,dtype=dtype)
                for edge in self.donor.internal_edges:
                    if edge["dst"]!=orig:continue
                    src=self.donor.node_to_local[edge["src"]]
                    source=(cur_native_o[src] if (not edge["recurrent"] and src<len(cur_native_o)) else apply_tensor(prev_r[:,t,src,:],self._m(bundle,"state",src))@self.donor.wo[src].T if "state" in enabled else prev_r[:,t,src,:]@self.donor.wo[src].T)
                    msg=msg+edge["weight"]*source
                ext=trace.external_messages[:,t,li,:].detach().cpu();glob=trace.global_terms[:,t,li,:].detach().cpu()
                if "external_messages" in enabled: ext=apply_tensor(ext,self._m(bundle,"external_messages",li))
                if "global_term" in enabled: glob=apply_tensor(glob,self._m(bundle,"global_term",li))
                if "projected_input" in enabled:
                    projected=apply_tensor(trace.projected_inputs[:,t,li,:].detach().cpu(),self._m(bundle,"projected_input",li))
                else: projected=F.linear(x_t,self.donor.proj_w[li],self.donor.proj_b[li])
                total=ext+msg; local=ds@self.donor.ws[li].T; mterm=total@self.donor.wm[li].T; inp=projected@self.donor.wx[li].T
                cand_native=torch.tanh(local+glob+mterm+inp+self.donor.b[li]);gate=trace.gates[:,t,:].detach().cpu()
                if "gate" in enabled: gate=apply_gate_tensor(gate,self._m(bundle,"gate"))
                prop_native=gate*ds+(1.0-gate)*cand_native;out_native=prop_native@self.donor.wo[li].T;active=(t<lengths).to(dtype).unsqueeze(1)
                # Padded timesteps preserve recipient history in recipient coordinates.
                rs_new=inverse_tensor(prop_native,self._m(bundle,"state",li)) if "state" in enabled else prop_native
                cand_r=inverse_tensor(cand_native,self._m(bundle,"state",li)) if "state" in enabled else cand_native
                out_r=apply_tensor(out_native,self._m(bundle,"output",li)) if "output" in enabled else out_native
                rs_new=active*rs_new+(1-active)*rs; out_r=active*out_r+(1-active)*(trace.outputs[:,t,rnodes[li],:].detach().cpu())
                cur_native_s.append(prop_native);cur_native_o.append(out_native);cur_s.append(rs_new);cur_c.append(cand_r);cur_o.append(out_r);cur_i.append(active*msg)
            native_states.append(torch.stack(cur_native_s,1));native_outputs.append(torch.stack(cur_native_o,1));states.append(torch.stack(cur_s,1));cands.append(torch.stack(cur_c,1));outs.append(torch.stack(cur_o,1));internals.append(torch.stack(cur_i,1))
        return AlignedReplay(states=torch.stack(states,1),candidates=torch.stack(cands,1),outputs=torch.stack(outs,1),donor_native_states=torch.stack(native_states,1),donor_native_outputs=torch.stack(native_outputs,1),internal_messages=torch.stack(internals,1))
