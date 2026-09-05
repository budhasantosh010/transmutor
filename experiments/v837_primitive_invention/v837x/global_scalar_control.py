from __future__ import annotations
import math
import torch
from torch import nn
from experiments.v837_primitive_invention.common.substrate import ForwardTrace, NeutralGraphModel

CONDITIONS={'X0_historical_direct','X1_local_scalar_carry','X2_global_scalar_carry','X2C_global_scale_candidate_control'}
MODES={'INPUT_ONLY_GLOBAL_SCALAR','STATE_ONLY_GLOBAL_SCALAR','JOINT_INPUT_STATE_GLOBAL_SCALAR'}

class GlobalScalarNeutralModel(NeutralGraphModel):
    """Frozen 10x4 neutral substrate with one authorized scalar controller."""
    def __init__(self,graph,*,condition:str,authorized_mode:str='JOINT_INPUT_STATE_GLOBAL_SCALAR',obs_dim:int=6):
        if condition not in CONDITIONS: raise ValueError(condition)
        if authorized_mode not in MODES: raise ValueError(authorized_mode)
        self.global_condition=condition; self.authorized_mode=authorized_mode
        modulation='dynamic_scalar_candidate' if condition=='X1_local_scalar_carry' else 'none'
        super().__init__(graph,obs_dim=obs_dim,state_dim=4,message_dim=4,state_update_mode='direct',interaction_mode='none',state_modulation_mode=modulation)
        if condition in {'X2_global_scalar_carry','X2C_global_scale_candidate_control'}:
            if authorized_mode in {'STATE_ONLY_GLOBAL_SCALAR','JOINT_INPUT_STATE_GLOBAL_SCALAR'}:
                self.global_ws=nn.Parameter(torch.randn(40)*(0.20/math.sqrt(40)))
            else: self.register_parameter('global_ws',None)
            if authorized_mode in {'INPUT_ONLY_GLOBAL_SCALAR','JOINT_INPUT_STATE_GLOBAL_SCALAR'}:
                self.global_wx=nn.Parameter(torch.randn(obs_dim)*(0.20/math.sqrt(obs_dim)))
            else: self.register_parameter('global_wx',None)
            self.global_b=nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('global_ws',None); self.register_parameter('global_wx',None); self.register_parameter('global_b',None)
    @property
    def controller_param_count(self):
        if self.global_condition=='X1_local_scalar_carry': return 150
        if self.global_condition not in {'X2_global_scalar_carry','X2C_global_scale_candidate_control'}: return 0
        return {'INPUT_ONLY_GLOBAL_SCALAR':7,'STATE_ONLY_GLOBAL_SCALAR':41,'JOINT_INPUT_STATE_GLOBAL_SCALAR':47}[self.authorized_mode]
    @property
    def controller_macs(self):
        if self.global_condition=='X1_local_scalar_carry': return 140
        if self.global_condition not in {'X2_global_scalar_carry','X2C_global_scale_candidate_control'}: return 0
        return {'INPUT_ONLY_GLOBAL_SCALAR':6,'STATE_ONLY_GLOBAL_SCALAR':40,'JOINT_INPUT_STATE_GLOBAL_SCALAR':46}[self.authorized_mode]
    def _global_gate(self,prev_states,x_t):
        terms=0.0
        if self.global_ws is not None: terms=terms+torch.sum(torch.cat(prev_states,dim=1)*self.global_ws.view(1,-1),dim=1,keepdim=True)
        if self.global_wx is not None: terms=terms+torch.sum(x_t*self.global_wx.view(1,-1),dim=1,keepdim=True)
        return torch.sigmoid(terms+self.global_b)
    def forward(self,observations,lengths=None,*,return_trace=False):
        if self.global_condition=='X0_historical_direct': return super().forward(observations,lengths,return_trace=return_trace)
        if observations.ndim!=3: raise ValueError('observations must be [B,T,D]')
        batch,steps,dim=observations.shape
        if dim!=self.obs_dim: raise ValueError('observation dimension mismatch')
        n=len(self.graph.cells); device=observations.device
        prev_states=[torch.zeros(batch,4,device=device) for _ in range(n)]; prev_outputs=[torch.zeros(batch,4,device=device) for _ in range(n)]
        sts=[]; cands=[]; outs=[]; msgs=[]; recs=[]; mts=[]; its=[]; gates=[]
        for t in range(steps):
            x_t=observations[:,t,:]
            global_gate=self._global_gate(prev_states,x_t) if self.global_condition.startswith('X2') else None
            cs=[]; cc=[]; co=[]; cm=[]; cr=[]; cmt=[]; cit=[]; cg=[]
            for i in range(n):
                message=torch.zeros(batch,4,device=device)
                for ei,e in enumerate(self.graph.edges):
                    if e.dst!=i: continue
                    src=prev_outputs[e.src] if e.recurrent or e.src>=len(co) else co[e.src]
                    message=message+self.edge_weights[ei]*src
                if self.input_access_mode=='none': visible_x=torch.zeros_like(x_t)
                elif self.input_access_mode=='broadcast': visible_x=x_t
                else: visible_x=x_t*self.input_access_mask[:,i].view(1,-1)
                if self.global_condition=='X1_local_scalar_carry':
                    g=torch.sigmoid(torch.sum(prev_states[i]*self.cell_gs[i].view(1,-1),1,keepdim=True)+torch.sum(message*self.cell_gm[i].view(1,-1),1,keepdim=True)+torch.sum(visible_x*self.cell_gx[i].view(1,-1),1,keepdim=True)+self.cell_gb[i])
                else: g=global_gate
                rt=prev_states[i]@self.cell_ws[i].T; mt=message@self.cell_wm[i].T; it=visible_x@self.cell_wx[i].T; cand=torch.tanh(rt+mt+it+self.cell_b[i])
                proposed=(g*prev_states[i]+(1-g)*cand) if self.global_condition in {'X1_local_scalar_carry','X2_global_scalar_carry'} else g*cand
                po=proposed@self.cell_wo[i].T
                if lengths is not None:
                    active=(t<lengths).to(observations.dtype).unsqueeze(1); state=active*proposed+(1-active)*prev_states[i]; output=active*po+(1-active)*prev_outputs[i]
                    if return_trace: message=active*message; rt=active*rt; mt=active*mt; it=active*it; g=active*g+(1-active)
                else: state=proposed; output=po
                cs.append(state); cc.append(cand); co.append(output)
                if return_trace: cm.append(message); cr.append(rt); cmt.append(mt); cit.append(it); cg.append(g)
            prev_states=cs; prev_outputs=co
            if return_trace: sts.append(torch.stack(cs,1)); cands.append(torch.stack(cc,1)); outs.append(torch.stack(co,1)); msgs.append(torch.stack(cm,1)); recs.append(torch.stack(cr,1)); mts.append(torch.stack(cmt,1)); its.append(torch.stack(cit,1)); gates.append(torch.stack(cg,1))
        pred=torch.tanh(self.readout(torch.cat(prev_states,dim=1))).squeeze(-1)
        if not return_trace: return pred
        return pred,ForwardTrace(states=torch.stack(sts,1),candidate_states=torch.stack(cands,1),outputs=torch.stack(outs,1),messages=torch.stack(msgs,1),recurrent_terms=torch.stack(recs,1),message_terms=torch.stack(mts,1),input_terms=torch.stack(its,1),state_modulators=torch.stack(gates,1))
