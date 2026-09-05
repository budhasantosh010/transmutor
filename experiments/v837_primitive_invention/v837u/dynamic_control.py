from __future__ import annotations

import torch

from experiments.v837_primitive_invention.common.substrate import ForwardTrace, NeutralGraphModel


CONDITIONS = {
    "U0_historical_direct",
    "U1_v837p_scalar_candidate",
    "U2_dynamic_scalar_carry",
    "U2C_scalar_scale_candidate_control",
}


class NeutralDynamicCarryModel(NeutralGraphModel):
    """Authorized V837u scalar-carry transfer with an exact same-controller control."""

    def __init__(self, graph, *, condition: str, obs_dim: int = 6):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown V837u condition: {condition}")
        self.followup_condition = condition
        modulation = "none" if condition == "U0_historical_direct" else "dynamic_scalar_candidate"
        super().__init__(
            graph, obs_dim=obs_dim, state_dim=4, message_dim=4,
            state_update_mode="direct", interaction_mode="none", state_modulation_mode=modulation,
        )

    @property
    def controller_output_dim(self) -> int:
        return 0 if self.followup_condition == "U0_historical_direct" else 1

    @property
    def controller_parameter_count(self) -> int:
        return 0 if self.controller_output_dim == 0 else 150

    @property
    def controller_macs_per_timestep(self) -> int:
        return 0 if self.controller_output_dim == 0 else 140

    @property
    def state_modulation_location(self) -> str:
        return {
            "U0_historical_direct": "none",
            "U1_v837p_scalar_candidate": "pre_transform_candidate_state_access",
            "U2_dynamic_scalar_carry": "state_carry_preserve_replace",
            "U2C_scalar_scale_candidate_control": "candidate_output_scale_without_old_state_carry",
        }[self.followup_condition]

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, return_trace: bool = False):
        if self.followup_condition in {"U0_historical_direct", "U1_v837p_scalar_candidate"}:
            return super().forward(observations, lengths, return_trace=return_trace)
        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.obs_dim:
            raise ValueError(f"observation dimension {observed_dim} != model obs_dim {self.obs_dim}")
        n=len(self.graph.cells); device=observations.device
        prev_states=[torch.zeros(batch,self.state_dim,device=device) for _ in range(n)]
        prev_outputs=[torch.zeros(batch,self.message_dim,device=device) for _ in range(n)]
        st=[]; cand=[]; outs=[]; msgs=[]; recs=[]; mterms=[]; iterms=[]; mods=[]
        for t in range(steps):
            x_t=observations[:,t,:]; cs=[]; cc=[]; co=[]; cm=[]; cr=[]; cmt=[]; cit=[]; cg=[]
            for i in range(n):
                message=torch.zeros(batch,self.message_dim,device=device)
                for edge_index,edge in enumerate(self.graph.edges):
                    if edge.dst!=i: continue
                    source=prev_outputs[edge.src] if edge.recurrent or edge.src>=len(co) else co[edge.src]
                    message=message+self.edge_weights[edge_index]*source
                if self.input_access_mode=="none": visible_x=torch.zeros_like(x_t)
                elif self.input_access_mode=="broadcast": visible_x=x_t
                else: visible_x=x_t*self.input_access_mask[:,i].view(1,-1)
                g=torch.sigmoid(
                    torch.sum(prev_states[i]*self.cell_gs[i].view(1,-1),dim=1,keepdim=True)
                    +torch.sum(message*self.cell_gm[i].view(1,-1),dim=1,keepdim=True)
                    +torch.sum(visible_x*self.cell_gx[i].view(1,-1),dim=1,keepdim=True)
                    +self.cell_gb[i]
                )
                recurrent_term=prev_states[i]@self.cell_ws[i].T
                message_term=message@self.cell_wm[i].T
                input_term=visible_x@self.cell_wx[i].T
                candidate=torch.tanh(recurrent_term+message_term+input_term+self.cell_b[i])
                if self.followup_condition=="U2_dynamic_scalar_carry": proposed=g*prev_states[i]+(1.0-g)*candidate
                else: proposed=g*candidate
                proposed_output=proposed@self.cell_wo[i].T
                if lengths is not None:
                    active=(t<lengths).to(observations.dtype).unsqueeze(1)
                    state=active*proposed+(1-active)*prev_states[i]
                    output=active*proposed_output+(1-active)*prev_outputs[i]
                    if return_trace:
                        message=active*message; recurrent_term=active*recurrent_term; message_term=active*message_term; input_term=active*input_term; g=active*g+(1-active)
                else: state=proposed; output=proposed_output
                cs.append(state); cc.append(candidate); co.append(output)
                if return_trace: cm.append(message); cr.append(recurrent_term); cmt.append(message_term); cit.append(input_term); cg.append(g)
            prev_states=cs; prev_outputs=co
            if return_trace:
                st.append(torch.stack(cs,1)); cand.append(torch.stack(cc,1)); outs.append(torch.stack(co,1)); msgs.append(torch.stack(cm,1)); recs.append(torch.stack(cr,1)); mterms.append(torch.stack(cmt,1)); iterms.append(torch.stack(cit,1)); mods.append(torch.stack(cg,1))
        prediction=torch.tanh(self.readout(torch.cat(prev_states,dim=1))).squeeze(-1)
        if not return_trace: return prediction
        return prediction, ForwardTrace(
            states=torch.stack(st,1), candidate_states=torch.stack(cand,1), outputs=torch.stack(outs,1), messages=torch.stack(msgs,1),
            recurrent_terms=torch.stack(recs,1), message_terms=torch.stack(mterms,1), input_terms=torch.stack(iterms,1), state_modulators=torch.stack(mods,1),
        )
