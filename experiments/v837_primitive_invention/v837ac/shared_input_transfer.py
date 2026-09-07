from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837y.candidate_interaction import CandidateInteractionSpec, ControlledCandidateInteractionModel

AUTHORIZED_MODE = "TRAINABLE_CONTROLLER_INPUT_FACTORIZATION"
CONDITIONS = {"AC0_y3_parent", "AC1_controller_input_factorization", "AC1F_folded_control"}


def exact_t2_projection_from_seed(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the first nn.Linear(6,6) initialization created by T2 under a frozen init seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        layer = nn.Linear(6, 6)
        return layer.weight.detach().clone(), layer.bias.detach().clone()


class ControllerInputTransferY3(ControlledCandidateInteractionModel):
    """Exact Y3 with only the globally consumed controller input parameterization varied."""

    architecture_name = "v837ac_controller_input_transfer_y3"

    def __init__(self, graph, *, condition: str, coupling_initialization_seed: int, projection_seed: int):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown V837ac condition: {condition}")
        self.transfer_condition = condition
        spec = CandidateInteractionSpec(global_scalar_control=True, candidate_coupling_mode="rank4_cross_block", coupling_rank=4)
        super().__init__(graph, spec=spec, coupling_initialization_seed=int(coupling_initialization_seed), obs_dim=6)
        self.shared_input_projection: nn.Linear | None
        if condition == "AC1_controller_input_factorization":
            effective_w = self.global_wx.detach().clone()
            effective_b = self.global_b.detach().clone()
            A, a = exact_t2_projection_from_seed(projection_seed)
            q = torch.linalg.solve(A.T, effective_w)
            d = effective_b - torch.dot(q, a).view_as(effective_b)
            self.shared_input_projection = nn.Linear(6, 6)
            with torch.no_grad():
                self.shared_input_projection.weight.copy_(A)
                self.shared_input_projection.bias.copy_(a)
                self.global_wx.copy_(q)
                self.global_b.copy_(d)
            self.register_buffer("initial_projection_weight", A.clone(), persistent=False)
            self.register_buffer("initial_projection_bias", a.clone(), persistent=False)
        else:
            self.shared_input_projection = None
            self.register_buffer("initial_projection_weight", torch.empty(0), persistent=False)
            self.register_buffer("initial_projection_bias", torch.empty(0), persistent=False)

    @property
    def projection_active(self) -> bool:
        return self.shared_input_projection is not None

    @property
    def projection_specific_macs(self) -> int:
        return 36 if self.projection_active else 0

    @property
    def total_recurrent_controller_projection_macs(self) -> int:
        return int(self.recurrent_controller_macs + self.projection_specific_macs)

    def effective_controller_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.projection_active:
            return self.global_wx, self.global_b
        A = self.shared_input_projection.weight
        a = self.shared_input_projection.bias
        q = self.global_wx
        d = self.global_b
        return A.T @ q, d + torch.dot(q, a).view_as(d)

    def projection_diagnostics(self) -> dict:
        if not self.projection_active:
            w,b=self.effective_controller_input()
            return {"active":False,"singular_values":[],"condition_number":None,"frobenius_norm":0.0,"bias_norm":0.0,"projection_drift":0.0,"effective_controller_input_norm":float(torch.linalg.vector_norm(w.detach()).item()),"effective_controller_bias":float(b.detach().item())}
        A=self.shared_input_projection.weight.detach().cpu(); a=self.shared_input_projection.bias.detach().cpu(); singular=torch.linalg.svdvals(A); mn=float(singular.min()); mx=float(singular.max()); w,b=self.effective_controller_input()
        return {"active":True,"trainable":True,"singular_values":[float(v) for v in singular.tolist()],"condition_number":float(mx/max(mn,1e-12)),"frobenius_norm":float(torch.linalg.vector_norm(A).item()),"bias_norm":float(torch.linalg.vector_norm(a).item()),"projection_drift":float(torch.linalg.vector_norm(A-self.initial_projection_weight.detach().cpu()).item()),"projection_bias_drift":float(torch.linalg.vector_norm(a-self.initial_projection_bias.detach().cpu()).item()),"effective_controller_input_norm":float(torch.linalg.vector_norm(w.detach()).item()),"effective_controller_bias":float(b.detach().item())}

    def _global_gate(self, snapshot_states: list[torch.Tensor], x_t: torch.Tensor) -> torch.Tensor:
        if not self.projection_active:
            return super()._global_gate(snapshot_states, x_t)
        stacked=torch.cat(snapshot_states,dim=1)
        state_term=torch.sum(stacked*self.global_ws.view(1,-1),dim=1,keepdim=True)
        projected=self.shared_input_projection(x_t)
        input_term=torch.sum(projected*self.global_wx.view(1,-1),dim=1,keepdim=True)
        return torch.sigmoid(state_term+input_term+self.global_b)
