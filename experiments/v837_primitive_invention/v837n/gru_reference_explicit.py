from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.v837_primitive_invention.common.reference_models import LearnedReferenceModel
from experiments.v837_primitive_invention.common.task_interface import OBS_DIM


CONDITIONS = {
    "full_gru",
    "static_update_vector",
    "static_update_scalar",
    "no_update",
    "no_reset",
    "static_reset_vector",
    "no_update_no_reset",
}


@dataclass
class ExplicitGRUTrace:
    states: torch.Tensor
    candidates: torch.Tensor
    updates: torch.Tensor
    resets: torch.Tensor
    update_input_components: torch.Tensor
    update_state_components: torch.Tensor
    reset_input_components: torch.Tensor
    reset_state_components: torch.Tensor


class ExplicitGRUReferenceModel(LearnedReferenceModel):
    """PyTorch-GRUCell-equivalent recurrence with independently ablatable mechanisms.

    The full condition preserves torch.nn.GRUCell's exact convention:

      r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
      z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
      n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
      h' = (1-z) * n + z * h

    The input projection and readout are identical to V837j/l GRUReferenceModel.
    """

    architecture_name = "explicit_gru_reference"

    def __init__(self, hidden_size: int = 13, input_dim: int = OBS_DIM, *, condition: str = "full_gru"):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown explicit GRU condition: {condition}")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.condition = condition

        # Creation/reset order intentionally mirrors GRUReferenceModel:
        # input projection -> GRUCell parameters/reset -> readout.
        self.input_projection = nn.Linear(self.input_dim, self.input_dim)
        self.weight_ih = nn.Parameter(torch.empty(3 * self.hidden_size, self.input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * self.hidden_size, self.hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * self.hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * self.hidden_size))
        self.reset_gru_parameters()
        self.readout = nn.Linear(self.hidden_size, 1)

        # Static controls are initialized after all shared parameters so paired
        # conditions start with bit-identical shared weights/readout.
        if condition == "static_update_vector":
            self.static_update_logit = nn.Parameter(torch.zeros(self.hidden_size))
        elif condition == "static_update_scalar":
            self.static_update_logit = nn.Parameter(torch.zeros(1))
        elif condition == "static_reset_vector":
            self.static_reset_logit = nn.Parameter(torch.zeros(self.hidden_size))

    def reset_gru_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for parameter in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh):
            nn.init.uniform_(parameter, -stdv, stdv)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        # Dynamic update/reset slices each contain H*I + H*H + 2H parameters.
        gate_slice = self.hidden_size * self.input_dim + self.hidden_size * self.hidden_size + 2 * self.hidden_size
        active = self.parameter_count()
        if self.condition in {"static_update_vector", "static_update_scalar", "no_update", "no_update_no_reset"}:
            active -= gate_slice
        if self.condition in {"no_reset", "static_reset_vector", "no_update_no_reset"}:
            active -= gate_slice
        return int(active)

    def load_from_framework_gru(self, reference: nn.Module) -> None:
        """Copy V837j GRUReferenceModel weights for equation-equivalence tests."""
        with torch.no_grad():
            self.input_projection.load_state_dict(reference.input_projection.state_dict())
            self.weight_ih.copy_(reference.cell.weight_ih)
            self.weight_hh.copy_(reference.cell.weight_hh)
            self.bias_ih.copy_(reference.cell.bias_ih)
            self.bias_hh.copy_(reference.cell.bias_hh)
            self.readout.load_state_dict(reference.readout.state_dict())

    def _gate_values(self, projected: torch.Tensor, state: torch.Tensor):
        gi = F.linear(projected, self.weight_ih, self.bias_ih)
        gh = F.linear(state, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        dynamic_reset = torch.sigmoid(i_r + h_r)
        dynamic_update = torch.sigmoid(i_z + h_z)

        if self.condition in {"no_reset", "no_update_no_reset"}:
            reset = torch.ones_like(dynamic_reset)
        elif self.condition == "static_reset_vector":
            reset = torch.sigmoid(self.static_reset_logit).view(1, -1).expand_as(dynamic_reset)
        else:
            reset = dynamic_reset

        candidate = torch.tanh(i_n + reset * h_n)

        if self.condition in {"no_update", "no_update_no_reset"}:
            update = torch.zeros_like(dynamic_update)
        elif self.condition == "static_update_vector":
            update = torch.sigmoid(self.static_update_logit).view(1, -1).expand_as(dynamic_update)
        elif self.condition == "static_update_scalar":
            update = torch.sigmoid(self.static_update_logit).view(1, 1).expand_as(dynamic_update)
        else:
            update = dynamic_update

        return {
            "candidate": candidate,
            "update": update,
            "reset": reset,
            "update_input_component": torch.sigmoid(i_z),
            "update_state_component": torch.sigmoid(h_z),
            "reset_input_component": torch.sigmoid(i_r),
            "reset_state_component": torch.sigmoid(h_r),
        }

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        update_override: torch.Tensor | None = None,
        reset_override: torch.Tensor | None = None,
    ):
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.input_dim:
            raise ValueError(f"observation dimension {observed_dim} != input_dim {self.input_dim}")
        if update_override is not None and tuple(update_override.shape) != (batch, steps, self.hidden_size):
            raise ValueError("update_override must have shape [B,T,H]")
        if reset_override is not None and tuple(reset_override.shape) != (batch, steps, self.hidden_size):
            raise ValueError("reset_override must have shape [B,T,H]")

        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        candidates: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        resets: list[torch.Tensor] = []
        update_inputs: list[torch.Tensor] = []
        update_states: list[torch.Tensor] = []
        reset_inputs: list[torch.Tensor] = []
        reset_states: list[torch.Tensor] = []

        for t in range(steps):
            projected = self.input_projection(observations[:, t, :])
            values = self._gate_values(projected, state)
            reset = reset_override[:, t, :] if reset_override is not None else values["reset"]
            # Reset override must also affect candidate construction.
            if reset_override is not None:
                gi = F.linear(projected, self.weight_ih, self.bias_ih)
                gh = F.linear(state, self.weight_hh, self.bias_hh)
                _, _, i_n = gi.chunk(3, dim=1)
                _, _, h_n = gh.chunk(3, dim=1)
                candidate = torch.tanh(i_n + reset * h_n)
            else:
                candidate = values["candidate"]
            update = update_override[:, t, :] if update_override is not None else values["update"]
            proposed = (1.0 - update) * candidate + update * state
            if lengths is None:
                state = proposed
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed + (1.0 - active) * state
            states.append(state)
            if return_trace:
                candidates.append(candidate)
                updates.append(update)
                resets.append(reset)
                update_inputs.append(values["update_input_component"])
                update_states.append(values["update_state_component"])
                reset_inputs.append(values["reset_input_component"])
                reset_states.append(values["reset_state_component"])

        stacked_states = torch.stack(states, dim=1)
        prediction = torch.tanh(self.readout(states[-1])).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, ExplicitGRUTrace(
            states=stacked_states,
            candidates=torch.stack(candidates, dim=1),
            updates=torch.stack(updates, dim=1),
            resets=torch.stack(resets, dim=1),
            update_input_components=torch.stack(update_inputs, dim=1),
            update_state_components=torch.stack(update_states, dim=1),
            reset_input_components=torch.stack(reset_inputs, dim=1),
            reset_state_components=torch.stack(reset_states, dim=1),
        )
