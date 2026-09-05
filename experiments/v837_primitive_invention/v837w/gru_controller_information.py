from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU, scalarize_dynamic_gate


INFORMATION_MODES = {
    "W0_joint_input_state": "joint",
    "W1_input_only": "input_only",
    "W2_state_only": "state_only",
    "W3_bias_only": "bias_only",
}


@dataclass
class ControllerInformationTrace:
    states: torch.Tensor
    candidates: torch.Tensor
    updates: torch.Tensor
    raw_update_vectors: torch.Tensor
    input_logits: torch.Tensor
    state_logits: torch.Tensor
    bias_logits: torch.Tensor


class ControllerInformationGRU(DynamicGranularityGRU):
    """Exact T2 no-reset scalarized GRU with only update-controller information ablated."""

    architecture_name = "v837w_controller_information_gru"

    def __init__(self, hidden_size: int = 13, input_dim: int = 6, *, condition: str):
        if condition not in INFORMATION_MODES:
            raise ValueError(f"unknown V837w condition: {condition}")
        self.information_condition = condition
        self.information_mode = INFORMATION_MODES[condition]
        super().__init__(hidden_size=hidden_size, input_dim=input_dim, condition="T2_scalarized_update_no_reset")
        self.condition = condition

    @property
    def nominal_update_controller_parameters(self) -> int:
        return self.hidden_size * self.input_dim + self.hidden_size * self.hidden_size + 2 * self.hidden_size

    @property
    def active_update_controller_parameters(self) -> int:
        input_weights = self.hidden_size * self.input_dim
        state_weights = self.hidden_size * self.hidden_size
        biases = 2 * self.hidden_size
        if self.information_mode == "joint":
            return input_weights + state_weights + biases
        if self.information_mode == "input_only":
            return input_weights + biases
        if self.information_mode == "state_only":
            return state_weights + biases
        return biases

    def active_parameter_count(self) -> int:
        # Candidate/input-projection/readout core is 329 parameters for H=13,I=6;
        # reset is disabled in all conditions. Only update-controller information
        # source changes across W0-W3.
        core_without_update = super().active_parameter_count() - self.nominal_update_controller_parameters
        return int(core_without_update + self.active_update_controller_parameters)

    def _decomposed_components(self, projected: torch.Tensor, state: torch.Tensor, *, mode: str | None = None) -> dict[str, torch.Tensor]:
        mode = mode or self.information_mode
        # Execute the frozen T2 fused projections first. This preserves the
        # exact accumulation order for W0 and the candidate pathway.
        gi = F.linear(projected, self.weight_ih, self.bias_ih)
        gh = F.linear(state, self.weight_hh, self.bias_hh)
        _, i_z, i_n = gi.chunk(3, dim=1)
        _, h_z, h_n = gh.chunk(3, dim=1)

        # Separately expose the clean source decomposition required by V837w.
        _, w_iz, _ = self.weight_ih.chunk(3, dim=0)
        _, w_hz, _ = self.weight_hh.chunk(3, dim=0)
        _, b_iz, _ = self.bias_ih.chunk(3, dim=0)
        _, b_hz, _ = self.bias_hh.chunk(3, dim=0)
        input_logit = F.linear(projected, w_iz, None)
        state_logit = F.linear(state, w_hz, None)
        bias_logit = (b_iz + b_hz).view(1, -1).expand_as(input_logit)

        if mode == "joint":
            # Exact T2 anchor: use the fused biased chunks, not an
            # algebraically-equivalent re-association of the three terms.
            raw = torch.sigmoid(i_z + h_z)
        elif mode == "input_only":
            raw = torch.sigmoid(input_logit + bias_logit)
        elif mode == "state_only":
            raw = torch.sigmoid(state_logit + bias_logit)
        elif mode == "bias_only":
            raw = torch.sigmoid(bias_logit)
        else:
            raise ValueError(mode)
        update = scalarize_dynamic_gate(raw)

        # No reset: exact T2 candidate with reset fixed to one, using the
        # same fused projection chunks as the historical implementation.
        candidate = torch.tanh(i_n + h_n)
        return {
            "candidate": candidate,
            "update": update,
            "raw_update": raw,
            "input_logit": input_logit,
            "state_logit": state_logit,
            "bias_logit": bias_logit,
        }

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        information_mode_override: str | None = None,
    ):
        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.input_dim:
            raise ValueError(f"observation dimension {observed_dim} != input_dim {self.input_dim}")
        mode = information_mode_override or self.information_mode
        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states, candidates, updates, raw_updates, input_logits, state_logits, bias_logits = [], [], [], [], [], [], []
        for t in range(steps):
            projected = self.input_projection(observations[:, t, :])
            values = self._decomposed_components(projected, state, mode=mode)
            proposed = (1.0 - values["update"]) * values["candidate"] + values["update"] * state
            if lengths is None:
                state = proposed
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed + (1.0 - active) * state
            states.append(state)
            if return_trace:
                candidates.append(values["candidate"])
                updates.append(values["update"])
                raw_updates.append(values["raw_update"])
                input_logits.append(values["input_logit"])
                state_logits.append(values["state_logit"])
                bias_logits.append(values["bias_logit"])
        prediction = torch.tanh(self.readout(states[-1])).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, ControllerInformationTrace(
            states=torch.stack(states, dim=1),
            candidates=torch.stack(candidates, dim=1),
            updates=torch.stack(updates, dim=1),
            raw_update_vectors=torch.stack(raw_updates, dim=1),
            input_logits=torch.stack(input_logits, dim=1),
            state_logits=torch.stack(state_logits, dim=1),
            bias_logits=torch.stack(bias_logits, dim=1),
        )
