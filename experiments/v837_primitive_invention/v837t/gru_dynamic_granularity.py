from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837n.gru_reference_explicit import ExplicitGRUReferenceModel


CONDITION_GRANULARITY = {
    "T0_full_vector_gru": ("vector", "vector"),
    "T1_vector_update_no_reset": ("vector", "off"),
    "T2_scalarized_update_no_reset": ("scalarized", "off"),
    "T3_no_update_vector_reset": ("off", "vector"),
    "T4_no_update_scalarized_reset": ("off", "scalarized"),
    "T5_dual_scalarized": ("scalarized", "scalarized"),
}


def scalarize_dynamic_gate(gate: torch.Tensor) -> torch.Tensor:
    """Remove only inter-dimensional gate variation after sigmoid."""
    if gate.ndim < 2:
        raise ValueError("dynamic gate must include a hidden dimension")
    scalar = gate.mean(dim=-1, keepdim=True)
    return scalar.expand_as(gate)


@dataclass
class DynamicGranularityTrace:
    states: torch.Tensor
    candidates: torch.Tensor
    updates: torch.Tensor
    resets: torch.Tensor
    raw_dynamic_updates: torch.Tensor
    raw_dynamic_resets: torch.Tensor


class DynamicGranularityGRU(ExplicitGRUReferenceModel):
    """Frozen explicit GRU with only post-sigmoid gate-output granularity varied."""

    architecture_name = "v837t_dynamic_granularity_gru"

    def __init__(self, hidden_size: int = 13, input_dim: int = 6, *, condition: str):
        if condition not in CONDITION_GRANULARITY:
            raise ValueError(f"unknown V837t condition: {condition}")
        self.granularity_condition = condition
        self.update_granularity, self.reset_granularity = CONDITION_GRANULARITY[condition]
        super().__init__(hidden_size, input_dim, condition="full_gru")
        self.condition = condition

    def nominal_parameter_count(self) -> int:
        return self.parameter_count()

    def active_parameter_count(self) -> int:
        core = (
            self.input_dim * self.input_dim + self.input_dim
            + self.hidden_size * self.input_dim + self.hidden_size * self.hidden_size + 2 * self.hidden_size
            + self.hidden_size + 1
        )
        gate_slice = self.hidden_size * self.input_dim + self.hidden_size * self.hidden_size + 2 * self.hidden_size
        active = core
        if self.update_granularity != "off":
            active += gate_slice
        if self.reset_granularity != "off":
            active += gate_slice
        return int(active)

    @staticmethod
    def _apply_granularity(dynamic: torch.Tensor, level: str, *, off_value: float) -> torch.Tensor:
        if level == "vector":
            return dynamic
        if level == "scalarized":
            return scalarize_dynamic_gate(dynamic)
        if level == "off":
            return torch.full_like(dynamic, float(off_value))
        raise ValueError(level)

    def _components(self, projected: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        gi = F.linear(projected, self.weight_ih, self.bias_ih)
        gh = F.linear(state, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        raw_reset = torch.sigmoid(i_r + h_r)
        raw_update = torch.sigmoid(i_z + h_z)
        reset = self._apply_granularity(raw_reset, self.reset_granularity, off_value=1.0)
        update = self._apply_granularity(raw_update, self.update_granularity, off_value=0.0)
        # Exact PyTorch-compatible reset ordering: post hidden transform.
        candidate = torch.tanh(i_n + reset * h_n)
        return {
            "candidate": candidate,
            "update": update,
            "reset": reset,
            "raw_update": raw_update,
            "raw_reset": raw_reset,
            "hidden_candidate": h_n,
            "input_candidate": i_n,
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
        expected = (batch, steps, self.hidden_size)
        if update_override is not None and tuple(update_override.shape) != expected:
            raise ValueError("update_override must have shape [B,T,H]")
        if reset_override is not None and tuple(reset_override.shape) != expected:
            raise ValueError("reset_override must have shape [B,T,H]")

        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        candidates: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        resets: list[torch.Tensor] = []
        raw_updates: list[torch.Tensor] = []
        raw_resets: list[torch.Tensor] = []

        for t in range(steps):
            projected = self.input_projection(observations[:, t, :])
            values = self._components(projected, state)
            reset = reset_override[:, t, :] if reset_override is not None else values["reset"]
            candidate = (
                torch.tanh(values["input_candidate"] + reset * values["hidden_candidate"])
                if reset_override is not None else values["candidate"]
            )
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
                raw_updates.append(values["raw_update"])
                raw_resets.append(values["raw_reset"])

        prediction = torch.tanh(self.readout(states[-1])).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, DynamicGranularityTrace(
            states=torch.stack(states, dim=1),
            candidates=torch.stack(candidates, dim=1),
            updates=torch.stack(updates, dim=1),
            resets=torch.stack(resets, dim=1),
            raw_dynamic_updates=torch.stack(raw_updates, dim=1),
            raw_dynamic_resets=torch.stack(raw_resets, dim=1),
        )
