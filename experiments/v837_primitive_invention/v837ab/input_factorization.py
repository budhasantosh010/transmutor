from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import (
    DynamicGranularityGRU,
    scalarize_dynamic_gate,
)


HIDDEN_SIZE = 13
INPUT_DIM = 6
T2_CONDITION = "T2_scalarized_update_no_reset"
CONDITIONS = {
    "AB0_exact_factorized_t2",
    "AB1_fully_folded_equivalent",
    "AB2_candidate_factorized_update_folded",
    "AB3_candidate_folded_update_factorized",
    "AB4_frozen_shared_projection",
    "AB5_naive_projection_free",
}
FACTORIZED_GATES = {
    "AB0_exact_factorized_t2": {"r", "z", "n"},
    "AB1_fully_folded_equivalent": set(),
    "AB2_candidate_factorized_update_folded": {"n"},
    "AB3_candidate_folded_update_factorized": {"z"},
    "AB4_frozen_shared_projection": {"r", "z", "n"},
    "AB5_naive_projection_free": set(),
}


@dataclass
class InputFactorizationTrace:
    states: torch.Tensor
    candidates: torch.Tensor
    updates: torch.Tensor
    resets: torch.Tensor
    raw_dynamic_updates: torch.Tensor
    raw_dynamic_resets: torch.Tensor
    candidate_input_terms: torch.Tensor
    update_input_terms: torch.Tensor
    reset_input_terms: torch.Tensor


def fold_linear(weight: torch.Tensor, bias: torch.Tensor, projection_weight: torch.Tensor, projection_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly fold y=W(Ax+a)+b into y=(WA)x+(b+Wa)."""
    return weight @ projection_weight, bias + weight @ projection_bias


def fold_full_weight_ih(
    weight_ih: torch.Tensor,
    bias_ih: torch.Tensor,
    projection_weight: torch.Tensor,
    projection_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fold_linear(weight_ih, bias_ih, projection_weight, projection_bias)


def _source_from_state(source_state: Mapping[str, torch.Tensor] | None = None) -> DynamicGranularityGRU:
    source = DynamicGranularityGRU(HIDDEN_SIZE, INPUT_DIM, condition=T2_CONDITION)
    if source_state is not None:
        source.load_state_dict(dict(source_state), strict=True)
    return source


class InputFactorizationT2(nn.Module):
    """T2 with only linear input parameterization varied.

    Every non-naive condition is derived from one exact T2 source initialization.
    AB0 keeps the factorization. AB1 folds it. AB2/AB3 keep exactly one active
    pathway factorized. AB4 freezes the shared projection. AB5 removes the
    projection but uses the ordinary GRU input-matrix initialization rather than
    the composed effective initialization.
    """

    architecture_name = "v837ab_input_factorization_t2"

    def __init__(self, *, condition: str, source_state: Mapping[str, torch.Tensor] | None = None):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown V837ab condition: {condition}")
        self.condition = condition
        self.input_dim = INPUT_DIM
        self.hidden_size = HIDDEN_SIZE
        self.factorized_gates = frozenset(FACTORIZED_GATES[condition])

        source = _source_from_state(source_state)
        A = source.input_projection.weight.detach().clone()
        a = source.input_projection.bias.detach().clone()
        w_ir, w_iz, w_in = source.weight_ih.detach().clone().chunk(3, dim=0)
        b_ir, b_iz, b_in = source.bias_ih.detach().clone().chunk(3, dim=0)
        u_hr, u_hz, u_hn = source.weight_hh.detach().clone().chunk(3, dim=0)
        c_hr, c_hz, c_hn = source.bias_hh.detach().clone().chunk(3, dim=0)

        if self.factorized_gates:
            self.projection_weight = nn.Parameter(A, requires_grad=condition != "AB4_frozen_shared_projection")
            self.projection_bias = nn.Parameter(a, requires_grad=condition != "AB4_frozen_shared_projection")
        else:
            self.register_parameter("projection_weight", None)
            self.register_parameter("projection_bias", None)

        source_inputs = {"r": (w_ir, b_ir), "z": (w_iz, b_iz), "n": (w_in, b_in)}
        for gate, (weight, bias) in source_inputs.items():
            if condition == "AB5_naive_projection_free":
                derived_weight, derived_bias = weight, bias
            elif gate in self.factorized_gates:
                derived_weight, derived_bias = weight, bias
            else:
                derived_weight, derived_bias = fold_linear(weight, bias, A, a)
            setattr(self, f"weight_i{gate}", nn.Parameter(derived_weight.clone()))
            setattr(self, f"bias_i{gate}", nn.Parameter(derived_bias.clone()))

        for gate, weight, bias in (
            ("r", u_hr, c_hr), ("z", u_hz, c_hz), ("n", u_hn, c_hn),
        ):
            setattr(self, f"weight_h{gate}", nn.Parameter(weight.clone()))
            setattr(self, f"bias_h{gate}", nn.Parameter(bias.clone()))

        self.readout_weight = nn.Parameter(source.readout.weight.detach().clone())
        self.readout_bias = nn.Parameter(source.readout.bias.detach().clone())

    @classmethod
    def from_trained_t2(cls, source: DynamicGranularityGRU, *, condition: str = "AB1_fully_folded_equivalent") -> "InputFactorizationT2":
        if source.granularity_condition != T2_CONDITION:
            raise ValueError("source must be exact T2")
        # Constructor RNG is irrelevant because source_state overwrites the temporary source.
        return cls(condition=condition, source_state=source.state_dict())

    @property
    def projection_active(self) -> bool:
        return self.projection_weight is not None

    @property
    def projection_trainable(self) -> bool:
        return self.projection_weight is not None and bool(self.projection_weight.requires_grad)

    @property
    def projection_specific_macs(self) -> int:
        return 36 if self.projection_active else 0

    @property
    def active_reference_macs_per_timestep(self) -> int:
        # Candidate + update input maps (2*13*6), hidden maps (2*13*13), plus shared 6x6 projection when present.
        return 2 * HIDDEN_SIZE * INPUT_DIM + 2 * HIDDEN_SIZE * HIDDEN_SIZE + self.projection_specific_macs

    def nominal_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def active_parameter_count(self) -> int:
        # Match V837t semantics: inactive reset slice is excluded, projection retained if used by any active path.
        reset_slice = HIDDEN_SIZE * INPUT_DIM + HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
        return self.nominal_parameter_count() - reset_slice

    def executed_parameter_count(self) -> int:
        # Raw reset is still computed diagnostically, so every stored model parameter participates in the forward.
        return self.nominal_parameter_count()

    def executed_frozen_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if not p.requires_grad))

    def parameter_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))

    def _project(self, x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        if self.projection_weight is None or self.projection_bias is None:
            raise RuntimeError("projection requested in projection-free condition")
        weight = self.projection_weight.detach() if detach else self.projection_weight
        bias = self.projection_bias.detach() if detach else self.projection_bias
        return F.linear(x, weight, bias)

    def _input_term(
        self,
        gate: str,
        x: torch.Tensor,
        projected: torch.Tensor | None,
        *,
        detach_projection: bool = False,
    ) -> torch.Tensor:
        weight = getattr(self, f"weight_i{gate}")
        bias = getattr(self, f"bias_i{gate}")
        if gate in self.factorized_gates:
            if detach_projection:
                projected = self._project(x, detach=True)
            elif projected is None:
                projected = self._project(x)
            return F.linear(projected, weight, bias)
        return F.linear(x, weight, bias)

    def effective_input_map(self, gate: str) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self, f"weight_i{gate}")
        bias = getattr(self, f"bias_i{gate}")
        if gate in self.factorized_gates:
            assert self.projection_weight is not None and self.projection_bias is not None
            return fold_linear(weight, bias, self.projection_weight, self.projection_bias)
        return weight, bias

    def projection_diagnostics(self) -> dict:
        if not self.projection_active:
            return {"active": False, "singular_values": [], "condition_number": None, "determinant_abs": None, "frobenius_norm": 0.0, "bias_norm": 0.0}
        with torch.no_grad():
            singular = torch.linalg.svdvals(self.projection_weight.detach().cpu())
            minimum = float(singular.min().item()) if singular.numel() else 0.0
            maximum = float(singular.max().item()) if singular.numel() else 0.0
            return {
                "active": True,
                "trainable": self.projection_trainable,
                "singular_values": [float(v) for v in singular.tolist()],
                "condition_number": float(maximum / max(minimum, 1e-12)),
                "determinant_abs": float(torch.abs(torch.linalg.det(self.projection_weight.detach().cpu())).item()),
                "frobenius_norm": float(torch.linalg.vector_norm(self.projection_weight.detach().cpu()).item()),
                "bias_norm": float(torch.linalg.vector_norm(self.projection_bias.detach().cpu()).item()),
            }

    def factorization_snapshot(self) -> dict:
        with torch.no_grad():
            output = {"projection": self.projection_diagnostics(), "effective": {}}
            for gate in ("n", "z"):
                effective_w, effective_b = self.effective_input_map(gate)
                singular = torch.linalg.svdvals(effective_w.detach().cpu())
                minimum = float(singular.min().item()) if singular.numel() else 0.0
                maximum = float(singular.max().item()) if singular.numel() else 0.0
                raw_w = getattr(self, f"weight_i{gate}")
                output["effective"][gate] = {
                    "frobenius_norm": float(torch.linalg.vector_norm(effective_w.detach().cpu()).item()),
                    "bias_norm": float(torch.linalg.vector_norm(effective_b.detach().cpu()).item()),
                    "spectral_norm": maximum,
                    "singular_values": [float(v) for v in singular.tolist()],
                    "condition_number": float(maximum / max(minimum, 1e-12)),
                    "downstream_weight_norm": float(torch.linalg.vector_norm(raw_w.detach().cpu()).item()),
                }
            return output

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        detach_projection_for_candidate: bool = False,
        detach_projection_for_update: bool = False,
    ):
        if observations.ndim != 3 or observations.shape[-1] != INPUT_DIM:
            raise ValueError("observations must be [B,T,6]")
        batch, steps, _ = observations.shape
        state = torch.zeros(batch, HIDDEN_SIZE, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        candidates: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        resets: list[torch.Tensor] = []
        raw_updates: list[torch.Tensor] = []
        raw_resets: list[torch.Tensor] = []
        candidate_inputs: list[torch.Tensor] = []
        update_inputs: list[torch.Tensor] = []
        reset_inputs: list[torch.Tensor] = []
        for t in range(steps):
            x_t = observations[:, t, :]
            projected = self._project(x_t) if self.projection_active else None
            i_r = self._input_term("r", x_t, projected)
            i_z = self._input_term("z", x_t, projected, detach_projection=detach_projection_for_update)
            i_n = self._input_term("n", x_t, projected, detach_projection=detach_projection_for_candidate)
            h_r = F.linear(state, self.weight_hr, self.bias_hr)
            h_z = F.linear(state, self.weight_hz, self.bias_hz)
            h_n = F.linear(state, self.weight_hn, self.bias_hn)
            raw_reset = torch.sigmoid(i_r + h_r)
            raw_update = torch.sigmoid(i_z + h_z)
            reset = torch.ones_like(raw_reset)
            update = scalarize_dynamic_gate(raw_update)
            candidate = torch.tanh(i_n + h_n)
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
                raw_updates.append(raw_update)
                raw_resets.append(raw_reset)
                candidate_inputs.append(i_n)
                update_inputs.append(i_z)
                reset_inputs.append(i_r)
        prediction = torch.tanh(F.linear(states[-1], self.readout_weight, self.readout_bias)).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, InputFactorizationTrace(
            states=torch.stack(states, dim=1),
            candidates=torch.stack(candidates, dim=1),
            updates=torch.stack(updates, dim=1),
            resets=torch.stack(resets, dim=1),
            raw_dynamic_updates=torch.stack(raw_updates, dim=1),
            raw_dynamic_resets=torch.stack(raw_resets, dim=1),
            candidate_input_terms=torch.stack(candidate_inputs, dim=1),
            update_input_terms=torch.stack(update_inputs, dim=1),
            reset_input_terms=torch.stack(reset_inputs, dim=1),
        )


def max_trace_errors(reference: DynamicGranularityGRU, folded: InputFactorizationT2, observations: torch.Tensor, lengths: torch.Tensor | None = None) -> dict:
    reference.eval(); folded.eval()
    with torch.no_grad():
        p_ref, t_ref = reference(observations, lengths, return_trace=True)
        p_fold, t_fold = folded(observations, lengths, return_trace=True)
    return {
        "prediction": float(torch.max(torch.abs(p_ref - p_fold)).item()),
        "state": float(torch.max(torch.abs(t_ref.states - t_fold.states)).item()),
        "candidate": float(torch.max(torch.abs(t_ref.candidates - t_fold.candidates)).item()),
        "update": float(torch.max(torch.abs(t_ref.updates - t_fold.updates)).item()),
        "raw_update": float(torch.max(torch.abs(t_ref.raw_dynamic_updates - t_fold.raw_dynamic_updates)).item()),
        "reset": float(torch.max(torch.abs(t_ref.resets - t_fold.resets)).item()),
        "raw_reset": float(torch.max(torch.abs(t_ref.raw_dynamic_resets - t_fold.raw_dynamic_resets)).item()),
    }
