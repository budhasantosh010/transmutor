from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837n.gru_reference_explicit import ExplicitGRUReferenceModel


CONDITION_FACTORS = {
    "G0_full_dynamic": ("dynamic", "dynamic"),
    "G1_dynamic_update_no_reset": ("dynamic", "off"),
    "G2_no_update_dynamic_reset": ("off", "dynamic"),
    "G3_static_update_vector_no_reset": ("static_vector", "off"),
    "G4_no_update_static_reset_vector": ("off", "static_vector"),
    "G5_static_update_vector_static_reset_vector": ("static_vector", "static_vector"),
    "G6_static_update_scalar_static_reset_vector": ("static_scalar", "static_vector"),
    "G7_static_update_vector_static_reset_scalar": ("static_vector", "static_scalar"),
    "G8_static_update_scalar_static_reset_scalar": ("static_scalar", "static_scalar"),
    "G9_no_update_no_reset": ("off", "off"),
}


@dataclass(frozen=True)
class FactorialSpec:
    condition: str
    update_level: str
    reset_level: str


def condition_spec(condition: str) -> FactorialSpec:
    try:
        update_level, reset_level = CONDITION_FACTORS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown V837o factorial condition: {condition}") from exc
    return FactorialSpec(condition, update_level, reset_level)


class FactorialGRUReferenceModel(ExplicitGRUReferenceModel):
    """Frozen V837n explicit GRU with factorial update/reset factor levels."""

    architecture_name = "v837o_factorial_gru_reference"

    def __init__(self, hidden_size: int = 13, input_dim: int = 6, *, condition: str):
        self.factorial_spec = condition_spec(condition)
        super().__init__(hidden_size, input_dim, condition="full_gru")
        self.condition = condition
        # Static coefficients reuse the retained GRU gate-bias slices. Zero
        # logits give coefficient 0.5 and do not add nominal parameters.
        with torch.no_grad():
            r_bias, z_bias, _ = self.bias_hh.chunk(3)
            if self.factorial_spec.update_level in {"static_vector", "static_scalar"}:
                z_bias.zero_()
            if self.factorial_spec.reset_level in {"static_vector", "static_scalar"}:
                r_bias.zero_()

    @property
    def update_level(self) -> str:
        return self.factorial_spec.update_level

    @property
    def reset_level(self) -> str:
        return self.factorial_spec.reset_level

    def nominal_parameter_count(self) -> int:
        return self.parameter_count()

    def active_parameter_count(self) -> int:
        core = (
            self.input_dim * self.input_dim
            + self.input_dim
            + self.hidden_size * self.input_dim
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size
            + self.hidden_size
            + 1
        )
        gate_slice = self.hidden_size * self.input_dim + self.hidden_size * self.hidden_size + 2 * self.hidden_size

        def factor_count(level: str) -> int:
            if level == "dynamic":
                return gate_slice
            if level == "static_vector":
                return self.hidden_size
            if level == "static_scalar":
                return 1
            if level == "off":
                return 0
            raise AssertionError(level)

        return int(core + factor_count(self.update_level) + factor_count(self.reset_level))

    def _static_logit(self, factor: str) -> torch.Tensor:
        r_bias, z_bias, _ = self.bias_hh.chunk(3)
        if factor == "update":
            return z_bias
        if factor == "reset":
            return r_bias
        raise ValueError(factor)

    def static_coefficient(self, factor: str) -> torch.Tensor | None:
        level = self.update_level if factor == "update" else self.reset_level
        if level == "static_vector":
            return torch.sigmoid(self._static_logit(factor))
        if level == "static_scalar":
            scalar = torch.sigmoid(self._static_logit(factor)[0:1])
            return scalar.expand(self.hidden_size)
        return None

    def _factor_value(self, *, factor: str, level: str, dynamic: torch.Tensor, off_value: float) -> torch.Tensor:
        if level == "dynamic":
            return dynamic
        if level == "off":
            return torch.full_like(dynamic, float(off_value))
        logits = self._static_logit(factor)
        if level == "static_vector":
            return torch.sigmoid(logits).view(1, -1).expand_as(dynamic)
        if level == "static_scalar":
            return torch.sigmoid(logits[0]).view(1, 1).expand_as(dynamic)
        raise ValueError(level)

    def _gate_values(self, projected: torch.Tensor, state: torch.Tensor):
        gi = F.linear(projected, self.weight_ih, self.bias_ih)
        gh = F.linear(state, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        dynamic_reset = torch.sigmoid(i_r + h_r)
        dynamic_update = torch.sigmoid(i_z + h_z)
        reset = self._factor_value(factor="reset", level=self.reset_level, dynamic=dynamic_reset, off_value=1.0)
        update = self._factor_value(factor="update", level=self.update_level, dynamic=dynamic_update, off_value=0.0)
        candidate = torch.tanh(i_n + reset * h_n)
        zero = torch.zeros_like(dynamic_update)
        return {
            "candidate": candidate,
            "update": update,
            "reset": reset,
            "update_input_component": torch.sigmoid(i_z) if self.update_level == "dynamic" else zero,
            "update_state_component": torch.sigmoid(h_z) if self.update_level == "dynamic" else zero,
            "reset_input_component": torch.sigmoid(i_r) if self.reset_level == "dynamic" else zero,
            "reset_state_component": torch.sigmoid(h_r) if self.reset_level == "dynamic" else zero,
        }
