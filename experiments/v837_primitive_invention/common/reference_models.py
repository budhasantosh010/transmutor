from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .task_interface import OBS_DIM


@dataclass
class ReferenceTrace:
    states: torch.Tensor


class LearnedReferenceModel(nn.Module):
    architecture_name: str = "reference"

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def _finish(self, states: list[torch.Tensor], readout: nn.Module, *, return_trace: bool):
        stacked = torch.stack(states, dim=1)
        prediction = torch.tanh(readout(states[-1])).squeeze(-1)
        if return_trace:
            return prediction, ReferenceTrace(states=stacked)
        return prediction


class GRUReferenceModel(LearnedReferenceModel):
    architecture_name = "gru_reference"

    def __init__(self, hidden_size: int, input_dim: int = OBS_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.input_projection = nn.Linear(self.input_dim, self.input_dim)
        self.cell = nn.GRUCell(self.input_dim, self.hidden_size)
        self.readout = nn.Linear(self.hidden_size, 1)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, return_trace: bool = False):
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.input_dim:
            raise ValueError(f"observation dimension {observed_dim} != input_dim {self.input_dim}")
        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        for t in range(steps):
            projected = self.input_projection(observations[:, t, :])
            candidate = self.cell(projected, state)
            if lengths is None:
                state = candidate
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * candidate + (1.0 - active) * state
            states.append(state)
        return self._finish(states, self.readout, return_trace=return_trace)


class ResidualRecurrentMLPReferenceModel(LearnedReferenceModel):
    architecture_name = "residual_rnn_reference"

    def __init__(self, hidden_size: int, input_dim: int = OBS_DIM, beta: float = 0.25):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.beta = float(beta)
        self.state_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_size, bias=True)
        self.readout = nn.Linear(self.hidden_size, 1)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, return_trace: bool = False):
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.input_dim:
            raise ValueError(f"observation dimension {observed_dim} != input_dim {self.input_dim}")
        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        for t in range(steps):
            candidate = torch.tanh(self.state_projection(state) + self.input_projection(observations[:, t, :]))
            proposed = state + self.beta * candidate
            if lengths is None:
                state = proposed
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed + (1.0 - active) * state
            states.append(state)
        return self._finish(states, self.readout, return_trace=return_trace)


class VanillaRNNReferenceModel(LearnedReferenceModel):
    architecture_name = "vanilla_rnn_reference"

    def __init__(self, hidden_size: int, input_dim: int = OBS_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.state_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_size, bias=True)
        self.readout = nn.Linear(self.hidden_size, 1)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, return_trace: bool = False):
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.input_dim:
            raise ValueError(f"observation dimension {observed_dim} != input_dim {self.input_dim}")
        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        states: list[torch.Tensor] = []
        for t in range(steps):
            proposed = torch.tanh(self.state_projection(state) + self.input_projection(observations[:, t, :]))
            if lengths is None:
                state = proposed
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed + (1.0 - active) * state
            states.append(state)
        return self._finish(states, self.readout, return_trace=return_trace)


REFERENCE_BUILDERS: dict[str, Callable[[int, int], LearnedReferenceModel]] = {
    "gru_reference": lambda hidden, input_dim: GRUReferenceModel(hidden, input_dim),
    "residual_rnn_reference": lambda hidden, input_dim: ResidualRecurrentMLPReferenceModel(hidden, input_dim),
    "vanilla_rnn_reference": lambda hidden, input_dim: VanillaRNNReferenceModel(hidden, input_dim),
}


def build_reference_model(architecture_type: str, hidden_size: int, input_dim: int = OBS_DIM) -> LearnedReferenceModel:
    try:
        builder = REFERENCE_BUILDERS[architecture_type]
    except KeyError as exc:
        raise ValueError(f"unknown learned reference architecture: {architecture_type}") from exc
    return builder(int(hidden_size), int(input_dim))


def select_hidden_size_for_parameter_target(*, input_dim: int, target_parameter_count: int, architecture_type: str, minimum_hidden_size: int = 1, maximum_hidden_size: int = 256) -> dict:
    candidates: list[tuple[int, int, int]] = []
    for hidden_size in range(int(minimum_hidden_size), int(maximum_hidden_size) + 1):
        model = build_reference_model(architecture_type, hidden_size, input_dim)
        count = model.parameter_count()
        candidates.append((abs(count - int(target_parameter_count)), hidden_size, count))
    _, hidden_size, parameter_count = min(candidates, key=lambda item: (item[0], item[1]))
    target = int(target_parameter_count)
    return {
        "architecture_type": architecture_type,
        "hidden_size": int(hidden_size),
        "parameter_count": int(parameter_count),
        "target_parameter_count": target,
        "difference": int(parameter_count - target),
        "percent_difference": float((parameter_count - target) / target * 100.0) if target else 0.0,
    }
