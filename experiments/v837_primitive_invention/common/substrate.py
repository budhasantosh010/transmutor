from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .graph import GraphSpec


@dataclass
class ForwardTrace:
    states: torch.Tensor  # [B,T,N,D]
    outputs: torch.Tensor  # [B,T,N,D]
    messages: torch.Tensor  # [B,T,N,M]
    recurrent_terms: torch.Tensor  # [B,T,N,D]
    message_terms: torch.Tensor  # [B,T,N,D]
    input_terms: torch.Tensor  # [B,T,N,D]


class NeutralGraphModel(nn.Module):
    """One generic continuous-cell implementation used for every V837 task family."""

    def __init__(self, graph: GraphSpec, obs_dim: int = 6, state_dim: int = 4, message_dim: int = 4):
        super().__init__()
        if state_dim != message_dim:
            raise ValueError("V837 initial substrate keeps state_dim == message_dim")
        graph.validate()
        self.graph = graph.clone()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.message_dim = int(message_dim)
        if self.graph.input_access is not None and self.graph.input_access.observation_dim != self.obs_dim:
            raise ValueError("graph input-access observation dimension does not match model obs_dim")
        if self.graph.input_access is None or self.graph.input_access.mode == "broadcast":
            input_mask = torch.ones(self.obs_dim, len(self.graph.cells), dtype=torch.float32)
            self.input_access_mode = "broadcast"
        elif self.graph.input_access.mode == "none":
            input_mask = torch.zeros(self.obs_dim, len(self.graph.cells), dtype=torch.float32)
            self.input_access_mode = "none"
        else:
            input_mask = torch.tensor(np.asarray(self.graph.input_access.mask, dtype=np.float32), dtype=torch.float32)
            self.input_access_mode = self.graph.input_access.mode
        # The mask belongs to GraphSpec serialization. Keeping this buffer
        # non-persistent preserves compatibility with historical model bundles.
        self.register_buffer("input_access_mask", input_mask, persistent=False)
        self.cell_ws = nn.ParameterList()
        self.cell_wm = nn.ParameterList()
        self.cell_wx = nn.ParameterList()
        self.cell_b = nn.ParameterList()
        self.cell_wo = nn.ParameterList()
        for cell in self.graph.cells:
            generator = torch.Generator(device="cpu").manual_seed(int(cell.param_seed) + 137)
            scale = 1.0 / math.sqrt(max(1, state_dim))
            recurrent_gain = 0.20 + (int(cell.param_seed) % 1000) / 1000.0 * 0.85
            recurrent_base = recurrent_gain * torch.eye(state_dim) + torch.randn(state_dim, state_dim, generator=generator) * (0.12 * scale)
            message_base = torch.randn(state_dim, message_dim, generator=generator) * (0.35 * scale)
            input_base = torch.randn(state_dim, obs_dim, generator=generator) * (0.45 / math.sqrt(obs_dim))
            output_base = torch.eye(message_dim, state_dim) + torch.randn(message_dim, state_dim, generator=generator) * (0.12 * scale)
            self.cell_ws.append(nn.Parameter(recurrent_base))
            self.cell_wm.append(nn.Parameter(message_base))
            self.cell_wx.append(nn.Parameter(input_base))
            self.cell_b.append(nn.Parameter(torch.zeros(state_dim)))
            self.cell_wo.append(nn.Parameter(output_base))
        self.edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(float(edge.weight), dtype=torch.float32)) for edge in self.graph.edges])
        self.readout = nn.Linear(len(self.graph.cells) * state_dim, 1)

    @property
    def input_edge_count(self) -> int:
        return int(torch.sum(self.input_access_mask).item())

    @property
    def internal_message_edge_count(self) -> int:
        return len(self.graph.edges)

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        disabled_cells: set[int] | None = None,
        disabled_raw_input_cells: set[int] | None = None,
        disabled_message_cells: set[int] | None = None,
        disable_messages: bool = False,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ForwardTrace]:
        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.obs_dim:
            raise ValueError(f"observation dimension {observed_dim} != model obs_dim {self.obs_dim}")
        n = len(self.graph.cells)
        device = observations.device
        disabled = disabled_cells or set()
        raw_disabled = disabled_raw_input_cells or set()
        message_disabled = disabled_message_cells or set()
        prev_states = [torch.zeros(batch, self.state_dim, device=device) for _ in range(n)]
        prev_outputs = [torch.zeros(batch, self.message_dim, device=device) for _ in range(n)]
        state_trace: list[torch.Tensor] = []
        output_trace: list[torch.Tensor] = []
        aggregate_message_trace: list[torch.Tensor] = []
        recurrent_term_trace: list[torch.Tensor] = []
        message_term_trace: list[torch.Tensor] = []
        input_term_trace: list[torch.Tensor] = []
        for t in range(steps):
            x_t = observations[:, t, :]
            current_states: list[torch.Tensor] = []
            current_outputs: list[torch.Tensor] = []
            current_messages: list[torch.Tensor] = []
            current_recurrent_terms: list[torch.Tensor] = []
            current_message_terms: list[torch.Tensor] = []
            current_input_terms: list[torch.Tensor] = []
            for cell_index in range(n):
                if cell_index in disabled:
                    state = torch.zeros(batch, self.state_dim, device=device)
                    output = torch.zeros(batch, self.message_dim, device=device)
                    message = torch.zeros(batch, self.message_dim, device=device)
                    recurrent_term = torch.zeros(batch, self.state_dim, device=device)
                    message_term = torch.zeros(batch, self.state_dim, device=device)
                    input_term = torch.zeros(batch, self.state_dim, device=device)
                else:
                    message = torch.zeros(batch, self.message_dim, device=device)
                    if not disable_messages and cell_index not in message_disabled:
                        for edge_index, edge in enumerate(self.graph.edges):
                            if edge.dst != cell_index:
                                continue
                            if edge.recurrent or edge.src >= len(current_outputs):
                                source = prev_outputs[edge.src]
                            else:
                                source = current_outputs[edge.src]
                            message = message + self.edge_weights[edge_index] * source
                    recurrent_term = prev_states[cell_index] @ self.cell_ws[cell_index].T
                    message_term = message @ self.cell_wm[cell_index].T
                    if cell_index in raw_disabled or self.input_access_mode == "none":
                        visible_x = torch.zeros_like(x_t)
                    elif self.input_access_mode == "broadcast":
                        # Preserve the historical arithmetic path exactly.
                        visible_x = x_t
                    else:
                        visible_x = x_t * self.input_access_mask[:, cell_index].view(1, -1)
                    input_term = visible_x @ self.cell_wx[cell_index].T
                    proposed_state = torch.tanh(
                        recurrent_term
                        + message_term
                        + input_term
                        + self.cell_b[cell_index]
                    )
                    proposed_output = proposed_state @ self.cell_wo[cell_index].T
                    if lengths is not None:
                        active = (t < lengths).to(observations.dtype).unsqueeze(1)
                        state = active * proposed_state + (1.0 - active) * prev_states[cell_index]
                        output = active * proposed_output + (1.0 - active) * prev_outputs[cell_index]
                        if return_trace:
                            message = active * message
                            recurrent_term = active * recurrent_term
                            message_term = active * message_term
                            input_term = active * input_term
                    else:
                        state = proposed_state
                        output = proposed_output
                current_states.append(state)
                current_outputs.append(output)
                if return_trace:
                    current_messages.append(message)
                    current_recurrent_terms.append(recurrent_term)
                    current_message_terms.append(message_term)
                    current_input_terms.append(input_term)
            prev_states = current_states
            prev_outputs = current_outputs
            if return_trace:
                state_trace.append(torch.stack(current_states, dim=1))
                output_trace.append(torch.stack(current_outputs, dim=1))
                aggregate_message_trace.append(torch.stack(current_messages, dim=1))
                recurrent_term_trace.append(torch.stack(current_recurrent_terms, dim=1))
                message_term_trace.append(torch.stack(current_message_terms, dim=1))
                input_term_trace.append(torch.stack(current_input_terms, dim=1))
        stacked = torch.cat(prev_states, dim=1)
        prediction = torch.tanh(self.readout(stacked)).squeeze(-1)
        if return_trace:
            trace = ForwardTrace(
                states=torch.stack(state_trace, dim=1),
                outputs=torch.stack(output_trace, dim=1),
                messages=torch.stack(aggregate_message_trace, dim=1),
                recurrent_terms=torch.stack(recurrent_term_trace, dim=1),
                message_terms=torch.stack(message_term_trace, dim=1),
                input_terms=torch.stack(input_term_trace, dim=1),
            )
            return prediction, trace
        return prediction

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())


def clone_with_state(model: NeutralGraphModel) -> NeutralGraphModel:
    clone = NeutralGraphModel(model.graph, obs_dim=model.obs_dim, state_dim=model.state_dim, message_dim=model.message_dim)
    clone.load_state_dict(model.state_dict())
    return clone
