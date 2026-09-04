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
    candidate_states: torch.Tensor  # [B,T,N,D], tanh candidate before any carry/transport
    outputs: torch.Tensor  # [B,T,N,D]
    messages: torch.Tensor  # [B,T,N,M]
    recurrent_terms: torch.Tensor  # [B,T,N,D]
    message_terms: torch.Tensor  # [B,T,N,D]
    input_terms: torch.Tensor  # [B,T,N,D]
    state_modulators: torch.Tensor | None = None  # [B,T,N,1] for scalar dynamic modulation


class NeutralGraphModel(nn.Module):
    """One generic continuous-cell implementation used for every V837 task family."""

    def __init__(
        self,
        graph: GraphSpec,
        obs_dim: int = 6,
        state_dim: int = 4,
        message_dim: int = 4,
        *,
        state_update_mode: str = "direct",
        alpha_init: float = 0.5,
        transport_rho: float = 0.95,
        interaction_mode: str = "none",
        interaction_rank: int = 2,
        state_modulation_mode: str = "none",
    ):
        super().__init__()
        if state_dim != message_dim:
            raise ValueError("V837 initial substrate keeps state_dim == message_dim")
        graph.validate()
        self.graph = graph.clone()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.message_dim = int(message_dim)
        if state_update_mode not in {"direct", "learned_leaky", "linear_transport", "transport_matched_additive"}:
            raise ValueError(f"unknown state_update_mode: {state_update_mode}")
        if not 0.0 < float(alpha_init) < 1.0:
            raise ValueError("alpha_init must be strictly between 0 and 1")
        if not 0.0 < float(transport_rho) <= 1.0:
            raise ValueError("transport_rho must be in (0,1]")
        self.state_update_mode = state_update_mode
        self.alpha_init = float(alpha_init)
        self.transport_rho = float(transport_rho)
        if interaction_mode not in {"none", "low_rank_multiplicative", "parameter_matched_additive"}:
            raise ValueError(f"unknown interaction_mode: {interaction_mode}")
        if int(interaction_rank) <= 0:
            raise ValueError("interaction_rank must be positive")
        self.interaction_mode = interaction_mode
        self.interaction_rank = int(interaction_rank)
        if state_modulation_mode not in {"none", "dynamic_scalar_candidate", "dynamic_scalar_matched_additive"}:
            raise ValueError(f"unknown state_modulation_mode: {state_modulation_mode}")
        if state_modulation_mode != "none" and interaction_mode != "none":
            raise ValueError("dynamic state modulation is isolated from interaction-mode experiments")
        self.state_modulation_mode = state_modulation_mode
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
        self.cell_alpha_logits = nn.ParameterList()
        self.cell_transport_raw = nn.ParameterList()
        self.cell_as = nn.ParameterList(); self.cell_am = nn.ParameterList(); self.cell_ax = nn.ParameterList()
        self.cell_bs = nn.ParameterList(); self.cell_bm = nn.ParameterList(); self.cell_bx = nn.ParameterList()
        self.cell_cu = nn.ParameterList(); self.cell_cv = nn.ParameterList(); self.cell_ci = nn.ParameterList()
        self.cell_gs = nn.ParameterList(); self.cell_gm = nn.ParameterList(); self.cell_gx = nn.ParameterList(); self.cell_gb = nn.ParameterList()
        for cell in self.graph.cells:
            generator = torch.Generator(device="cpu").manual_seed(int(cell.param_seed) + 137)
            scale = 1.0 / math.sqrt(max(1, state_dim))
            recurrent_gain = 0.20 + (int(cell.param_seed) % 1000) / 1000.0 * 0.85
            if self.interaction_mode == "none":
                # Preserve the historical random-draw order exactly.
                recurrent_base = recurrent_gain * torch.eye(state_dim) + torch.randn(state_dim, state_dim, generator=generator) * (0.12 * scale)
                message_base = torch.randn(state_dim, message_dim, generator=generator) * (0.35 * scale)
                input_base = torch.randn(state_dim, obs_dim, generator=generator) * (0.45 / math.sqrt(obs_dim))
                output_base = torch.eye(message_dim, state_dim) + torch.randn(message_dim, state_dim, generator=generator) * (0.12 * scale)
                self.cell_ws.append(nn.Parameter(recurrent_base))
                self.cell_wm.append(nn.Parameter(message_base))
                self.cell_wx.append(nn.Parameter(input_base))
            else:
                rank = self.interaction_rank
                branch_scale = 1.0 / math.sqrt(max(1, rank))
                self.cell_as.append(nn.Parameter(torch.randn(rank, state_dim, generator=generator) * (0.35 * scale)))
                self.cell_am.append(nn.Parameter(torch.randn(rank, message_dim, generator=generator) * (0.35 * scale)))
                self.cell_ax.append(nn.Parameter(torch.randn(rank, obs_dim, generator=generator) * (0.45 / math.sqrt(obs_dim))))
                self.cell_bs.append(nn.Parameter(torch.randn(rank, state_dim, generator=generator) * (0.35 * scale)))
                self.cell_bm.append(nn.Parameter(torch.randn(rank, message_dim, generator=generator) * (0.35 * scale)))
                self.cell_bx.append(nn.Parameter(torch.randn(rank, obs_dim, generator=generator) * (0.45 / math.sqrt(obs_dim))))
                self.cell_cu.append(nn.Parameter(torch.randn(state_dim, rank, generator=generator) * branch_scale))
                self.cell_cv.append(nn.Parameter(torch.randn(state_dim, rank, generator=generator) * branch_scale))
                self.cell_ci.append(nn.Parameter(torch.randn(state_dim, rank, generator=generator) * branch_scale))
                output_base = torch.eye(message_dim, state_dim) + torch.randn(message_dim, state_dim, generator=generator) * (0.12 * scale)
            self.cell_b.append(nn.Parameter(torch.zeros(state_dim)))
            self.cell_wo.append(nn.Parameter(output_base))
            if self.state_update_mode == "learned_leaky":
                logit = math.log(self.alpha_init / (1.0 - self.alpha_init))
                self.cell_alpha_logits.append(nn.Parameter(torch.tensor(logit, dtype=torch.float32)))
            elif self.state_update_mode in {"linear_transport", "transport_matched_additive"}:
                self.cell_transport_raw.append(nn.Parameter(torch.eye(state_dim, dtype=torch.float32)))
            if self.state_modulation_mode != "none":
                # V837p uses the smallest mechanism licensed by V837o: one
                # input/state/message-conditioned scalar modulator per cell.
                mod_generator = torch.Generator(device="cpu").manual_seed(int(cell.param_seed) + 9137)
                self.cell_gs.append(nn.Parameter(torch.randn(state_dim, generator=mod_generator) * (0.20 / math.sqrt(state_dim))))
                self.cell_gm.append(nn.Parameter(torch.randn(message_dim, generator=mod_generator) * (0.20 / math.sqrt(message_dim))))
                self.cell_gx.append(nn.Parameter(torch.randn(obs_dim, generator=mod_generator) * (0.20 / math.sqrt(obs_dim))))
                self.cell_gb.append(nn.Parameter(torch.zeros(1)))
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
        candidate_state_trace: list[torch.Tensor] = []
        output_trace: list[torch.Tensor] = []
        aggregate_message_trace: list[torch.Tensor] = []
        recurrent_term_trace: list[torch.Tensor] = []
        message_term_trace: list[torch.Tensor] = []
        input_term_trace: list[torch.Tensor] = []
        state_modulator_trace: list[torch.Tensor] = []
        for t in range(steps):
            x_t = observations[:, t, :]
            current_states: list[torch.Tensor] = []
            current_candidates: list[torch.Tensor] = []
            current_outputs: list[torch.Tensor] = []
            current_messages: list[torch.Tensor] = []
            current_recurrent_terms: list[torch.Tensor] = []
            current_message_terms: list[torch.Tensor] = []
            current_input_terms: list[torch.Tensor] = []
            current_state_modulators: list[torch.Tensor] = []
            for cell_index in range(n):
                if cell_index in disabled:
                    state = torch.zeros(batch, self.state_dim, device=device)
                    candidate_state = torch.zeros(batch, self.state_dim, device=device)
                    output = torch.zeros(batch, self.message_dim, device=device)
                    message = torch.zeros(batch, self.message_dim, device=device)
                    recurrent_term = torch.zeros(batch, self.state_dim, device=device)
                    message_term = torch.zeros(batch, self.state_dim, device=device)
                    input_term = torch.zeros(batch, self.state_dim, device=device)
                    state_modulator = torch.ones(batch, 1, device=device)
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
                    if cell_index in raw_disabled or self.input_access_mode == "none":
                        visible_x = torch.zeros_like(x_t)
                    elif self.input_access_mode == "broadcast":
                        # Preserve the historical arithmetic path exactly.
                        visible_x = x_t
                    else:
                        visible_x = x_t * self.input_access_mask[:, cell_index].view(1, -1)
                    if self.interaction_mode == "none":
                        if self.state_modulation_mode != "none":
                            modulator_pre = (
                                torch.sum(prev_states[cell_index] * self.cell_gs[cell_index].view(1, -1), dim=1, keepdim=True)
                                + torch.sum(message * self.cell_gm[cell_index].view(1, -1), dim=1, keepdim=True)
                                + torch.sum(visible_x * self.cell_gx[cell_index].view(1, -1), dim=1, keepdim=True)
                                + self.cell_gb[cell_index]
                            )
                            state_modulator = torch.sigmoid(modulator_pre)
                        else:
                            state_modulator = torch.ones(batch, 1, device=device, dtype=observations.dtype)
                        recurrent_source = (
                            state_modulator * prev_states[cell_index]
                            if self.state_modulation_mode == "dynamic_scalar_candidate"
                            else prev_states[cell_index]
                        )
                        recurrent_term = recurrent_source @ self.cell_ws[cell_index].T
                        message_term = message @ self.cell_wm[cell_index].T
                        input_term = visible_x @ self.cell_wx[cell_index].T
                        preactivation = recurrent_term + message_term + input_term + self.cell_b[cell_index]
                        if self.state_modulation_mode == "dynamic_scalar_matched_additive":
                            # Same dynamic scalar network and parameter count,
                            # but the coefficient cannot multiplicatively gate
                            # access to the previous state.
                            preactivation = preactivation + state_modulator
                    else:
                        state_modulator = torch.ones(batch, 1, device=device, dtype=observations.dtype)
                        u_s = prev_states[cell_index] @ self.cell_as[cell_index].T
                        u_m = message @ self.cell_am[cell_index].T
                        u_x = visible_x @ self.cell_ax[cell_index].T
                        v_s = prev_states[cell_index] @ self.cell_bs[cell_index].T
                        v_m = message @ self.cell_bm[cell_index].T
                        v_x = visible_x @ self.cell_bx[cell_index].T
                        u = u_s + u_m + u_x
                        v = v_s + v_m + v_x
                        recurrent_term = u_s @ self.cell_cu[cell_index].T + v_s @ self.cell_cv[cell_index].T
                        message_term = u_m @ self.cell_cu[cell_index].T + v_m @ self.cell_cv[cell_index].T
                        input_term = u_x @ self.cell_cu[cell_index].T + v_x @ self.cell_cv[cell_index].T
                        if self.interaction_mode == "low_rank_multiplicative":
                            extra = (u * v) @ self.cell_ci[cell_index].T
                        else:
                            # Same parameter count as the multiplicative branch,
                            # but no elementwise interaction.
                            extra = (u + v) @ self.cell_ci[cell_index].T
                        preactivation = u @ self.cell_cu[cell_index].T + v @ self.cell_cv[cell_index].T + extra + self.cell_b[cell_index]
                    if self.state_update_mode == "transport_matched_additive":
                        transport = self._stable_transport_matrix(cell_index)
                        preactivation = preactivation + prev_states[cell_index] @ transport.T
                    candidate_state = torch.tanh(preactivation)
                    if self.state_update_mode == "direct":
                        proposed_state = candidate_state
                    elif self.state_update_mode == "learned_leaky":
                        alpha = torch.sigmoid(self.cell_alpha_logits[cell_index])
                        proposed_state = (1.0 - alpha) * prev_states[cell_index] + alpha * candidate_state
                    elif self.state_update_mode == "linear_transport":
                        transport = self._stable_transport_matrix(cell_index)
                        proposed_state = prev_states[cell_index] @ transport.T + candidate_state
                    else:
                        proposed_state = candidate_state
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
                            state_modulator = active * state_modulator + (1.0 - active)
                    else:
                        state = proposed_state
                        output = proposed_output
                current_states.append(state)
                current_candidates.append(candidate_state)
                current_outputs.append(output)
                if return_trace:
                    current_messages.append(message)
                    current_recurrent_terms.append(recurrent_term)
                    current_message_terms.append(message_term)
                    current_input_terms.append(input_term)
                    current_state_modulators.append(state_modulator)
            prev_states = current_states
            prev_outputs = current_outputs
            if return_trace:
                state_trace.append(torch.stack(current_states, dim=1))
                candidate_state_trace.append(torch.stack(current_candidates, dim=1))
                output_trace.append(torch.stack(current_outputs, dim=1))
                aggregate_message_trace.append(torch.stack(current_messages, dim=1))
                recurrent_term_trace.append(torch.stack(current_recurrent_terms, dim=1))
                message_term_trace.append(torch.stack(current_message_terms, dim=1))
                input_term_trace.append(torch.stack(current_input_terms, dim=1))
                state_modulator_trace.append(torch.stack(current_state_modulators, dim=1))
        stacked = torch.cat(prev_states, dim=1)
        prediction = torch.tanh(self.readout(stacked)).squeeze(-1)
        if return_trace:
            trace = ForwardTrace(
                states=torch.stack(state_trace, dim=1),
                candidate_states=torch.stack(candidate_state_trace, dim=1),
                outputs=torch.stack(output_trace, dim=1),
                messages=torch.stack(aggregate_message_trace, dim=1),
                recurrent_terms=torch.stack(recurrent_term_trace, dim=1),
                message_terms=torch.stack(message_term_trace, dim=1),
                input_terms=torch.stack(input_term_trace, dim=1),
                state_modulators=torch.stack(state_modulator_trace, dim=1),
            )
            return prediction, trace
        return prediction

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def state_update_coefficients(self) -> list[float]:
        if self.state_update_mode != "learned_leaky":
            return [1.0] * len(self.graph.cells)
        return [float(torch.sigmoid(value).detach().cpu().item()) for value in self.cell_alpha_logits]

    def _stable_transport_matrix(self, cell_index: int) -> torch.Tensor:
        raw = self.cell_transport_raw[cell_index]
        spectral_norm = torch.linalg.matrix_norm(raw, ord=2)
        return self.transport_rho * raw / torch.clamp(spectral_norm, min=1e-6)

    def transport_diagnostics(self) -> list[dict[str, float]]:
        if self.state_update_mode not in {"linear_transport", "transport_matched_additive"}:
            return []
        output: list[dict[str, float]] = []
        with torch.no_grad():
            for index in range(len(self.cell_transport_raw)):
                matrix = self._stable_transport_matrix(index).detach().cpu()
                eigenvalues = torch.linalg.eigvals(matrix)
                output.append({
                    "cell_index": float(index),
                    "spectral_norm": float(torch.linalg.matrix_norm(matrix, ord=2).item()),
                    "spectral_radius": float(torch.max(torch.abs(eigenvalues)).item()),
                })
        return output


def clone_with_state(model: NeutralGraphModel) -> NeutralGraphModel:
    clone = NeutralGraphModel(
        model.graph,
        obs_dim=model.obs_dim,
        state_dim=model.state_dim,
        message_dim=model.message_dim,
        state_update_mode=model.state_update_mode,
        alpha_init=model.alpha_init,
        transport_rho=model.transport_rho,
        interaction_mode=model.interaction_mode,
        interaction_rank=model.interaction_rank,
        state_modulation_mode=model.state_modulation_mode,
    )
    clone.load_state_dict(model.state_dict())
    return clone
