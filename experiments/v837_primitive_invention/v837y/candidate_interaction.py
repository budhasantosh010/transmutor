from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from experiments.v837_primitive_invention.v837r.recurrent_coupling import (
    LOCAL_STATE_DIM,
    MESSAGE_DIM,
    NUM_CELLS,
    TOTAL_STATE_DIM,
    CouplingTrace,
    GloballyCoupledNeutralGraphModel,
    RecurrentCouplingSpec,
    coupling_actual_macs,
    local_recurrent_macs,
)


ALLOWED_CANDIDATE_MODES = {"none", "rank4_cross_block", "rank4_matched_local"}


@dataclass(frozen=True)
class CandidateInteractionSpec:
    global_scalar_control: bool
    candidate_coupling_mode: str
    coupling_rank: int | None = None
    matched_local_rank: int | None = None

    def validate(self) -> None:
        if self.candidate_coupling_mode not in ALLOWED_CANDIDATE_MODES:
            raise ValueError(f"unsupported candidate coupling mode: {self.candidate_coupling_mode}")
        if self.candidate_coupling_mode == "none":
            if self.coupling_rank is not None or self.matched_local_rank is not None:
                raise ValueError("none candidate coupling may not declare a rank")
        elif self.candidate_coupling_mode == "rank4_cross_block":
            if self.coupling_rank != 4 or self.matched_local_rank is not None:
                raise ValueError("V837y cross-block coupling is frozen to rank 4")
        else:
            if self.matched_local_rank != 4 or self.coupling_rank is not None:
                raise ValueError("V837y matched-local coupling is frozen to rank 4")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "global_scalar_control": bool(self.global_scalar_control),
            "candidate_coupling_mode": self.candidate_coupling_mode,
            "coupling_rank": self.coupling_rank,
            "matched_local_rank": self.matched_local_rank,
        }


@dataclass
class CandidateInteractionTrace(CouplingTrace):
    global_gates: torch.Tensor | None = None  # [B,T,1]


class ControlledCandidateInteractionModel(GloballyCoupledNeutralGraphModel):
    """Frozen V837r candidate branch optionally composed with the frozen V837x global carry controller.

    Both the global scalar controller and rank-4 candidate branch read the same
    snapshotted previous 10x4 recurrent state at the start of each timestep.
    Historical mixed same-step/recurrent message scheduling is preserved.
    """

    architecture_name = "v837y_controlled_candidate_interaction"

    def __init__(
        self,
        graph,
        *,
        spec: CandidateInteractionSpec,
        coupling_initialization_seed: int,
        obs_dim: int = 6,
    ):
        spec.validate()
        self.interaction_spec = spec
        if spec.candidate_coupling_mode == "none":
            coupling = RecurrentCouplingSpec(
                mode="none",
                cross_block_only=True,
                initialization_seed=int(coupling_initialization_seed),
            )
        elif spec.candidate_coupling_mode == "rank4_cross_block":
            coupling = RecurrentCouplingSpec(
                mode="low_rank",
                rank=4,
                cross_block_only=True,
                scaling=1.0,
                initialization_seed=int(coupling_initialization_seed),
            )
        else:
            coupling = RecurrentCouplingSpec(
                mode="parameter_matched_local",
                matched_local_rank=4,
                cross_block_only=True,
                scaling=1.0,
                initialization_seed=int(coupling_initialization_seed),
            )
        super().__init__(graph, coupling, obs_dim=obs_dim, state_modulation_mode="none")
        if spec.global_scalar_control:
            # Exact V837x JOINT_INPUT_STATE_GLOBAL_SCALAR parameterization and scale.
            self.global_ws = nn.Parameter(torch.randn(TOTAL_STATE_DIM) * (0.20 / math.sqrt(TOTAL_STATE_DIM)))
            self.global_wx = nn.Parameter(torch.randn(obs_dim) * (0.20 / math.sqrt(obs_dim)))
            self.global_b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("global_ws", None)
            self.register_parameter("global_wx", None)
            self.register_parameter("global_b", None)

    @property
    def global_scalar_control(self) -> bool:
        return bool(self.interaction_spec.global_scalar_control)

    @property
    def controller_param_count(self) -> int:
        return 47 if self.global_scalar_control else 0

    @property
    def controller_macs(self) -> int:
        return 46 if self.global_scalar_control else 0

    @property
    def candidate_branch_param_count(self) -> int:
        if self.interaction_spec.candidate_coupling_mode == "none":
            return 0
        return 320

    @property
    def candidate_branch_macs(self) -> int:
        return int(coupling_actual_macs(self.coupling))

    @property
    def recurrent_controller_macs(self) -> int:
        return int(local_recurrent_macs() + self.candidate_branch_macs + self.controller_macs)

    def _global_gate(self, snapshot_states: list[torch.Tensor], x_t: torch.Tensor) -> torch.Tensor:
        if not self.global_scalar_control:
            raise RuntimeError("global controller requested while disabled")
        stacked = torch.cat(snapshot_states, dim=1)
        state_term = torch.sum(stacked * self.global_ws.view(1, -1), dim=1, keepdim=True)
        input_term = torch.sum(x_t * self.global_wx.view(1, -1), dim=1, keepdim=True)
        return torch.sigmoid(state_term + input_term + self.global_b)

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        disable_messages: bool = False,
        zero_coupling_source_cell: int | None = None,
        global_gate_override: float | torch.Tensor | None = None,
        return_trace: bool = False,
    ):
        # For Y0/Y2, dispatch directly through the frozen V837r implementation.
        if not self.global_scalar_control:
            result = super().forward(
                observations,
                lengths,
                disable_messages=disable_messages,
                zero_coupling_source_cell=zero_coupling_source_cell,
                return_trace=return_trace,
            )
            if not return_trace:
                return result
            prediction, trace = result
            return prediction, CandidateInteractionTrace(
                states=trace.states,
                candidate_states=trace.candidate_states,
                outputs=trace.outputs,
                messages=trace.messages,
                recurrent_terms=trace.recurrent_terms,
                global_recurrent_terms=trace.global_recurrent_terms,
                matched_local_terms=trace.matched_local_terms,
                message_terms=trace.message_terms,
                input_terms=trace.input_terms,
                state_modulators=trace.state_modulators,
                global_gates=None,
            )

        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.obs_dim:
            raise ValueError(f"observation dimension {observed_dim} != model obs_dim {self.obs_dim}")
        device = observations.device
        dtype = observations.dtype
        prev_states = [torch.zeros(batch, LOCAL_STATE_DIM, device=device, dtype=dtype) for _ in range(NUM_CELLS)]
        prev_outputs = [torch.zeros(batch, MESSAGE_DIM, device=device, dtype=dtype) for _ in range(NUM_CELLS)]
        traces: dict[str, list[torch.Tensor]] = {name: [] for name in (
            "states", "candidates", "outputs", "messages", "local", "global", "matched", "message_terms", "input_terms", "gates"
        )}
        for t in range(steps):
            x_t = observations[:, t, :]
            snapshot_states = prev_states
            global_terms = self._global_terms(snapshot_states, zero_coupling_source_cell=zero_coupling_source_cell)
            matched_terms = self._matched_local_terms(snapshot_states)
            gate = self._global_gate(snapshot_states, x_t)
            if global_gate_override is not None:
                if isinstance(global_gate_override, torch.Tensor):
                    gate = global_gate_override.to(device=device, dtype=dtype)
                    if gate.ndim == 0:
                        gate = gate.expand(batch, 1)
                    elif gate.ndim == 1:
                        gate = gate.view(-1, 1)
                else:
                    gate = torch.full((batch, 1), float(global_gate_override), device=device, dtype=dtype)
            current_states: list[torch.Tensor] = []
            current_candidates: list[torch.Tensor] = []
            current_outputs: list[torch.Tensor] = []
            current_messages: list[torch.Tensor] = []
            current_local_terms: list[torch.Tensor] = []
            current_message_terms: list[torch.Tensor] = []
            current_input_terms: list[torch.Tensor] = []
            for cell_index in range(NUM_CELLS):
                message = torch.zeros(batch, MESSAGE_DIM, device=device, dtype=dtype)
                if not disable_messages:
                    for edge_index, edge in enumerate(self.graph.edges):
                        if edge.dst != cell_index:
                            continue
                        source = prev_outputs[edge.src] if edge.recurrent or edge.src >= len(current_outputs) else current_outputs[edge.src]
                        message = message + self.base.edge_weights[edge_index] * source
                visible_x = self._visible_input(x_t, cell_index)
                local_term = snapshot_states[cell_index] @ self.base.cell_ws[cell_index].T
                message_term = message @ self.base.cell_wm[cell_index].T
                input_term = visible_x @ self.base.cell_wx[cell_index].T
                preactivation = (
                    local_term
                    + global_terms[cell_index]
                    + matched_terms[cell_index]
                    + message_term
                    + input_term
                    + self.base.cell_b[cell_index]
                )
                candidate = torch.tanh(preactivation)
                proposed_state = gate * snapshot_states[cell_index] + (1.0 - gate) * candidate
                proposed_output = proposed_state @ self.base.cell_wo[cell_index].T
                if lengths is not None:
                    active = (t < lengths).to(dtype).unsqueeze(1)
                    state = active * proposed_state + (1.0 - active) * snapshot_states[cell_index]
                    output = active * proposed_output + (1.0 - active) * prev_outputs[cell_index]
                    message_for_trace = active * message
                    local_for_trace = active * local_term
                    global_for_trace = active * global_terms[cell_index]
                    matched_for_trace = active * matched_terms[cell_index]
                    message_term_for_trace = active * message_term
                    input_for_trace = active * input_term
                    gate_for_trace = active * gate + (1.0 - active)
                else:
                    state = proposed_state
                    output = proposed_output
                    message_for_trace = message
                    local_for_trace = local_term
                    global_for_trace = global_terms[cell_index]
                    matched_for_trace = matched_terms[cell_index]
                    message_term_for_trace = message_term
                    input_for_trace = input_term
                    gate_for_trace = gate
                current_states.append(state)
                current_candidates.append(candidate)
                current_outputs.append(output)
                if return_trace:
                    current_messages.append(message_for_trace)
                    current_local_terms.append(local_for_trace)
                    current_message_terms.append(message_term_for_trace)
                    current_input_terms.append(input_for_trace)
            prev_states = current_states
            prev_outputs = current_outputs
            if return_trace:
                traces["states"].append(torch.stack(current_states, dim=1))
                traces["candidates"].append(torch.stack(current_candidates, dim=1))
                traces["outputs"].append(torch.stack(current_outputs, dim=1))
                traces["messages"].append(torch.stack(current_messages, dim=1))
                traces["local"].append(torch.stack(current_local_terms, dim=1))
                traces["global"].append(torch.stack([
                    ((t < lengths).to(dtype).unsqueeze(1) * term if lengths is not None else term)
                    for term in global_terms
                ], dim=1))
                traces["matched"].append(torch.stack([
                    ((t < lengths).to(dtype).unsqueeze(1) * term if lengths is not None else term)
                    for term in matched_terms
                ], dim=1))
                traces["message_terms"].append(torch.stack(current_message_terms, dim=1))
                traces["input_terms"].append(torch.stack(current_input_terms, dim=1))
                traces["gates"].append(gate_for_trace)
        prediction = torch.tanh(self.base.readout(torch.cat(prev_states, dim=1))).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, CandidateInteractionTrace(
            states=torch.stack(traces["states"], dim=1),
            candidate_states=torch.stack(traces["candidates"], dim=1),
            outputs=torch.stack(traces["outputs"], dim=1),
            messages=torch.stack(traces["messages"], dim=1),
            recurrent_terms=torch.stack(traces["local"], dim=1),
            global_recurrent_terms=torch.stack(traces["global"], dim=1),
            matched_local_terms=torch.stack(traces["matched"], dim=1),
            message_terms=torch.stack(traces["message_terms"], dim=1),
            input_terms=torch.stack(traces["input_terms"], dim=1),
            state_modulators=torch.stack(traces["gates"], dim=1).unsqueeze(2).expand(-1, -1, NUM_CELLS, -1),
            global_gates=torch.stack(traces["gates"], dim=1),
        )
