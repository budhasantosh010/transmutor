from __future__ import annotations

import torch

from experiments.v837_primitive_invention.v837r.recurrent_coupling import LOCAL_STATE_DIM, MESSAGE_DIM, NUM_CELLS
from experiments.v837_primitive_invention.v837y.candidate_interaction import (
    CandidateInteractionSpec,
    CandidateInteractionTrace,
    ControlledCandidateInteractionModel,
)

STAGE_MODES = {"historical_mixed", "fully_synchronous"}


class CandidateStageModel(ControlledCandidateInteractionModel):
    """Frozen V837y Y3 parent with only message/candidate stage timing varied."""

    architecture_name = "v837z_candidate_stage"

    def __init__(self, graph, *, stage_mode: str, coupling_initialization_seed: int, obs_dim: int = 6):
        if stage_mode not in STAGE_MODES:
            raise ValueError(f"unsupported stage mode: {stage_mode}")
        self.stage_mode = stage_mode
        super().__init__(
            graph,
            spec=CandidateInteractionSpec(
                global_scalar_control=True,
                candidate_coupling_mode="rank4_cross_block",
                coupling_rank=4,
            ),
            coupling_initialization_seed=coupling_initialization_seed,
            obs_dim=obs_dim,
        )

    @property
    def synchronous_candidate_stage(self) -> bool:
        return self.stage_mode == "fully_synchronous"

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
        if self.stage_mode == "historical_mixed":
            return super().forward(
                observations,
                lengths,
                disable_messages=disable_messages,
                zero_coupling_source_cell=zero_coupling_source_cell,
                global_gate_override=global_gate_override,
                return_trace=return_trace,
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
            snapshot_outputs = prev_outputs
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

            # Z1's only scientific change: every edge reads the previous-output snapshot.
            messages: list[torch.Tensor] = []
            for cell_index in range(NUM_CELLS):
                message = torch.zeros(batch, MESSAGE_DIM, device=device, dtype=dtype)
                if not disable_messages:
                    for edge_index, edge in enumerate(self.graph.edges):
                        if edge.dst == cell_index:
                            message = message + self.base.edge_weights[edge_index] * snapshot_outputs[edge.src]
                messages.append(message)

            current_states: list[torch.Tensor] = []
            current_candidates: list[torch.Tensor] = []
            current_outputs: list[torch.Tensor] = []
            current_local_terms: list[torch.Tensor] = []
            current_message_terms: list[torch.Tensor] = []
            current_input_terms: list[torch.Tensor] = []
            current_trace_messages: list[torch.Tensor] = []

            for cell_index in range(NUM_CELLS):
                visible_x = self._visible_input(x_t, cell_index)
                local_term = snapshot_states[cell_index] @ self.base.cell_ws[cell_index].T
                message_term = messages[cell_index] @ self.base.cell_wm[cell_index].T
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
                    output = active * proposed_output + (1.0 - active) * snapshot_outputs[cell_index]
                    message_for_trace = active * messages[cell_index]
                    local_for_trace = active * local_term
                    message_term_for_trace = active * message_term
                    input_for_trace = active * input_term
                else:
                    state = proposed_state
                    output = proposed_output
                    message_for_trace = messages[cell_index]
                    local_for_trace = local_term
                    message_term_for_trace = message_term
                    input_for_trace = input_term
                current_states.append(state)
                current_candidates.append(candidate)
                current_outputs.append(output)
                if return_trace:
                    current_trace_messages.append(message_for_trace)
                    current_local_terms.append(local_for_trace)
                    current_message_terms.append(message_term_for_trace)
                    current_input_terms.append(input_for_trace)

            # Simultaneous commit: no candidate/output produced above is visible within this timestep.
            prev_states = current_states
            prev_outputs = current_outputs
            if return_trace:
                traces["states"].append(torch.stack(current_states, dim=1))
                traces["candidates"].append(torch.stack(current_candidates, dim=1))
                traces["outputs"].append(torch.stack(current_outputs, dim=1))
                traces["messages"].append(torch.stack(current_trace_messages, dim=1))
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
                gate_for_trace = (t < lengths).to(dtype).unsqueeze(1) * gate + (1.0 - (t < lengths).to(dtype).unsqueeze(1)) if lengths is not None else gate
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


def historical_candidate_depths(graph) -> list[int]:
    """Same-step nonlinear depth induced by executable non-recurrent edges."""
    depths = [1 for _ in range(len(graph.cells))]
    for dst in range(len(graph.cells)):
        parents = [edge.src for edge in graph.edges if edge.dst == dst and not edge.recurrent and edge.src < dst]
        if parents:
            depths[dst] = 1 + max(depths[src] for src in parents)
    return depths


def synchronous_candidate_depths(graph) -> list[int]:
    return [1 for _ in graph.cells]
