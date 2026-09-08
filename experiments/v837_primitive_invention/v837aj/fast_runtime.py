from __future__ import annotations

from types import MethodType

import torch

from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3
from experiments.v837_primitive_invention.v837r.recurrent_coupling import LOCAL_STATE_DIM, MESSAGE_DIM, NUM_CELLS


def _fast_no_trace_forward(
    self: CandidateInputFactorizationY3,
    observations: torch.Tensor,
    lengths: torch.Tensor | None = None,
    *,
    disable_messages: bool = False,
    zero_coupling_source_cell: int | None = None,
    global_gate_override: float | torch.Tensor | None = None,
    return_trace: bool = False,
):
    """Bit-equivalent V837aj AF1D fast path for the common no-trace case.

    The frozen historical implementation remains the reference path whenever a
    trace is requested.  This adapter changes no parameters or arithmetic
    ordering that contributes to predictions: it only hoists the per-timestep
    active mask, avoids trace-only tensor products when no trace is requested,
    and pre-indexes incoming message edges while preserving original edge order.
    """
    if return_trace:
        return CandidateInputFactorizationY3.forward(
            self,
            observations,
            lengths,
            disable_messages=disable_messages,
            zero_coupling_source_cell=zero_coupling_source_cell,
            global_gate_override=global_gate_override,
            return_trace=True,
        )
    if not self.global_scalar_control:
        return CandidateInputFactorizationY3.forward(
            self,
            observations,
            lengths,
            disable_messages=disable_messages,
            zero_coupling_source_cell=zero_coupling_source_cell,
            global_gate_override=global_gate_override,
            return_trace=False,
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
    incoming = self._v837aj_incoming_edge_indices

    # CandidateInputFactorizationY3.forward clears only the shared-projection
    # cache. AF1D is de-shared, but mirror the reference state exactly anyway.
    self._shared_projection_cache_key = None
    self._shared_projection_cache_value = None

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
        if lengths is not None:
            active = (t < lengths).to(dtype).unsqueeze(1)
            inactive = 1.0 - active
        current_states: list[torch.Tensor] = []
        current_outputs: list[torch.Tensor] = []
        for cell_index in range(NUM_CELLS):
            message = torch.zeros(batch, MESSAGE_DIM, device=device, dtype=dtype)
            if not disable_messages:
                for edge_index in incoming[cell_index]:
                    edge = self.graph.edges[edge_index]
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
                state = active * proposed_state + inactive * snapshot_states[cell_index]
                output = active * proposed_output + inactive * prev_outputs[cell_index]
            else:
                state = proposed_state
                output = proposed_output
            current_states.append(state)
            current_outputs.append(output)
        prev_states = current_states
        prev_outputs = current_outputs
    return torch.tanh(self.base.readout(torch.cat(prev_states, dim=1))).squeeze(-1)


def enable_fast_runtime(model: CandidateInputFactorizationY3) -> CandidateInputFactorizationY3:
    if type(model) is not CandidateInputFactorizationY3:
        raise TypeError("V837aj fast runtime requires exact CandidateInputFactorizationY3")
    incoming: list[tuple[int, ...]] = []
    for cell_index in range(NUM_CELLS):
        incoming.append(tuple(i for i, edge in enumerate(model.graph.edges) if edge.dst == cell_index))
    model._v837aj_incoming_edge_indices = tuple(incoming)
    model.forward = MethodType(_fast_no_trace_forward, model)
    return model
