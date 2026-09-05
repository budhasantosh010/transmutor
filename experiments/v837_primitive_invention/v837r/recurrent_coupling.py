from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from experiments.v837_primitive_invention.common.graph import GraphSpec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel


TOTAL_STATE_DIM = 40
NUM_CELLS = 10
LOCAL_STATE_DIM = 4
MESSAGE_DIM = 4


@dataclass(frozen=True)
class RecurrentCouplingSpec:
    mode: str
    total_state_dim: int = TOTAL_STATE_DIM
    local_block_dim: int = LOCAL_STATE_DIM
    rank: int | None = None
    cross_block_only: bool = True
    scaling: float = 1.0
    initialization_seed: int = 0
    matched_local_rank: int | None = None

    def validate(self) -> None:
        if self.mode not in {"none", "low_rank", "dense", "parameter_matched_local"}:
            raise ValueError(f"unsupported coupling mode: {self.mode}")
        if self.total_state_dim != TOTAL_STATE_DIM:
            raise ValueError(f"V837r requires total_state_dim={TOTAL_STATE_DIM}")
        if self.local_block_dim != LOCAL_STATE_DIM:
            raise ValueError(f"V837r requires local_block_dim={LOCAL_STATE_DIM}")
        if self.total_state_dim % self.local_block_dim:
            raise ValueError("total_state_dim must divide into fixed local blocks")
        if self.mode == "low_rank" and (self.rank is None or int(self.rank) <= 0):
            raise ValueError("low_rank coupling requires positive rank")
        if self.mode == "dense" and self.rank is not None:
            raise ValueError("dense coupling must not declare a rank")
        if self.mode == "parameter_matched_local" and (self.matched_local_rank is None or int(self.matched_local_rank) <= 0):
            raise ValueError("parameter_matched_local requires matched_local_rank")
        if self.mode == "none" and (self.rank is not None or self.matched_local_rank is not None):
            raise ValueError("none coupling may not carry rank metadata")
        if not math.isfinite(float(self.scaling)) or float(self.scaling) <= 0.0:
            raise ValueError("coupling scaling must be positive and finite")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "total_state_dim": self.total_state_dim,
            "local_block_dim": self.local_block_dim,
            "rank": self.rank,
            "cross_block_only": self.cross_block_only,
            "scaling": self.scaling,
            "initialization_seed": self.initialization_seed,
            "matched_local_rank": self.matched_local_rank,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RecurrentCouplingSpec":
        spec = cls(
            mode=str(payload["mode"]),
            total_state_dim=int(payload.get("total_state_dim", TOTAL_STATE_DIM)),
            local_block_dim=int(payload.get("local_block_dim", LOCAL_STATE_DIM)),
            rank=None if payload.get("rank") is None else int(payload["rank"]),
            cross_block_only=bool(payload.get("cross_block_only", True)),
            scaling=float(payload.get("scaling", 1.0)),
            initialization_seed=int(payload.get("initialization_seed", 0)),
            matched_local_rank=None if payload.get("matched_local_rank") is None else int(payload["matched_local_rank"]),
        )
        spec.validate()
        return spec


def cross_block_mask(total_state_dim: int = TOTAL_STATE_DIM, local_block_dim: int = LOCAL_STATE_DIM) -> torch.Tensor:
    if total_state_dim % local_block_dim:
        raise ValueError("state dimension must divide into local blocks")
    mask = torch.ones(total_state_dim, total_state_dim, dtype=torch.float32)
    for start in range(0, total_state_dim, local_block_dim):
        mask[start : start + local_block_dim, start : start + local_block_dim] = 0.0
    return mask


def coupling_core_macs(spec: RecurrentCouplingSpec) -> int:
    """Requested architectural coupling-core estimate before cross-block exclusion overhead."""
    spec.validate()
    if spec.mode == "none":
        return 0
    if spec.mode == "low_rank":
        return 2 * spec.total_state_dim * int(spec.rank)
    if spec.mode == "dense":
        return spec.total_state_dim * spec.total_state_dim
    return 2 * NUM_CELLS * LOCAL_STATE_DIM * int(spec.matched_local_rank)


def coupling_actual_macs(spec: RecurrentCouplingSpec) -> int:
    """Approximate implemented recurrent MACs including cross-block exclusion when applicable."""
    spec.validate()
    if spec.mode == "none":
        return 0
    if spec.mode == "low_rank":
        # Full factorized product plus subtraction of the ten local block products.
        return 4 * spec.total_state_dim * int(spec.rank)
    if spec.mode == "dense":
        if spec.cross_block_only:
            return spec.total_state_dim * spec.total_state_dim - NUM_CELLS * LOCAL_STATE_DIM * LOCAL_STATE_DIM
        return spec.total_state_dim * spec.total_state_dim
    return 2 * NUM_CELLS * LOCAL_STATE_DIM * int(spec.matched_local_rank)


def local_recurrent_macs() -> int:
    return NUM_CELLS * LOCAL_STATE_DIM * LOCAL_STATE_DIM


def coupling_scaling_complexity(spec: RecurrentCouplingSpec) -> str:
    if spec.mode == "none":
        return "O(D) block-local"
    if spec.mode == "low_rank":
        return "O(D*r) low-rank global"
    if spec.mode == "dense":
        return "O(D^2) dense global"
    return "O(D*r_local) local matched control"


@dataclass
class CouplingTrace:
    states: torch.Tensor  # [B,T,N,4]
    candidate_states: torch.Tensor  # [B,T,N,4]
    outputs: torch.Tensor  # [B,T,N,4]
    messages: torch.Tensor  # [B,T,N,4]
    recurrent_terms: torch.Tensor  # historical local recurrent terms [B,T,N,4]
    global_recurrent_terms: torch.Tensor  # added cross-dimensional terms [B,T,N,4]
    matched_local_terms: torch.Tensor  # parameter-matched local branch terms [B,T,N,4]
    message_terms: torch.Tensor  # [B,T,N,4]
    input_terms: torch.Tensor  # [B,T,N,4]
    state_modulators: torch.Tensor | None = None  # [B,T,N,1] for V837s dynamic-scalar factorial


class GloballyCoupledNeutralGraphModel(nn.Module):
    """Historical local neutral graph plus one isolated recurrent-coupling branch.

    The local state remains ten independent 4D tensors.  Global coupling reads
    only the snapshotted previous 40D state and is added to each cell candidate.
    Historical message ordering and the 40-wide readout are preserved.
    """

    def __init__(
        self,
        graph: GraphSpec,
        coupling: RecurrentCouplingSpec,
        *,
        obs_dim: int = 6,
        state_modulation_mode: str = "none",
    ):
        super().__init__()
        graph.validate()
        coupling.validate()
        if state_modulation_mode not in {"none", "dynamic_scalar_candidate", "dynamic_scalar_matched_additive"}:
            raise ValueError(f"unsupported state modulation mode: {state_modulation_mode}")
        if len(graph.cells) != NUM_CELLS:
            raise ValueError("V837r requires the calibrated ten-cell graph")
        self.graph = graph.clone()
        self.coupling = coupling
        self.obs_dim = int(obs_dim)
        self.state_dim = LOCAL_STATE_DIM
        self.message_dim = MESSAGE_DIM
        self.state_modulation_mode = state_modulation_mode
        self.base = NeutralGraphModel(
            self.graph,
            obs_dim=self.obs_dim,
            state_dim=LOCAL_STATE_DIM,
            message_dim=MESSAGE_DIM,
            state_update_mode="direct",
            interaction_mode="none",
            state_modulation_mode=state_modulation_mode,
        )
        self.register_buffer("cross_block_mask", cross_block_mask(), persistent=True)

        if coupling.mode == "low_rank":
            rank = int(coupling.rank)
            generator = torch.Generator(device="cpu").manual_seed(int(coupling.initialization_seed))
            sigma = (1.0 / float(TOTAL_STATE_DIM * rank)) ** 0.25
            self.global_u = nn.Parameter(torch.randn(TOTAL_STATE_DIM, rank, generator=generator) * sigma)
            self.global_v = nn.Parameter(torch.randn(TOTAL_STATE_DIM, rank, generator=generator) * sigma)
        elif coupling.mode == "dense":
            generator = torch.Generator(device="cpu").manual_seed(int(coupling.initialization_seed))
            self.global_dense = nn.Parameter(torch.randn(TOTAL_STATE_DIM, TOTAL_STATE_DIM, generator=generator) / math.sqrt(TOTAL_STATE_DIM))
        elif coupling.mode == "parameter_matched_local":
            local_rank = int(coupling.matched_local_rank)
            sigma = (1.0 / float(LOCAL_STATE_DIM * local_rank)) ** 0.25
            self.local_extra_u = nn.ParameterList()
            self.local_extra_v = nn.ParameterList()
            for cell_index in range(NUM_CELLS):
                generator = torch.Generator(device="cpu").manual_seed(
                    deterministic_int("v837r-local-control-init", coupling.initialization_seed, cell_index, local_rank)
                )
                self.local_extra_u.append(nn.Parameter(torch.randn(LOCAL_STATE_DIM, local_rank, generator=generator) * sigma))
                self.local_extra_v.append(nn.Parameter(torch.randn(LOCAL_STATE_DIM, local_rank, generator=generator) * sigma))

    @property
    def input_edge_count(self) -> int:
        return self.base.input_edge_count

    @property
    def internal_message_edge_count(self) -> int:
        return self.base.internal_message_edge_count

    @property
    def readout_input_width(self) -> int:
        return int(self.base.readout.in_features)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def parameter_bytes(self) -> int:
        return int(sum(parameter.numel() * parameter.element_size() for parameter in self.parameters()))

    def added_parameter_count(self) -> int:
        return self.parameter_count() - self.base.parameter_count()

    def serialization_metadata(self) -> dict:
        return {
            "coupling": self.coupling.to_dict(),
            "state_layout": "local_10x4",
            "state_modulation_mode": self.state_modulation_mode,
            "total_state_dim": TOTAL_STATE_DIM,
            "readout_input_width": self.readout_input_width,
            "parameter_count": self.parameter_count(),
            "added_parameter_count": self.added_parameter_count(),
            "coupling_core_macs": coupling_core_macs(self.coupling),
            "coupling_actual_macs": coupling_actual_macs(self.coupling),
            "local_recurrent_macs": local_recurrent_macs(),
            "scaling_complexity": coupling_scaling_complexity(self.coupling),
        }

    def effective_global_matrix(self) -> torch.Tensor:
        if self.coupling.mode == "low_rank":
            matrix = self.global_u @ self.global_v.T
        elif self.coupling.mode == "dense":
            matrix = self.global_dense
        else:
            return torch.zeros(TOTAL_STATE_DIM, TOTAL_STATE_DIM, device=self.base.readout.weight.device)
        if self.coupling.cross_block_only:
            matrix = matrix * self.cross_block_mask
        return float(self.coupling.scaling) * matrix

    def coupling_diagnostics(self) -> dict:
        if self.coupling.mode not in {"low_rank", "dense"}:
            return {
                "configured_rank": self.coupling.rank,
                "effective_rank": 0,
                "singular_values": [],
                "spectral_norm": 0.0,
                "frobenius_norm": 0.0,
                "cross_block_energy": 0.0,
                "diagonal_block_energy": 0.0,
                "offdiag_fraction": 0.0,
            }
        with torch.no_grad():
            matrix = self.effective_global_matrix().detach().cpu()
            singular = torch.linalg.svdvals(matrix)
            tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * float(singular.max().item() if singular.numel() else 0.0)
            effective_rank = int(torch.sum(singular > tolerance).item())
            squared = matrix * matrix
            mask = self.cross_block_mask.detach().cpu()
            cross_energy = float(torch.sum(squared * mask).item())
            diag_energy = float(torch.sum(squared * (1.0 - mask)).item())
            total = cross_energy + diag_energy
            return {
                "configured_rank": self.coupling.rank,
                "effective_rank": effective_rank,
                "singular_values": [float(v) for v in singular.tolist()],
                "spectral_norm": float(singular.max().item()) if singular.numel() else 0.0,
                "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
                "cross_block_energy": cross_energy,
                "diagonal_block_energy": diag_energy,
                "offdiag_fraction": 0.0 if total <= 1e-12 else cross_energy / total,
            }

    def _visible_input(self, x_t: torch.Tensor, cell_index: int) -> torch.Tensor:
        if self.base.input_access_mode == "none":
            return torch.zeros_like(x_t)
        if self.base.input_access_mode == "broadcast":
            return x_t
        return x_t * self.base.input_access_mask[:, cell_index].view(1, -1)

    def _global_terms(self, snapshot_states: list[torch.Tensor], zero_coupling_source_cell: int | None = None) -> list[torch.Tensor]:
        batch = snapshot_states[0].shape[0]
        device = snapshot_states[0].device
        dtype = snapshot_states[0].dtype
        if self.coupling.mode not in {"low_rank", "dense"}:
            return [torch.zeros(batch, LOCAL_STATE_DIM, device=device, dtype=dtype) for _ in range(NUM_CELLS)]
        stacked = torch.cat(snapshot_states, dim=1)
        if zero_coupling_source_cell is not None:
            stacked = stacked.clone()
            start = int(zero_coupling_source_cell) * LOCAL_STATE_DIM
            stacked[:, start : start + LOCAL_STATE_DIM] = 0.0
        matrix = self.effective_global_matrix()
        global_flat = stacked @ matrix.T
        return list(global_flat.split(LOCAL_STATE_DIM, dim=1))

    def _matched_local_terms(self, snapshot_states: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.coupling.mode != "parameter_matched_local":
            return [torch.zeros_like(state) for state in snapshot_states]
        terms = []
        for cell_index, state in enumerate(snapshot_states):
            latent = state @ self.local_extra_v[cell_index]
            terms.append(float(self.coupling.scaling) * (latent @ self.local_extra_u[cell_index].T))
        return terms

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        disable_messages: bool = False,
        zero_coupling_source_cell: int | None = None,
        return_trace: bool = False,
    ):
        if self.coupling.mode == "none" and zero_coupling_source_cell is None:
            if not return_trace:
                return self.base(observations, lengths, disable_messages=disable_messages)
            prediction, trace = self.base(observations, lengths, disable_messages=disable_messages, return_trace=True)
            zeros = torch.zeros_like(trace.recurrent_terms)
            return prediction, CouplingTrace(
                states=trace.states,
                candidate_states=trace.candidate_states,
                outputs=trace.outputs,
                messages=trace.messages,
                recurrent_terms=trace.recurrent_terms,
                global_recurrent_terms=zeros,
                matched_local_terms=zeros,
                message_terms=trace.message_terms,
                input_terms=trace.input_terms,
                state_modulators=trace.state_modulators,
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
            "states", "candidates", "outputs", "messages", "local", "global", "matched", "message_terms", "input_terms", "modulators"
        )}
        for t in range(steps):
            x_t = observations[:, t, :]
            snapshot_states = prev_states
            global_terms = self._global_terms(snapshot_states, zero_coupling_source_cell=zero_coupling_source_cell)
            matched_terms = self._matched_local_terms(snapshot_states)
            current_states: list[torch.Tensor] = []
            current_candidates: list[torch.Tensor] = []
            current_outputs: list[torch.Tensor] = []
            current_messages: list[torch.Tensor] = []
            current_local_terms: list[torch.Tensor] = []
            current_message_terms: list[torch.Tensor] = []
            current_input_terms: list[torch.Tensor] = []
            current_modulators: list[torch.Tensor] = []
            for cell_index in range(NUM_CELLS):
                message = torch.zeros(batch, MESSAGE_DIM, device=device, dtype=dtype)
                if not disable_messages:
                    for edge_index, edge in enumerate(self.graph.edges):
                        if edge.dst != cell_index:
                            continue
                        if edge.recurrent or edge.src >= len(current_outputs):
                            source = prev_outputs[edge.src]
                        else:
                            source = current_outputs[edge.src]
                        message = message + self.base.edge_weights[edge_index] * source
                visible_x = self._visible_input(x_t, cell_index)
                if self.state_modulation_mode != "none":
                    modulator_pre = (
                        torch.sum(snapshot_states[cell_index] * self.base.cell_gs[cell_index].view(1, -1), dim=1, keepdim=True)
                        + torch.sum(message * self.base.cell_gm[cell_index].view(1, -1), dim=1, keepdim=True)
                        + torch.sum(visible_x * self.base.cell_gx[cell_index].view(1, -1), dim=1, keepdim=True)
                        + self.base.cell_gb[cell_index]
                    )
                    state_modulator = torch.sigmoid(modulator_pre)
                else:
                    state_modulator = torch.ones(batch, 1, device=device, dtype=dtype)
                recurrent_source = (
                    state_modulator * snapshot_states[cell_index]
                    if self.state_modulation_mode == "dynamic_scalar_candidate"
                    else snapshot_states[cell_index]
                )
                local_term = recurrent_source @ self.base.cell_ws[cell_index].T
                message_term = message @ self.base.cell_wm[cell_index].T
                input_term = visible_x @ self.base.cell_wx[cell_index].T
                preactivation = (
                    local_term
                    + message_term
                    + input_term
                    + global_terms[cell_index]
                    + matched_terms[cell_index]
                    + self.base.cell_b[cell_index]
                )
                if self.state_modulation_mode == "dynamic_scalar_matched_additive":
                    preactivation = preactivation + state_modulator
                candidate = torch.tanh(preactivation)
                proposed_output = candidate @ self.base.cell_wo[cell_index].T
                if lengths is not None:
                    active = (t < lengths).to(dtype).unsqueeze(1)
                    state = active * candidate + (1.0 - active) * snapshot_states[cell_index]
                    output = active * proposed_output + (1.0 - active) * prev_outputs[cell_index]
                    message_for_trace = active * message
                    local_for_trace = active * local_term
                    global_for_trace = active * global_terms[cell_index]
                    matched_for_trace = active * matched_terms[cell_index]
                    message_term_for_trace = active * message_term
                    input_for_trace = active * input_term
                    modulator_for_trace = active * state_modulator + (1.0 - active)
                else:
                    state = candidate
                    output = proposed_output
                    message_for_trace = message
                    local_for_trace = local_term
                    global_for_trace = global_terms[cell_index]
                    matched_for_trace = matched_terms[cell_index]
                    message_term_for_trace = message_term
                    input_for_trace = input_term
                    modulator_for_trace = state_modulator
                current_states.append(state)
                current_candidates.append(candidate)
                current_outputs.append(output)
                current_messages.append(message_for_trace)
                current_local_terms.append(local_for_trace)
                current_message_terms.append(message_term_for_trace)
                current_input_terms.append(input_for_trace)
                current_modulators.append(modulator_for_trace)
                if return_trace:
                    # global/matched traces are added below from the snapshotted recurrent sources
                    pass
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
                traces["modulators"].append(torch.stack(current_modulators, dim=1))
        prediction = torch.tanh(self.base.readout(torch.cat(prev_states, dim=1))).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, CouplingTrace(
            states=torch.stack(traces["states"], dim=1),
            candidate_states=torch.stack(traces["candidates"], dim=1),
            outputs=torch.stack(traces["outputs"], dim=1),
            messages=torch.stack(traces["messages"], dim=1),
            recurrent_terms=torch.stack(traces["local"], dim=1),
            global_recurrent_terms=torch.stack(traces["global"], dim=1),
            matched_local_terms=torch.stack(traces["matched"], dim=1),
            message_terms=torch.stack(traces["message_terms"], dim=1),
            input_terms=torch.stack(traces["input_terms"], dim=1),
            state_modulators=torch.stack(traces["modulators"], dim=1),
        )
