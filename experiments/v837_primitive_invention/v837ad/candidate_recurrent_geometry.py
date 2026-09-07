from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import (
    DynamicGranularityGRU,
    scalarize_dynamic_gate,
)

INPUT_DIM = 6
T2_CONDITION = "T2_scalarized_update_no_reset"
H13 = 13
H40 = 40
BLOCK_COORD = 4
NUM_LOGICAL_BLOCKS = 10

SPARSE_TOPOLOGIES: dict[str, dict[str, tuple[int, ...]]] = {
    "S0": {"offsets": (1, 3, 5, 7), "shifts": (0, 1, 2, 3)},
    "S1": {"offsets": (1, 2, 6, 8), "shifts": (1, 3, 0, 2)},
    "S2": {"offsets": (2, 4, 6, 9), "shifts": (2, 0, 3, 1)},
    "S3": {"offsets": (1, 4, 7, 9), "shifts": (3, 2, 1, 0)},
    "S4": {"offsets": (2, 3, 5, 8), "shifts": (0, 2, 1, 3)},
}


@dataclass(frozen=True)
class CandidateGeometrySpec:
    name: str
    hidden_size: int
    candidate_mask_mode: str
    block_size: int | None = None
    sparse_topology_id: str | None = None

    def validate(self) -> None:
        if self.name == "AD0_H13_dense":
            if self.hidden_size != H13 or self.candidate_mask_mode != "dense" or self.block_size is not None or self.sparse_topology_id is not None:
                raise ValueError("AD0 must be exact H13 dense T2")
            return
        if self.hidden_size != H40:
            raise ValueError("all post-anchor V837ad conditions are H40")
        if self.candidate_mask_mode == "dense":
            if self.block_size is not None or self.sparse_topology_id is not None:
                raise ValueError("dense condition may not declare block/topology")
            return
        if self.candidate_mask_mode == "block_diagonal":
            if self.block_size not in {20, 8, 4} or self.sparse_topology_id is not None:
                raise ValueError("block geometry must be frozen 20/8/4")
            if H40 % int(self.block_size) != 0:
                raise ValueError("block size must divide H40")
            return
        if self.candidate_mask_mode == "degree4_global_sparse":
            if self.block_size is not None or self.sparse_topology_id not in SPARSE_TOPOLOGIES:
                raise ValueError("sparse geometry must use frozen S0-S4 topology")
            return
        raise ValueError(f"unsupported candidate mask mode: {self.candidate_mask_mode}")


CONDITION_SPECS: dict[str, CandidateGeometrySpec] = {
    "AD0_H13_dense": CandidateGeometrySpec("AD0_H13_dense", H13, "dense"),
    "AD1_H40_dense": CandidateGeometrySpec("AD1_H40_dense", H40, "dense"),
    "AD2_H40_2x20": CandidateGeometrySpec("AD2_H40_2x20", H40, "block_diagonal", block_size=20),
    "AD3_H40_5x8": CandidateGeometrySpec("AD3_H40_5x8", H40, "block_diagonal", block_size=8),
    "AD4_H40_10x4": CandidateGeometrySpec("AD4_H40_10x4", H40, "block_diagonal", block_size=4),
    **{
        f"AD4S_{sid}": CandidateGeometrySpec(f"AD4S_{sid}", H40, "degree4_global_sparse", sparse_topology_id=sid)
        for sid in SPARSE_TOPOLOGIES
    },
}


def dense_mask(hidden_size: int) -> torch.Tensor:
    return torch.ones(hidden_size, hidden_size, dtype=torch.float32)


def block_diagonal_mask(hidden_size: int, block_size: int) -> torch.Tensor:
    if hidden_size % block_size:
        raise ValueError("block size must divide hidden size")
    mask = torch.zeros(hidden_size, hidden_size, dtype=torch.float32)
    for start in range(0, hidden_size, block_size):
        mask[start : start + block_size, start : start + block_size] = 1.0
    return mask


def degree4_global_sparse_mask(topology_id: str) -> torch.Tensor:
    if topology_id not in SPARSE_TOPOLOGIES:
        raise ValueError(f"unknown sparse topology {topology_id}")
    spec = SPARSE_TOPOLOGIES[topology_id]
    offsets, shifts = spec["offsets"], spec["shifts"]
    mask = torch.zeros(H40, H40, dtype=torch.float32)
    for dest_block in range(NUM_LOGICAL_BLOCKS):
        for dest_coord in range(BLOCK_COORD):
            dest = dest_block * BLOCK_COORD + dest_coord
            for offset, shift in zip(offsets, shifts):
                src_block = (dest_block + offset) % NUM_LOGICAL_BLOCKS
                src_coord = (dest_coord + shift) % BLOCK_COORD
                src = src_block * BLOCK_COORD + src_coord
                mask[dest, src] = 1.0
    return mask


def candidate_mask(spec: CandidateGeometrySpec) -> torch.Tensor:
    spec.validate()
    if spec.candidate_mask_mode == "dense":
        return dense_mask(spec.hidden_size)
    if spec.candidate_mask_mode == "block_diagonal":
        assert spec.block_size is not None
        return block_diagonal_mask(spec.hidden_size, spec.block_size)
    assert spec.sparse_topology_id is not None
    return degree4_global_sparse_mask(spec.sparse_topology_id)


def _adjacency(mask: torch.Tensor) -> list[list[int]]:
    # Edge source -> destination because W[dest, source] is active.
    n = int(mask.shape[0])
    adj = [[] for _ in range(n)]
    nz = torch.nonzero(mask > 0, as_tuple=False)
    for dest, src in nz.tolist():
        adj[src].append(dest)
    return adj


def strongly_connected(mask: torch.Tensor) -> bool:
    adj = _adjacency(mask)
    rev = [[] for _ in adj]
    for src, dsts in enumerate(adj):
        for dst in dsts:
            rev[dst].append(src)

    def visit(graph: list[list[int]]) -> int:
        seen = {0}; q = deque([0])
        while q:
            node = q.popleft()
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt); q.append(nxt)
        return len(seen)

    return visit(adj) == len(adj) and visit(rev) == len(adj)


def weak_components(mask: torch.Tensor) -> int:
    adj = _adjacency(mask)
    und = [set(v) for v in adj]
    for src, dsts in enumerate(adj):
        for dst in dsts:
            und[dst].add(src)
    unseen = set(range(len(adj))); count = 0
    while unseen:
        count += 1; seed = next(iter(unseen)); q = [seed]; unseen.remove(seed)
        while q:
            node = q.pop()
            for nxt in und[node]:
                if nxt in unseen:
                    unseen.remove(nxt); q.append(nxt)
    return count


def directed_diameter(mask: torch.Tensor) -> int | None:
    if not strongly_connected(mask):
        return None
    adj = _adjacency(mask); diameter = 0
    for start in range(len(adj)):
        dist = [-1] * len(adj); dist[start] = 0; q = deque([start])
        while q:
            node = q.popleft()
            for nxt in adj[node]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[node] + 1; q.append(nxt)
        diameter = max(diameter, max(dist))
    return diameter


def mask_integrity(spec: CandidateGeometrySpec) -> dict:
    mask = candidate_mask(spec)
    active = int(mask.sum().item())
    fan_in = mask.sum(dim=1).to(torch.int64)
    fan_out = mask.sum(dim=0).to(torch.int64)
    same_block = 0
    if spec.hidden_size == H40:
        for dest, src in torch.nonzero(mask > 0, as_tuple=False).tolist():
            same_block += int(dest // BLOCK_COORD == src // BLOCK_COORD)
    return {
        "condition": spec.name,
        "mode": spec.candidate_mask_mode,
        "hidden_size": spec.hidden_size,
        "block_size": spec.block_size,
        "sparse_topology_id": spec.sparse_topology_id,
        "active_weights": active,
        "density": active / float(spec.hidden_size * spec.hidden_size),
        "fan_in": [int(v) for v in fan_in.tolist()],
        "fan_out": [int(v) for v in fan_out.tolist()],
        "same_logical_4d_block_edges": int(same_block),
        "weak_components": weak_components(mask),
        "strongly_connected": strongly_connected(mask),
        "directed_diameter": directed_diameter(mask),
        "mask_trainable": False,
    }


@dataclass
class CandidateGeometryTrace:
    states: torch.Tensor
    candidates: torch.Tensor
    updates: torch.Tensor
    resets: torch.Tensor
    raw_dynamic_updates: torch.Tensor
    raw_dynamic_resets: torch.Tensor
    candidate_input_terms: torch.Tensor
    candidate_hidden_terms: torch.Tensor
    update_input_terms: torch.Tensor
    update_hidden_terms: torch.Tensor


class CandidateRecurrentGeometryGRU(nn.Module):
    """Exact T2 semantics with only hidden size and W_hn candidate geometry varied."""

    architecture_name = "v837ad_candidate_recurrent_geometry"

    def __init__(self, *, spec: CandidateGeometrySpec):
        super().__init__(); spec.validate(); self.spec = spec
        self.hidden_size = int(spec.hidden_size); self.input_dim = INPUT_DIM
        source = DynamicGranularityGRU(self.hidden_size, self.input_dim, condition=T2_CONDITION)
        self.input_projection_weight = nn.Parameter(source.input_projection.weight.detach().clone())
        self.input_projection_bias = nn.Parameter(source.input_projection.bias.detach().clone())
        self.weight_ih = nn.Parameter(source.weight_ih.detach().clone())
        self.weight_hh = nn.Parameter(source.weight_hh.detach().clone())
        self.bias_ih = nn.Parameter(source.bias_ih.detach().clone())
        self.bias_hh = nn.Parameter(source.bias_hh.detach().clone())
        self.readout_weight = nn.Parameter(source.readout.weight.detach().clone())
        self.readout_bias = nn.Parameter(source.readout.bias.detach().clone())
        self.register_buffer("candidate_mask", candidate_mask(spec), persistent=True)

    @property
    def active_candidate_recurrent_weights(self) -> int:
        return int(self.candidate_mask.sum().item())

    @property
    def masked_candidate_recurrent_weights(self) -> int:
        return self.hidden_size * self.hidden_size - self.active_candidate_recurrent_weights

    @property
    def candidate_recurrent_macs(self) -> int:
        return self.active_candidate_recurrent_weights

    @property
    def total_active_macs_per_timestep(self) -> int:
        h = self.hidden_size
        return INPUT_DIM * INPUT_DIM + h * INPUT_DIM + self.candidate_recurrent_macs + h * INPUT_DIM + h * h

    def nominal_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_raw_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def active_parameter_count(self) -> int:
        h = self.hidden_size
        reset_slice = h * INPUT_DIM + h + h * h + h
        return self.nominal_parameter_count() - reset_slice - self.masked_candidate_recurrent_weights

    def parameter_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()) + self.candidate_mask.numel() * self.candidate_mask.element_size())

    def candidate_weight_raw(self) -> torch.Tensor:
        return self.weight_hh[2 * self.hidden_size : 3 * self.hidden_size]

    def candidate_weight_effective(self) -> torch.Tensor:
        return self.candidate_weight_raw() * self.candidate_mask

    def update_weight(self) -> torch.Tensor:
        return self.weight_hh[self.hidden_size : 2 * self.hidden_size]

    def candidate_input_weight(self) -> torch.Tensor:
        return self.weight_ih[2 * self.hidden_size : 3 * self.hidden_size]

    def update_input_weight(self) -> torch.Tensor:
        return self.weight_ih[self.hidden_size : 2 * self.hidden_size]

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, return_trace: bool = False):
        if observations.ndim != 3 or observations.shape[-1] != INPUT_DIM:
            raise ValueError("observations must be [B,T,6]")
        batch, steps, _ = observations.shape
        state = torch.zeros(batch, self.hidden_size, dtype=observations.dtype, device=observations.device)
        traces = {k: [] for k in ("states","candidates","updates","resets","raw_updates","raw_resets","candidate_inputs","candidate_hidden","update_inputs","update_hidden")}
        for t in range(steps):
            x_t = observations[:, t, :]
            projected = F.linear(x_t, self.input_projection_weight, self.input_projection_bias)
            gi = F.linear(projected, self.weight_ih, self.bias_ih)
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r = F.linear(state, self.weight_hh[: self.hidden_size], self.bias_hh[: self.hidden_size])
            h_z = F.linear(state, self.update_weight(), self.bias_hh[self.hidden_size : 2 * self.hidden_size])
            h_n = F.linear(state, self.candidate_weight_effective(), self.bias_hh[2 * self.hidden_size : 3 * self.hidden_size])
            raw_reset = torch.sigmoid(i_r + h_r)
            raw_update = torch.sigmoid(i_z + h_z)
            reset = torch.ones_like(raw_reset)
            update = scalarize_dynamic_gate(raw_update)
            candidate = torch.tanh(i_n + h_n)
            proposed = update * state + (1.0 - update) * candidate
            if lengths is None:
                state = proposed
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed + (1.0 - active) * state
            if return_trace:
                traces["states"].append(state); traces["candidates"].append(candidate); traces["updates"].append(update); traces["resets"].append(reset)
                traces["raw_updates"].append(raw_update); traces["raw_resets"].append(raw_reset); traces["candidate_inputs"].append(i_n)
                traces["candidate_hidden"].append(h_n); traces["update_inputs"].append(i_z); traces["update_hidden"].append(h_z)
        prediction = torch.tanh(F.linear(state, self.readout_weight, self.readout_bias)).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, CandidateGeometryTrace(
            states=torch.stack(traces["states"], dim=1), candidates=torch.stack(traces["candidates"], dim=1),
            updates=torch.stack(traces["updates"], dim=1), resets=torch.stack(traces["resets"], dim=1),
            raw_dynamic_updates=torch.stack(traces["raw_updates"], dim=1), raw_dynamic_resets=torch.stack(traces["raw_resets"], dim=1),
            candidate_input_terms=torch.stack(traces["candidate_inputs"], dim=1), candidate_hidden_terms=torch.stack(traces["candidate_hidden"], dim=1),
            update_input_terms=torch.stack(traces["update_inputs"], dim=1), update_hidden_terms=torch.stack(traces["update_hidden"], dim=1),
        )

    def geometry_diagnostics(self) -> dict:
        with torch.no_grad():
            w = self.candidate_weight_effective().detach().cpu(); singular = torch.linalg.svdvals(w)
            eig = torch.linalg.eigvals(w) if w.numel() else torch.empty(0, dtype=torch.complex64)
            rank = int(torch.linalg.matrix_rank(w).item())
            within = 0.0; cross = 0.0
            if self.hidden_size == H40:
                for dest in range(H40):
                    for src in range(H40):
                        e = float(w[dest, src].item()) ** 2
                        if dest // BLOCK_COORD == src // BLOCK_COORD: within += e
                        else: cross += e
            return {
                **mask_integrity(self.spec),
                "spectral_radius": float(torch.max(torch.abs(eig)).item()) if eig.numel() else 0.0,
                "spectral_norm": float(singular.max().item()) if singular.numel() else 0.0,
                "frobenius_norm": float(torch.linalg.vector_norm(w).item()),
                "singular_values": [float(v) for v in singular.tolist()],
                "numerical_rank": rank,
                "within_4d_block_energy": within,
                "cross_4d_block_energy": cross,
            }


def raw_initialization_signature(model: CandidateRecurrentGeometryGRU) -> dict[str, torch.Tensor]:
    return {
        "input_projection_weight": model.input_projection_weight.detach().clone(),
        "input_projection_bias": model.input_projection_bias.detach().clone(),
        "weight_ih": model.weight_ih.detach().clone(),
        "weight_hh": model.weight_hh.detach().clone(),
        "bias_ih": model.bias_ih.detach().clone(),
        "bias_hh": model.bias_hh.detach().clone(),
        "readout_weight": model.readout_weight.detach().clone(),
        "readout_bias": model.readout_bias.detach().clone(),
    }
