from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from experiments.v837_primitive_invention.common.graph import GraphSpec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel


LOCAL_VIEW_DIM = 4
TOTAL_STATE_DIM = 40


@dataclass(frozen=True)
class StateLayoutSpec:
    mode: str
    num_cells: int
    total_state_dim: int
    num_state_groups: int
    group_assignment: tuple[int, ...]
    group_dims: tuple[int, ...]
    projection_seed: int

    def validate(self) -> None:
        if self.mode not in {"cell_local", "group_shared", "fully_shared"}:
            raise ValueError(f"unsupported state-layout mode: {self.mode}")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if self.total_state_dim != TOTAL_STATE_DIM:
            raise ValueError(f"V837q requires total_state_dim={TOTAL_STATE_DIM}")
        if len(self.group_assignment) != self.num_cells:
            raise ValueError("group_assignment length must equal num_cells")
        if len(self.group_dims) != self.num_state_groups:
            raise ValueError("group_dims length must equal num_state_groups")
        if sum(self.group_dims) != self.total_state_dim:
            raise ValueError("group dimensions must sum to total_state_dim")
        if set(self.group_assignment) != set(range(self.num_state_groups)):
            raise ValueError("every state group must own at least one cell")
        for group_index, group_dim in enumerate(self.group_dims):
            members = sum(1 for value in self.group_assignment if value == group_index)
            if group_dim != members * LOCAL_VIEW_DIM:
                raise ValueError("V837q primary layouts require group_dim == member_count * local_view_dim")
        if self.mode == "cell_local" and self.num_state_groups != self.num_cells:
            raise ValueError("cell_local layout requires one state group per cell")
        if self.mode == "fully_shared" and self.num_state_groups != 1:
            raise ValueError("fully_shared layout requires one state group")

    def members(self, group_index: int) -> tuple[int, ...]:
        return tuple(i for i, group in enumerate(self.group_assignment) if group == group_index)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "num_cells": self.num_cells,
            "total_state_dim": self.total_state_dim,
            "num_state_groups": self.num_state_groups,
            "group_assignment": list(self.group_assignment),
            "group_dims": list(self.group_dims),
            "projection_seed": self.projection_seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StateLayoutSpec":
        spec = cls(
            mode=str(payload["mode"]),
            num_cells=int(payload["num_cells"]),
            total_state_dim=int(payload["total_state_dim"]),
            num_state_groups=int(payload["num_state_groups"]),
            group_assignment=tuple(int(v) for v in payload["group_assignment"]),
            group_dims=tuple(int(v) for v in payload["group_dims"]),
            projection_seed=int(payload["projection_seed"]),
        )
        spec.validate()
        return spec


def standard_state_layout(name: str, *, projection_seed: int) -> StateLayoutSpec:
    if name == "Q0_local_10x4":
        spec = StateLayoutSpec(
            mode="cell_local",
            num_cells=10,
            total_state_dim=40,
            num_state_groups=10,
            group_assignment=tuple(range(10)),
            group_dims=(4,) * 10,
            projection_seed=int(projection_seed),
        )
    elif name == "Q1_group5_5x8":
        spec = StateLayoutSpec(
            mode="group_shared",
            num_cells=10,
            total_state_dim=40,
            num_state_groups=5,
            group_assignment=(0, 0, 1, 1, 2, 2, 3, 3, 4, 4),
            group_dims=(8,) * 5,
            projection_seed=int(projection_seed),
        )
    elif name == "Q2_group2_2x20":
        spec = StateLayoutSpec(
            mode="group_shared",
            num_cells=10,
            total_state_dim=40,
            num_state_groups=2,
            group_assignment=(0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
            group_dims=(20, 20),
            projection_seed=int(projection_seed),
        )
    elif name == "Q3_shared_1x40":
        spec = StateLayoutSpec(
            mode="fully_shared",
            num_cells=10,
            total_state_dim=40,
            num_state_groups=1,
            group_assignment=(0,) * 10,
            group_dims=(40,),
            projection_seed=int(projection_seed),
        )
    else:
        raise ValueError(f"unknown V837q state-layout condition: {name}")
    spec.validate()
    return spec


def build_fixed_state_projection(group_dim: int, local_dim: int, cell_id: int, seed: int) -> torch.Tensor:
    """Deterministic task-independent orthonormal-row projection [local_dim, group_dim]."""
    group_dim = int(group_dim)
    local_dim = int(local_dim)
    if group_dim < local_dim:
        raise ValueError("group_dim must be >= local_dim")
    generator_seed = deterministic_int("v837q-fixed-state-projection", int(seed), group_dim, local_dim, int(cell_id))
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    raw = torch.randn(group_dim, local_dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    projection = q.T.contiguous()
    return projection


def projection_norm_error(projection: torch.Tensor) -> float:
    identity = torch.eye(projection.shape[0], dtype=projection.dtype, device=projection.device)
    return float(torch.max(torch.abs(projection @ projection.T - identity)).item())


def group_write_normalization(group_dim: int, member_count: int, local_dim: int = LOCAL_VIEW_DIM) -> float:
    """Frame-energy normalization predeclared before V837q runs.

    The primary layouts satisfy group_dim == member_count*local_dim.  The
    resulting factor is exactly one, which preserves expected per-coordinate
    energy for independent orthonormal-row projections without adding a
    learned or family-specific scale.
    """
    return float(math.sqrt(float(group_dim) / float(member_count * local_dim)))


@dataclass
class SharedStateTrace:
    states: torch.Tensor  # [B,T,40]
    candidates: torch.Tensor  # [B,T,40] group-state proposals before length masking
    local_views: torch.Tensor  # [B,T,N,4] views read from the snapshotted previous group state
    cell_candidates: torch.Tensor  # [B,T,N,4]
    outputs: torch.Tensor  # [B,T,N,4] committed-state message projections
    messages: torch.Tensor  # [B,T,N,4]


class SharedStateNeutralGraphModel(nn.Module):
    """Historical neutral cell/edge computation with explicit recurrent-state ownership.

    Q0 delegates exactly to NeutralGraphModel. Q1-Q3 reuse the same trainable
    historical cell, edge, and 40-wide readout parameters. The only new
    objects are fixed non-trainable state read/write projections.
    """

    def __init__(
        self,
        graph: GraphSpec,
        state_layout: StateLayoutSpec,
        *,
        obs_dim: int = 6,
        local_view_dim: int = LOCAL_VIEW_DIM,
        message_dim: int = 4,
    ):
        super().__init__()
        graph.validate()
        state_layout.validate()
        if len(graph.cells) != state_layout.num_cells:
            raise ValueError("graph cell count does not match state layout")
        if local_view_dim != LOCAL_VIEW_DIM or message_dim != LOCAL_VIEW_DIM:
            raise ValueError("V837q preserves historical 4D cell/message transforms")
        self.graph = graph.clone()
        self.state_layout = state_layout
        self.obs_dim = int(obs_dim)
        self.local_view_dim = int(local_view_dim)
        self.message_dim = int(message_dim)
        self.total_state_dim = int(state_layout.total_state_dim)
        self.base = NeutralGraphModel(
            self.graph,
            obs_dim=self.obs_dim,
            state_dim=self.local_view_dim,
            message_dim=self.message_dim,
            state_update_mode="direct",
            interaction_mode="none",
            state_modulation_mode="none",
        )
        self._projection_names: list[str] = []
        for cell_index in range(state_layout.num_cells):
            group_index = state_layout.group_assignment[cell_index]
            group_dim = state_layout.group_dims[group_index]
            if state_layout.mode == "cell_local":
                projection = torch.eye(self.local_view_dim, dtype=torch.float32)
            else:
                projection = build_fixed_state_projection(
                    group_dim,
                    self.local_view_dim,
                    cell_index,
                    state_layout.projection_seed,
                )
            name = f"state_projection_{cell_index}"
            self.register_buffer(name, projection, persistent=True)
            self._projection_names.append(name)

    @property
    def input_edge_count(self) -> int:
        return self.base.input_edge_count

    @property
    def internal_message_edge_count(self) -> int:
        return self.base.internal_message_edge_count

    @property
    def readout_input_width(self) -> int:
        return int(self.base.readout.in_features)

    @property
    def non_trainable_projection_elements(self) -> int:
        return int(sum(self.projection(i).numel() for i in range(self.state_layout.num_cells)))

    def projection(self, cell_index: int) -> torch.Tensor:
        return getattr(self, self._projection_names[int(cell_index)])

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def parameter_bytes(self) -> int:
        return int(sum(parameter.numel() * parameter.element_size() for parameter in self.parameters()))

    def serialization_metadata(self) -> dict:
        return {
            "state_layout": self.state_layout.to_dict(),
            "local_view_dim": self.local_view_dim,
            "message_dim": self.message_dim,
            "readout_input_width": self.readout_input_width,
            "non_trainable_projection_elements": self.non_trainable_projection_elements,
        }

    def group_write_normalizations(self) -> tuple[float, ...]:
        return tuple(
            group_write_normalization(
                self.state_layout.group_dims[group_index],
                len(self.state_layout.members(group_index)),
                self.local_view_dim,
            )
            for group_index in range(self.state_layout.num_state_groups)
        )

    def _visible_input(self, x_t: torch.Tensor, cell_index: int) -> torch.Tensor:
        if self.base.input_access_mode == "none":
            return torch.zeros_like(x_t)
        if self.base.input_access_mode == "broadcast":
            return x_t
        return x_t * self.base.input_access_mask[:, cell_index].view(1, -1)

    def _forward_shared(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None,
        *,
        disable_messages: bool,
        disabled_contribution_cells: set[int] | None,
        return_trace: bool,
    ):
        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.obs_dim:
            raise ValueError(f"observation dimension {observed_dim} != model obs_dim {self.obs_dim}")
        device = observations.device
        n = len(self.graph.cells)
        disabled_contributions = disabled_contribution_cells or set()
        prev_group_states = [
            torch.zeros(batch, dim, dtype=observations.dtype, device=device)
            for dim in self.state_layout.group_dims
        ]
        prev_outputs = [torch.zeros(batch, self.message_dim, dtype=observations.dtype, device=device) for _ in range(n)]
        state_trace: list[torch.Tensor] = []
        candidate_group_trace: list[torch.Tensor] = []
        local_view_trace: list[torch.Tensor] = []
        cell_candidate_trace: list[torch.Tensor] = []
        output_trace: list[torch.Tensor] = []
        message_trace: list[torch.Tensor] = []

        for t in range(steps):
            x_t = observations[:, t, :]
            snapshot = prev_group_states
            local_views = [
                snapshot[self.state_layout.group_assignment[cell_index]] @ self.projection(cell_index).T
                for cell_index in range(n)
            ]
            current_candidate_outputs: list[torch.Tensor] = []
            current_candidates: list[torch.Tensor] = []
            current_messages: list[torch.Tensor] = []

            for cell_index in range(n):
                message = torch.zeros(batch, self.message_dim, dtype=observations.dtype, device=device)
                if not disable_messages:
                    for edge_index, edge in enumerate(self.graph.edges):
                        if edge.dst != cell_index:
                            continue
                        if edge.recurrent or edge.src >= len(current_candidate_outputs):
                            source = prev_outputs[edge.src]
                        else:
                            source = current_candidate_outputs[edge.src]
                        message = message + self.base.edge_weights[edge_index] * source
                visible_x = self._visible_input(x_t, cell_index)
                candidate = torch.tanh(
                    local_views[cell_index] @ self.base.cell_ws[cell_index].T
                    + message @ self.base.cell_wm[cell_index].T
                    + visible_x @ self.base.cell_wx[cell_index].T
                    + self.base.cell_b[cell_index]
                )
                candidate_output = candidate @ self.base.cell_wo[cell_index].T
                if lengths is not None:
                    active = (t < lengths).to(observations.dtype).unsqueeze(1)
                    candidate_output = active * candidate_output + (1.0 - active) * prev_outputs[cell_index]
                current_candidates.append(candidate)
                current_candidate_outputs.append(candidate_output)
                current_messages.append(message)

            proposed_groups: list[torch.Tensor] = []
            for group_index in range(self.state_layout.num_state_groups):
                group_dim = self.state_layout.group_dims[group_index]
                proposed = torch.zeros(batch, group_dim, dtype=observations.dtype, device=device)
                members = self.state_layout.members(group_index)
                for cell_index in members:
                    if cell_index in disabled_contributions:
                        continue
                    proposed = proposed + current_candidates[cell_index] @ self.projection(cell_index)
                proposed = proposed * group_write_normalization(group_dim, len(members), self.local_view_dim)
                proposed_groups.append(proposed)

            if lengths is None:
                current_groups = proposed_groups
            else:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                current_groups = [
                    active * proposed + (1.0 - active) * previous
                    for proposed, previous in zip(proposed_groups, snapshot)
                ]

            committed_views = [
                current_groups[self.state_layout.group_assignment[cell_index]] @ self.projection(cell_index).T
                for cell_index in range(n)
            ]
            committed_outputs = [
                committed_views[cell_index] @ self.base.cell_wo[cell_index].T
                for cell_index in range(n)
            ]
            prev_group_states = current_groups
            prev_outputs = committed_outputs

            if return_trace:
                state_trace.append(torch.cat(current_groups, dim=1))
                candidate_group_trace.append(torch.cat(proposed_groups, dim=1))
                local_view_trace.append(torch.stack(local_views, dim=1))
                cell_candidate_trace.append(torch.stack(current_candidates, dim=1))
                output_trace.append(torch.stack(committed_outputs, dim=1))
                message_trace.append(torch.stack(current_messages, dim=1))

        final_state = torch.cat(prev_group_states, dim=1)
        prediction = torch.tanh(self.base.readout(final_state)).squeeze(-1)
        if return_trace:
            return prediction, SharedStateTrace(
                states=torch.stack(state_trace, dim=1),
                candidates=torch.stack(candidate_group_trace, dim=1),
                local_views=torch.stack(local_view_trace, dim=1),
                cell_candidates=torch.stack(cell_candidate_trace, dim=1),
                outputs=torch.stack(output_trace, dim=1),
                messages=torch.stack(message_trace, dim=1),
            )
        return prediction

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        disable_messages: bool = False,
        disabled_contribution_cells: set[int] | None = None,
        return_trace: bool = False,
    ):
        if self.state_layout.mode == "cell_local" and not disabled_contribution_cells:
            return self.base(
                observations,
                lengths,
                disable_messages=disable_messages,
                return_trace=return_trace,
            )
        if self.state_layout.mode == "cell_local" and disabled_contribution_cells:
            # Diagnostic-only proxy for the local baseline: a local state is its
            # cell's contribution, so disabling that contribution disables the
            # corresponding historical cell. Primary training never uses this.
            return self.base(
                observations,
                lengths,
                disable_messages=disable_messages,
                disabled_cells=set(disabled_contribution_cells),
                return_trace=return_trace,
            )
        return self._forward_shared(
            observations,
            lengths,
            disable_messages=disable_messages,
            disabled_contribution_cells=disabled_contribution_cells,
            return_trace=return_trace,
        )
