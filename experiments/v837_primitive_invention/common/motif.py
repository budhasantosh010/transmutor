from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .graph import CellSpec, EdgeSpec, GraphSpec, connected_subsets
from .guards import assert_primitive_mining_allowed
from .metrics import permutation_rate_difference
from .substrate import NeutralGraphModel
from .trainer import episodes_to_batch


@dataclass
class MotifOccurrence:
    motif_hash: str
    graph_id: str
    nodes: tuple[int, ...]
    size: int
    signature: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "motif_hash": self.motif_hash,
            "graph_id": self.graph_id,
            "nodes": list(self.nodes),
            "size": self.size,
            "signature": self.signature,
        }


def _bin(value: float, edges=(-0.5, -0.1, 0.1, 0.5)) -> int:
    return int(np.digitize([value], edges)[0])


def cell_dynamic_bins(model: NeutralGraphModel, episodes) -> dict[int, tuple[int, int, int]]:
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    states = trace.states.detach().cpu().numpy()
    bins: dict[int, tuple[int, int, int]] = {}
    for cell in range(states.shape[2]):
        values = states[:, :, cell, :].reshape(-1)
        mean = float(np.mean(values))
        variance = float(np.var(values))
        if states.shape[1] > 1:
            a = states[:, :-1, cell, :].reshape(-1)
            b = states[:, 1:, cell, :].reshape(-1)
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
        else:
            corr = 0.0
        bins[cell] = (_bin(mean), _bin(variance, (0.01, 0.05, 0.15, 0.4)), _bin(corr, (-0.5, -0.1, 0.1, 0.5)))
    return bins


def canonical_motif_signature(model: NeutralGraphModel, nodes: tuple[int, ...], dynamic_bins: dict[int, tuple[int, int, int]]) -> dict[str, Any]:
    nodes = tuple(sorted(nodes))
    k = len(nodes)
    internal = []
    node_set = set(nodes)
    for edge_index, edge in enumerate(model.graph.edges):
        if edge.src in node_set and edge.dst in node_set:
            weight = float(model.edge_weights[edge_index].detach().cpu().item())
            internal.append((edge.src, edge.dst, bool(edge.recurrent), _bin(weight)))
    best: str | None = None
    best_payload: dict[str, Any] | None = None
    for permutation in itertools.permutations(nodes):
        mapping = {old: new for new, old in enumerate(permutation)}
        payload = {
            "size": k,
            "cells": [dynamic_bins[old] for old in permutation],
            "edges": sorted((mapping[src], mapping[dst], recurrent, weight_bin) for src, dst, recurrent, weight_bin in internal),
        }
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if best is None or text < best:
            best = text
            best_payload = payload
    assert best_payload is not None
    return best_payload


def motif_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def extract_motifs(model: NeutralGraphModel, episodes, *, min_cells: int = 2, max_cells: int = 6, limit: int = 256) -> list[MotifOccurrence]:
    dynamics = cell_dynamic_bins(model, episodes)
    occurrences: list[MotifOccurrence] = []
    for nodes in connected_subsets(model.graph, min_size=min_cells, max_size=max_cells, limit=limit):
        signature = canonical_motif_signature(model, nodes, dynamics)
        occurrences.append(MotifOccurrence(motif_hash(signature), model.graph.graph_id, nodes, len(nodes), signature))
    return occurrences


def recurrence_candidates(success_occurrences: list[list[MotifOccurrence]], random_occurrences: list[list[MotifOccurrence]], *, min_fraction: float = 0.20, min_organisms: int = 5, seed: int = 837) -> list[dict]:
    success_sets = [set(item.motif_hash for item in occurrences) for occurrences in success_occurrences]
    random_sets = [set(item.motif_hash for item in occurrences) for occurrences in random_occurrences]
    all_hashes = sorted(set().union(*success_sets)) if success_sets else []
    output = []
    for item_hash in all_hashes:
        success_hits = np.asarray([item_hash in group for group in success_sets], dtype=float)
        random_hits = np.asarray([item_hash in group for group in random_sets], dtype=float)
        count = int(success_hits.sum())
        fraction = float(success_hits.mean()) if len(success_hits) else 0.0
        if count < min_organisms or fraction < min_fraction:
            continue
        permutation = permutation_rate_difference(success_hits, random_hits, permutations=999, seed=seed)
        example = next(item for group in success_occurrences for item in group if item.motif_hash == item_hash)
        output.append(
            {
                "motif_hash": item_hash,
                "size": example.size,
                "success_count": count,
                "success_fraction": fraction,
                "random_count": int(random_hits.sum()),
                "random_fraction": float(random_hits.mean()) if len(random_hits) else 0.0,
                "enrichment": permutation,
                "signature": example.signature,
            }
        )
    return sorted(output, key=lambda row: (-row["success_fraction"], row["enrichment"]["p_value"], row["motif_hash"]))


class CallablePrimitive(nn.Module):
    """Callable wrapper around an extracted neutral-cell subgraph; it carries no semantic task label."""

    def __init__(self, primitive_id: str, graph: GraphSpec, state_dim: int, message_dim: int, obs_dim: int, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.primitive_id = primitive_id
        self.model = NeutralGraphModel(graph, obs_dim=obs_dim, state_dim=state_dim, message_dim=message_dim)
        self.model.load_state_dict(state_dict)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        _, trace = self.model(observations, lengths, return_trace=True)
        return trace.states[:, -1].reshape(observations.shape[0], -1)


def extract_isolated_subgraph(model: NeutralGraphModel, nodes: tuple[int, ...]) -> tuple[GraphSpec, dict[str, torch.Tensor]]:
    nodes = tuple(sorted(nodes))
    mapping = {old: new for new, old in enumerate(nodes)}
    cells = [CellSpec(mapping[old], param_seed=model.graph.cells[old].param_seed, birth_generation=0) for old in nodes]
    internal_edges = []
    source_edge_indices = []
    for edge_index, edge in enumerate(model.graph.edges):
        if edge.src in mapping and edge.dst in mapping:
            internal_edges.append(EdgeSpec(mapping[edge.src], mapping[edge.dst], weight=float(model.edge_weights[edge_index].detach().item()), recurrent=edge.recurrent))
            source_edge_indices.append(edge_index)
    graph = GraphSpec(cells=cells, edges=internal_edges, parent_id=model.graph.graph_id, generation=0)
    submodel = NeutralGraphModel(graph, obs_dim=model.obs_dim, state_dim=model.state_dim, message_dim=model.message_dim)
    with torch.no_grad():
        for new, old in enumerate(nodes):
            submodel.cell_ws[new].copy_(model.cell_ws[old])
            submodel.cell_wm[new].copy_(model.cell_wm[old])
            submodel.cell_wx[new].copy_(model.cell_wx[old])
            submodel.cell_b[new].copy_(model.cell_b[old])
            submodel.cell_wo[new].copy_(model.cell_wo[old])
        for new_index, old_index in enumerate(source_edge_indices):
            submodel.edge_weights[new_index].copy_(model.edge_weights[old_index])
        submodel.readout.weight.zero_(); submodel.readout.bias.zero_()
    return graph, {key: value.detach().cpu().clone() for key, value in submodel.state_dict().items()}


def randomized_isolated_subgraph(
    model: NeutralGraphModel,
    nodes: tuple[int, ...],
    *,
    seed: int,
) -> tuple[GraphSpec, dict[str, torch.Tensor]]:
    """Size/topology-matched random replacement for causal/random-macro controls.

    The extracted topology and edge count are preserved while all floating
    parameters are randomized deterministically. No task semantics enter the
    control construction.
    """
    graph, state = extract_isolated_subgraph(model, nodes)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    randomized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if value.is_floating_point():
            randomized[key] = torch.randn(value.shape, generator=generator, dtype=value.dtype) * 0.2
        else:
            randomized[key] = value.clone()
    return graph, randomized


def primitive_equivalence(primitive: CallablePrimitive, expanded: CallablePrimitive, observations: torch.Tensor, lengths: torch.Tensor, tolerance: float = 1e-6) -> dict:
    primitive.eval(); expanded.eval()
    with torch.no_grad():
        a = primitive(observations, lengths)
        b = expanded(observations, lengths)
    max_error = float(torch.max(torch.abs(a - b)).item())
    return {"max_absolute_error": max_error, "tolerance": tolerance, "pass": max_error <= tolerance}


def begin_scientific_motif_pipeline() -> None:
    """Guard the scientific motif pipeline while representation recovery is open.

    Low-level motif utilities remain testable, but an experiment must call this
    gate before treating motif extraction/promotion as scientific evidence.
    """
    assert_primitive_mining_allowed()
