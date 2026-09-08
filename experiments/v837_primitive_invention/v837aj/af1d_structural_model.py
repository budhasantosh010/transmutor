from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.v837_primitive_invention.common.graph import CellSpec, EdgeSpec, GraphSpec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3
from experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency import (
    base_seed as historical_base_seed,
    coupling_seed as historical_coupling_seed,
    projection_seed as historical_projection_seed,
)
from experiments.v837_primitive_invention.v837aj.topology import (
    NUM_CELLS,
    SearchTopology,
    historical_anchor_topology,
    semantic_edge_initial_weight,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONDITION = "AF1D_deshared_candidate_input_factorization"


def _candidate_seed(namespace: str, family: str, run_index: int, slot: int, part: str) -> int:
    return deterministic_int(namespace, family, int(run_index), int(slot), part)


def candidate_common_seed(family: str, run_index: int, slot: int) -> int:
    return _candidate_seed("v837aj-candidate-common-init", family, run_index, slot, "torch")


def candidate_coupling_seed(family: str, run_index: int, slot: int) -> int:
    return _candidate_seed("v837aj-candidate-common-init", family, run_index, slot, "coupling")


def candidate_projection_seed(family: str, run_index: int, slot: int) -> int:
    return _candidate_seed("v837aj-candidate-common-init", family, run_index, slot, "projection")


def candidate_cell_seed(family: str, run_index: int, slot: int, cell_index: int) -> int:
    return deterministic_int("v837aj-candidate-cell-init", family, int(run_index), int(slot), int(cell_index)) % 2_000_000_000


def finalization_common_seed(family: str, run_index: int) -> int:
    return deterministic_int("v837aj-finalize", family, int(run_index), "torch")


def finalization_coupling_seed(family: str, run_index: int) -> int:
    return deterministic_int("v837aj-finalize", family, int(run_index), "coupling")


def finalization_projection_seed(family: str, run_index: int) -> int:
    return deterministic_int("v837aj-finalize", family, int(run_index), "projection")


def finalization_cell_seed(family: str, run_index: int, cell_index: int) -> int:
    return deterministic_int("v837aj-finalize", family, int(run_index), "cell", int(cell_index)) % 2_000_000_000


def _graph_for_topology(
    topology: SearchTopology,
    *,
    family: str,
    run_index: int,
    slot: int | None,
    finalization: bool,
) -> GraphSpec:
    cells = []
    for cell_index in range(NUM_CELLS):
        seed = (
            finalization_cell_seed(family, run_index, cell_index)
            if finalization
            else candidate_cell_seed(family, run_index, int(slot), cell_index)
        )
        cells.append(CellSpec(cell_index, param_seed=int(seed)))
    edges = [
        EdgeSpec(
            edge.src,
            edge.dst,
            weight=float(semantic_edge_initial_weight(family, run_index, edge)),
            recurrent=edge.recurrent,
        )
        for edge in topology.edges
    ]
    graph = GraphSpec(cells=cells, edges=edges, generation=0, parent_id="V837AJ_SEARCH_TOPOLOGY")
    graph.validate(max_cells=NUM_CELLS, max_edges=int(CONFIG["max_message_edges"]))
    return graph


def build_candidate_model(topology: SearchTopology, family: str, run_index: int, slot: int) -> CandidateInputFactorizationY3:
    common = candidate_common_seed(family, run_index, slot)
    torch.manual_seed(int(common))
    np.random.seed(int(common) % (2**32 - 1))
    graph = _graph_for_topology(topology, family=family, run_index=run_index, slot=slot, finalization=False)
    return CandidateInputFactorizationY3(
        graph,
        condition=CONDITION,
        coupling_initialization_seed=int(candidate_coupling_seed(family, run_index, slot)),
        projection_seed=int(candidate_projection_seed(family, run_index, slot)),
    )


def build_finalization_model(topology: SearchTopology, family: str, run_index: int) -> CandidateInputFactorizationY3:
    common = finalization_common_seed(family, run_index)
    torch.manual_seed(int(common))
    np.random.seed(int(common) % (2**32 - 1))
    graph = _graph_for_topology(topology, family=family, run_index=run_index, slot=None, finalization=True)
    return CandidateInputFactorizationY3(
        graph,
        condition=CONDITION,
        coupling_initialization_seed=int(finalization_coupling_seed(family, run_index)),
        projection_seed=int(finalization_projection_seed(family, run_index)),
    )


def build_exact_anchor_model(family: str, replicate: int) -> CandidateInputFactorizationY3:
    """Exact historical AF1D initialization, including historical edge weights."""
    from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
    seed = historical_base_seed(family, replicate)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    return CandidateInputFactorizationY3(
        high_capacity_generic_graph(int(replicate)),
        condition=CONDITION,
        coupling_initialization_seed=int(historical_coupling_seed(replicate)),
        projection_seed=int(historical_projection_seed(family, replicate)),
    )


def build_exact_anchor_wrapper(family: str, replicate: int) -> CandidateInputFactorizationY3:
    """Historical anchor materialized through the V837aj SearchTopology wrapper."""
    from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
    source = high_capacity_generic_graph(int(replicate))
    topology = SearchTopology.from_graph(source)
    seed = historical_base_seed(family, replicate)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    wrapped_graph = topology.to_graph(int(replicate), semantic_edge_initialization=False)
    return CandidateInputFactorizationY3(
        wrapped_graph,
        condition=CONDITION,
        coupling_initialization_seed=int(historical_coupling_seed(replicate)),
        projection_seed=int(historical_projection_seed(family, replicate)),
    )


def non_edge_parameter_tensors(model: CandidateInputFactorizationY3) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("base.edge_weights.")
    }


def semantic_edge_tensor_map(model: CandidateInputFactorizationY3) -> dict[tuple[int, int, bool], torch.Tensor]:
    mapping: dict[tuple[int, int, bool], torch.Tensor] = {}
    for edge, parameter in zip(model.graph.edges, model.base.edge_weights):
        mapping[(int(edge.src), int(edge.dst), bool(edge.recurrent))] = parameter.detach().cpu().clone()
    return mapping


def initialization_fingerprint(model: CandidateInputFactorizationY3, *, include_edges: bool = True) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not include_edges and name.startswith("base.edge_weights."):
            continue
        arr = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(arr.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(str(tuple(arr.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def common_initialization_equal(a: CandidateInputFactorizationY3, b: CandidateInputFactorizationY3) -> bool:
    pa, pb = non_edge_parameter_tensors(a), non_edge_parameter_tensors(b)
    if pa.keys() != pb.keys():
        return False
    if not all(torch.equal(pa[name], pb[name]) for name in pa):
        return False
    ea, eb = semantic_edge_tensor_map(a), semantic_edge_tensor_map(b)
    for key in ea.keys() & eb.keys():
        if not torch.equal(ea[key], eb[key]):
            return False
    return True


def model_compute(topology: SearchTopology) -> dict:
    edge_params = int(topology.edge_count)
    active_params = int(CONFIG["active_parameters_without_message_edges"]) + edge_params
    recurrent_controller_projection = int(CONFIG["frozen_af1d_recurrent_controller_projection_macs"])
    edge_macs = int(CONFIG["message_edge_mac_per_timestep"]) * edge_params
    return {
        "active_parameters": active_params,
        "edge_parameters": edge_params,
        "projection_parameters": 420,
        "recurrent_controller_projection_macs": recurrent_controller_projection,
        "message_edge_macs": edge_macs,
        "total_modeled_macs_per_timestep": recurrent_controller_projection + edge_macs,
    }


def architecture_lock() -> dict:
    model = build_exact_anchor_wrapper("conditional_routing", 0)
    topology = historical_anchor_topology()
    checks = {
        "exact_model_class": type(model) is CandidateInputFactorizationY3,
        "condition": model.transfer_condition == CONDITION,
        "ten_cells": len(model.graph.cells) == 10,
        "rank4_coupling": model.interaction_spec.candidate_coupling_mode == "rank4_cross_block" and model.interaction_spec.coupling_rank == 4,
        "global_controller": model.global_scalar_control and model.controller_param_count == 47,
        "ten_deshared_projections": model.cell_candidate_projections is not None and len(model.cell_candidate_projections) == 10,
        "readout_shape": tuple(model.base.readout.weight.shape) == (1, 40),
        "anchor_edges": topology.edge_count == 55,
        "anchor_parameters": model.parameter_count() == 1643,
        "recurrent_controller_projection_macs": model.total_recurrent_controller_projection_macs == 1206,
    }
    return {"compatible": all(checks.values()), "checks": checks, "model_class": type(model).__name__}
