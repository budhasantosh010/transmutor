from __future__ import annotations

from dataclasses import replace

from .graph import CellSpec, EdgeSpec, GraphSpec, renormalize
from .seeds import rng

ALLOWED_MUTATIONS = (
    "ADD_CELL",
    "REMOVE_CELL",
    "ADD_EDGE",
    "REMOVE_EDGE",
    "PERTURB_EDGE_WEIGHT",
    "PERTURB_CELL_PARAMETERS",
    "ADD_RECURRENT_EDGE",
    "REMOVE_RECURRENT_EDGE",
    "DUPLICATE_SUBGRAPH",
)


def mutate(graph: GraphSpec, seed: int, *, max_cells: int = 16, max_edges: int = 64, forced_op: str | None = None) -> tuple[GraphSpec, str]:
    random = rng("mutation", graph.graph_id, seed)
    ops = list(ALLOWED_MUTATIONS)
    if forced_op is not None:
        if forced_op not in ALLOWED_MUTATIONS:
            raise ValueError(forced_op)
        ops = [forced_op]
    for _ in range(20):
        op = random.choice(ops)
        candidate = graph.clone()
        candidate.parent_id = graph.graph_id
        candidate.generation = graph.generation + 1
        changed = _apply(candidate, op, random, max_cells=max_cells, max_edges=max_edges)
        if changed:
            candidate = renormalize(candidate)
            candidate.mutation_history.append({"generation": candidate.generation, "operation": op, "seed": seed})
            candidate.validate(max_cells=max_cells, max_edges=max_edges)
            return candidate, op
    return graph.clone(), "NO_OP"


def _apply(graph: GraphSpec, op: str, random, *, max_cells: int, max_edges: int) -> bool:
    n = len(graph.cells)
    if op == "ADD_CELL":
        if n >= max_cells:
            return False
        new_id = n
        graph.cells.append(CellSpec(new_id, param_seed=random.randrange(1, 2**20), birth_generation=graph.generation + 1))
        graph.birth_history.append({"generation": graph.generation + 1, "cells": [new_id]})
        return True
    if op == "REMOVE_CELL":
        if n <= 2:
            return False
        victim = random.randrange(n)
        graph.cells = [cell for cell in graph.cells if cell.id != victim]
        graph.edges = [edge for edge in graph.edges if edge.src != victim and edge.dst != victim]
        return True
    if op in {"ADD_EDGE", "ADD_RECURRENT_EDGE"}:
        if len(graph.edges) >= max_edges:
            return False
        recurrent = op == "ADD_RECURRENT_EDGE"
        existing = {(edge.src, edge.dst, edge.recurrent) for edge in graph.edges}
        possibilities = [(a, b) for a in range(n) for b in range(n) if (a, b, recurrent) not in existing]
        if not possibilities:
            return False
        src, dst = random.choice(possibilities)
        graph.edges.append(EdgeSpec(src, dst, weight=random.uniform(-1.0, 1.0), recurrent=recurrent))
        return True
    if op == "REMOVE_EDGE":
        if not graph.edges:
            return False
        graph.edges.pop(random.randrange(len(graph.edges)))
        return True
    if op == "REMOVE_RECURRENT_EDGE":
        recurrent_indices = [i for i, edge in enumerate(graph.edges) if edge.recurrent]
        if not recurrent_indices:
            return False
        graph.edges.pop(random.choice(recurrent_indices))
        return True
    if op == "PERTURB_EDGE_WEIGHT":
        if not graph.edges:
            return False
        index = random.randrange(len(graph.edges))
        edge = graph.edges[index]
        graph.edges[index] = replace(edge, weight=max(-2.0, min(2.0, edge.weight + random.gauss(0.0, 0.25))))
        return True
    if op == "PERTURB_CELL_PARAMETERS":
        if not graph.cells:
            return False
        index = random.randrange(len(graph.cells))
        cell = graph.cells[index]
        graph.cells[index] = replace(cell, param_seed=random.randrange(1, 2**20))
        return True
    if op == "DUPLICATE_SUBGRAPH":
        if n >= max_cells:
            return False
        anchors = sorted({edge.src for edge in graph.edges} | {edge.dst for edge in graph.edges})
        if not anchors:
            anchors = list(range(n))
        source = random.choice(anchors)
        neighborhood = {source}
        for edge in graph.edges:
            if edge.src == source:
                neighborhood.add(edge.dst)
            if edge.dst == source:
                neighborhood.add(edge.src)
        selected = sorted(neighborhood)[: min(3, max_cells - n)]
        mapping: dict[int, int] = {}
        for old in selected:
            new_id = len(graph.cells)
            mapping[old] = new_id
            graph.cells.append(CellSpec(new_id, param_seed=graph.cells[old].param_seed, birth_generation=graph.generation + 1))
        internal = [edge for edge in graph.edges if edge.src in mapping and edge.dst in mapping]
        for edge in internal:
            if len(graph.edges) >= max_edges:
                break
            graph.edges.append(EdgeSpec(mapping[edge.src], mapping[edge.dst], weight=edge.weight, recurrent=edge.recurrent))
        graph.birth_history.append({"generation": graph.generation + 1, "cells": list(mapping.values()), "duplicated_from": selected})
        return bool(mapping)
    raise ValueError(op)
