from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CellSpec:
    id: int
    param_seed: int = 0
    birth_generation: int = 0


@dataclass(frozen=True)
class EdgeSpec:
    src: int
    dst: int
    weight: float = 1.0
    recurrent: bool = False


@dataclass
class InputAccessSpec:
    """Task-agnostic graph-level raw-observation connectivity.

    The mask is stored as [observation_dim, num_cells]. A value of 1 means
    that raw observation dimension is visible to that ordinary cell. The cell
    implementation itself is unchanged.
    """

    mode: str
    observation_dim: int
    num_cells: int
    mask: list[list[float]]
    density: float
    seed: int = 0

    def validate(self) -> None:
        if self.mode not in {"broadcast", "fixed_sparse", "none"}:
            raise ValueError(f"unsupported input access mode: {self.mode}")
        if self.observation_dim <= 0 or self.num_cells <= 0:
            raise ValueError("input access dimensions must be positive")
        array = np.asarray(self.mask, dtype=np.float32)
        if array.shape != (self.observation_dim, self.num_cells):
            raise ValueError(
                f"input mask shape {array.shape} != {(self.observation_dim, self.num_cells)}"
            )
        if not np.all(np.isin(array, [0.0, 1.0])):
            raise ValueError("input mask must be binary")
        effective = float(np.mean(array))
        if self.mode == "broadcast" and not np.all(array == 1.0):
            raise ValueError("broadcast mode requires an all-ones mask")
        if self.mode == "none" and not np.all(array == 0.0):
            raise ValueError("none mode requires an all-zero mask")
        if abs(float(self.density) - effective) > 1e-8:
            raise ValueError(f"input density metadata drift: {self.density} != {effective}")

    @property
    def effective_density(self) -> float:
        return float(np.mean(np.asarray(self.mask, dtype=np.float32)))

    @property
    def edge_count(self) -> int:
        return int(np.asarray(self.mask, dtype=np.float32).sum())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode,
            "observation_dim": int(self.observation_dim),
            "num_cells": int(self.num_cells),
            "mask": [[float(value) for value in row] for row in self.mask],
            "density": float(self.density),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputAccessSpec":
        spec = cls(
            mode=str(data["mode"]),
            observation_dim=int(data["observation_dim"]),
            num_cells=int(data["num_cells"]),
            mask=[[float(value) for value in row] for row in data["mask"]],
            density=float(data["density"]),
            seed=int(data.get("seed", 0)),
        )
        spec.validate()
        return spec


def build_fixed_sparse_mask(
    observation_dim: int,
    num_cells: int,
    density: float,
    seed: int,
    *,
    ensure_each_input_connected: bool = True,
    ensure_each_cell_connected: bool = True,
) -> np.ndarray:
    """Build a deterministic task-independent binary input mask.

    The requested edge count is rounded to the nearest feasible integer and
    raised only when necessary to satisfy the coverage constraints.
    """

    observation_dim = int(observation_dim)
    num_cells = int(num_cells)
    density = float(density)
    if observation_dim <= 0 or num_cells <= 0:
        raise ValueError("observation_dim and num_cells must be positive")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must lie in [0,1]")
    total = observation_dim * num_cells
    target_edges = int(round(total * density))
    minimum = 0
    if ensure_each_input_connected:
        minimum = max(minimum, observation_dim)
    if ensure_each_cell_connected:
        minimum = max(minimum, num_cells)
    target_edges = min(total, max(target_edges, minimum))
    generator = np.random.default_rng(int(seed))
    mask = np.zeros((observation_dim, num_cells), dtype=np.float32)

    input_order = generator.permutation(observation_dim).tolist()
    cell_order = generator.permutation(num_cells).tolist()

    if ensure_each_cell_connected:
        for position, cell in enumerate(cell_order):
            input_index = input_order[position % observation_dim]
            mask[input_index, cell] = 1.0

    if ensure_each_input_connected:
        for position, input_index in enumerate(input_order):
            if mask[input_index].sum() == 0:
                cell = cell_order[position % num_cells]
                mask[input_index, cell] = 1.0

    current = int(mask.sum())
    candidates = [(row, col) for row in range(observation_dim) for col in range(num_cells) if mask[row, col] == 0.0]
    generator.shuffle(candidates)
    for row, col in candidates[: max(0, target_edges - current)]:
        mask[row, col] = 1.0

    if ensure_each_input_connected and np.any(mask.sum(axis=1) == 0):
        raise RuntimeError("failed to connect every observation dimension")
    if ensure_each_cell_connected and np.any(mask.sum(axis=0) == 0):
        raise RuntimeError("failed to connect every cell")
    return mask


def build_input_access_spec(
    mode: str,
    observation_dim: int,
    num_cells: int,
    *,
    density: float = 1.0,
    seed: int = 0,
) -> InputAccessSpec:
    if mode == "broadcast":
        mask = np.ones((observation_dim, num_cells), dtype=np.float32)
    elif mode == "none":
        mask = np.zeros((observation_dim, num_cells), dtype=np.float32)
    elif mode == "fixed_sparse":
        mask = build_fixed_sparse_mask(observation_dim, num_cells, density, seed)
    else:
        raise ValueError(f"unsupported input access mode: {mode}")
    spec = InputAccessSpec(
        mode=mode,
        observation_dim=int(observation_dim),
        num_cells=int(num_cells),
        mask=mask.tolist(),
        density=float(mask.mean()),
        seed=int(seed),
    )
    spec.validate()
    return spec


def degree_preserving_shuffled_mask(mask: np.ndarray | list[list[float]], seed: int, *, attempts: int = 5000) -> np.ndarray:
    """Randomize connectivity with degree-preserving 2x2 edge swaps."""

    array = np.asarray(mask, dtype=np.float32).copy()
    if array.ndim != 2:
        raise ValueError("mask must be 2D")
    original = array.copy()
    generator = np.random.default_rng(int(seed))
    rows, cols = array.shape
    successful = 0
    for _ in range(int(attempts)):
        r1, r2 = generator.choice(rows, size=2, replace=False)
        c1, c2 = generator.choice(cols, size=2, replace=False)
        block = (array[r1, c1], array[r1, c2], array[r2, c1], array[r2, c2])
        if block == (1.0, 0.0, 0.0, 1.0):
            array[r1, c1] = 0.0; array[r2, c2] = 0.0
            array[r1, c2] = 1.0; array[r2, c1] = 1.0
            successful += 1
        elif block == (0.0, 1.0, 1.0, 0.0):
            array[r1, c1] = 1.0; array[r2, c2] = 1.0
            array[r1, c2] = 0.0; array[r2, c1] = 0.0
            successful += 1
        if successful >= max(8, int(array.sum())):
            break
    if not np.array_equal(array.sum(axis=0), original.sum(axis=0)):
        raise RuntimeError("column degree changed during shuffle")
    if not np.array_equal(array.sum(axis=1), original.sum(axis=1)):
        raise RuntimeError("row degree changed during shuffle")
    return array


@dataclass
class GraphSpec:
    cells: list[CellSpec]
    edges: list[EdgeSpec]
    input_access: InputAccessSpec | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    birth_history: list[dict[str, Any]] = field(default_factory=list)
    mutation_history: list[dict[str, Any]] = field(default_factory=list)
    parent_id: str = ""
    generation: int = 0

    def validate(self, max_cells: int = 16, max_edges: int = 64) -> None:
        ids = [cell.id for cell in self.cells]
        if ids != list(range(len(ids))):
            raise ValueError(f"cell ids must be contiguous 0..N-1, got {ids}")
        if not (1 <= len(ids) <= max_cells):
            raise ValueError(f"invalid cell count {len(ids)}")
        if len(self.edges) > max_edges:
            raise ValueError(f"edge cap exceeded: {len(self.edges)} > {max_edges}")
        seen: set[tuple[int, int, bool]] = set()
        for edge in self.edges:
            if edge.src not in ids or edge.dst not in ids:
                raise ValueError(f"edge references missing cell: {edge}")
            key = (edge.src, edge.dst, edge.recurrent)
            if key in seen:
                raise ValueError(f"duplicate edge {key}")
            seen.add(key)
        if self.input_access is not None:
            self.input_access.validate()
            if self.input_access.num_cells != len(self.cells):
                raise ValueError("input access num_cells does not match graph")

    def canonical_structure(self) -> dict[str, Any]:
        cells = [{"id": idx, "param_seed": int(cell.param_seed)} for idx, cell in enumerate(self.cells)]
        edges = sorted(
            (
                {
                    "src": int(edge.src),
                    "dst": int(edge.dst),
                    "weight": round(float(edge.weight), 8),
                    "recurrent": bool(edge.recurrent),
                }
                for edge in self.edges
            ),
            key=lambda item: (item["src"], item["dst"], item["recurrent"], item["weight"]),
        )
        payload: dict[str, Any] = {"cells": cells, "edges": edges}
        # Broadcast is the historical implicit default. Omitting it here keeps
        # historical graph IDs and therefore historical training seeds stable.
        if self.input_access is not None and self.input_access.mode != "broadcast":
            payload["input_access"] = self.input_access.to_dict()
        return payload

    @property
    def graph_id(self) -> str:
        payload = json.dumps(self.canonical_structure(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [asdict(cell) for cell in self.cells],
            "edges": [asdict(edge) for edge in self.edges],
            "input_access": self.input_access.to_dict() if self.input_access is not None else None,
            "parameters": self.parameters,
            "birth_history": self.birth_history,
            "mutation_history": self.mutation_history,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "graph_id": self.graph_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphSpec":
        access_data = data.get("input_access")
        graph = cls(
            cells=[CellSpec(**{k: v for k, v in row.items() if k in {"id", "param_seed", "birth_generation"}}) for row in data["cells"]],
            edges=[EdgeSpec(**{k: v for k, v in row.items() if k in {"src", "dst", "weight", "recurrent"}}) for row in data["edges"]],
            input_access=InputAccessSpec.from_dict(access_data) if access_data else None,
            parameters=dict(data.get("parameters", {})),
            birth_history=list(data.get("birth_history", [])),
            mutation_history=list(data.get("mutation_history", [])),
            parent_id=str(data.get("parent_id", "")),
            generation=int(data.get("generation", 0)),
        )
        graph.validate()
        return graph

    def clone(self) -> "GraphSpec":
        return GraphSpec.from_dict(self.to_dict())

    def descriptors(self) -> dict[str, float]:
        n = len(self.cells)
        e = len(self.edges)
        recurrent = sum(1 for edge in self.edges if edge.recurrent)
        self_loops = sum(1 for edge in self.edges if edge.src == edge.dst)
        indegree = [0] * n
        outdegree = [0] * n
        adjacency = {i: set() for i in range(n)}
        for edge in self.edges:
            indegree[edge.dst] += 1
            outdegree[edge.src] += 1
            adjacency[edge.src].add(edge.dst)
        scc_count = _strongly_connected_component_count(n, self.edges)
        cycle_count = _simple_cycle_proxy(n, self.edges)
        path_length = _average_reachable_path_length(n, adjacency)
        return {
            "cell_count": float(n),
            "edge_count": float(e),
            "recurrent_edge_fraction": float(recurrent / e) if e else 0.0,
            "self_loop_fraction": float(self_loops / max(1, e)),
            "mean_indegree": float(sum(indegree) / max(1, n)),
            "mean_outdegree": float(sum(outdegree) / max(1, n)),
            "max_indegree": float(max(indegree) if indegree else 0),
            "max_outdegree": float(max(outdegree) if outdegree else 0),
            "strongly_connected_components": float(scc_count),
            "cycle_count_proxy": float(cycle_count),
            "average_reachable_path_length": float(path_length),
        }


def initial_graph() -> GraphSpec:
    graph = GraphSpec(
        cells=[CellSpec(0, param_seed=0, birth_generation=0), CellSpec(1, param_seed=1, birth_generation=0)],
        edges=[EdgeSpec(0, 1, weight=0.5, recurrent=False)],
        parameters={},
        birth_history=[{"generation": 0, "cells": [0, 1]}],
        mutation_history=[],
        parent_id="",
        generation=0,
    )
    graph.validate()
    return graph


def renormalize(graph: GraphSpec) -> GraphSpec:
    old_ids = sorted(cell.id for cell in graph.cells)
    mapping = {old: new for new, old in enumerate(old_ids)}
    cells = [
        CellSpec(mapping[cell.id], param_seed=cell.param_seed, birth_generation=cell.birth_generation)
        for cell in sorted(graph.cells, key=lambda item: item.id)
    ]
    edges = [
        EdgeSpec(mapping[edge.src], mapping[edge.dst], weight=edge.weight, recurrent=edge.recurrent)
        for edge in graph.edges
        if edge.src in mapping and edge.dst in mapping
    ]
    access = None
    if graph.input_access is not None:
        old_mask = np.asarray(graph.input_access.mask, dtype=np.float32)
        new_mask = old_mask[:, old_ids]
        access = InputAccessSpec(
            mode=graph.input_access.mode,
            observation_dim=graph.input_access.observation_dim,
            num_cells=len(cells),
            mask=new_mask.tolist(),
            density=float(new_mask.mean()),
            seed=graph.input_access.seed,
        )
    out = GraphSpec(
        cells=cells,
        edges=edges,
        input_access=access,
        parameters=dict(graph.parameters),
        birth_history=list(graph.birth_history),
        mutation_history=list(graph.mutation_history),
        parent_id=graph.parent_id,
        generation=graph.generation,
    )
    out.validate()
    return out


def connected_subsets(graph: GraphSpec, min_size: int = 2, max_size: int = 6, limit: int = 256) -> list[tuple[int, ...]]:
    neighbors = {i: set() for i in range(len(graph.cells))}
    for edge in graph.edges:
        neighbors[edge.src].add(edge.dst)
        neighbors[edge.dst].add(edge.src)
    found: set[tuple[int, ...]] = set()
    frontier = [tuple(sorted((edge.src, edge.dst))) for edge in graph.edges if edge.src != edge.dst]
    for item in frontier:
        if len(item) >= min_size:
            found.add(item)
    idx = 0
    while idx < len(frontier) and len(found) < limit:
        subset = frontier[idx]
        idx += 1
        if len(subset) >= max_size:
            continue
        adjacent = set().union(*(neighbors[node] for node in subset)) - set(subset)
        for node in sorted(adjacent):
            grown = tuple(sorted((*subset, node)))
            if grown not in found:
                if len(grown) >= min_size:
                    found.add(grown)
                frontier.append(grown)
                if len(found) >= limit:
                    break
    return sorted(found, key=lambda item: (len(item), item))[:limit]


def _strongly_connected_component_count(n: int, edges: list[EdgeSpec]) -> int:
    adjacency = {i: [] for i in range(n)}
    reverse = {i: [] for i in range(n)}
    for edge in edges:
        adjacency[edge.src].append(edge.dst)
        reverse[edge.dst].append(edge.src)
    seen: set[int] = set()
    order: list[int] = []

    def dfs(node: int) -> None:
        seen.add(node)
        for nxt in adjacency[node]:
            if nxt not in seen:
                dfs(nxt)
        order.append(node)

    for node in range(n):
        if node not in seen:
            dfs(node)
    seen.clear()
    count = 0

    def rdfs(node: int) -> None:
        seen.add(node)
        for nxt in reverse[node]:
            if nxt not in seen:
                rdfs(nxt)

    for node in reversed(order):
        if node not in seen:
            count += 1
            rdfs(node)
    return count


def _simple_cycle_proxy(n: int, edges: list[EdgeSpec]) -> int:
    edge_set = {(edge.src, edge.dst) for edge in edges}
    count = sum(1 for src, dst in edge_set if src == dst)
    for a, b in itertools.combinations(range(n), 2):
        if (a, b) in edge_set and (b, a) in edge_set:
            count += 1
    return count


def _average_reachable_path_length(n: int, adjacency: dict[int, set[int]]) -> float:
    lengths: list[int] = []
    for source in range(n):
        dist = {source: 0}
        queue = [source]
        for node in queue:
            for nxt in adjacency[node]:
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        lengths.extend(value for node, value in dist.items() if node != source)
    return sum(lengths) / len(lengths) if lengths else 0.0
