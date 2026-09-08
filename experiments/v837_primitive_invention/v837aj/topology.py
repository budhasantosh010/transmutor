from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.v837_primitive_invention.common.graph import EdgeSpec, GraphSpec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
NUM_CELLS = int(CONFIG["cell_count"])
MAX_EDGES = int(CONFIG["max_message_edges"])


@dataclass(frozen=True, order=True)
class SearchEdge:
    src: int
    dst: int
    recurrent: bool

    def __post_init__(self) -> None:
        if not (0 <= int(self.src) < NUM_CELLS and 0 <= int(self.dst) < NUM_CELLS):
            raise ValueError(f"edge endpoint outside 0..{NUM_CELLS - 1}: {self}")
        if not bool(self.recurrent) and int(self.src) >= int(self.dst):
            raise ValueError(f"same-step edge requires src < dst: {self}")

    def to_dict(self) -> dict:
        return {"src": int(self.src), "dst": int(self.dst), "recurrent": bool(self.recurrent)}


@dataclass(frozen=True)
class SearchTopology:
    edges: tuple[SearchEdge, ...]

    def __post_init__(self) -> None:
        canonical = tuple(sorted(self.edges, key=lambda edge: (bool(edge.recurrent), int(edge.src), int(edge.dst))))
        if len(canonical) != len(set(canonical)):
            raise ValueError("duplicate structural edge")
        if len(canonical) > MAX_EDGES:
            raise ValueError(f"topology exceeds {MAX_EDGES}-edge cap")
        object.__setattr__(self, "edges", canonical)

    @property
    def topology_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def same_step_count(self) -> int:
        return sum(not edge.recurrent for edge in self.edges)

    @property
    def recurrent_count(self) -> int:
        return sum(edge.recurrent for edge in self.edges)

    def canonical_payload(self) -> dict:
        return {"cell_count": NUM_CELLS, "edges": [edge.to_dict() for edge in self.edges]}

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))

    def to_dict(self, *, include_descriptors: bool = True) -> dict:
        payload = {
            "topology_id": self.topology_id,
            "cell_count": NUM_CELLS,
            "edges": [edge.to_dict() for edge in self.edges],
            "edge_count": self.edge_count,
            "same_step_edge_count": self.same_step_count,
            "recurrent_edge_count": self.recurrent_count,
        }
        if include_descriptors:
            payload["descriptors"] = topology_descriptors(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SearchTopology":
        return cls(tuple(SearchEdge(int(e["src"]), int(e["dst"]), bool(e["recurrent"])) for e in payload["edges"]))

    @classmethod
    def from_graph(cls, graph: GraphSpec) -> "SearchTopology":
        return cls(tuple(SearchEdge(int(edge.src), int(edge.dst), bool(edge.recurrent)) for edge in graph.edges))

    def to_graph(self, restart: int, *, semantic_edge_initialization: bool = False, family: str = "", run_index: int = 0) -> GraphSpec:
        source = high_capacity_generic_graph(int(restart))
        edges: list[EdgeSpec] = []
        for edge in self.edges:
            if semantic_edge_initialization:
                weight = semantic_edge_initial_weight(family, run_index, edge)
            else:
                weight = 0.55 if edge.recurrent else 0.35
            edges.append(EdgeSpec(edge.src, edge.dst, weight=float(weight), recurrent=edge.recurrent))
        graph = GraphSpec(
            cells=list(source.cells),
            edges=edges,
            input_access=source.input_access,
            generation=0,
            parent_id="V837AJ_STRUCTURE",
        )
        graph.validate(max_cells=NUM_CELLS, max_edges=MAX_EDGES)
        return graph


SAME_STEP_UNIVERSE = tuple(SearchEdge(src, dst, False) for src in range(NUM_CELLS) for dst in range(src + 1, NUM_CELLS))
RECURRENT_UNIVERSE = tuple(SearchEdge(src, dst, True) for src in range(NUM_CELLS) for dst in range(NUM_CELLS))


def semantic_edge_seed(family: str, run_index: int, edge: SearchEdge) -> int:
    return deterministic_int("v837aj-edge-init", family, int(run_index), int(edge.src), int(edge.dst), int(edge.recurrent))


def semantic_edge_initial_weight(family: str, run_index: int, edge: SearchEdge) -> float:
    spec = CONFIG["semantic_edge_initialization"]
    generator = torch.Generator(device="cpu").manual_seed(int(semantic_edge_seed(family, run_index, edge)))
    base = float(spec["recurrent_mean"] if edge.recurrent else spec["same_step_mean"])
    return float(base + float(spec["std"]) * torch.randn((), generator=generator).item())


def historical_anchor_topology() -> SearchTopology:
    return SearchTopology.from_graph(high_capacity_generic_graph(0))


def minimal_topology() -> SearchTopology:
    edges = [SearchEdge(i, i + 1, False) for i in range(NUM_CELLS - 1)]
    edges.extend(SearchEdge(i, i, True) for i in range(NUM_CELLS))
    return SearchTopology(tuple(edges))


def zero_message_topology() -> SearchTopology:
    return SearchTopology(tuple())


def _rng(namespace: str, *parts: object) -> np.random.Generator:
    return np.random.default_rng(deterministic_int(namespace, *parts) % (2**63 - 1))


def _delete_fraction(topology: SearchTopology, fraction: float, label: str) -> SearchTopology:
    edges = list(topology.edges)
    count = max(1, int(round(len(edges) * float(fraction))))
    rng = _rng("v837aj-panel-delete", label)
    chosen = set(int(i) for i in rng.choice(len(edges), size=min(count, len(edges)), replace=False).tolist())
    return SearchTopology(tuple(edge for i, edge in enumerate(edges) if i not in chosen))


def _temporal_rewire(topology: SearchTopology, fraction: float, label: str) -> SearchTopology:
    edges = set(topology.edges)
    candidates = [edge for edge in topology.edges if not edge.recurrent and SearchEdge(edge.src, edge.dst, True) not in edges]
    count = max(1, int(round(len(topology.edges) * float(fraction))))
    count = min(count, len(candidates))
    rng = _rng("v837aj-panel-temporal", label)
    chosen = rng.choice(len(candidates), size=count, replace=False).tolist() if count else []
    for index in chosen:
        old = candidates[int(index)]
        edges.remove(old)
        edges.add(SearchEdge(old.src, old.dst, True))
    return SearchTopology(tuple(edges))


def _endpoint_rewire(topology: SearchTopology, fraction: float, label: str) -> SearchTopology:
    edges = set(topology.edges)
    requested = max(1, int(round(len(topology.edges) * float(fraction))))
    rng = _rng("v837aj-panel-endpoint", label)
    # Only select edges whose temporal class has an absent legal endpoint.
    # The historical anchor saturates the 45-edge SAME_STEP universe, so its
    # endpoint rewires necessarily operate on recurrent edges while preserving
    # the recurrent/same-step counts exactly.
    candidates = []
    for old in topology.edges:
        universe = RECURRENT_UNIVERSE if old.recurrent else SAME_STEP_UNIVERSE
        if any(edge not in edges and edge != old for edge in universe):
            candidates.append(old)
    count = min(requested, len(candidates))
    selected = [candidates[int(i)] for i in rng.choice(len(candidates), size=count, replace=False).tolist()] if count else []
    for old in selected:
        universe = RECURRENT_UNIVERSE if old.recurrent else SAME_STEP_UNIVERSE
        absent = [edge for edge in universe if edge not in edges and edge != old]
        if not absent:
            raise RuntimeError("endpoint rewire candidate unexpectedly became non-rewirable")
        new = absent[int(rng.integers(0, len(absent)))]
        edges.remove(old)
        edges.add(new)
    return SearchTopology(tuple(edges))


def sample_topology_with_counts(same_step_count: int, recurrent_count: int, *, namespace: str, parts: tuple[object, ...]) -> SearchTopology:
    same_step_count = int(same_step_count)
    recurrent_count = int(recurrent_count)
    if same_step_count < 0 or same_step_count > len(SAME_STEP_UNIVERSE):
        raise ValueError("invalid same-step count")
    if recurrent_count < 0 or recurrent_count > len(RECURRENT_UNIVERSE):
        raise ValueError("invalid recurrent count")
    if same_step_count + recurrent_count > MAX_EDGES:
        raise ValueError("edge count exceeds cap")
    rng = _rng(namespace, *parts)
    edges: list[SearchEdge] = []
    if same_step_count:
        edges.extend(SAME_STEP_UNIVERSE[int(i)] for i in rng.choice(len(SAME_STEP_UNIVERSE), size=same_step_count, replace=False).tolist())
    if recurrent_count:
        edges.extend(RECURRENT_UNIVERSE[int(i)] for i in rng.choice(len(RECURRENT_UNIVERSE), size=recurrent_count, replace=False).tolist())
    return SearchTopology(tuple(edges))


def calibration_panel() -> dict[str, SearchTopology]:
    anchor = historical_anchor_topology()
    panel = {
        "P0": anchor,
        "P1": minimal_topology(),
        "P2": zero_message_topology(),
        "P3": _delete_fraction(anchor, 0.15, "P3"),
        "P4": _delete_fraction(anchor, 0.30, "P4"),
        "P5": _delete_fraction(anchor, 0.45, "P5"),
        "P6": _delete_fraction(anchor, 0.60, "P6"),
        "P7": _temporal_rewire(anchor, 0.15, "P7"),
        "P8": _temporal_rewire(anchor, 0.30, "P8"),
        "P9": _endpoint_rewire(anchor, 0.15, "P9"),
        "P10": _endpoint_rewire(anchor, 0.30, "P10"),
        "P11": sample_topology_with_counts(23, 5, namespace="v837aj-panel-random", parts=("P11",)),
    }
    if list(panel) != CONFIG["calibration"]["panel_ids"]:
        raise RuntimeError("calibration panel order drift")
    if len({topology.topology_id for topology in panel.values()}) != 12:
        raise RuntimeError("calibration panel contains duplicate topologies")
    return panel


def _component_count(adjacency: list[set[int]], *, directed: bool) -> int:
    if not directed:
        seen: set[int] = set()
        count = 0
        for start in range(NUM_CELLS):
            if start in seen:
                continue
            count += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adjacency[node] - seen)
        return count
    # Kosaraju, including isolated vertices.
    seen: set[int] = set()
    order: list[int] = []
    def visit(node: int) -> None:
        seen.add(node)
        for nxt in adjacency[node]:
            if nxt not in seen:
                visit(nxt)
        order.append(node)
    for node in range(NUM_CELLS):
        if node not in seen:
            visit(node)
    reverse = [set() for _ in range(NUM_CELLS)]
    for src in range(NUM_CELLS):
        for dst in adjacency[src]:
            reverse[dst].add(src)
    seen.clear(); count = 0
    for start in reversed(order):
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(reverse[node] - seen)
    return count


def _same_step_depth(topology: SearchTopology) -> int:
    depth = [1] * NUM_CELLS
    incoming = [[] for _ in range(NUM_CELLS)]
    for edge in topology.edges:
        if not edge.recurrent:
            incoming[edge.dst].append(edge.src)
    for node in range(NUM_CELLS):
        if incoming[node]:
            depth[node] = 1 + max(depth[src] for src in incoming[node])
    return max(depth) if depth else 0


def _directed_diameter(edges: list[SearchEdge]) -> int | None:
    adjacency = [set() for _ in range(NUM_CELLS)]
    for edge in edges:
        adjacency[edge.src].add(edge.dst)
    distances: list[int] = []
    for start in range(NUM_CELLS):
        dist = [-1] * NUM_CELLS
        dist[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        distances.extend(value for value in dist if value > 0)
    return max(distances) if distances else None


def _undirected_clustering(topology: SearchTopology) -> float:
    adjacency = [set() for _ in range(NUM_CELLS)]
    for edge in topology.edges:
        if edge.src == edge.dst:
            continue
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)
    values: list[float] = []
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            continue
        links = 0
        ordered = sorted(neighbors)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if b in adjacency[a]:
                    links += 1
        values.append(links / (degree * (degree - 1) / 2.0))
    return float(np.mean(values)) if values else 0.0


def topology_descriptors(topology: SearchTopology) -> dict:
    out_degree = [0] * NUM_CELLS
    in_degree = [0] * NUM_CELLS
    recurrent_out = [0] * NUM_CELLS
    recurrent_in = [0] * NUM_CELLS
    directed = [set() for _ in range(NUM_CELLS)]
    undirected = [set() for _ in range(NUM_CELLS)]
    recurrent_edges: list[SearchEdge] = []
    for edge in topology.edges:
        out_degree[edge.src] += 1
        in_degree[edge.dst] += 1
        directed[edge.src].add(edge.dst)
        undirected[edge.src].add(edge.dst)
        undirected[edge.dst].add(edge.src)
        if edge.recurrent:
            recurrent_out[edge.src] += 1
            recurrent_in[edge.dst] += 1
            recurrent_edges.append(edge)
    legal_total = len(SAME_STEP_UNIVERSE) + len(RECURRENT_UNIVERSE)
    return {
        "edge_count": topology.edge_count,
        "same_step_edge_count": topology.same_step_count,
        "recurrent_edge_count": topology.recurrent_count,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "recurrent_in_degree": recurrent_in,
        "recurrent_out_degree": recurrent_out,
        "density": topology.edge_count / float(legal_total),
        "strongly_connected_components": _component_count(directed, directed=True),
        "weakly_connected_components": _component_count(undirected, directed=False),
        "same_step_dag_depth": _same_step_depth(topology),
        "recurrent_graph_diameter": _directed_diameter(recurrent_edges),
        "clustering_coefficient": _undirected_clustering(topology),
    }


def edge_jaccard(a: SearchTopology, b: SearchTopology, *, recurrent: bool | None = None) -> float:
    sa = {edge for edge in a.edges if recurrent is None or edge.recurrent is recurrent}
    sb = {edge for edge in b.edges if recurrent is None or edge.recurrent is recurrent}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / float(len(sa | sb))


def topology_edit_distance(a: SearchTopology, b: SearchTopology) -> int:
    return len(set(a.edges) ^ set(b.edges))


def degree_vector(topology: SearchTopology) -> np.ndarray:
    desc = topology_descriptors(topology)
    return np.asarray(desc["in_degree"] + desc["out_degree"] + desc["recurrent_in_degree"] + desc["recurrent_out_degree"], dtype=np.float64)


def degree_cosine(a: SearchTopology, b: SearchTopology) -> float:
    va, vb = degree_vector(a), degree_vector(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 1.0
