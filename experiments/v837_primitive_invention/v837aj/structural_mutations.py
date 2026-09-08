from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.v837aj.topology import (
    MAX_EDGES,
    RECURRENT_UNIVERSE,
    SAME_STEP_UNIVERSE,
    SearchEdge,
    SearchTopology,
    historical_anchor_topology,
    minimal_topology,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

ALLOWED_MUTATIONS = (
    "ADD_SAME_STEP_EDGE",
    "REMOVE_SAME_STEP_EDGE",
    "ADD_RECURRENT_EDGE",
    "REMOVE_RECURRENT_EDGE",
    "REWIRE_SAME_STEP_EDGE",
    "REWIRE_RECURRENT_EDGE",
)
MUTATION_PROBABILITIES = {name: float(CONFIG["mutation_probabilities"][name]) for name in ALLOWED_MUTATIONS}
if abs(sum(MUTATION_PROBABILITIES.values()) - 1.0) > 1e-12:
    raise RuntimeError("V837aj mutation probabilities do not sum to one")


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(deterministic_int("v837aj-structural-mutation", *parts) % (2**63 - 1))


def _choice(rng: np.random.Generator, values: list[SearchEdge]) -> SearchEdge | None:
    if not values:
        return None
    return values[int(rng.integers(0, len(values)))]


def apply_mutation(topology: SearchTopology, operator: str, rng: np.random.Generator) -> SearchTopology | None:
    if operator not in ALLOWED_MUTATIONS:
        raise ValueError(f"prohibited V837aj mutation: {operator}")
    edges = set(topology.edges)
    same = [edge for edge in topology.edges if not edge.recurrent]
    recurrent = [edge for edge in topology.edges if edge.recurrent]

    if operator == "ADD_SAME_STEP_EDGE":
        if len(edges) >= MAX_EDGES:
            return None
        candidate = _choice(rng, [edge for edge in SAME_STEP_UNIVERSE if edge not in edges])
        if candidate is None:
            return None
        edges.add(candidate)
    elif operator == "REMOVE_SAME_STEP_EDGE":
        candidate = _choice(rng, same)
        if candidate is None:
            return None
        edges.remove(candidate)
    elif operator == "ADD_RECURRENT_EDGE":
        if len(edges) >= MAX_EDGES:
            return None
        candidate = _choice(rng, [edge for edge in RECURRENT_UNIVERSE if edge not in edges])
        if candidate is None:
            return None
        edges.add(candidate)
    elif operator == "REMOVE_RECURRENT_EDGE":
        candidate = _choice(rng, recurrent)
        if candidate is None:
            return None
        edges.remove(candidate)
    elif operator == "REWIRE_SAME_STEP_EDGE":
        old = _choice(rng, same)
        if old is None:
            return None
        absent = [edge for edge in SAME_STEP_UNIVERSE if edge not in edges and edge != old]
        new = _choice(rng, absent)
        if new is None:
            return None
        edges.remove(old); edges.add(new)
    else:
        old = _choice(rng, recurrent)
        if old is None:
            return None
        absent = [edge for edge in RECURRENT_UNIVERSE if edge not in edges and edge != old]
        new = _choice(rng, absent)
        if new is None:
            return None
        edges.remove(old); edges.add(new)
    return SearchTopology(tuple(edges))


def mutate_unique(
    parent: SearchTopology,
    *,
    family: str,
    run_index: int,
    proposal_index: int,
    evaluated_ids: set[str],
    retry_limit: int | None = None,
) -> SearchTopology | None:
    limit = int(retry_limit or CONFIG["search"]["mutation_retry_limit"])
    names = list(ALLOWED_MUTATIONS)
    probabilities = np.asarray([MUTATION_PROBABILITIES[name] for name in names], dtype=np.float64)
    for retry in range(limit):
        rng = _rng(family, int(run_index), int(proposal_index), int(retry))
        operator = str(rng.choice(names, p=probabilities))
        child = apply_mutation(parent, operator, rng)
        if child is None or child.topology_id in evaluated_ids:
            continue
        return child
    return None


def constructive_initial_population(family: str, run_index: int) -> list[SearchTopology]:
    target = int(CONFIG["search"]["population_size"])
    base = minimal_topology()
    anchor_id = historical_anchor_topology().topology_id
    population = [base]
    seen = {base.topology_id}
    attempt = 0
    while len(population) < target:
        rng = _rng("initial", family, int(run_index), attempt)
        mutations = int(rng.integers(int(CONFIG["search"]["initial_mutations_min"]), int(CONFIG["search"]["initial_mutations_max"]) + 1))
        candidate = base
        valid = True
        local_seen = set(seen)
        for step in range(mutations):
            child = mutate_unique(
                candidate,
                family=family,
                run_index=run_index,
                proposal_index=100000 + attempt * 10 + step,
                evaluated_ids=local_seen,
            )
            if child is None:
                valid = False
                break
            candidate = child
            local_seen.add(candidate.topology_id)
        attempt += 1
        if not valid or candidate.topology_id in seen or candidate.topology_id == anchor_id:
            if attempt > 10000:
                raise RuntimeError("unable to construct 16 unique anchor-free initial topologies")
            continue
        population.append(candidate); seen.add(candidate.topology_id)
    if anchor_id in {topology.topology_id for topology in population}:
        raise RuntimeError("historical anchor leaked into constructive initial population")
    return population
