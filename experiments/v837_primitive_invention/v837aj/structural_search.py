from __future__ import annotations

import json
from pathlib import Path

from experiments.v837_primitive_invention.v837aj.fidelity_calibration import STAGE_SEARCH, train_proxy_candidate
from experiments.v837_primitive_invention.v837aj.structural_mutations import constructive_initial_population, mutate_unique
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology, historical_anchor_topology

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def require_valid_proxy() -> str:
    path = HERE / "diagnostics/fidelity_decision.json"
    if not path.is_file():
        raise RuntimeError("V837aj-B blocked: Stage-A fidelity decision missing")
    decision = json.loads(path.read_text(encoding="utf-8"))
    fidelity = decision.get("selected_search_fidelity")
    if decision.get("proxy_valid") is not True or not fidelity:
        raise RuntimeError("V837aj-B blocked: SEARCH_FIDELITY_PROXY_INVALID")
    if fidelity not in {"F0", "F1", "F2", "F3"}:
        raise RuntimeError("V837aj-B blocked: invalid selected fidelity")
    return str(fidelity)


def _sort_key(record: dict) -> tuple:
    return (
        float(record["fitness"]),
        -float(record["selection_success"]),
        int(record["topology"]["edge_count"]),
        str(record["topology_id"]),
    )


def _decorate(row: dict, *, evaluation_index: int, generation: int, parent_topology_id: str | None) -> dict:
    row = dict(row)
    row["evaluation_index"] = int(evaluation_index)
    row["generation"] = int(generation)
    row["parent_topology_id"] = parent_topology_id
    return row


def _best_so_far(records: list[dict]) -> list[dict]:
    best: dict | None = None
    curve = []
    for record in sorted(records, key=lambda r: int(r["evaluation_index"])):
        if best is None or _sort_key(record) < _sort_key(best):
            best = record
        assert best is not None
        curve.append({
            "evaluation_index": int(record["evaluation_index"]),
            "best_fitness": float(best["fitness"]),
            "best_selection_success": float(best["selection_success"]),
            "best_edge_count": int(best["topology"]["edge_count"]),
            "best_topology_id": str(best["topology_id"]),
        })
    return curve


def run_directed_search(family: str, run_index: int, fidelity: str | None = None) -> dict:
    fidelity = fidelity or require_valid_proxy()
    budget = int(CONFIG["search"]["candidate_budget"])
    population_size = int(CONFIG["search"]["population_size"])
    offspring_per_generation = int(CONFIG["search"]["offspring_per_generation"])
    generations = int(CONFIG["search"]["generations"])
    if population_size + offspring_per_generation * generations != budget:
        raise RuntimeError("V837aj directed candidate budget drift")

    initial = constructive_initial_population(family, int(run_index))
    anchor_id = historical_anchor_topology().topology_id
    if anchor_id in {topology.topology_id for topology in initial}:
        raise RuntimeError("historical anchor seeded into constructive search")

    records: list[dict] = []
    evaluated_ids: set[str] = set()
    population: list[dict] = []
    for slot, topology in enumerate(initial):
        row = train_proxy_candidate(
            topology,
            family=family,
            run_index=int(run_index),
            candidate_slot=int(slot),
            fidelity=fidelity,
            stage=STAGE_SEARCH,
        )
        record = _decorate(row, evaluation_index=slot, generation=0, parent_topology_id=None)
        records.append(record); population.append(record); evaluated_ids.add(topology.topology_id)

    evaluation_index = population_size
    proposal_counter = 0
    for generation in range(1, generations + 1):
        population.sort(key=_sort_key)
        parent_pool = population[: int(CONFIG["search"]["parent_pool_size"])]
        offspring: list[dict] = []
        for offspring_index in range(offspring_per_generation):
            parent = parent_pool[(generation * offspring_per_generation + offspring_index) % len(parent_pool)]
            parent_topology = SearchTopology.from_dict(parent["topology"])
            child = None
            for attempt in range(1000):
                child = mutate_unique(
                    parent_topology,
                    family=family,
                    run_index=int(run_index),
                    proposal_index=proposal_counter,
                    evaluated_ids=evaluated_ids,
                )
                proposal_counter += 1
                if child is not None and child.topology_id not in evaluated_ids:
                    break
                child = None
            if child is None:
                raise RuntimeError("unable to generate unique directed-search offspring")
            row = train_proxy_candidate(
                child,
                family=family,
                run_index=int(run_index),
                candidate_slot=int(evaluation_index),
                fidelity=fidelity,
                stage=STAGE_SEARCH,
            )
            record = _decorate(
                row,
                evaluation_index=evaluation_index,
                generation=generation,
                parent_topology_id=str(parent["topology_id"]),
            )
            records.append(record); offspring.append(record); evaluated_ids.add(child.topology_id)
            evaluation_index += 1
        population = sorted(population + offspring, key=_sort_key)[:population_size]

    if len(records) != budget or len({r["topology_id"] for r in records}) != budget:
        raise RuntimeError("directed search did not train exactly 64 unique topologies")
    champion = min(records, key=_sort_key)
    return {
        "version": "V837aj",
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": family,
        "run_index": int(run_index),
        "fidelity": fidelity,
        "candidate_budget": budget,
        "anchor_seeded": False,
        "anchor_seen_during_search": anchor_id in {r["topology_id"] for r in records},
        "records": records,
        "best_so_far": _best_so_far(records),
        "champion": {
            "topology": champion["topology"],
            "topology_id": champion["topology_id"],
            "search_fitness": champion["fitness"],
            "search_selection_success": champion["selection_success"],
            "search_development_success": champion["development_success"],
            "edge_count": champion["topology"]["edge_count"],
            "recurrent_edge_count": champion["topology"]["recurrent_edge_count"],
            "selected_evaluation_index": champion["evaluation_index"],
            "selected_before_final_validation": True,
        },
    }
