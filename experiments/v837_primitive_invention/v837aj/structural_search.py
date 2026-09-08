from __future__ import annotations

import json
import os
import time
from pathlib import Path

from experiments.v837_primitive_invention.v837aj.fidelity_calibration import STAGE_SEARCH, train_proxy_candidate
from experiments.v837_primitive_invention.v837aj.structural_mutations import constructive_initial_population, mutate_unique
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology, historical_anchor_topology

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PROGRESS_DIR = HERE / "raw" / "cache"


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


def _progress_path(family: str, run_index: int) -> Path:
    return PROGRESS_DIR / f"progress_search_{family}_{int(run_index)}.json"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def _checkpoint(
    *,
    path: Path | None,
    family: str,
    run_index: int,
    fidelity: str,
    phase: str,
    initial_next_slot: int,
    records: list[dict],
    population: list[dict],
    evaluated_ids: set[str],
    evaluation_index: int,
    proposal_counter: int,
    generation: int,
    parent_pool: list[dict],
    offspring: list[dict],
    offspring_index: int,
    complete: bool = False,
) -> None:
    if path is None:
        return
    _atomic_json(path, {
        "version": "V837aj",
        "engine": "DIRECTED_STRUCTURAL_SEARCH",
        "family": family,
        "run_index": int(run_index),
        "fidelity": fidelity,
        "phase": phase,
        "initial_next_slot": int(initial_next_slot),
        "records": records,
        "population": population,
        "evaluated_ids": sorted(evaluated_ids),
        "evaluation_index": int(evaluation_index),
        "proposal_counter": int(proposal_counter),
        "generation": int(generation),
        "parent_pool": parent_pool,
        "offspring": offspring,
        "offspring_index": int(offspring_index),
        "complete": bool(complete),
    })


def _result_from_records(family: str, run_index: int, fidelity: str, records: list[dict], anchor_id: str) -> dict:
    budget = int(CONFIG["search"]["candidate_budget"])
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


def run_directed_search(family: str, run_index: int, fidelity: str | None = None, *, checkpoint: bool = False) -> dict:
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

    progress_path = _progress_path(family, run_index) if checkpoint else None
    if progress_path is not None and progress_path.is_file():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("version") != "V837aj" or state.get("engine") != "DIRECTED_STRUCTURAL_SEARCH":
            raise RuntimeError("invalid directed-search progress checkpoint")
        if state.get("family") != family or int(state.get("run_index", -1)) != int(run_index) or state.get("fidelity") != fidelity:
            raise RuntimeError("directed-search progress checkpoint identity mismatch")
        records = list(state.get("records", []))
        population = list(state.get("population", []))
        evaluated_ids = set(state.get("evaluated_ids", []))
        phase = str(state.get("phase", "initial"))
        initial_next_slot = int(state.get("initial_next_slot", 0))
        evaluation_index = int(state.get("evaluation_index", len(records)))
        proposal_counter = int(state.get("proposal_counter", 0))
        generation = int(state.get("generation", 1))
        parent_pool = list(state.get("parent_pool", []))
        offspring = list(state.get("offspring", []))
        offspring_index = int(state.get("offspring_index", 0))
        if state.get("complete") is True:
            return _result_from_records(family, run_index, fidelity, records, anchor_id)
    else:
        records = []
        population = []
        evaluated_ids: set[str] = set()
        phase = "initial"
        initial_next_slot = 0
        evaluation_index = 0
        proposal_counter = 0
        generation = 1
        parent_pool = []
        offspring = []
        offspring_index = 0

    if phase == "initial":
        for slot in range(initial_next_slot, population_size):
            topology = initial[slot]
            if topology.topology_id in evaluated_ids:
                raise RuntimeError("directed checkpoint duplicated an initial topology")
            row = train_proxy_candidate(
                topology,
                family=family,
                run_index=int(run_index),
                candidate_slot=int(slot),
                fidelity=fidelity,
                stage=STAGE_SEARCH,
            )
            record = _decorate(row, evaluation_index=slot, generation=0, parent_topology_id=None)
            records.append(record)
            population.append(record)
            evaluated_ids.add(topology.topology_id)
            initial_next_slot = slot + 1
            evaluation_index = initial_next_slot
            _checkpoint(
                path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
                phase="initial", initial_next_slot=initial_next_slot, records=records,
                population=population, evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
                proposal_counter=proposal_counter, generation=generation, parent_pool=[], offspring=[],
                offspring_index=0,
            )
        phase = "evolution"
        evaluation_index = population_size
        generation = 1
        parent_pool = []
        offspring = []
        offspring_index = 0
        _checkpoint(
            path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
            phase=phase, initial_next_slot=population_size, records=records, population=population,
            evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
            proposal_counter=proposal_counter, generation=generation, parent_pool=parent_pool,
            offspring=offspring, offspring_index=offspring_index,
        )

    while generation <= generations:
        if not parent_pool:
            population.sort(key=_sort_key)
            parent_pool = population[: int(CONFIG["search"]["parent_pool_size"])]
            offspring = []
            offspring_index = 0
            _checkpoint(
                path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
                phase="evolution", initial_next_slot=population_size, records=records,
                population=population, evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
                proposal_counter=proposal_counter, generation=generation, parent_pool=parent_pool,
                offspring=offspring, offspring_index=offspring_index,
            )

        while offspring_index < offspring_per_generation:
            parent = parent_pool[(generation * offspring_per_generation + offspring_index) % len(parent_pool)]
            parent_topology = SearchTopology.from_dict(parent["topology"])
            child = None
            for _attempt in range(1000):
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
            records.append(record)
            offspring.append(record)
            evaluated_ids.add(child.topology_id)
            evaluation_index += 1
            offspring_index += 1
            _checkpoint(
                path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
                phase="evolution", initial_next_slot=population_size, records=records,
                population=population, evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
                proposal_counter=proposal_counter, generation=generation, parent_pool=parent_pool,
                offspring=offspring, offspring_index=offspring_index,
            )

        population = sorted(population + offspring, key=_sort_key)[:population_size]
        generation += 1
        parent_pool = []
        offspring = []
        offspring_index = 0
        _checkpoint(
            path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
            phase="evolution", initial_next_slot=population_size, records=records,
            population=population, evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
            proposal_counter=proposal_counter, generation=generation, parent_pool=parent_pool,
            offspring=offspring, offspring_index=offspring_index,
        )

    result = _result_from_records(family, run_index, fidelity, records, anchor_id)
    _checkpoint(
        path=progress_path, family=family, run_index=run_index, fidelity=fidelity,
        phase="complete", initial_next_slot=population_size, records=records,
        population=population, evaluated_ids=evaluated_ids, evaluation_index=evaluation_index,
        proposal_counter=proposal_counter, generation=generation, parent_pool=[], offspring=[],
        offspring_index=0, complete=True,
    )
    return result
