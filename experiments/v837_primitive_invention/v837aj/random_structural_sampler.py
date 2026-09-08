from __future__ import annotations

import json
from pathlib import Path

from experiments.v837_primitive_invention.v837aj.fidelity_calibration import STAGE_RANDOM_SEARCH, train_proxy_candidate
from experiments.v837_primitive_invention.v837aj.structural_search import _best_so_far, _sort_key, require_valid_proxy
from experiments.v837_primitive_invention.v837aj.topology import sample_topology_with_counts

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def run_random_sampler(directed_run: dict, fidelity: str | None = None) -> dict:
    fidelity = fidelity or require_valid_proxy()
    family = str(directed_run["family"])
    run_index = int(directed_run["run_index"])
    directed_records = sorted(directed_run["records"], key=lambda r: int(r["evaluation_index"]))
    budget = int(CONFIG["search"]["candidate_budget"])
    if len(directed_records) != budget:
        raise RuntimeError("random sampler requires exact 64-slot directed budget")
    records: list[dict] = []
    seen: set[str] = set()
    for slot, directed in enumerate(directed_records):
        same_count = int(directed["topology"]["same_step_edge_count"])
        recurrent_count = int(directed["topology"]["recurrent_edge_count"])
        topology = None
        for attempt in range(10000):
            candidate = sample_topology_with_counts(
                same_count,
                recurrent_count,
                namespace="v837aj-random-structure",
                parts=(family, run_index, slot, attempt),
            )
            if candidate.topology_id not in seen:
                topology = candidate
                break
        if topology is None:
            raise RuntimeError("unable to generate unique complexity-matched random topology")
        row = train_proxy_candidate(
            topology,
            family=family,
            run_index=run_index,
            candidate_slot=slot,
            fidelity=fidelity,
            stage=STAGE_RANDOM_SEARCH,
        )
        row["evaluation_index"] = slot
        row["matched_directed_topology_id"] = directed["topology_id"]
        row["matched_same_step_edge_count"] = same_count
        row["matched_recurrent_edge_count"] = recurrent_count
        records.append(row); seen.add(topology.topology_id)
    if len(records) != budget or len(seen) != budget:
        raise RuntimeError("random sampler did not train exactly 64 unique topologies")
    for directed, random_row in zip(directed_records, records):
        if int(directed["topology"]["edge_count"]) != int(random_row["topology"]["edge_count"]):
            raise RuntimeError("random total edge count mismatch")
        if int(directed["topology"]["recurrent_edge_count"]) != int(random_row["topology"]["recurrent_edge_count"]):
            raise RuntimeError("random recurrent edge count mismatch")
        if int(directed["candidate_initialization_slot"]) != int(random_row["candidate_initialization_slot"]):
            raise RuntimeError("search/random candidate initialization slot mismatch")
    champion = min(records, key=_sort_key)
    return {
        "version": "V837aj",
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": family,
        "run_index": run_index,
        "fidelity": fidelity,
        "candidate_budget": budget,
        "complexity_matched_per_slot": True,
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
