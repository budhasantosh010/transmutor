from __future__ import annotations

import json
import os
import time
from pathlib import Path

from experiments.v837_primitive_invention.v837aj.fidelity_calibration import STAGE_RANDOM_SEARCH, train_proxy_candidate
from experiments.v837_primitive_invention.v837aj.structural_search import _best_so_far, _sort_key, require_valid_proxy
from experiments.v837_primitive_invention.v837aj.topology import sample_topology_with_counts

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PROGRESS_DIR = HERE / "raw" / "cache"


def _progress_path(family: str, run_index: int) -> Path:
    return PROGRESS_DIR / f"progress_random_{family}_{int(run_index)}.json"


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


def _save_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    _atomic_json(path, payload)


def _result(directed_run: dict, fidelity: str, records: list[dict]) -> dict:
    family = str(directed_run["family"])
    run_index = int(directed_run["run_index"])
    directed_records = sorted(directed_run["records"], key=lambda r: int(r["evaluation_index"]))
    budget = int(CONFIG["search"]["candidate_budget"])
    if len(records) != budget or len({row["topology_id"] for row in records}) != budget:
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


def run_random_sampler(directed_run: dict, fidelity: str | None = None, *, checkpoint: bool = False) -> dict:
    fidelity = fidelity or require_valid_proxy()
    family = str(directed_run["family"])
    run_index = int(directed_run["run_index"])
    directed_records = sorted(directed_run["records"], key=lambda r: int(r["evaluation_index"]))
    budget = int(CONFIG["search"]["candidate_budget"])
    if len(directed_records) != budget:
        raise RuntimeError("random sampler requires exact 64-slot directed budget")

    progress_path = _progress_path(family, run_index) if checkpoint else None
    if progress_path is not None and progress_path.is_file():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("version") != "V837aj" or state.get("engine") != "RANDOM_STRUCTURAL_SAMPLER":
            raise RuntimeError("invalid random-search progress checkpoint")
        if state.get("family") != family or int(state.get("run_index", -1)) != run_index or state.get("fidelity") != fidelity:
            raise RuntimeError("random-search progress checkpoint identity mismatch")
        records = list(state.get("records", []))
        seen = set(state.get("seen", []))
        next_slot = int(state.get("next_slot", len(records)))
        if state.get("complete") is True:
            return _result(directed_run, fidelity, records)
    else:
        records = []
        seen: set[str] = set()
        next_slot = 0

    for slot in range(next_slot, budget):
        directed = directed_records[slot]
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
        records.append(row)
        seen.add(topology.topology_id)
        _save_progress(progress_path, {
            "version": "V837aj",
            "engine": "RANDOM_STRUCTURAL_SAMPLER",
            "family": family,
            "run_index": run_index,
            "fidelity": fidelity,
            "records": records,
            "seen": sorted(seen),
            "next_slot": slot + 1,
            "complete": False,
        })

    result = _result(directed_run, fidelity, records)
    _save_progress(progress_path, {
        "version": "V837aj",
        "engine": "RANDOM_STRUCTURAL_SAMPLER",
        "family": family,
        "run_index": run_index,
        "fidelity": fidelity,
        "records": records,
        "seen": sorted(seen),
        "next_slot": budget,
        "complete": True,
    })
    return result
