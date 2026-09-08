from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.v837aj.fidelity_calibration import FAMILIES
from experiments.v837_primitive_invention.v837aj.random_structural_sampler import run_random_sampler
from experiments.v837_primitive_invention.v837aj.structural_search import require_valid_proxy, run_directed_search

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CACHE = HERE / "raw/cache"


def _extension_allowed() -> bool:
    path = HERE / "diagnostics/primary_decision.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("robustness_extension_required") is True


def _run_indices(extension: bool) -> range:
    return range(5, 10) if extension else range(0, 5)


def _directed_worker(family: str, run_index: int, fidelity: str) -> dict:
    return run_directed_search(family, run_index, fidelity, checkpoint=True)


def _random_worker(directed_run: dict, fidelity: str) -> dict:
    return run_random_sampler(directed_run, fidelity, checkpoint=True)


def _paired_worker(family: str, run_index: int, fidelity: str) -> tuple[dict, dict]:
    directed = run_directed_search(family, run_index, fidelity, checkpoint=True)
    random = run_random_sampler(directed, fidelity, checkpoint=True)
    return directed, random


def _cache_path(engine: str, family: str, run_index: int) -> Path:
    return CACHE / f"{engine}_{family}_{int(run_index)}.json"


def _load_complete(engine: str, family: str, run_index: int) -> dict | None:
    path = _cache_path(engine, family, run_index)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("candidate_budget", -1)) != 64 or len(payload.get("records", [])) != 64:
        return None
    if len({row["topology_id"] for row in payload["records"]}) != 64:
        return None
    return payload


def _aggregate_proxy(engine: str) -> None:
    runs = []
    for family in FAMILIES:
        for run_index in range(10):
            payload = _load_complete(engine, family, run_index)
            if payload is not None:
                runs.append(payload)
    runs.sort(key=lambda row: (row["family"], int(row["run_index"])))
    name = "search_proxy_runs.json" if engine == "search" else "random_proxy_runs.json"
    write_json(HERE / "raw" / name, {"version":"V837aj","engine":engine,"runs":runs,"candidate_evaluations":sum(len(run["records"]) for run in runs)})
    champions = []
    for run in runs:
        champions.append({"family":run["family"],"run_index":run["run_index"],"engine":run["engine"],**run["champion"]})
    champion_name = "search_champions.json" if engine == "search" else "random_champions.json"
    write_json(HERE / "raw" / champion_name, {"version":"V837aj","champions":champions,"champions_frozen_before_final_validation":True})


def run_directed(extension: bool) -> int:
    fidelity = require_valid_proxy()
    if extension and not _extension_allowed():
        raise SystemExit("V837aj robustness directed search blocked: primary machine trigger absent")
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for family in FAMILIES:
        for run_index in _run_indices(extension):
            if _load_complete("search", family, run_index) is None:
                jobs.append((family, run_index, fidelity))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(25, max(1, int(os.environ.get("V837AJ_MAX_WORKERS", os.cpu_count() or 1))))) as pool:
            futures = {pool.submit(_directed_worker, *job): job for job in jobs}
            for future in as_completed(futures):
                run = future.result(); path = _cache_path("search", run["family"], run["run_index"]); write_json(path, run)
                print(f"directed {run['family']} r{run['run_index']}: champion fit={run['champion']['search_fitness']:.6f} sel={run['champion']['search_selection_success']:.6f} edges={run['champion']['edge_count']}", flush=True)
    _aggregate_proxy("search")
    return 0


def run_random(extension: bool) -> int:
    fidelity = require_valid_proxy()
    if extension and not _extension_allowed():
        raise SystemExit("V837aj robustness random search blocked: primary machine trigger absent")
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for family in FAMILIES:
        for run_index in _run_indices(extension):
            directed = _load_complete("search", family, run_index)
            if directed is None:
                raise SystemExit(f"random baseline blocked: missing directed slot template {family} r{run_index}")
            if _load_complete("random", family, run_index) is None:
                jobs.append((directed, fidelity))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(25, max(1, int(os.environ.get("V837AJ_MAX_WORKERS", os.cpu_count() or 1))))) as pool:
            futures = {pool.submit(_random_worker, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                run = future.result(); path = _cache_path("random", run["family"], run["run_index"]); write_json(path, run)
                print(f"random {run['family']} r{run['run_index']}: champion fit={run['champion']['search_fitness']:.6f} sel={run['champion']['search_selection_success']:.6f} edges={run['champion']['edge_count']}", flush=True)
    _aggregate_proxy("random")
    return 0


def run_paired(extension: bool) -> int:
    fidelity = require_valid_proxy()
    if extension and not _extension_allowed():
        raise SystemExit("V837aj robustness paired search blocked: primary machine trigger absent")
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for family in FAMILIES:
        for run_index in _run_indices(extension):
            if _load_complete("random", family, run_index) is None:
                jobs.append((family, run_index, fidelity))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(25, max(1, int(os.environ.get("V837AJ_MAX_WORKERS", os.cpu_count() or 1))))) as pool:
            futures = {pool.submit(_paired_worker, *job): job for job in jobs}
            for future in as_completed(futures):
                directed, random = future.result()
                write_json(_cache_path("search", directed["family"], directed["run_index"]), directed)
                write_json(_cache_path("random", random["family"], random["run_index"]), random)
                print(
                    f"paired {directed['family']} r{directed['run_index']}: "
                    f"directed sel={directed['champion']['search_selection_success']:.6f} "
                    f"random sel={random['champion']['search_selection_success']:.6f}",
                    flush=True,
                )
    _aggregate_proxy("search")
    _aggregate_proxy("random")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("directed","random","paired"), required=True); parser.add_argument("--extension", action="store_true")
    args = parser.parse_args()
    if args.phase == "directed": return run_directed(bool(args.extension))
    if args.phase == "random": return run_random(bool(args.extension))
    return run_paired(bool(args.extension))

if __name__ == "__main__": raise SystemExit(main())
