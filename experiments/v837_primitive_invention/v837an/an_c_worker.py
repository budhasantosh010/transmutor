from __future__ import annotations

import argparse
import numpy as np

from .an_a_causal_macrovariables import _carrier_values, _eligibility
from .an_c_distributed_synergy import _load_scan_checkpoint, _save_scan_checkpoint, _trigger_families
from .authorization import MIN_ELIGIBLE, PARTITIONS
from .coalition_scan import minimum_sufficient, scan_organism_coalitions
from .instrumented_af1d import load_population_rows
from .synergy_metrics import characterize
from .trace_cache import clear_memory, pair_traces
from .utils import range_list


def run_worker(shard: int, shards: int) -> int:
    if shards < 1 or shard < 0 or shard >= shards:
        raise ValueError("invalid AN-C shard")
    triggers = _trigger_families()
    competent = [r for r in load_population_rows() if bool(r["competent"])]
    fit_seeds = range_list(PARTITIONS["AN_FIT"])
    select_seeds = range_list(PARTITIONS["AN_SELECT"])
    assigned = [(i, o) for i, o in enumerate(competent, 1) if (i - 1) % shards == shard and triggers[o["family"]]["trigger"]]
    completed = 0
    for index, organism in assigned:
        cached = _load_scan_checkpoint(organism)
        if cached is not None:
            completed += 1
            print(f"V837an AN-C shard {shard}/{shards} {index}/{len(competent)} {organism['organism_id']} cached", flush=True)
            continue
        family = organism["family"]
        fd = pair_traces(organism["organism_id"], family, fit_seeds)
        sd = pair_traces(organism["organism_id"], family, select_seeds)
        fm, _ = _eligibility(fd, family); sm, _ = _eligibility(sd, family)
        fi = np.flatnonzero(fm); si = np.flatnonzero(sm)
        if len(fi) < MIN_ELIGIBLE["AN_FIT"] or len(si) < MIN_ELIGIBLE["AN_SELECT"]:
            print(f"V837an AN-C shard {shard}/{shards} {index}/{len(competent)} {organism['organism_id']} underpowered", flush=True)
            clear_memory()
            continue
        ft = np.asarray([p.primary_phase for p in fd["pairs"]], dtype=np.int64)
        fbase = _carrier_values(fd["base_trace"], "STATE40", ft)[fi]
        fcf = _carrier_values(fd["cf_trace"], "STATE40", ft)[fi]
        reference = np.concatenate([fbase, fcf], axis=0)
        rows = scan_organism_coalitions(sd, si, reference)
        winner = minimum_sufficient(rows)
        syn = characterize(rows, winner)
        _save_scan_checkpoint(organism, rows, winner, syn, len(fi), len(si))
        completed += 1
        print(f"V837an AN-C shard {shard}/{shards} {index}/{len(competent)} {organism['organism_id']} complete", flush=True)
        clear_memory()
    print(f"V837an AN-C shard {shard}/{shards} done: {completed}/{len(assigned)} powered-or-cached scans", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    return run_worker(args.shard, args.shards)


if __name__ == "__main__":
    raise SystemExit(main())
