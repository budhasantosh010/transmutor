from __future__ import annotations

import math
from pathlib import Path

from .authorization import PARTITIONS
from .trace_cache import cache_manifest
from .utils import HERE, read_json, write_json

RAW = HERE / "raw"
DIAG = HERE / "diagnostics"


def _read(rel: str, default):
    p = HERE / rel
    return read_json(p) if p.is_file() else default


def _count_selected_rows(payload: dict) -> int:
    return len(payload.get("organism_results", []))


def _partition_seed_ids(final_validation_run: bool) -> list[int]:
    names = ["AN_FIT", "AN_SELECT", "AN_ROUTING_FIT", "AN_ROUTING_SELECT", "AN_META_CONFIRM", "AN_FINAL_DEV"]
    if final_validation_run:
        names.append("AN_FINAL_VALIDATION")
    ids: set[int] = set()
    for name in names:
        lo, hi = PARTITIONS[name]
        ids.update(range(int(lo), int(hi) + 1))
    return sorted(ids)


def compute_resource_accounting() -> dict:
    a = _read("raw/an_a_selection.json", {})
    b = _read("raw/routing_selection.json", {})
    c = _read("raw/synergy_results.json", {})
    meta = _read("raw/meta_confirmation.json", {})
    dev = _read("raw/final_dev_confirmation.json", {})
    val = _read("raw/final_validation.json", {})
    runtime_obs = _read("diagnostics/an_a_runtime_observation.json", {})
    a_diag = _read("diagnostics/phase_stability.json", {})

    a_rows = len(a.get("organism_results", []))
    a_powered = len({r.get("organism_id") for r in a.get("organism_results", []) if r.get("select_eligible", 0)})
    # Each AN-A organism/config performs source-swap f/r, a batched 32-random
    # control f/r, compiler f/r, shuffled-semantic, same-norm-noise, and
    # orthogonal-residual forwards: nine instrumented forwards/config.
    an_a_causal_patch_calls = 9 * a_rows
    an_a_random_control_calls = 5 * a_rows
    an_a_svds = 2 * a_rows
    an_a_ridge_solves = 2 * a_rows
    an_a_permutation_tests = 2 * a_rows

    # Secondary AN-A diagnostics run in one process and reuse FIT/SELECT traces
    # through trace_cache. Their intervention calls are still real compute and
    # must not disappear from accounting just because they are diagnostic-only.
    phase_rows = [r for r in a_diag.get("phase_rows", []) if r.get("powered")]
    routing_selected_rows = [r for r in a_diag.get("routing_selected_value", []) if r.get("run")]
    coupling_rows = list(a_diag.get("coupling_factor_diagnostic", []))
    single_cell_rows = list(a_diag.get("single_cell_state_controls", []))
    diagnostic_candidate_patch_calls = 2 * (len(phase_rows) + len(routing_selected_rows) + len(single_cell_rows))
    diagnostic_random_control_calls = 2 * (len(phase_rows) + len(routing_selected_rows) + len(single_cell_rows))
    diagnostic_svds = len(routing_selected_rows) + len(coupling_rows) + len(single_cell_rows)
    diagnostic_ridge_solves = len(coupling_rows)
    diagnostic_permutation_tests = len(phase_rows) + len(routing_selected_rows) + len(single_cell_rows)
    diagnostic_trace_organisms = len(_read("raw/counterfactual_power.json", {}).get("rows", [])) if a_diag else 0

    b_powered_rows = [r for r in b.get("organism_results", []) if r.get("powered")]
    b_routing_calls = 0
    b_permutation_tests = 0
    for row in b_powered_rows:
        ncfg = len(row.get("configs", {}))
        # singleton ranking f/r + all frozen configs f/r + 64 matched controls
        # f/r per config. Candidate sets are internally batch-vectorized.
        b_routing_calls += 4 + 2 * ncfg
        b_permutation_tests += ncfg

    c_scans = _read("raw/coalition_scan.json", {}).get("scans", [])
    coalition_forward_calls = 64 * len(c_scans)  # ceil(1023/32)=32 chunks × f/r
    coalition_permutation_tests = 1023 * len(c_scans)

    # Trace generation is performed once per organism/partition inside each
    # standalone stage process. Count base and counterfactual forwards
    # separately rather than counting episodes.
    an_a_orgs = len(_read("raw/counterfactual_power.json", {}).get("rows", []))
    an_b_orgs = len(b.get("organism_results", []))
    an_c_orgs = len(c.get("organism_results", []))
    meta_org_partitions = len({(r.get("family"), x.get("organism_id")) for r in meta.get("results", []) for x in r.get("organism_results", [])})
    dev_org_partitions = len({(r.get("family"), x.get("organism_id")) for r in dev.get("results", []) for x in r.get("organism_results", [])})
    val_org_partitions = len({(r.get("family"), x.get("organism_id")) for r in val.get("results", []) for x in r.get("organism_results", [])}) if val.get("run") else 0
    pair_trace_partition_calls = 2 * an_a_orgs + 2 * diagnostic_trace_organisms + 2 * an_b_orgs + 2 * an_c_orgs + meta_org_partitions + dev_org_partitions + val_org_partitions
    confirmation_results = list(meta.get("results", [])) + list(dev.get("results", [])) + (list(val.get("results", [])) if val.get("run") else [])
    confirmation_powered_organism_evaluations = sum(int(r.get("powered_organisms", 0) or 0) for r in confirmation_results)
    # Every frozen confirmation evaluation, regardless of branch, uses two
    # candidate-direction interventions plus two matched-control batched
    # interventions per powered organism, with one paired permutation test.
    confirmation_intervention_forward_calls = 4 * confirmation_powered_organism_evaluations
    confirmation_permutation_tests = confirmation_powered_organism_evaluations

    seed_ids = _partition_seed_ids(bool(val.get("run")))
    cache = cache_manifest()
    wall_seconds = sum(float(x.get("wall_seconds", 0.0) or 0.0) for x in (a, b, c, meta, dev, val))
    cpu_seconds = sum(float(x.get("cpu_seconds", 0.0) or 0.0) for x in (a, b, c, meta, dev, val))
    cpu_exact = True
    cpu_lower_bound = cpu_seconds
    if not a.get("cpu_seconds"):
        exact = runtime_obs.get("an_a_cpu_seconds_exact")
        lower = runtime_obs.get("an_a_cpu_seconds_lower_bound")
        if exact is not None:
            cpu_seconds += float(exact); cpu_lower_bound += float(exact)
        elif lower is not None:
            cpu_exact = False; cpu_lower_bound += float(lower)

    payload = {
        "version": "V837an",
        "new_organism_fits": 0,
        "optimizer_steps": 0,
        "adapter_gradient_steps": 0,
        "processed_training_examples": 0,
        "natural_trace_forward_calls": pair_trace_partition_calls,
        "counterfactual_trace_forward_calls": pair_trace_partition_calls,
        "causal_patch_forward_calls": an_a_causal_patch_calls + diagnostic_candidate_patch_calls,
        "random_control_forward_calls": an_a_random_control_calls + diagnostic_random_control_calls,
        "routing_forward_calls": b_routing_calls,
        "coalition_forward_calls": coalition_forward_calls,
        "confirmation_intervention_forward_calls": confirmation_intervention_forward_calls,
        "svd_count": an_a_svds + diagnostic_svds,
        "ridge_solve_count": an_a_ridge_solves + diagnostic_ridge_solves,
        "permutation_tests": an_a_permutation_tests + diagnostic_permutation_tests + b_permutation_tests + coalition_permutation_tests + confirmation_permutation_tests,
        "secondary_diagnostic_forward_calls": diagnostic_candidate_patch_calls + diagnostic_random_control_calls,
        "secondary_diagnostic_svds": diagnostic_svds,
        "secondary_diagnostic_ridge_solves": diagnostic_ridge_solves,
        "secondary_diagnostic_permutation_tests": diagnostic_permutation_tests,
        "cpu_seconds": cpu_seconds if cpu_exact and cpu_seconds > 0 else None,
        "cpu_seconds_exact": cpu_exact,
        "cpu_seconds_lower_bound": cpu_lower_bound if cpu_lower_bound > 0 else None,
        "an_a_cpu_accounting_note": runtime_obs.get("reason_exact_cpu_unavailable"),
        "wall_seconds": wall_seconds if wall_seconds > 0 else None,
        "gpu_seconds": 0.0,
        "cache_hits": None,
        "cache_misses": pair_trace_partition_calls,
        "cache_manifest": cache,
        "unique_task_seed_ids_consumed": len(seed_ids),
        "task_seed_id_min": min(seed_ids) if seed_ids else None,
        "task_seed_id_max": max(seed_ids) if seed_ids else None,
        "fresh_audit_seed_ids_consumed": 0,
        "forward_call_accounting_semantics": "One model invocation counts as one forward call even when candidate/control conditions are batch-vectorized within that invocation.",
        "an_a_config_organism_rows": a_rows,
        "an_a_powered_organisms": a_powered,
        "an_a_phase_diagnostic_rows": len(phase_rows),
        "an_a_routing_selected_diagnostic_rows": len(routing_selected_rows),
        "an_a_coupling_factor_diagnostic_rows": len(coupling_rows),
        "an_a_single_cell_control_rows": len(single_cell_rows),
        "an_b_powered_organisms": len(b_powered_rows),
        "an_c_scanned_organisms": len(c_scans),
        "confirmation_powered_organism_evaluations": confirmation_powered_organism_evaluations,
    }
    write_json(DIAG / "resource_accounting.json", payload)
    write_json(HERE / "v837an_resource_accounting.json", payload)
    write_json(HERE.parent / "v837an_resource_accounting.json", payload)
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(compute_resource_accounting(), indent=2))
