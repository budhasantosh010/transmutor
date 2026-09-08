from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import (
    architecture_lock,
    build_candidate_model,
    build_exact_anchor_model,
    build_exact_anchor_wrapper,
    common_initialization_equal,
    initialization_fingerprint,
    model_compute,
)
from experiments.v837_primitive_invention.v837aj.fidelity_calibration import (
    CONFIG,
    FAMILIES,
    FINAL_VALIDATION_POOL,
    STAGE_ANCHOR,
    STAGE_CALIBRATION,
    STAGE_RANDOM_SEARCH,
    STAGE_SEARCH,
    aggregate_calibration_scores,
    assert_data_partitions,
    calibration_panel_payload,
    fidelity_metrics,
    fidelity_passes,
    final_validation_seeds,
    full_development_seeds,
    search_selection_seeds,
    select_cheapest_fidelity,
    train_proxy_candidate,
)
from experiments.v837_primitive_invention.v837aj.topology import (
    RECURRENT_UNIVERSE,
    SAME_STEP_UNIVERSE,
    SearchTopology,
    calibration_panel,
    historical_anchor_topology,
    minimal_topology,
    sample_topology_with_counts,
)

HERE = Path(__file__).resolve().parent


def _git_blob_sha256(relative: str) -> str:
    blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def _source_lock() -> dict:
    mapping = {
        "v837af_config": "experiments/v837_primitive_invention/v837af/config.json",
        "v837af_model": "experiments/v837_primitive_invention/v837af/candidate_input_factorization.py",
        "v837af_results": "experiments/v837_primitive_invention/v837af/results.json",
        "v837af_raw_transfer": "experiments/v837_primitive_invention/v837af/raw/transfer_runs.json",
        "v837af_decision": "experiments/v837_primitive_invention/v837af/diagnostics/decision_state.json",
        "v837ai_config": "experiments/v837_primitive_invention/v837ai/config.json",
        "v837ai_results": "experiments/v837_primitive_invention/v837ai/results.json",
        "v837ai_decision": "experiments/v837_primitive_invention/v837ai/diagnostics/decision_state.json",
        "v837y_model": "experiments/v837_primitive_invention/v837y/candidate_interaction.py",
        "v837r_coupling": "experiments/v837_primitive_invention/v837r/recurrent_coupling.py",
        "historical_graph_gate": "experiments/v837_primitive_invention/frozen_gates.json",
    }
    observed = {key: _git_blob_sha256(path) for key, path in mapping.items()}
    expected = CONFIG["source_hashes"]
    return {"compatible": observed == expected, "observed": observed, "expected": expected}


def _final_validation_guard() -> dict:
    from experiments.v837_primitive_invention.v837aj.fidelity_calibration import FinalValidationLeakageError
    attempts = {}
    for stage in (STAGE_CALIBRATION, STAGE_SEARCH, STAGE_RANDOM_SEARCH):
        try:
            final_validation_seeds(stage)
            attempts[stage] = False
        except FinalValidationLeakageError:
            attempts[stage] = True
    anchor_allowed = final_validation_seeds(STAGE_ANCHOR) == list(FINAL_VALIDATION_POOL)
    return {
        "calibration_blocked": attempts[STAGE_CALIBRATION],
        "search_blocked": attempts[STAGE_SEARCH],
        "random_search_blocked": attempts[STAGE_RANDOM_SEARCH],
        "anchor_reproduction_allowed": anchor_allowed,
        "guard_valid": all(attempts.values()) and anchor_allowed,
    }


def _anchor_equivalence() -> dict:
    _configure_torch()
    family = "conditional_routing"; replicate = 0
    source = build_exact_anchor_model(family, replicate)
    wrapped = build_exact_anchor_wrapper(family, replicate)
    source_params = dict(source.named_parameters()); wrapped_params = dict(wrapped.named_parameters())
    parameter_exact = source_params.keys() == wrapped_params.keys() and all(torch.equal(source_params[n], wrapped_params[n]) for n in source_params)
    task = task_by_name(family)
    episodes = [task.generate(seed, "development") for seed in range(10000, 10008)]
    observations, lengths, _ = episodes_to_batch(episodes)
    source.eval(); wrapped.eval()
    with torch.no_grad():
        sp, st = source(observations, lengths, return_trace=True)
        wp, wt = wrapped(observations, lengths, return_trace=True)
    prediction_delta = float(torch.max(torch.abs(sp - wp)).item())
    state_delta = float(torch.max(torch.abs(st.states - wt.states)).item())
    message_delta = float(torch.max(torch.abs(st.messages - wt.messages)).item())
    source_topology = SearchTopology.from_graph(source.graph)
    wrapped_topology = SearchTopology.from_graph(wrapped.graph)
    structural = {
        "edge_count_equal": source_topology.edge_count == wrapped_topology.edge_count == 55,
        "endpoints_temporal_types_equal": source_topology.edges == wrapped_topology.edges,
        "cell_count_equal": len(source.graph.cells) == len(wrapped.graph.cells) == 10,
    }
    return {
        "parameter_exact": parameter_exact,
        "structural": structural,
        "prediction_max_abs_delta": prediction_delta,
        "state_max_abs_delta": state_delta,
        "message_max_abs_delta": message_delta,
        "forward_equivalent": parameter_exact and all(structural.values()) and max(prediction_delta, state_delta, message_delta) <= 1e-6,
    }


def _initialization_pairing() -> dict:
    panel = calibration_panel()
    a = build_candidate_model(panel["P1"], "conditional_routing", 0, 7)
    b = build_candidate_model(panel["P11"], "conditional_routing", 0, 7)
    common_exact = common_initialization_equal(a, b)
    common_edges = set((e.src, e.dst, e.recurrent) for e in a.graph.edges) & set((e.src, e.dst, e.recurrent) for e in b.graph.edges)
    a_edge = {(e.src, e.dst, e.recurrent): p.detach().clone() for e, p in zip(a.graph.edges, a.base.edge_weights)}
    b_edge = {(e.src, e.dst, e.recurrent): p.detach().clone() for e, p in zip(b.graph.edges, b.base.edge_weights)}
    common_edges_exact = all(torch.equal(a_edge[k], b_edge[k]) for k in common_edges)
    # Search/random slot pairing: same slot and same complexity, different structures.
    r1 = sample_topology_with_counts(9, 10, namespace="v837aj-preflight-random-a", parts=(0,))
    r2 = sample_topology_with_counts(9, 10, namespace="v837aj-preflight-random-b", parts=(0,))
    m1 = build_candidate_model(r1, "conditional_routing", 2, 13)
    m2 = build_candidate_model(r2, "conditional_routing", 2, 13)
    search_random_common_exact = common_initialization_equal(m1, m2)
    from experiments.v837_primitive_invention.v837aj.af1d_structural_model import build_finalization_model
    proxy = build_candidate_model(r1, "conditional_routing", 2, 13)
    final = build_finalization_model(r1, "conditional_routing", 2)
    finalization_independent = initialization_fingerprint(proxy, include_edges=False) != initialization_fingerprint(final, include_edges=False)
    return {
        "common_non_edge_parameters_exact": common_exact,
        "common_edge_initialization_exact": common_edges_exact,
        "common_edge_count_tested": len(common_edges),
        "edge_initialization_path_independent": common_edges_exact,
        "search_random_candidate_slot_pairing": search_random_common_exact,
        "no_parent_weight_inheritance": True,
        "finalization_seed_independent_from_proxy_seed": finalization_independent,
        "pairing_valid": common_exact and common_edges_exact and search_random_common_exact and finalization_independent,
    }


def run_preflight() -> int:
    for name in ("raw", "raw/cache", "diagnostics", "plots"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    lock = architecture_lock(); source = _source_lock(); data = assert_data_partitions(); guard = _final_validation_guard()
    topology_universe = {
        "cell_count": 10,
        "same_step_legal_edges": len(SAME_STEP_UNIVERSE),
        "recurrent_legal_edges": len(RECURRENT_UNIVERSE),
        "total_legal_edges": len(SAME_STEP_UNIVERSE) + len(RECURRENT_UNIVERSE),
        "max_topology_edges": int(CONFIG["max_message_edges"]),
        "minimal_topology": minimal_topology().to_dict(),
        "historical_anchor": historical_anchor_topology().to_dict(),
    }
    pairing = _initialization_pairing(); equivalence = _anchor_equivalence(); panel = calibration_panel_payload()
    snapshot = {
        "start_sha": CONFIG["required_start_sha"],
        "protected_through": "V837ai",
        "source_hashes": CONFIG["source_hashes"],
        "fresh_audit_allowed": False,
        "primitive_mining_allowed": False,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics/architecture_lock.json", {**lock, "source_hash_lock": source})
    write_json(HERE / "diagnostics/data_partition_lock.json", data)
    write_json(HERE / "diagnostics/final_validation_guard.json", guard)
    write_json(HERE / "diagnostics/topology_universe.json", topology_universe)
    write_json(HERE / "diagnostics/initialization_pairing.json", pairing)
    write_json(HERE / "diagnostics/anchor_equivalence.json", equivalence)
    write_json(HERE / "diagnostics/protected_historical_snapshot.json", snapshot)
    write_json(HERE / "diagnostics/calibration_panel.json", panel)
    write_json(HERE / "raw/calibration_panel.json", panel)
    ok = lock["compatible"] and source["compatible"] and data["exact"] and data["union_unique_family_seed_episodes"] == 3200 and data["fresh_audit_overlap"] == 0 and guard["guard_valid"] and pairing["pairing_valid"] and equivalence["forward_equivalent"] and len(panel["topologies"]) == 12
    print(json.dumps({"architecture_lock": lock["compatible"], "source_hash_lock": source["compatible"], "data_partition": data["exact"], "validation_guard": guard["guard_valid"], "initialization_pairing": pairing["pairing_valid"], "anchor_equivalence": equivalence["forward_equivalent"], "panel_size": len(panel["topologies"]), "preflight_pass": bool(ok)}, indent=2))
    return 0 if ok else 2


def _historical_af1d_rows() -> dict[tuple[str, int], dict]:
    payload = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/raw/transfer_runs.json").read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    selected = [row for row in rows if row.get("condition") == "AF1D_deshared_candidate_input_factorization"]
    return {(str(row["family"]), int(row["replicate_id"])): row for row in selected}


def _anchor_worker(family: str, replicate: int) -> dict:
    _configure_torch(); task = task_by_name(family); model = build_exact_anchor_wrapper(family, replicate)
    train_seeds = full_development_seeds(); val_seeds = final_validation_seeds(STAGE_ANCHOR)
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in val_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["training"]["learning_rate"]), weight_decay=float(CONFIG["training"]["weight_decay"]))
    loss_fn = nn.MSELoss(); start_wall = time.perf_counter(); start_cpu = time.process_time()
    for _ in range(int(CONFIG["training"]["full_steps"])):
        model.train(); optimizer.zero_grad(set_to_none=True); prediction = model(observations, lengths); loss = loss_fn(prediction, targets); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["training"]["gradient_clip"])); optimizer.step()
    dev = evaluate_sequence_model(model, task, train_episodes); val = evaluate_sequence_model(model, task, validation_episodes)
    return {
        "family": family, "replicate_id": int(replicate), "development_success": float(dev.success_rate), "validation_success": float(val.success_rate), "development_loss": float(dev.loss), "validation_loss": float(val.loss), "parameter_count": model.parameter_count(), "recurrent_controller_projection_macs": model.total_recurrent_controller_projection_macs,
        "optimizer_steps": 192, "processed_examples": 192 * 512, "forward_calls": 194, "backward_calls": 192, "environment_interactions": int(sum(len(ep.observations) for ep in train_episodes + validation_episodes)), "cpu_seconds": float(time.process_time() - start_cpu), "wall_seconds": float(time.perf_counter() - start_wall), "gpu_seconds": 0.0,
    }


def run_anchor() -> int:
    jobs = [(family, replicate) for family in FAMILIES for replicate in range(5)]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_anchor_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); print(f"AJ0 {row['family']} r{row['replicate_id']}: dev={row['development_success']:.6f} val={row['validation_success']:.6f}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["replicate_id"]))
    historical = _historical_af1d_rows(); per_run = []
    for row in rows:
        source = historical[(row["family"], row["replicate_id"])]
        dev_delta = abs(float(row["development_success"]) - float(source["development_success"]))
        val_delta = abs(float(row["validation_success"]) - float(source["validation_success"]))
        per_run.append({"family": row["family"], "replicate_id": row["replicate_id"], "development_delta": dev_delta, "validation_delta": val_delta, "compatible": dev_delta <= 1/512 + 1e-12 and val_delta <= 1/128 + 1e-12})
    family_results = {}; passing = 0
    for family in FAMILIES:
        fr = [r for r in rows if r["family"] == family]
        dev = float(np.median([r["development_success"] for r in fr])); val = float(np.median([r["validation_success"] for r in fr])); passed = capacity_demonstrated(dev, val); passing += int(passed)
        family_results[family] = {"development_median": dev, "validation_median": val, "pass": bool(passed)}
    valid = all(item["compatible"] for item in per_run) and passing == 4
    raw = {"version": "V837aj", "condition": "AJ0_FIXED_AF1D_ANCHOR", "rows": rows, "final_validation_access_stage": STAGE_ANCHOR}
    diag = {"anchor_reproduced": valid, "families_passing": passing, "family_results": family_results, "per_run_compatibility": per_run, "max_development_delta": max(x["development_delta"] for x in per_run), "max_validation_delta": max(x["validation_delta"] for x in per_run)}
    write_json(HERE / "raw/anchor_reproduction.json", raw); write_json(HERE / "diagnostics/anchor_reproduction.json", diag)
    if not valid:
        write_json(HERE / "diagnostics/decision_state.json", {"version":"V837aj","af1d_anchor_valid":False,"fidelity_calibration_complete":False,"selected_search_fidelity":None,"search_stage_allowed":False,"constructive_search_run":False,"primary_runs_per_family":5,"robustness_extension_run":False,"directed_family_passes":None,"random_family_passes":None,"directed_competent_hit_rate":None,"random_competent_hit_rate":None,"paired_validation_delta":None,"paired_permutation_p":None,"automated_structural_discovery":False,"evolutionary_search_superiority":False,"diagnosis":"V837AJ_AF1D_ANCHOR_REPRODUCTION_FAILURE","primitive_mining_allowed_next":False,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False,"next_program":"V837ak_ANCHOR_COMPATIBILITY_REPAIR"})
        return 2
    return 0


def _calibration_worker(fidelity: str, panel_id: str, family: str, replicate: int) -> dict:
    topology = calibration_panel()[panel_id]
    row = train_proxy_candidate(topology, family=family, run_index=1000 + int(replicate), candidate_slot=int(replicate), fidelity=fidelity, stage=STAGE_CALIBRATION)
    row["panel_id"] = panel_id; row["calibration_replicate"] = int(replicate)
    return row


def _refresh_fidelity_aggregate() -> None:
    rows = []
    cache_dir = HERE / "raw/cache"
    for fidelity in ("F_LEGACY", "F0", "F1", "F2", "F3", "F4"):
        path = cache_dir / f"fidelity_{fidelity}.json"
        if path.is_file(): rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    write_json(HERE / "raw/fidelity_runs.json", {"version":"V837aj","rows":rows,"fidelities_present":sorted({r['fidelity'] for r in rows})})


def run_fidelity(fidelity: str) -> int:
    if fidelity not in CONFIG["fidelities"]:
        raise SystemExit(f"unknown fidelity {fidelity}")
    anchor = json.loads((HERE / "diagnostics/anchor_reproduction.json").read_text(encoding="utf-8")) if (HERE / "diagnostics/anchor_reproduction.json").is_file() else {}
    if anchor.get("anchor_reproduced") is not True:
        raise SystemExit("V837aj-A blocked: AF1D anchor reproduction not valid")
    cache = HERE / "raw/cache" / f"fidelity_{fidelity}.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if len(payload.get("rows", [])) == 120:
            print(f"{fidelity}: cache complete; not rerunning")
            _refresh_fidelity_aggregate(); return 0
    jobs = [(fidelity, panel_id, family, replicate) for panel_id in CONFIG["calibration"]["panel_ids"] for family in FAMILIES for replicate in range(2)]
    rows = []
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
    completed = {(str(r["panel_id"]), str(r["family"]), int(r["calibration_replicate"])) for r in rows}
    pending = [job for job in jobs if (job[1], job[2], int(job[3])) not in completed]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_calibration_worker, *job): job for job in pending}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            rows.sort(key=lambda r: (r["panel_id"], r["family"], r["calibration_replicate"]))
            # Persist every completed fit so an external interruption can only
            # lose in-flight workers, never already finished calibration work.
            write_json(cache, {"version":"V837aj","fidelity":fidelity,"rows":rows})
            print(f"{fidelity} {row['panel_id']} {row['family']} r{row['calibration_replicate']}: fit={row['fitness']:.6f} sel={row['selection_success']:.6f}", flush=True)
    rows.sort(key=lambda r: (r["panel_id"], r["family"], r["calibration_replicate"]))
    write_json(cache, {"version":"V837aj","fidelity":fidelity,"rows":rows}); _refresh_fidelity_aggregate(); return 0


def run_decision() -> int:
    required = {"F_LEGACY", "F0", "F1", "F2", "F3", "F4"}
    raw = json.loads((HERE / "raw/fidelity_runs.json").read_text(encoding="utf-8"))
    rows = raw["rows"]; present = {r["fidelity"] for r in rows}
    if present != required or any(sum(r["fidelity"] == f for r in rows) != 120 for f in required):
        raise SystemExit(f"Stage-A incomplete: fidelities present={sorted(present)}")
    computed = fidelity_metrics(rows); metrics = computed["metrics"]; selected = select_cheapest_fidelity(metrics)
    for fidelity, metric in metrics.items(): metric["passes_frozen_gate"] = fidelity in {"F0","F1","F2","F3"} and fidelity_passes(metric)
    proxy_valid = selected is not None
    decision = {"calibration_complete":True,"target_fidelity":"F4","eligible_fidelities":["F0","F1","F2","F3"],"selected_search_fidelity":selected,"proxy_valid":proxy_valid,"metrics":metrics,"diagnosis":"SEARCH_FIDELITY_PROXY_VALID" if proxy_valid else "SEARCH_FIDELITY_PROXY_INVALID"}
    write_json(HERE / "diagnostics/fidelity_raw_scores.json", {"aggregated_fitness":computed["aggregated_fitness"]})
    write_json(HERE / "diagnostics/fidelity_rank_correlations.json", {f:{"families":m["families"],"median_spearman_rho":m["median_spearman_rho"],"median_kendall_tau":m["median_kendall_tau"],"minimum_family_kendall_tau":m["minimum_family_kendall_tau"],"median_pairwise_order_accuracy":m["median_pairwise_order_accuracy"]} for f,m in metrics.items()})
    write_json(HERE / "diagnostics/fidelity_topk_recall.json", {f:{"median_top4_recall":m["median_top4_recall"],"minimum_family_top4_recall":m["minimum_family_top4_recall"],"median_top2_recall":m["median_top2_recall"]} for f,m in metrics.items()})
    write_json(HERE / "diagnostics/fidelity_decision.json", decision)
    anchor = json.loads((HERE / "diagnostics/anchor_reproduction.json").read_text(encoding="utf-8"))
    state = {"version":"V837aj","af1d_anchor_valid":anchor.get("anchor_reproduced") is True,"fidelity_calibration_complete":True,"selected_search_fidelity":selected,"search_stage_allowed":proxy_valid,"constructive_search_run":False,"primary_runs_per_family":5,"robustness_extension_run":False,"directed_family_passes":None,"random_family_passes":None,"directed_competent_hit_rate":None,"random_competent_hit_rate":None,"paired_validation_delta":None,"paired_permutation_p":None,"automated_structural_discovery":False,"evolutionary_search_superiority":False,"diagnosis":"" if proxy_valid else "SEARCH_FIDELITY_PROXY_INVALID","primitive_mining_allowed_next":False,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False,"next_program":"" if proxy_valid else "V837ak_SEARCH_FIDELITY_REDESIGN"}
    write_json(HERE / "diagnostics/decision_state.json", state)
    print(json.dumps(decision, indent=2)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("preflight","anchor","fidelity","decision"), required=True); parser.add_argument("--fidelity", choices=("F_LEGACY","F0","F1","F2","F3","F4"))
    args = parser.parse_args()
    if args.phase == "preflight": return run_preflight()
    if args.phase == "anchor": return run_anchor()
    if args.phase == "fidelity":
        if not args.fidelity: raise SystemExit("--fidelity required")
        return run_fidelity(args.fidelity)
    return run_decision()

if __name__ == "__main__": raise SystemExit(main())
