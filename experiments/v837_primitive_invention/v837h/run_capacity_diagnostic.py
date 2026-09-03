from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.evaluator import representation_diagnostics
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.graph import build_input_access_spec
from experiments.v837_primitive_invention.common.metrics import paired_bootstrap_difference
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import train_graph
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837d.experiment import summarize_condition

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
TRAIN_SEEDS = list(range(30000, 30128))
VALIDATION_SEEDS = list(range(30200, 30328))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = ["parameter_matched_additive", "low_rank_multiplicative"]


def _worker(condition: str, family: str, replicate: int) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    task = task_by_name(family)
    base_graph = high_capacity_generic_graph(replicate)
    graph = base_graph.clone()
    graph.input_access = build_input_access_spec("broadcast", 6, 10, density=1.0, seed=0)
    run_seed = deterministic_int("blocker-reference-train", family, replicate)
    init_seed = deterministic_int("train", base_graph.graph_id, task.name, run_seed)
    trained = train_graph(
        graph,
        task,
        TRAIN_SEEDS,
        VALIDATION_SEEDS,
        run_seed=run_seed,
        state_dim=4,
        message_dim=4,
        steps=int(CONFIG["training"]["steps"]),
        learning_rate=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
        training_scope="full_adamw",
        initialization_seed_override=init_seed,
        state_update_mode="direct",
        interaction_mode=condition,
        interaction_rank=int(CONFIG["interaction_rank"]),
    )
    diagnostic_episodes = [task.generate(seed, "validation") for seed in VALIDATION_SEEDS[:32]]
    diagnostics = representation_diagnostics(trained.model, task, diagnostic_episodes, include_cell_ablations=True)
    return {
        "condition": condition,
        "family": family,
        "replicate": replicate,
        "requested_density": 1.0,
        "effective_density": 1.0,
        "development_success": trained.development.success_rate,
        "development_loss": trained.development.loss,
        "validation_success": trained.validation.success_rate,
        "validation_loss": trained.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(trained.development.success_rate, trained.validation.success_rate),
        "representation_diagnostics": diagnostics,
        "resources": trained.resources.to_dict(),
    }


def _paired(left_rows, right_rows, label):
    out = {}
    for family in FAMILIES:
        left = sorted([r for r in left_rows if r["family"] == family], key=lambda r: r["replicate"])
        right = sorted([r for r in right_rows if r["family"] == family], key=lambda r: r["replicate"])
        out[family] = paired_bootstrap_difference(
            np.asarray([r["validation_success"] for r in left]),
            np.asarray([r["validation_success"] for r in right]),
            seed=deterministic_int("v837h", label, family),
        )
    return out


def main() -> int:
    jobs = [(condition, family, replicate) for condition in CONDITIONS for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates_per_family"]))]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate"]))
    additive_rows = [r for r in rows if r["condition"] == "parameter_matched_additive"]
    mult_rows = [r for r in rows if r["condition"] == "low_rank_multiplicative"]
    additive = summarize_condition(additive_rows, seed=83781)
    mult = summarize_condition(mult_rows, seed=83782)
    historical_payload = json.loads((ROOT / "experiments/v837_primitive_invention/v837d/diagnostics/broadcast_capacity.json").read_text(encoding="utf-8"))
    historical = historical_payload["summary"]
    historical_rows = historical_payload["rows"]
    passed = int(mult["families_passing_aggregate"]) >= int(CONFIG["pass_gate"]["families_required"])
    mult_vs_add = _paired(mult_rows, additive_rows, "mult-vs-add")
    mult_vs_hist = _paired(mult_rows, historical_rows, "mult-vs-historical")
    same_parameter_count = int(mult["resource_accounting"]["parameter_count"]) == int(additive["resource_accounting"]["parameter_count"])
    historical_params = int(historical["resource_accounting"]["parameter_count"])
    new_params = int(mult["resource_accounting"]["parameter_count"])
    payload = {
        "version": "V837h",
        "parent": "V837g",
        "single_change": CONFIG["single_change"],
        "representation_change": CONFIG["representation_change"],
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "conditions": {
            "historical_additive": historical,
            "parameter_matched_additive": additive,
            "low_rank_multiplicative": mult,
        },
        "parameter_matching": {
            "historical_parameter_count": historical_params,
            "additive_control_parameter_count": int(additive["resource_accounting"]["parameter_count"]),
            "multiplicative_parameter_count": new_params,
            "additive_equals_multiplicative": same_parameter_count,
            "percent_increase_vs_historical": 100.0 * (new_params - historical_params) / historical_params,
        },
        "paired_validation_delta_multiplicative_minus_additive": mult_vs_add,
        "paired_validation_delta_multiplicative_minus_historical": mult_vs_hist,
        "capacity_results": mult["per_family"],
        "representation_diagnostics": {
            "message_dependency_ratio": mult["representation_diagnostics"]["message_dependency_ratio"],
            "pairwise_state_correlation": mult["representation_diagnostics"]["mean_pairwise_state_corr"],
            "state_saturation_fraction": mult["representation_diagnostics"]["state_saturation_fraction"],
        },
        "resource_accounting": {
            "multiplicative": mult["resource_accounting"],
            "parameter_matched_additive": additive["resource_accounting"],
        },
        "pass_gate": CONFIG["pass_gate"],
        "pass": passed,
        "failure_classification": [] if passed else ["INTERACTION_BASIS_FAILURE", "CAPACITY_WITHOUT_GENERALIZATION"],
        "interpretation": "Rank-2 multiplicative interaction restored the high-capacity screen." if passed else "Rank-2 multiplicative interaction did not restore >=4/5 capacity; after V837d and V837g this triggers the representation-recovery stop rule.",
        "next_experiment": "Freeze the recovered interaction substrate and rerun the original full neutral structural search." if passed else "STOP representation stacking and close the recovery line with REPRESENTATION_BLOCKER_ANALYSIS.md.",
        "rows": rows,
    }
    write_json(HERE / "results.json", payload)
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    doc.write_text(
        ("# V837h PASS\n\nRepresentation-recovery candidate only; primitive mining remains blocked until full structural-search competence passes.\n\n" if passed else "# V837h FAILURE\n\n")
        + f"Multiplicative families passing: {mult['families_passing_aggregate']}/5. Parameter-matched additive families passing: {additive['families_passing_aggregate']}/5.\n\n"
        + payload["interpretation"] + "\n\nFresh-audit seeds consumed: 0. Primitive promotions: 0.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
