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
from experiments.v837_primitive_invention.common.metrics import continuous_summary, paired_bootstrap_difference
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


def _worker(family: str, replicate: int) -> dict:
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
        state_update_mode="learned_leaky",
        alpha_init=float(CONFIG["alpha_init"]),
    )
    diagnostic_episodes = [task.generate(seed, "validation") for seed in VALIDATION_SEEDS[:32]]
    diagnostics = representation_diagnostics(trained.model, task, diagnostic_episodes, include_cell_ablations=True)
    return {
        "condition": "learned_alpha",
        "family": family,
        "replicate": replicate,
        "requested_density": 1.0,
        "effective_density": 1.0,
        "development_success": trained.development.success_rate,
        "development_loss": trained.development.loss,
        "validation_success": trained.validation.success_rate,
        "validation_loss": trained.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(trained.development.success_rate, trained.validation.success_rate),
        "alpha_coefficients": trained.model.state_update_coefficients(),
        "representation_diagnostics": diagnostics,
        "resources": trained.resources.to_dict(),
    }


def main() -> int:
    jobs = [(family, replicate) for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates_per_family"]))]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, family, replicate): (family, replicate) for family, replicate in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"learned_alpha {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f} alpha={np.mean(row['alpha_coefficients']):.3f}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["replicate"]))
    learned = summarize_condition(rows, seed=8377)
    learned["alpha_coefficients"] = continuous_summary([alpha for row in rows for alpha in row["alpha_coefficients"]])

    historical = json.loads((ROOT / "experiments/v837_primitive_invention/v837d/diagnostics/broadcast_capacity.json").read_text(encoding="utf-8"))
    control = historical["summary"]
    paired = {}
    historical_rows = historical["rows"]
    for family in FAMILIES:
        left = sorted([r for r in rows if r["family"] == family], key=lambda r: r["replicate"])
        right = sorted([r for r in historical_rows if r["family"] == family], key=lambda r: r["replicate"])
        paired[family] = paired_bootstrap_difference(
            np.asarray([r["validation_success"] for r in left]),
            np.asarray([r["validation_success"] for r in right]),
            seed=deterministic_int("v837g", family),
        )

    passed = int(learned["families_passing_aggregate"]) >= int(CONFIG["pass_gate"]["families_required"])
    payload = {
        "version": "V837g",
        "parent": "V837d",
        "single_change": CONFIG["single_change"],
        "representation_change": CONFIG["representation_change"],
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "conditions": {"historical_alpha_1": control, "learned_alpha": learned},
        "paired_validation_delta_learned_minus_historical": paired,
        "capacity_results": learned["per_family"],
        "representation_diagnostics": {
            "alpha_coefficients": learned["alpha_coefficients"],
            "message_dependency_ratio": learned["representation_diagnostics"]["message_dependency_ratio"],
            "pairwise_state_correlation": learned["representation_diagnostics"]["mean_pairwise_state_corr"],
            "state_saturation_fraction": learned["representation_diagnostics"]["state_saturation_fraction"],
        },
        "resource_accounting": learned["resource_accounting"],
        "pass_gate": CONFIG["pass_gate"],
        "pass": passed,
        "failure_classification": [] if passed else ["STATE_UPDATE_FAILURE", "CAPACITY_WITHOUT_GENERALIZATION"],
        "interpretation": "Learned generic state persistence restored the high-capacity representation screen." if passed else "Learned generic state persistence did not restore >=4/5 high-capacity competence under the unchanged gate.",
        "next_experiment": "Freeze this representation and rerun the original full neutral structural search." if passed else "Test one low-rank multiplicative interaction branch with a parameter-matched additive control (V837h).",
    }
    write_json(HERE / "results.json", payload)
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    doc.write_text(
        ("# V837g PASS\n\nThis is a representation-recovery candidate only, not a primitive-invention PASS.\n" if passed else "# V837g FAILURE\n\n")
        + f"Families passing aggregate capacity gate: {learned['families_passing_aggregate']}/5.\n\n"
        + payload["interpretation"] + "\n\nFresh-audit seeds consumed: 0. Primitive promotions: 0.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
