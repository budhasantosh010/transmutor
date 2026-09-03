from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.graph import CellSpec, EdgeSpec, GraphSpec
from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.seeds import deterministic_int, frozen_gates, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import train_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"


def high_capacity_generic_graph(restart: int) -> GraphSpec:
    """A deliberately overpowered but task-agnostic reference within the same cell substrate.

    It stays under the V837 hard caps and contains no semantic operator/module.
    The topology is identical for every family; only deterministic parameter
    seeds differ across diagnostic restarts.
    """
    n = 10
    cells = [CellSpec(i, param_seed=deterministic_int("blocker-reference-cell", restart, i) % 2_000_000_000) for i in range(n)]
    edges: list[EdgeSpec] = []
    # Dense acyclic message mixing plus one recurrent self edge per cell: 55 edges total.
    for src in range(n):
        for dst in range(src + 1, n):
            edges.append(EdgeSpec(src, dst, weight=0.35, recurrent=False))
    for i in range(n):
        edges.append(EdgeSpec(i, i, weight=0.55, recurrent=True))
    graph = GraphSpec(cells=cells, edges=edges, generation=0, parent_id="BLOCKER_DIAGNOSTIC")
    graph.validate(max_cells=16, max_edges=64)
    return graph


def _worker(family: str, restart: int) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    task = task_by_name(family)
    # Entire diagnostic stays in the predeclared ablation seed region.
    train_seeds = list(range(30000, 30128))
    validation_seeds = list(range(30200, 30328))
    graph = high_capacity_generic_graph(restart)
    result = train_graph(
        graph,
        task,
        train_seeds,
        validation_seeds,
        run_seed=deterministic_int("blocker-reference-train", family, restart),
        state_dim=4,
        message_dim=4,
        steps=192,
        learning_rate=0.005,
        weight_decay=0.0001,
        training_scope="full_adamw",
    )
    return {
        "family": family,
        "restart": restart,
        "graph_id": graph.graph_id,
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "resources": result.resources.to_dict(),
    }


def main() -> int:
    if gate_sha256() != EXPECTED_GATE_HASH:
        raise SystemExit("frozen gate hash mismatch")
    families = [task.name for task in all_tasks()]
    jobs = [(family, restart) for family in families for restart in range(3)]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, family, restart): (family, restart) for family, restart in jobs}
        for future in as_completed(futures):
            family, restart = futures[future]
            row = future.result(); rows.append(row)
            print(f"diagnostic {family} restart {restart}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["restart"]))
    per_family = {}
    capable = 0
    for family in families:
        fr = [row for row in rows if row["family"] == family]
        best = max(fr, key=lambda row: (row["validation_success"], row["development_success"]))
        family_capable = best["development_success"] >= 0.90 and best["validation_success"] >= 0.85
        capable += int(family_capable)
        per_family[family] = {
            "best_restart": best["restart"],
            "best_development_success": best["development_success"],
            "best_validation_success": best["validation_success"],
            "validation_success_distribution": continuous_summary([row["validation_success"] for row in fr]),
            "capacity_demonstrated": family_capable,
        }
    if capable >= 4:
        conclusion = "SEARCH_GENERALIZATION_BOTTLENECK_SUPPORTED"
        classification = "SEARCH_FAILURE"
        explanation = "A single task-agnostic high-capacity topology inside the same neutral-cell substrate solved at least four families after strong parameter fitting, so the substrate can express the required behaviors. The failed evolutionary lineage therefore primarily reflects discovery/generalization efficiency rather than an immediate representational impossibility."
    else:
        conclusion = "SUBSTRATE_CAPACITY_NOT_DEMONSTRATED"
        classification = "REPRESENTATION_FAILURE"
        explanation = "Even a task-agnostic high-capacity reference within the same neutral-cell substrate did not demonstrate competence on four families under strong fitting, so structural-search breadth alone cannot explain the lineage failure; substrate/optimization representation remains the strongest blocker."
    payload = {
        "diagnostic": "V837 three-failure substrate-vs-search capacity probe",
        "gate_file_sha256": EXPECTED_GATE_HASH,
        "seed_region": "ablation 30000-30499 only",
        "reference_graph": {"cells": 10, "edges": 55, "task_specific_topology": False, "named_high_level_operators": False},
        "training": {"scope": "full_adamw", "steps": 192, "train_episodes": 128, "validation_episodes": 128, "restarts_per_family": 3},
        "per_family": per_family,
        "families_capacity_demonstrated": capable,
        "conclusion": conclusion,
        "failure_classification": classification,
        "explanation": explanation,
        "rows": rows,
    }
    write_json(HERE / "blocker_diagnostic_results.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
