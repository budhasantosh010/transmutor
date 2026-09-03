from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import train_graph
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"


def _worker(family: str, restart: int) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    task = task_by_name(family)
    # Single diagnostic change from the first capacity probe: increase fitting data 128 -> 300.
    # Both training and heldout diagnostic seeds remain wholly inside the predeclared ablation region.
    train_seeds = list(range(30000, 30300))
    validation_seeds = list(range(30400, 30500))
    graph = high_capacity_generic_graph(restart)
    result = train_graph(
        graph,
        task,
        train_seeds,
        validation_seeds,
        run_seed=deterministic_int("blocker-data-train", family, restart),
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
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, family, restart): (family, restart) for family in families for restart in range(3)}
        for future in as_completed(futures):
            family, restart = futures[future]
            row = future.result(); rows.append(row)
            print(f"data diagnostic {family} restart {restart}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["restart"]))
    per_family = {}
    capable = 0
    for family in families:
        fr = [row for row in rows if row["family"] == family]
        best = max(fr, key=lambda row: (row["validation_success"], row["development_success"]))
        family_capable = best["development_success"] >= 0.90 and best["validation_success"] >= 0.85
        capable += int(family_capable)
        per_family[family] = {
            "best_development_success": best["development_success"],
            "best_validation_success": best["validation_success"],
            "validation_success_distribution": continuous_summary([row["validation_success"] for row in fr]),
            "capacity_demonstrated": family_capable,
        }
    payload = {
        "diagnostic": "V837 blocker data-size probe",
        "single_change_from_first_blocker_probe": "increase ablation-region fitting episodes from 128 to 300; topology, optimizer, steps, dimensions, and success criterion unchanged",
        "gate_file_sha256": EXPECTED_GATE_HASH,
        "seed_region": "ablation 30000-30499 only; train 30000-30299, heldout 30400-30499",
        "families_capacity_demonstrated": capable,
        "per_family": per_family,
        "rows": rows,
    }
    if capable >= 4:
        payload["conclusion"] = "DATA_LIMITATION_SUPPORTED"
        payload["failure_classification"] = "GENERALIZATION_FAILURE"
        payload["explanation"] = "Increasing fitting data alone allows the same neutral substrate/reference topology to demonstrate at least four-family competence; V837's small per-candidate fitting sample is the strongest isolated bottleneck."
    else:
        payload["conclusion"] = "DATA_INCREASE_DID_NOT_RESCUE_SUBSTRATE_REFERENCE"
        payload["failure_classification"] = "REPRESENTATION_FAILURE"
        payload["explanation"] = "A >2x increase in fitting data did not make the same high-capacity neutral reference competent on four families. The blocker therefore lies deeper than the original small-data budget: the current substrate/optimization dynamics do not reliably represent/generalize the required family set under the tested regime."
    write_json(HERE / "blocker_data_diagnostic_results.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
