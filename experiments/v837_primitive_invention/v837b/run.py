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

from experiments.v837_primitive_invention.common.evaluator import graph_and_dynamic_descriptors
from experiments.v837_primitive_invention.common.metrics import binary_summary, continuous_summary
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting
from experiments.v837_primitive_invention.common.seeds import frozen_gates, gate_sha256
from experiments.v837_primitive_invention.common.serialization import load_model_bundle, save_model_bundle, write_json
from experiments.v837_primitive_invention.common.trainer import TrainedCandidate, evaluate_model, refine_candidate_full_adamw
from experiments.v837_primitive_invention.tasks import task_by_name

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "v837" / "results.json"
MODELS = HERE / "models"
PLOTS = HERE / "plots"
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"


def _candidate_from_bundle(path: str, family: str, train_seeds: list[int], validation_seeds: list[int]) -> TrainedCandidate:
    model, _ = load_model_bundle(ROOT / path)
    task = task_by_name(family)
    development = evaluate_model(model, task, [task.generate(seed, "development") for seed in train_seeds])
    validation = evaluate_model(model, task, [task.generate(seed, "validation") for seed in validation_seeds])
    return TrainedCandidate(model.graph, model, development, validation, ResourceAccounting())


def _worker(row: dict, refinement: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    family = row["task_family"]
    task = task_by_name(family)
    train_seeds = list(row["train_seeds"])
    validation_seeds = list(row["validation_seeds"])
    evolved = _candidate_from_bundle(row["best_model_path"], family, train_seeds, validation_seeds)
    random = _candidate_from_bundle(row["random_model_path"], family, train_seeds, validation_seeds)
    refined = refine_candidate_full_adamw(
        evolved, task, train_seeds, validation_seeds,
        steps=int(refinement["steps"]), learning_rate=float(refinement["learning_rate"]), weight_decay=float(refinement["weight_decay"]),
    )
    refined_random = refine_candidate_full_adamw(
        random, task, train_seeds, validation_seeds,
        steps=int(refinement["steps"]), learning_rate=float(refinement["learning_rate"]), weight_decay=float(refinement["weight_decay"]),
    )
    MODELS.mkdir(parents=True, exist_ok=True)
    best_path = MODELS / f"{family}_run{row['run_index']:02d}_best.pt"
    random_path = MODELS / f"{family}_run{row['run_index']:02d}_random.pt"
    save_model_bundle(best_path, refined.model, {"family": family, "run_index": row["run_index"], "condition": "EVOLVED_REFINED"})
    save_model_bundle(random_path, refined_random.model, {"family": family, "run_index": row["run_index"], "condition": "RANDOM_MATCHED_REFINED"})
    return {
        "task_family": family,
        "run_index": row["run_index"],
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "best_graph": refined.graph.to_dict(),
        "development": refined.development.to_dict(),
        "validation": refined.validation.to_dict(),
        "random_matched": {
            "graph": refined_random.graph.to_dict(),
            "development": refined_random.development.to_dict(),
            "validation": refined_random.validation.to_dict(),
            "resources": refined_random.resources.to_dict(),
        },
        "resources": refined.resources.to_dict(),
        "descriptors": graph_and_dynamic_descriptors(refined.model, task, validation_seeds, "validation"),
        "random_descriptors": graph_and_dynamic_descriptors(refined_random.model, task, validation_seeds, "validation"),
        "best_model_path": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "random_model_path": str(random_path.relative_to(ROOT)).replace("\\", "/"),
        "inherited_parent_graph_id": row["best_graph"]["graph_id"],
    }


def _aggregate(rows: list[dict], parent: dict, config: dict) -> dict:
    gates = frozen_gates()
    per_family = {}
    family_passes = 0
    final_cells = []
    evolved_all = []
    random_all = []
    for family in parent["task_families"]:
        family_rows = sorted([row for row in rows if row["task_family"] == family], key=lambda row: row["run_index"])
        dev_success = [row["development"]["success_rate"] >= gates["v837"]["development_success_rate_per_family"] for row in family_rows]
        val_success = [row["validation"]["success_rate"] >= gates["v837"]["heldout_validation_success_rate_per_family"] for row in family_rows]
        evolved = [row["validation"]["success_rate"] for row in family_rows]
        random = [row["random_matched"]["validation"]["success_rate"] for row in family_rows]
        cells = [len(row["best_graph"]["cells"]) for row in family_rows]
        dev_summary = binary_summary(dev_success)
        val_summary = binary_summary(val_success)
        family_pass = dev_summary["success_rate"] >= 0.90 and val_summary["success_rate"] >= 0.85
        family_passes += int(family_pass)
        per_family[family] = {
            "development_run_gate": dev_summary,
            "validation_run_gate": val_summary,
            "validation_episode_success": continuous_summary(evolved),
            "random_matched_validation_success": continuous_summary(random),
            "random_gap_points": 100.0 * (float(np.mean(evolved)) - float(np.mean(random))),
            "final_cells": continuous_summary(cells),
            "pass": family_pass,
        }
        final_cells.extend(cells)
        evolved_all.extend(evolved)
        random_all.extend(random)
    gap = 100.0 * (float(np.mean(evolved_all)) - float(np.mean(random_all)))
    median_cells = float(np.median(final_cells))
    size_pass = median_cells < gates["v837"]["median_final_graph_fraction_of_cap_max"] * gates["substrate"]["max_cells"]
    random_pass = gap >= gates["v837"]["random_matched_graph_min_gap_points"]
    passed = parent["metrics"]["benchmark_validity"]["pass"] and family_passes >= gates["v837"]["families_required"] and size_pass and random_pass
    inherited = parent["resource_accounting"]
    refinement_steps = sum(row["resources"]["optimizer_steps"] + row["random_matched"]["resources"]["optimizer_steps"] for row in rows)
    refinement_env = sum(row["resources"]["environment_steps"] + row["random_matched"]["resources"]["environment_steps"] for row in rows)
    refinement_wall = sum(row["resources"]["wall_seconds"] + row["random_matched"]["resources"]["wall_seconds"] for row in rows)
    return {
        "version": "V837b",
        "parent": "V837",
        "research_question": "Does full continuous-parameter refinement rescue competence of the exact structures found by V837 without changing topology, tasks, seeds, structural search, or pass gates?",
        "hypothesis": "If V837 mainly failed because readout-only adaptation underfit useful recurrent dynamics, equal full-parameter refinement of evolved and matched-random graphs should raise at least four families through the unchanged competence gate while retaining a structural advantage.",
        "single_change": config["single_change"],
        "substrate_version": "neutral_cell_v1",
        "gate_file_sha256": config["gate_file_sha256"],
        "task_families": parent["task_families"],
        "development_seeds": parent["development_seeds"],
        "validation_seeds": parent["validation_seeds"],
        "fresh_audit_seeds": parent["fresh_audit_seeds"],
        "baselines": parent["baselines"],
        "metrics": {
            "benchmark_validity": parent["metrics"]["benchmark_validity"],
            "per_family": per_family,
            "families_passing": family_passes,
            "median_final_cells": median_cells,
            "size_gate_pass": size_pass,
            "random_matched_gap_points": gap,
            "random_matched_gate_pass": random_pass,
        },
        "resource_accounting": {
            "inherited_v837_candidate_evaluations": inherited["candidate_evaluations"],
            "inherited_v837_optimizer_steps": inherited["optimizer_steps"],
            "added_refinement_optimizer_steps": refinement_steps,
            "added_refinement_environment_steps": refinement_env,
            "added_refinement_wall_seconds_sum_workers": refinement_wall,
            "candidate_evaluations": inherited["candidate_evaluations"],
            "optimizer_steps": inherited["optimizer_steps"] + refinement_steps,
            "environment_steps": inherited["environment_steps"] + refinement_env,
            "wall_seconds_sum_workers": inherited["wall_seconds_sum_workers"] + refinement_wall,
            "peak_cells": inherited["peak_cells"],
            "peak_edges": inherited["peak_edges"],
            "final_cells": median_cells,
            "final_edges": inherited["final_edges"],
        },
        "motifs": [],
        "primitive_archive": {},
        "pass_gate": gates["v837"],
        "pass": bool(passed),
        "failure_classification": [] if passed else ["SEARCH_FAILURE"],
        "caveats": [
            "Structural search still ranked candidates using V837 readout-only adaptation; this variant isolates post-search parameter optimization rather than jointly optimizing structure and all weights.",
            "The same full-parameter refinement budget is applied to evolved and matched-random controls."
        ],
        "next_question": "If competence passes, test differentiation on heldout runs. If it fails, test whether stronger structural search rather than parameter refinement is the limiting factor.",
        "runs": rows,
    }


def _write_plot(results: dict) -> None:
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    families = results["task_families"]
    evolved = [results["metrics"]["per_family"][family]["validation_episode_success"]["mean"] for family in families]
    random = [results["metrics"]["per_family"][family]["random_matched_validation_success"]["mean"] for family in families]
    x = np.arange(len(families))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, evolved, width, label="evolved + full refinement")
    ax.bar(x + width / 2, random, width, label="matched random + full refinement")
    ax.set_xticks(x, families, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("heldout success rate")
    ax.set_title("V837b: equal full-parameter refinement after V837 search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "v837b_competence_random_control.png", dpi=160)
    plt.close(fig)


def main() -> int:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if gate_sha256() != EXPECTED_GATE_HASH or config["gate_file_sha256"] != EXPECTED_GATE_HASH:
        raise SystemExit("frozen gate hash mismatch")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    source_rows = parent["runs"]
    workers = min(int(config["compute"]["workers"]), os.cpu_count() or 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, row, config["refinement"]): (row["task_family"], row["run_index"]) for row in source_rows}
        for future in as_completed(futures):
            family, run_index = futures[future]
            row = future.result()
            rows.append(row)
            print(f"completed {family} run {run_index:02d}: dev={row['development']['success_rate']:.3f} val={row['validation']['success_rate']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["task_family"], row["run_index"]))
    results = _aggregate(rows, parent, config)
    write_json(HERE / "results.json", results)
    _write_plot(results)
    status = HERE / ("PASS.md" if results["pass"] else "FAILURE.md")
    if results["pass"]:
        status.write_text("# V837b PASS\n\nFull continuous-parameter refinement rescued the neutral-substrate competence prerequisite under the unchanged scientific gate. Motif claims are still not made here.\n", encoding="utf-8")
        return 0
    failed = [family for family, row in results["metrics"]["per_family"].items() if not row["pass"]]
    status.write_text(
        "# V837b FAILURE\n\n"
        f"WHAT failed? The unchanged V837 competence gate still failed after full-parameter refinement; failing families: {failed}.\n\n"
        "WHERE? On the exact V837-selected topologies and matched-random controls.\n\n"
        "WHEN? After structural search, before any motif mining.\n\n"
        "WHY suspected? If refinement materially improves absolute performance but not the family-level reliability gate, structural search/representation rather than readout optimization remains limiting.\n\n"
        "HOW reproduced? Same V837 graphs, same paired seeds, same gate, equal 48-step all-parameter AdamW refinement for evolved and random controls.\n\n"
        "WHAT evidence supports diagnosis? `results.json` records before-lineage resource inheritance and every refined paired result.\n\n"
        "WHAT alternatives were ruled out? Benchmark validity, first-observation leakage, and simple readout-only underfitting.\n\n"
        "WHAT single change next? V837c strengthens structural search breadth only while returning to the original readout-only candidate adaptation.\n",
        encoding="utf-8",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
