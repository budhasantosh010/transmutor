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

from experiments.v837_primitive_invention.common.evaluator import (
    differentiation_classifier,
    first_observation_leakage,
    graph_and_dynamic_descriptors,
    oracle_validation,
)
from experiments.v837_primitive_invention.common.metrics import binary_summary, continuous_summary
from experiments.v837_primitive_invention.common.search import structural_search
from experiments.v837_primitive_invention.common.seeds import cyclic_seeds, frozen_gates, gate_sha256
from experiments.v837_primitive_invention.common.serialization import save_model_bundle, write_json
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
PLOTS = HERE / "plots"
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"


def benchmark_validity() -> dict:
    gates = frozen_gates()
    tasks = all_tasks()
    oracle_seeds = cyclic_seeds("validation", 100, offset=0)
    oracle = {task.name: oracle_validation(task, oracle_seeds, "validation") for task in tasks}
    leakage = first_observation_leakage(tasks, cyclic_seeds("development", 120, offset=0))
    oracle_pass = all(row["binary"]["success_rate"] >= gates["benchmark_validity"]["oracle_min_success_rate"] for row in oracle.values())
    leakage_pass = leakage["accuracy"] <= gates["benchmark_validity"]["first_observation_family_classifier_max_accuracy"]
    return {
        "gate_file_sha256": gate_sha256(),
        "oracle": oracle,
        "first_observation_leakage": leakage,
        "oracle_pass": oracle_pass,
        "leakage_pass": leakage_pass,
        "pass": oracle_pass and leakage_pass,
    }


def _worker(family: str, run_index: int, training_scope: str) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    task = task_by_name(family)
    result = structural_search(task, run_index, overrides={"parameter_training_scope": training_scope})
    MODELS.mkdir(parents=True, exist_ok=True)
    best_path = MODELS / f"{family}_run{run_index:02d}_best.pt"
    random_path = MODELS / f"{family}_run{run_index:02d}_random.pt"
    save_model_bundle(best_path, result.best.model, {"family": family, "run_index": run_index, "condition": "EVOLVED"})
    save_model_bundle(random_path, result.random_control.model, {"family": family, "run_index": run_index, "condition": "RANDOM_MATCHED"})
    descriptor_seeds = result.validation_seeds
    descriptors = graph_and_dynamic_descriptors(result.best.model, task, descriptor_seeds, "validation")
    random_descriptors = graph_and_dynamic_descriptors(result.random_control.model, task, descriptor_seeds, "validation")
    payload = result.to_dict()
    payload["descriptors"] = descriptors
    payload["random_descriptors"] = random_descriptors
    payload["best_model_path"] = str(best_path.relative_to(ROOT)).replace("\\", "/")
    payload["random_model_path"] = str(random_path.relative_to(ROOT)).replace("\\", "/")
    return payload


def _aggregate(records: list[dict], benchmark: dict, config: dict) -> dict:
    gates = frozen_gates()
    per_family = {}
    family_passes = 0
    all_final_cells = []
    evolved_validation = []
    random_validation = []
    for family in config["task_families"]:
        rows = sorted([row for row in records if row["task_family"] == family], key=lambda row: row["run_index"])
        dev_run_success = [row["development"]["success_rate"] >= gates["v837"]["development_success_rate_per_family"] for row in rows]
        val_run_success = [row["validation"]["success_rate"] >= gates["v837"]["heldout_validation_success_rate_per_family"] for row in rows]
        final_cells = [len(row["best_graph"]["cells"]) for row in rows]
        evolved = [row["validation"]["success_rate"] for row in rows]
        random = [row["random_matched"]["validation"]["success_rate"] for row in rows]
        dev_summary = binary_summary(dev_run_success)
        val_summary = binary_summary(val_run_success)
        gap_points = 100.0 * (float(np.mean(evolved)) - float(np.mean(random)))
        family_pass = dev_summary["success_rate"] >= 0.90 and val_summary["success_rate"] >= 0.85
        family_passes += int(family_pass)
        per_family[family] = {
            "development_run_gate": dev_summary,
            "validation_run_gate": val_summary,
            "validation_episode_success": continuous_summary(evolved),
            "random_matched_validation_success": continuous_summary(random),
            "random_gap_points": gap_points,
            "final_cells": continuous_summary(final_cells),
            "pass": family_pass,
        }
        all_final_cells.extend(final_cells)
        evolved_validation.extend(evolved)
        random_validation.extend(random)
    random_gap_points = 100.0 * (float(np.mean(evolved_validation)) - float(np.mean(random_validation)))
    median_final_cells = float(np.median(all_final_cells))
    size_pass = median_final_cells < gates["v837"]["median_final_graph_fraction_of_cap_max"] * gates["substrate"]["max_cells"]
    random_pass = random_gap_points >= gates["v837"]["random_matched_graph_min_gap_points"]
    pass_value = benchmark["pass"] and family_passes >= gates["v837"]["families_required"] and size_pass and random_pass
    failure_classes = [] if pass_value else ["SEARCH_FAILURE"]
    return {
        "version": "V837",
        "parent": None,
        "research_question": "Can the same neutral low-level continuous-cell substrate differentiate into task-useful structures across multiple task families without named high-level operators?",
        "hypothesis": "Bounded low-level structural search plus fixed parameter adaptation can discover compact graph structures that solve at least four of five families and outperform matched random structures.",
        "single_change": config["single_change"],
        "substrate_version": "neutral_cell_v1",
        "gate_file_sha256": config["gate_file_sha256"],
        "task_families": config["task_families"],
        "development_seeds": config["development_seed_range"],
        "validation_seeds": config["validation_seed_range"],
        "fresh_audit_seeds": config["fresh_audit_seed_range"],
        "baselines": {"B0": "initial fixed two-cell graph", "B1": "bounded neutral structural search", "random_matched": "size/recurrent-fraction matched random graph trained with identical parameter budget"},
        "metrics": {
            "benchmark_validity": benchmark,
            "per_family": per_family,
            "families_passing": family_passes,
            "median_final_cells": median_final_cells,
            "size_gate_pass": size_pass,
            "random_matched_gap_points": random_gap_points,
            "random_matched_gate_pass": random_pass,
        },
        "resource_accounting": {
            "candidate_evaluations": int(sum(row["resources"]["candidate_evaluations"] for row in records)),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in records)),
            "environment_steps": int(sum(row["resources"]["environment_steps"] for row in records)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in records)),
            "mutation_count": int(sum(row["resources"]["mutation_count"] for row in records)),
            "peak_cells": int(max(len(row["best_graph"]["cells"]) for row in records)),
            "peak_edges": int(max(len(row["best_graph"]["edges"]) for row in records)),
            "final_cells": median_final_cells,
            "final_edges": float(np.median([len(row["best_graph"]["edges"]) for row in records])),
            "max_candidate_budget_per_run": int(gates["search"]["population"] + gates["search"]["offspring_per_generation"] * gates["search"]["max_generations"]),
            "max_optimizer_step_budget_per_run": int((gates["search"]["population"] + gates["search"]["offspring_per_generation"] * gates["search"]["max_generations"]) * gates["search"]["candidate_train_steps"]),
        },
        "motifs": [],
        "primitive_archive": {},
        "pass_gate": gates["v837"],
        "pass": bool(pass_value),
        "failure_classification": failure_classes,
        "caveats": [
            "V837 is a prerequisite competence test, not primitive invention.",
            "Internal cell dynamics are searched through generic parameter-seed mutation; gradient adaptation in this first variant is restricted to the readout using AdamW for CPU discipline.",
            "No task-family label is present in model input; evaluation metadata retains family names only for analysis."
        ],
        "next_question": "If V837 passes, test generic structural/dynamic differentiation. If it fails, change one scientific variable in the next version and retain this failure.",
        "runs": records,
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
    ax.bar(x - width / 2, evolved, width, label="evolved")
    ax.bar(x + width / 2, random, width, label="matched random")
    ax.set_xticks(x, families, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("heldout success rate")
    ax.set_title("V837 neutral-substrate competence vs matched random graphs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "v837_competence_random_control.png", dpi=160)
    plt.close(fig)


def main() -> int:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    current_hash = gate_sha256()
    if current_hash != EXPECTED_GATE_HASH or current_hash != config["gate_file_sha256"]:
        raise SystemExit(f"frozen gate hash mismatch: {current_hash}")
    benchmark = benchmark_validity()
    write_json(HERE.parent / "benchmark_validity.json", benchmark)
    if not benchmark["pass"]:
        results = {
            "version": "V837",
            "parent": None,
            "research_question": "Neutral-substrate competence",
            "hypothesis": "Benchmark validity is a prerequisite.",
            "single_change": config["single_change"],
            "substrate_version": "neutral_cell_v1",
            "task_families": config["task_families"],
            "development_seeds": config["development_seed_range"],
            "validation_seeds": config["validation_seed_range"],
            "fresh_audit_seeds": config["fresh_audit_seed_range"],
            "baselines": {}, "metrics": {"benchmark_validity": benchmark}, "resource_accounting": {}, "motifs": [], "primitive_archive": {},
            "pass_gate": frozen_gates()["benchmark_validity"], "pass": False, "failure_classification": ["BENCHMARK_CONFOUND"], "caveats": [], "next_question": "Repair benchmark leakage/validity before structural search.",
            "gate_file_sha256": config["gate_file_sha256"]
        }
        write_json(HERE / "results.json", results)
        return 2
    jobs = [(family, run_index) for family in config["task_families"] for run_index in range(frozen_gates()["v837"]["independent_searches_per_family"])]
    workers = min(int(config["compute"]["workers"]), os.cpu_count() or 1)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, family, run_index, config["parameter_training_scope"]): (family, run_index) for family, run_index in jobs}
        for future in as_completed(futures):
            family, run_index = futures[future]
            row = future.result()
            records.append(row)
            print(f"completed {family} run {run_index:02d}: dev={row['development']['success_rate']:.3f} val={row['validation']['success_rate']:.3f} gen={row['generations_used']}", flush=True)
    records.sort(key=lambda row: (row["task_family"], row["run_index"]))
    results = _aggregate(records, benchmark, config)
    write_json(HERE / "results.json", results)
    _write_plot(results)
    status_path = HERE / ("PASS.md" if results["pass"] else "FAILURE.md")
    if results["pass"]:
        status_path.write_text("# V837 PASS\n\nThe neutral-substrate competence prerequisite passed the frozen V837 gate. This is not yet primitive invention.\n", encoding="utf-8")
        return 0
    failed = [family for family, row in results["metrics"]["per_family"].items() if not row["pass"]]
    status_path.write_text(
        "# V837 FAILURE\n\n"
        f"WHAT failed? Neutral-substrate competence gate; failing families: {failed}.\n\n"
        "WHERE? Independent bounded structural searches under the common generic cell substrate.\n\n"
        "WHEN? Before motif mining; no downstream primitive claim is attempted.\n\n"
        "WHY suspected? The initial fixed readout-only parameter-adaptation scope may underfit useful recurrent dynamics even when topology/parameter seeds are searched.\n\n"
        "HOW reproduced? Thirty independent searches per family with frozen development/validation partitions and matched random controls.\n\n"
        "WHAT evidence? `results.json` contains every run, search cost, matched-random result, and family-level gate.\n\n"
        "WHAT alternatives ruled out? Oracle benchmark validity and first-observation leakage are checked before search.\n\n"
        "WHAT single change next? V837b changes parameter adaptation only: full-cell AdamW under the same task, search, seed, structure, and pass gates.\n",
        encoding="utf-8",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
