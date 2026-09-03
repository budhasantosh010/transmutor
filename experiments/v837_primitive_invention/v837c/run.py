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

from experiments.v837_primitive_invention.common.evaluator import first_observation_leakage, graph_and_dynamic_descriptors, oracle_validation
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
    return {"gate_file_sha256": gate_sha256(), "oracle": oracle, "first_observation_leakage": leakage, "oracle_pass": oracle_pass, "leakage_pass": leakage_pass, "pass": oracle_pass and leakage_pass}


def _worker(family: str, run_index: int, overrides: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    task = task_by_name(family)
    result = structural_search(task, run_index, overrides=overrides)
    MODELS.mkdir(parents=True, exist_ok=True)
    best_path = MODELS / f"{family}_run{run_index:02d}_best.pt"
    random_path = MODELS / f"{family}_run{run_index:02d}_random.pt"
    save_model_bundle(best_path, result.best.model, {"family": family, "run_index": run_index, "condition": "EVOLVED_WIDER_SEARCH"})
    save_model_bundle(random_path, result.random_control.model, {"family": family, "run_index": run_index, "condition": "RANDOM_MATCHED_WIDER_SEARCH"})
    row = result.to_dict()
    row["descriptors"] = graph_and_dynamic_descriptors(result.best.model, task, result.validation_seeds, "validation")
    row["random_descriptors"] = graph_and_dynamic_descriptors(result.random_control.model, task, result.validation_seeds, "validation")
    row["best_model_path"] = str(best_path.relative_to(ROOT)).replace("\\", "/")
    row["random_model_path"] = str(random_path.relative_to(ROOT)).replace("\\", "/")
    return row


def _aggregate(rows: list[dict], benchmark: dict, config: dict) -> dict:
    gates = frozen_gates()
    families = [task.name for task in all_tasks()]
    per_family = {}
    family_passes = 0
    all_cells = []
    evolved_all = []
    random_all = []
    for family in families:
        fr = sorted([row for row in rows if row["task_family"] == family], key=lambda row: row["run_index"])
        dev = [row["development"]["success_rate"] >= gates["v837"]["development_success_rate_per_family"] for row in fr]
        val = [row["validation"]["success_rate"] >= gates["v837"]["heldout_validation_success_rate_per_family"] for row in fr]
        evolved = [row["validation"]["success_rate"] for row in fr]
        random = [row["random_matched"]["validation"]["success_rate"] for row in fr]
        cells = [len(row["best_graph"]["cells"]) for row in fr]
        ds = binary_summary(dev)
        vs = binary_summary(val)
        family_pass = ds["success_rate"] >= 0.90 and vs["success_rate"] >= 0.85
        family_passes += int(family_pass)
        per_family[family] = {
            "development_run_gate": ds,
            "validation_run_gate": vs,
            "validation_episode_success": continuous_summary(evolved),
            "random_matched_validation_success": continuous_summary(random),
            "random_gap_points": 100.0 * (float(np.mean(evolved)) - float(np.mean(random))),
            "final_cells": continuous_summary(cells),
            "pass": family_pass,
        }
        all_cells.extend(cells)
        evolved_all.extend(evolved)
        random_all.extend(random)
    gap = 100.0 * (float(np.mean(evolved_all)) - float(np.mean(random_all)))
    median_cells = float(np.median(all_cells))
    size_pass = median_cells < gates["v837"]["median_final_graph_fraction_of_cap_max"] * gates["substrate"]["max_cells"]
    random_pass = gap >= gates["v837"]["random_matched_graph_min_gap_points"]
    passed = benchmark["pass"] and family_passes >= gates["v837"]["families_required"] and size_pass and random_pass
    candidate_budget = gates["search"]["population"] + int(config["search_overrides"]["offspring_per_generation"]) * gates["search"]["max_generations"]
    return {
        "version": "V837c",
        "parent": "V837b",
        "research_question": "Does substantially broader low-level structural search rescue neutral-substrate competence under the original parameter-adaptation regime?",
        "hypothesis": "If V837 failed mainly because four offspring per generation under-sampled useful low-level structures, doubling structural breadth should materially increase reliable family competence without changing representation or gates.",
        "single_change": config["single_change"],
        "substrate_version": "neutral_cell_v1",
        "gate_file_sha256": config["gate_file_sha256"],
        "task_families": families,
        "development_seeds": gates["seed_ranges"]["development"],
        "validation_seeds": gates["seed_ranges"]["validation"],
        "fresh_audit_seeds": gates["seed_ranges"]["fresh_audit"],
        "baselines": {"B0": "initial fixed two-cell graph", "B1": "wider bounded neutral structural search", "random_matched": "matched random graph under identical candidate training"},
        "metrics": {"benchmark_validity": benchmark, "per_family": per_family, "families_passing": family_passes, "median_final_cells": median_cells, "size_gate_pass": size_pass, "random_matched_gap_points": gap, "random_matched_gate_pass": random_pass},
        "resource_accounting": {
            "candidate_evaluations": int(sum(row["resources"]["candidate_evaluations"] for row in rows)),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in rows)),
            "environment_steps": int(sum(row["resources"]["environment_steps"] for row in rows)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in rows)),
            "mutation_count": int(sum(row["resources"]["mutation_count"] for row in rows)),
            "peak_cells": int(max(len(row["best_graph"]["cells"]) for row in rows)),
            "peak_edges": int(max(len(row["best_graph"]["edges"]) for row in rows)),
            "final_cells": median_cells,
            "final_edges": float(np.median([len(row["best_graph"]["edges"]) for row in rows])),
            "max_candidate_budget_per_run": int(candidate_budget),
            "max_optimizer_step_budget_per_run": int(candidate_budget * gates["search"]["candidate_train_steps"]),
        },
        "motifs": [], "primitive_archive": {}, "pass_gate": gates["v837"], "pass": bool(passed),
        "failure_classification": [] if passed else ["SEARCH_FAILURE"],
        "caveats": ["This variant changes structural search breadth only. Candidate parameter adaptation returns to the original readout-only AdamW regime."],
        "next_question": "If the third competence variant fails, stop the primitive-invention chain and run blocker diagnostics rather than silently continuing to motif mining.",
        "runs": rows,
    }


def _plot(results: dict) -> None:
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    families = results["task_families"]
    evolved = [results["metrics"]["per_family"][family]["validation_episode_success"]["mean"] for family in families]
    random = [results["metrics"]["per_family"][family]["random_matched_validation_success"]["mean"] for family in families]
    x = np.arange(len(families)); width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.bar(x-width/2, evolved, width, label="wider evolved search"); ax.bar(x+width/2, random, width, label="matched random")
    ax.set_xticks(x, families, rotation=20, ha="right"); ax.set_ylim(0,1); ax.set_ylabel("heldout success rate"); ax.set_title("V837c: doubled low-level structural-search breadth"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "v837c_competence_random_control.png", dpi=160); plt.close(fig)


def main() -> int:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if gate_sha256() != EXPECTED_GATE_HASH or config["gate_file_sha256"] != EXPECTED_GATE_HASH:
        raise SystemExit("frozen gate hash mismatch")
    benchmark = benchmark_validity()
    if not benchmark["pass"]:
        raise SystemExit("benchmark validity unexpectedly failed; do not run structural search")
    gates = frozen_gates(); families = [task.name for task in all_tasks()]
    jobs = [(family, run_index) for family in families for run_index in range(gates["v837"]["independent_searches_per_family"])]
    workers = min(int(config["compute"]["workers"]), os.cpu_count() or 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, family, run_index, config["search_overrides"]): (family, run_index) for family, run_index in jobs}
        for future in as_completed(futures):
            family, run_index = futures[future]; row = future.result(); rows.append(row)
            print(f"completed {family} run {run_index:02d}: dev={row['development']['success_rate']:.3f} val={row['validation']['success_rate']:.3f} gen={row['generations_used']}", flush=True)
    rows.sort(key=lambda row: (row["task_family"], row["run_index"]))
    results = _aggregate(rows, benchmark, config); write_json(HERE / "results.json", results); _plot(results)
    status = HERE / ("PASS.md" if results["pass"] else "FAILURE.md")
    if results["pass"]:
        status.write_text("# V837c PASS\n\nDoubling low-level structural-search breadth rescued competence under the unchanged gate.\n", encoding="utf-8"); return 0
    failed = [family for family, row in results["metrics"]["per_family"].items() if not row["pass"]]
    status.write_text(
        "# V837c FAILURE\n\n"
        f"WHAT failed? Third consecutive neutral-substrate competence variant; failing families: {failed}.\n\n"
        "WHERE? Bounded evolutionary structural search with twice the original offspring breadth.\n\n"
        "WHEN? Before motif mining, causal validation, compression, or retrieval.\n\n"
        "WHY suspected? Competence is not reliably produced by the current neutral-cell representation/search/training combination.\n\n"
        "HOW reproduced? Thirty paired independent searches per family, unchanged seed regions and pass gates, with only offspring breadth doubled.\n\n"
        "WHAT evidence supports diagnosis? Compare V837, V837b, and V837c. V837b ruled out simple readout underfitting; V837c directly tests additional structural breadth.\n\n"
        "WHAT alternatives were ruled out? Benchmark confound, first-observation task leakage, one narrow optimizer regime, and the original structural-search breadth.\n\n"
        "WHAT single change next? STOP per the three-failure rule and run a blocker diagnostic that separates substrate capacity from search/generalization.\n",
        encoding="utf-8",
    ); return 1


if __name__ == "__main__": raise SystemExit(main())
