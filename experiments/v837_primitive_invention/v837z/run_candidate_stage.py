from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837y.run_candidate_interaction import _post_training_diagnostics
from experiments.v837_primitive_invention.v837z.candidate_stage import CandidateStageModel, historical_candidate_depths, synchronous_candidate_depths

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
Z0 = "Z0_historical_candidate_stage"
Z1 = "Z1_synchronous_candidate_stage"
CONDITIONS = [Z0, Z1]


def _blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"] or v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("frozen V837 gate/capacity criterion changed")
    parent_paths = {
        "experiments/v837_primitive_invention/v837y/results.json": CONFIG["v837y_results_sha256"],
        "experiments/v837_primitive_invention/v837y/diagnostics/decision_state.json": CONFIG["v837y_decision_sha256"],
        "experiments/v837_primitive_invention/v837y/candidate_interaction.py": CONFIG["v837y_model_sha256"],
        "experiments/v837_primitive_invention/v837y/config.json": CONFIG["v837y_config_sha256"],
    }
    for path, expected in parent_paths.items():
        if _blob_sha256(path) != expected:
            raise SystemExit(f"frozen V837y parent changed: {path}")
    decision = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/diagnostics/decision_state.json").read_text())
    if not (decision.get("v837y_complete") is True and decision.get("representation_adequacy_pass") is False and decision.get("v837z_allowed") is True):
        raise SystemExit("V837z not authorized by V837y")
    if decision.get("selected_v837z_parent") != CONFIG["selected_parent"]:
        raise SystemExit("selected V837z parent mismatch")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text())
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    val = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    return train, val


def _coupling_seed(replicate: int) -> int:
    tr = CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], replicate)


def _stage_mode(condition: str) -> str:
    return "historical_mixed" if condition == Z0 else "fully_synchronous"


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    seed = _coupling_seed(replicate)
    return lambda: CandidateStageModel(graph, stage_mode=_stage_mode(condition), coupling_initialization_seed=seed)


def _stage_diagnostics(model: CandidateStageModel, task, validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    active = torch.arange(trace.states.shape[1]).view(1, -1) < lengths.view(-1, 1)
    states = trace.states.detach().cpu()
    prev = torch.zeros_like(states); prev[:, 1:] = states[:, :-1]
    changes = torch.linalg.vector_norm(states - prev, dim=-1).numpy()
    vals = []
    for b in range(changes.shape[0]):
        for t in range(changes.shape[1]):
            if bool(active[b, t]):
                row = changes[b, t]
                if np.std(row) > 1e-12:
                    vals.append(float(np.mean(np.corrcoef(np.vstack([row, np.arange(len(row))]))[0, 1])))
    # A stable synchrony statistic: correlation of each cell's change magnitude with cell 0 across active positions.
    flat_by_cell = []
    mask = active.numpy()
    for i in range(changes.shape[2]):
        flat_by_cell.append(changes[:, :, i][mask])
    cors = []
    for i in range(1, len(flat_by_cell)):
        if np.std(flat_by_cell[0]) > 1e-12 and np.std(flat_by_cell[i]) > 1e-12:
            cors.append(float(np.corrcoef(flat_by_cell[0], flat_by_cell[i])[0, 1]))
    depths = historical_candidate_depths(model.graph) if model.stage_mode == "historical_mixed" else synchronous_candidate_depths(model.graph)
    return {
        "stage_mode": model.stage_mode,
        "candidate_effective_depth": {
            "per_cell": depths,
            "min": int(min(depths)),
            "median": float(np.median(depths)),
            "max": int(max(depths)),
        },
        "state_change_synchrony": {
            "mean_cell0_cross_cell_correlation": float(np.mean(cors)) if cors else 0.0,
            "median_cell0_cross_cell_correlation": float(np.median(cors)) if cors else 0.0,
        },
    }, 1


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, val_seeds = _seeds()
    tr = CONFIG["training"]
    init_seed = deterministic_int(tr["base_initialization_namespace"], family, replicate)
    result = train_sequence_model(
        model_factory=_model_factory(condition, replicate),
        task=task,
        train_seeds=train_seeds,
        validation_seeds=val_seeds,
        initialization_seed=init_seed,
        steps=tr["steps"], learning_rate=tr["learning_rate"], weight_decay=tr["weight_decay"], gradient_clip=tr["gradient_clip"],
        curve_steps=tuple(tr["curve_steps"]),
    )
    common_diag, common_calls = _post_training_diagnostics(result.model, task, train_seeds, val_seeds)
    stage_diag, stage_calls = _stage_diagnostics(result.model, task, val_seeds)
    resources = result.resources.to_dict()
    resources["training_forward_calls"] = int(resources.get("forward_calls", 0))
    resources["diagnostic_forward_calls"] = common_calls + stage_calls
    resources["forward_calls"] = int(resources.get("forward_calls", 0)) + common_calls + stage_calls
    model = result.model
    return {
        "version": "V837z", "condition": condition, "family": family, "replicate_id": replicate,
        "initialization_seed": init_seed, "coupling_initialization_seed": _coupling_seed(replicate),
        "development_success": result.development.success_rate, "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss, "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "loss_curve": result.learning_curve,
        "parameter_count": model.parameter_count(), "active_parameter_count": model.parameter_count(),
        "controller_param_count": model.controller_param_count, "candidate_branch_param_count": model.candidate_branch_param_count,
        "local_recurrent_macs": 160, "candidate_branch_macs": model.candidate_branch_macs, "controller_macs": model.controller_macs,
        "total_recurrent_controller_macs": model.recurrent_controller_macs,
        "diagnostics": {**common_diag, **stage_diag}, "resources": resources,
        "processed_examples": tr["steps"] * tr["train_episodes"],
        "unique_seed_defined_episode_policy": "same 3200 family/seed episodes reused across Z0/Z1 and replicates",
        "fresh_audit_consumed": False, "structural_search_allowed": False, "primitive_mining_allowed": False, "v838_started": False, "gpu_seconds": 0.0,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs = [(c, f, r) for c in conditions for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    out = {"families_passing": 0, "family_validation_medians": {}, "family_development_medians": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        dev = float(np.median([r["development_success"] for r in fr])); val = float(np.median([r["validation_success"] for r in fr]))
        out["family_development_medians"][family] = dev; out["family_validation_medians"][family] = val
        out["families_passing"] += int(capacity_demonstrated(dev, val))
    return out


def _z0_guard(rows: list[dict]) -> dict:
    y = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/results.json").read_text())["conditions"][CONFIG["selected_parent"]]
    expected = {f: float(v["validation"]["median"]) for f, v in y["family_results"].items()}
    observed = _summary(rows, Z0)
    deltas = {f: abs(observed["family_validation_medians"][f] - expected[f]) for f in FAMILIES}
    drifted = [f for f, d in deltas.items() if d > 0.10]
    return {
        "compatible": len(drifted) < 2,
        "expected_families_passing": int(y["families_passing"]),
        "observed_families_passing": observed["families_passing"],
        "expected_family_medians": expected,
        "observed_family_medians": observed["family_validation_medians"],
        "absolute_deltas": deltas,
        "materially_drifted_families": drifted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("z0", "z1", "all"), default="all"); args = parser.parse_args()
    _assert_locks()
    for name in ("raw", "diagnostics", "plots"): (HERE / name).mkdir(exist_ok=True)
    policy = {"development_seed_range": CONFIG["training"]["development_seed_range"], "validation_seed_range": CONFIG["training"]["validation_seed_range"], "unique_seed_defined_episodes": 3200, "reuse_policy": "same episodes reused across conditions and replicates"}
    if args.phase in {"z0", "all"}:
        rows = _run([Z0]); write_json(HERE / "raw/z0_runs.json", {"rows": rows, "seed_policy": policy})
        guard = _z0_guard(rows); write_json(HERE / "diagnostics/parent_compatibility.json", guard); print(json.dumps(guard, indent=2), flush=True)
        if not guard["compatible"]:
            (HERE / "FAILURE.md").write_text("# V837z parent baseline drift\n\nZ0 failed to reproduce the machine-selected V837y parent. Z1 interpretation blocked.\n", encoding="utf-8"); return 2
    if args.phase in {"z1", "all"}:
        guard_path = HERE / "diagnostics/parent_compatibility.json"
        if not guard_path.exists() or not json.loads(guard_path.read_text()).get("compatible"):
            raise SystemExit("Z1 blocked until Z0 parent reproduction passes")
        rows = _run([Z1]); write_json(HERE / "raw/z1_runs.json", {"rows": rows, "seed_policy": policy})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
