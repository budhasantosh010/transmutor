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
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting, WallTimer
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import (
    CONDITIONS,
    CandidateInputFactorizationY3,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
AF0 = "AF0_y3_parent"
TRANSFER = [
    "AF1_shared_candidate_input_factorization",
    "AF1F_folded_candidate_input_control",
    "AF1D_deshared_candidate_input_factorization",
]


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    frozen = {
        "experiments/v837_primitive_invention/v837y/config.json": CONFIG["v837y_config_sha256"],
        "experiments/v837_primitive_invention/v837y/candidate_interaction.py": CONFIG["v837y_model_sha256"],
        "experiments/v837_primitive_invention/v837y/results.json": CONFIG["v837y_results_sha256"],
        "experiments/v837_primitive_invention/v837y/raw/interaction_runs.json": CONFIG["v837y_raw_sha256"],
        "experiments/v837_primitive_invention/v837ab/results.json": CONFIG["v837ab_results_sha256"],
        "experiments/v837_primitive_invention/v837ac/results.json": CONFIG["v837ac_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/results.json": CONFIG["v837ad_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/diagnostics/decision_state.json": CONFIG["v837ad_decision_sha256"],
    }
    for path, expected in frozen.items():
        if _git_blob_sha256(path) != expected:
            raise SystemExit(f"frozen V837af dependency changed: {path}")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v837ae").exists():
        raise SystemExit("V837ae must remain absent")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    val = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    if len(train) != 512 or len(val) != 128:
        raise RuntimeError("frozen V837af seed ranges changed")
    return train, val


def _base_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["base_initialization_namespace"], family, replicate)


def _coupling_seed(replicate: int) -> int:
    return deterministic_int(
        CONFIG["training"]["coupling_seed_namespace"],
        CONFIG["training"]["coupling_seed_condition"],
        replicate,
    )


def _projection_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["projection_initialization_namespace"], family, replicate)


def _model(condition: str, family: str, replicate: int) -> CandidateInputFactorizationY3:
    return CandidateInputFactorizationY3(
        high_capacity_generic_graph(replicate),
        condition=condition,
        coupling_initialization_seed=_coupling_seed(replicate),
        projection_seed=_projection_seed(family, replicate),
    )


def _success_rate(task, prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(prediction.tolist(), targets.tolist())]))


def _active_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps).view(1, -1) < lengths.view(-1, 1)


def _gradient_norm(parameters) -> float:
    chunks = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0


def _trajectory(model: CandidateInputFactorizationY3, step: int) -> dict:
    return {
        "step": int(step),
        "projection": model.projection_diagnostics(),
        "effective_input_maps": model.effective_input_map_diagnostics(),
    }


def _post_diagnostics(model: CandidateInputFactorizationY3, task, validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline, trace = model(observations, lengths, return_trace=True)
        no_message = model(observations, lengths, disable_messages=True)
    success = _success_rate(task, baseline, targets)
    no_message_success = _success_rate(task, no_message, targets)
    active = _active_mask(lengths, trace.input_terms.shape[1])
    candidate_active = trace.candidate_states[active]
    input_active = trace.input_terms[active]

    per_cell = []
    jacobian_norms = []
    for cell in range(10):
        W_eff, _ = model.effective_candidate_input_map(cell)
        terms = input_active[:, cell, :]
        candidates = candidate_active[:, cell, :]
        # J = diag(1-candidate^2) @ W_eff, exact current-step derivative
        # of candidate wrt its already-visible input vector.
        deriv = 1.0 - candidates.pow(2)
        jac = deriv.unsqueeze(-1) * W_eff.detach().unsqueeze(0)
        jac_norm = float(torch.linalg.vector_norm(jac, dim=(1, 2)).mean().item())
        jacobian_norms.append(jac_norm)
        per_cell.append({
            "cell": cell,
            "input_term_norm_mean": float(torch.linalg.vector_norm(terms, dim=-1).mean().item()),
            "input_term_temporal_variance": float(torch.var(terms, dim=0, unbiased=False).mean().item()),
            "candidate_visible_input_jacobian_frobenius_mean": jac_norm,
        })
    return {
        "projection": model.projection_diagnostics(),
        "effective_input_maps": model.effective_input_map_diagnostics(),
        "candidate_input": {
            "per_cell": per_cell,
            "mean_input_term_norm": float(np.mean([r["input_term_norm_mean"] for r in per_cell])),
            "mean_temporal_variance": float(np.mean([r["input_term_temporal_variance"] for r in per_cell])),
            "mean_candidate_visible_input_jacobian_frobenius": float(np.mean(jacobian_norms)),
        },
        "message_dependency": {
            "baseline_success": success,
            "no_message_success": no_message_success,
            "success_drop": success - no_message_success,
            "mean_abs_prediction_delta": float(torch.abs(no_message - baseline).mean().item()),
        },
    }, 2


def _train(condition: str, family: str, replicate: int) -> tuple[CandidateInputFactorizationY3, dict]:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _seeds()
    tr = CONFIG["training"]
    initialization_seed = _base_seed(family, replicate)
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed % (2**32 - 1))
    model = _model(condition, family, replicate)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    resources = ResourceAccounting(
        candidate_evaluations=1,
        model_fits=1,
        environment_steps=sum(len(ep.observations) for ep in train_episodes + validation_episodes),
        parameter_count=model.parameter_count(),
        model_parameter_bytes=model.parameter_bytes(),
    )
    curve = []
    projection_trajectory = []
    requested = set(tr["curve_steps"])
    latest_gradient_norm = 0.0

    def record(step: int) -> None:
        dev = evaluate_sequence_model(model, task, train_episodes)
        val = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
        curve.append({
            "step": int(step),
            "training_loss": dev.loss,
            "training_success": dev.success_rate,
            "validation_loss": val.loss,
            "validation_success": val.success_rate,
            "gradient_norm": latest_gradient_norm,
        })
        projection_trajectory.append(_trajectory(model, step))

    with WallTimer() as timer:
        if 0 in requested:
            record(0)
        for step in range(1, int(tr["steps"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observations, lengths)
            resources.forward_calls += 1
            loss = loss_fn(prediction, targets)
            loss.backward()
            latest_gradient_norm = _gradient_norm(params)
            torch.nn.utils.clip_grad_norm_(params, float(tr["gradient_clip"]))
            optimizer.step()
            resources.optimizer_steps += 1
            resources.examples_processed += len(train_episodes)
            if step in requested:
                record(step)
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
    resources.wall_seconds = timer.seconds
    resources.cpu_seconds = timer.cpu_seconds
    diagnostics, extra_calls = _post_diagnostics(model, task, validation_seeds)
    resources.forward_calls += extra_calls
    return model, {
        "initialization_seed": initialization_seed,
        "development_success": development.success_rate,
        "validation_success": validation.success_rate,
        "development_loss": development.loss,
        "validation_loss": validation.loss,
        "loss_curve": curve,
        "projection_trajectory": projection_trajectory,
        "diagnostics": diagnostics,
        "resources": resources.to_dict(),
    }


def _worker(condition: str, family: str, replicate: int) -> dict:
    model, training = _train(condition, family, replicate)
    return {
        "version": "V837af",
        "condition": condition,
        "family": family,
        "replicate_id": replicate,
        "initialization_seed": training["initialization_seed"],
        "coupling_initialization_seed": _coupling_seed(replicate),
        "projection_initialization_seed": _projection_seed(family, replicate),
        "development_success": training["development_success"],
        "validation_success": training["validation_success"],
        "development_loss": training["development_loss"],
        "validation_loss": training["validation_loss"],
        "capacity_demonstrated": capacity_demonstrated(training["development_success"], training["validation_success"]),
        "loss_curve": training["loss_curve"],
        "projection_trajectory": training["projection_trajectory"],
        "diagnostics": training["diagnostics"],
        "parameter_count": model.parameter_count(),
        "active_parameter_count": model.parameter_count(),
        "projection_parameter_count": model.projection_parameter_count,
        "projection_specific_macs": model.projection_specific_macs,
        "recurrent_controller_projection_macs": model.total_recurrent_controller_projection_macs,
        "resources": training["resources"],
        "processed_examples": training["resources"]["examples_processed"],
        "unique_seed_defined_episode_policy": "same 3200 family/seed episodes reused across all conditions and replicates",
        "fresh_audit_consumed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "v838_started": False,
        "gpu_seconds": 0.0,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs = [(condition, family, replicate) for condition in conditions for family in FAMILIES for replicate in range(5)]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['condition']} {row['family']} r{row['replicate_id']}: "
                f"dev={row['development_success']:.6f} val={row['validation_success']:.6f}",
                flush=True,
            )
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    out = {"families_passing": 0, "family_validation_medians": {}, "family_development_medians": {}}
    for family in FAMILIES:
        fr = [row for row in rows if row["condition"] == condition and row["family"] == family]
        dev = float(np.median([r["development_success"] for r in fr]))
        val = float(np.median([r["validation_success"] for r in fr]))
        out["family_development_medians"][family] = dev
        out["family_validation_medians"][family] = val
        out["families_passing"] += int(capacity_demonstrated(dev, val))
    return out


def _historical_y3_rows() -> list[dict]:
    raw = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/raw/interaction_runs.json").read_text(encoding="utf-8"))
    return [r for r in raw["rows"] if r["condition"] == "Y3_global_control_rank4_candidate"]


def _parent_guard(rows: list[dict]) -> dict:
    historical = _historical_y3_rows()
    observed = _summary(rows, AF0)
    expected_result = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/results.json").read_text(encoding="utf-8"))["conditions"]["Y3_global_control_rank4_candidate"]
    expected_medians = {family: float(expected_result["family_results"][family]["validation"]["median"]) for family in FAMILIES}
    median_deltas = {family: abs(observed["family_validation_medians"][family] - expected_medians[family]) for family in FAMILIES}
    row_errors = []
    for expected in historical:
        got = next(r for r in rows if r["condition"] == AF0 and r["family"] == expected["family"] and r["replicate_id"] == expected["replicate_id"])
        row_errors.append({
            "family": expected["family"],
            "replicate_id": expected["replicate_id"],
            "development_abs_delta": abs(float(got["development_success"]) - float(expected["development_success"])),
            "validation_abs_delta": abs(float(got["validation_success"]) - float(expected["validation_success"])),
        })
    max_dev = max(r["development_abs_delta"] for r in row_errors)
    max_val = max(r["validation_abs_delta"] for r in row_errors)
    valid = (
        observed["families_passing"] == 3
        and max(median_deltas.values()) <= CONFIG["parent_drift_threshold"]
        and max_dev <= 1.0 / 512.0
        and max_val <= 1.0 / 128.0
    )
    return {
        "parent_reproduced": bool(valid),
        "expected_families_passing": 3,
        "observed": observed,
        "expected_family_validation_medians": expected_medians,
        "median_absolute_deltas": median_deltas,
        "row_level": row_errors,
        "max_development_abs_delta": max_dev,
        "max_validation_abs_delta": max_val,
    }


def _step0_equivalence() -> dict:
    train_seeds, _ = _seeds()
    comparison_pairs = [
        ("AF1_shared_candidate_input_factorization", "AF1F_folded_candidate_input_control"),
        ("AF1_shared_candidate_input_factorization", "AF1D_deshared_candidate_input_factorization"),
    ]
    metrics = ["projected_visible_input", "candidate_input_term", "candidate", "global_gate", "next_state", "output", "prediction"]
    maxima = {f"{a}__vs__{b}": {metric: 0.0 for metric in metrics} for a, b in comparison_pairs}
    rows = []
    for family in FAMILIES:
        task = task_by_name(family)
        episodes = [task.generate(seed, "development") for seed in train_seeds[:8]]
        observations, lengths, _ = episodes_to_batch(episodes)
        for replicate in range(5):
            seed = _base_seed(family, replicate)
            models = {}
            for condition in {c for pair in comparison_pairs for c in pair}:
                torch.manual_seed(seed)
                # Replay the exact stored float32 initialization in float64 so
                # the frozen 1e-6 gate measures algebraic parameterization
                # equivalence rather than ill-conditioned float32 association
                # noise from sequential versus folded matrix products.
                models[condition] = _model(condition, family, replicate).double()
            probe_observations = observations.double()
            traces = {}
            predictions = {}
            projected = {}
            with torch.no_grad():
                for condition, model in models.items():
                    predictions[condition], traces[condition] = model(probe_observations, lengths, return_trace=True)
                    chunks = []
                    for t in range(probe_observations.shape[1]):
                        x_t = probe_observations[:, t, :]
                        per_cell = []
                        for cell in range(10):
                            visible = model.historical_visible_input(x_t, cell)
                            per_cell.append(model.diagnostic_projected_visible_input(visible, cell))
                        chunks.append(torch.stack(per_cell, dim=1))
                    projected[condition] = torch.stack(chunks, dim=1)
            active = (torch.arange(observations.shape[1]).view(1, -1) < lengths.view(-1, 1)).to(torch.float64).view(observations.shape[0], observations.shape[1], 1, 1)
            for a, b in comparison_pairs:
                ta, tb = traces[a], traces[b]
                pa, pb = predictions[a], predictions[b]
                bias_a = torch.stack(list(models[a].base.cell_b), dim=0).view(1, 1, 10, 4)
                bias_b = torch.stack(list(models[b].base.cell_b), dim=0).view(1, 1, 10, 4)
                errors = {
                    "projected_visible_input": float(torch.max(torch.abs(projected[a] - projected[b])).item()),
                    "candidate_input_term": float(torch.max(torch.abs((ta.input_terms + active * bias_a) - (tb.input_terms + active * bias_b))).item()),
                    "candidate": float(torch.max(torch.abs(ta.candidate_states - tb.candidate_states)).item()),
                    "global_gate": float(torch.max(torch.abs(ta.global_gates - tb.global_gates)).item()),
                    "next_state": float(torch.max(torch.abs(ta.states - tb.states)).item()),
                    "output": float(torch.max(torch.abs(ta.outputs - tb.outputs)).item()),
                    "prediction": float(torch.max(torch.abs(pa - pb)).item()),
                }
                key = f"{a}__vs__{b}"
                for metric, value in errors.items():
                    maxima[key][metric] = max(maxima[key][metric], value)
                rows.append({"family": family, "replicate_id": replicate, "comparison": key, "errors": errors})
    maximum = max(value for per_compare in maxima.values() for value in per_compare.values())
    return {
        "step0_equivalence_proven": bool(maximum <= CONFIG["step0_tolerance"]),
        "tolerance": CONFIG["step0_tolerance"],
        "equivalence_precision": "float64 replay of exact stored float32 initialization",
        "maximum_error": maximum,
        "max_errors": maxima,
        "comparisons": rows,
        "af1_vs_af1f_required": True,
        "af1_vs_af1d_required": True,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "af0", "transfer", "all"), default="all")
    args = parser.parse_args()
    _assert_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(exist_ok=True)
    if args.phase in {"preflight", "all"}:
        step0 = _step0_equivalence()
        write_json(HERE / "diagnostics" / "step0_equivalence.json", step0)
        print(json.dumps({"step0": step0["step0_equivalence_proven"], "maximum_error": step0["maximum_error"]}, indent=2))
        if not step0["step0_equivalence_proven"]:
            return 2
    if args.phase in {"af0", "all"}:
        step0_path = HERE / "diagnostics" / "step0_equivalence.json"
        if not step0_path.exists() or not json.loads(step0_path.read_text(encoding="utf-8")).get("step0_equivalence_proven"):
            raise SystemExit("V837af AF0 blocked until step-zero equivalence passes")
        rows = _run([AF0])
        write_json(HERE / "raw" / "af0_runs.json", {"rows": rows, "unique_seed_defined_episodes": 3200})
        guard = _parent_guard(rows)
        write_json(HERE / "diagnostics" / "anchor_compatibility.json", guard)
        print(json.dumps(guard, indent=2))
        if not guard["parent_reproduced"]:
            return 3
    if args.phase in {"transfer", "all"}:
        guard_path = HERE / "diagnostics" / "anchor_compatibility.json"
        if not guard_path.exists() or not json.loads(guard_path.read_text(encoding="utf-8")).get("parent_reproduced"):
            raise SystemExit("V837af transfer blocked until AF0 reproduces exact Y3")
        rows = _run(TRANSFER)
        write_json(HERE / "raw" / "transfer_runs.json", {"rows": rows, "unique_seed_defined_episodes": 3200})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
