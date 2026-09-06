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
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch, _temporal_variance
from experiments.v837_primitive_invention.v837r.run_coupling_diagnostic import _active_mask, _success_rate, _term_norm
from experiments.v837_primitive_invention.v837y.candidate_interaction import CandidateInteractionSpec, ControlledCandidateInteractionModel

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
ANCHORS = ["Y0_historical", "Y1_global_control", "Y2_rank4_candidate"]
INTERACTION = ["Y3_global_control_rank4_candidate", "Y3C_global_control_matched_local"]
CONDITIONS = ANCHORS + INTERACTION


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _assert_science_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    parents = {
        "experiments/v837_primitive_invention/v837r/results.json": CONFIG["v837r_results_sha256"],
        "experiments/v837_primitive_invention/v837r/recurrent_coupling.py": CONFIG["v837r_coupling_sha256"],
        "experiments/v837_primitive_invention/v837x/results.json": CONFIG["v837x_results_sha256"],
        "experiments/v837_primitive_invention/v837x/global_scalar_control.py": CONFIG["v837x_controller_sha256"],
        "experiments/v837_primitive_invention/v837x/diagnostics/decision_state.json": CONFIG["v837x_decision_sha256"],
    }
    for path, expected in parents.items():
        if _git_blob_sha256(path) != expected:
            raise SystemExit(f"frozen parent changed: {path}")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _condition_spec(condition: str) -> CandidateInteractionSpec:
    row = CONFIG["conditions"][condition]
    mode = row["candidate_coupling"]
    return CandidateInteractionSpec(
        global_scalar_control=bool(row["global_scalar_control"]),
        candidate_coupling_mode=mode,
        coupling_rank=4 if mode == "rank4_cross_block" else None,
        matched_local_rank=4 if mode == "rank4_matched_local" else None,
    )


def _coupling_seed(replicate: int) -> int:
    tr = CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], replicate)


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    spec = _condition_spec(condition)
    seed = _coupling_seed(replicate)
    return lambda: ControlledCandidateInteractionModel(graph, spec=spec, coupling_initialization_seed=seed)


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    validation = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    if len(train) != tr["train_episodes"] or len(validation) != tr["validation_episodes"]:
        raise RuntimeError("configured seed ranges do not match episode counts")
    return train, validation


def _gradient_norm(parameters) -> float:
    chunks = [p.grad.detach().reshape(-1) for p in parameters if p is not None and p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0


def _gradient_diagnostics(model: ControlledCandidateInteractionModel, task, train_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.zero_grad(set_to_none=True)
    model.train()
    predictions = model(observations, lengths)
    torch.mean((predictions - targets) ** 2).backward()
    base_parameters = []
    for i in range(10):
        base_parameters += [model.base.cell_ws[i], model.base.cell_wm[i], model.base.cell_wx[i], model.base.cell_b[i], model.base.cell_wo[i]]
    cell_norms = []
    for i in range(10):
        cell_norms.append(_gradient_norm([model.base.cell_ws[i], model.base.cell_wm[i], model.base.cell_wx[i], model.base.cell_b[i], model.base.cell_wo[i]]))
    controller = [model.global_ws, model.global_wx, model.global_b] if model.global_scalar_control else []
    if model.coupling.mode == "low_rank":
        branch = [model.global_u, model.global_v]
    elif model.coupling.mode == "parameter_matched_local":
        branch = list(model.local_extra_u) + list(model.local_extra_v)
    else:
        branch = []
    result = {
        "base_cell_gradient_norm": _gradient_norm(base_parameters),
        "controller_gradient_norm": _gradient_norm(controller),
        "candidate_branch_gradient_norm": _gradient_norm(branch),
        "per_cell_gradient_norms": cell_norms,
        "cell_gradient_norm_variance": float(np.var(cell_norms)),
        "all_parameter_gradient_norm": _gradient_norm(list(model.parameters())),
    }
    model.zero_grad(set_to_none=True)
    return result


def _gate_stats(trace, lengths: torch.Tensor) -> dict:
    if trace.global_gates is None:
        return {"enabled": False, "mean": 0.0, "median": 0.0, "std": 0.0, "temporal_variance": 0.0, "p10": 0.0, "p90": 0.0, "near_zero_fraction": 0.0, "near_one_fraction": 0.0, "carry_fraction": 0.0, "rewrite_fraction": 1.0}
    active = _active_mask(trace.global_gates, lengths)
    values = trace.global_gates[active].detach().cpu().numpy().reshape(-1)
    mean = float(np.mean(values))
    return {
        "enabled": True,
        "mean": mean,
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "temporal_variance": float(_temporal_variance(trace.global_gates, lengths)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "near_zero_fraction": float(np.mean(values <= 0.05)),
        "near_one_fraction": float(np.mean(values >= 0.95)),
        "carry_fraction": mean,
        "rewrite_fraction": 1.0 - mean,
    }


def _post_training_diagnostics(model: ControlledCandidateInteractionModel, task, train_seeds: list[int], validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline, trace = model(observations, lengths, return_trace=True)
        no_message = model(observations, lengths, disable_messages=True)
    baseline_success = _success_rate(task, baseline, targets)
    local_norm = _term_norm(trace.recurrent_terms, lengths)
    global_norm = _term_norm(trace.global_recurrent_terms, lengths)
    matched_norm = _term_norm(trace.matched_local_terms, lengths)
    message_norm = _term_norm(trace.message_terms, lengths)
    input_norm = _term_norm(trace.input_terms, lengths)
    branch_norm = global_norm if model.coupling.mode == "low_rank" else matched_norm
    bias = torch.stack(list(model.base.cell_b), dim=0)
    bias_norm = float(torch.linalg.vector_norm(bias).item())
    calls = 2
    cross_rows = []
    if model.coupling.mode == "low_rank":
        with torch.no_grad():
            for source in range(10):
                intervened, itrace = model(observations, lengths, zero_coupling_source_cell=source, return_trace=True)
                other = torch.ones(10, dtype=torch.bool); other[source] = False
                cross_rows.append({
                    "source_cell": source,
                    "mean_abs_other_candidate_delta": float(torch.abs(itrace.candidate_states[:, :, other, :] - trace.candidate_states[:, :, other, :]).mean().item()),
                    "mean_abs_other_next_state_delta": float(torch.abs(itrace.states[:, :, other, :] - trace.states[:, :, other, :]).mean().item()),
                    "mean_abs_final_output_delta": float(torch.abs(intervened - baseline).mean().item()),
                    "success_delta": float(_success_rate(task, intervened, targets) - baseline_success),
                })
                calls += 1
    controller_interventions = {}
    if model.global_scalar_control:
        mean_gate = _gate_stats(trace, lengths)["mean"]
        with torch.no_grad():
            for label, value in (("g_0", 0.0), ("g_0_5", 0.5), ("g_1", 1.0), ("g_mean_training", mean_gate)):
                pred = model(observations, lengths, global_gate_override=value)
                controller_interventions[label] = {
                    "gate": float(value),
                    "validation_success": _success_rate(task, pred, targets),
                    "success_delta_vs_learned": float(_success_rate(task, pred, targets) - baseline_success),
                    "mean_abs_prediction_delta": float(torch.abs(pred - baseline).mean().item()),
                }
                calls += 1
    gradient = _gradient_diagnostics(model, task, train_seeds)
    calls += 1
    return {
        "validation_success_recomputed": baseline_success,
        "coupling_matrix": model.coupling_diagnostics(),
        "candidate_contributions": {
            "local_recurrent_norm": local_norm,
            "rank4_global_norm": global_norm,
            "matched_local_extra_norm": matched_norm,
            "message_norm": message_norm,
            "input_norm": input_norm,
            "bias_norm": bias_norm,
            "global_to_local_ratio": global_norm / (local_norm + 1e-12),
            "global_to_message_ratio": global_norm / (message_norm + 1e-12),
            "global_to_input_ratio": global_norm / (input_norm + 1e-12),
            "added_to_local_ratio": branch_norm / (local_norm + 1e-12),
        },
        "controller": _gate_stats(trace, lengths),
        "message_dependency": {
            "baseline_success": baseline_success,
            "no_message_success": _success_rate(task, no_message, targets),
            "success_drop": baseline_success - _success_rate(task, no_message, targets),
            "mean_abs_prediction_delta": float(torch.abs(no_message - baseline).mean().item()),
        },
        "cross_cell_causal_influence": {
            "semantics": "zero one source cell only in the rank4 global candidate branch; preserve ordinary local state/messages/candidate path",
            "per_source": cross_rows,
            "mean_abs_other_candidate_delta": float(np.mean([r["mean_abs_other_candidate_delta"] for r in cross_rows])) if cross_rows else 0.0,
            "mean_abs_other_next_state_delta": float(np.mean([r["mean_abs_other_next_state_delta"] for r in cross_rows])) if cross_rows else 0.0,
            "mean_abs_final_output_delta": float(np.mean([r["mean_abs_final_output_delta"] for r in cross_rows])) if cross_rows else 0.0,
        },
        "controller_causal_intervention": controller_interventions,
        "gradient": gradient,
    }, calls


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _seeds()
    tr = CONFIG["training"]
    initialization_seed = deterministic_int(tr["base_initialization_namespace"], family, replicate)
    result = train_sequence_model(
        model_factory=_model_factory(condition, replicate),
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=initialization_seed,
        steps=tr["steps"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        gradient_clip=tr["gradient_clip"],
        curve_steps=tuple(tr["curve_steps"]),
    )
    diagnostics, diagnostic_calls = _post_training_diagnostics(result.model, task, train_seeds, validation_seeds)
    resources = result.resources.to_dict()
    resources["training_forward_calls"] = int(resources.get("forward_calls", 0))
    resources["diagnostic_forward_calls"] = diagnostic_calls
    resources["forward_calls"] = int(resources.get("forward_calls", 0)) + diagnostic_calls
    model = result.model
    return {
        "version": "V837y",
        "condition": condition,
        "family": family,
        "replicate_id": replicate,
        "initialization_seed": initialization_seed,
        "coupling_initialization_seed": _coupling_seed(replicate),
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "loss_curve": result.learning_curve,
        "parameter_count": model.parameter_count(),
        "active_parameter_count": model.parameter_count(),
        "controller_param_count": model.controller_param_count,
        "candidate_branch_param_count": model.candidate_branch_param_count,
        "local_recurrent_macs": 160,
        "candidate_branch_macs": model.candidate_branch_macs,
        "controller_macs": model.controller_macs,
        "total_recurrent_controller_macs": model.recurrent_controller_macs,
        "diagnostics": diagnostics,
        "resources": resources,
        "processed_examples": tr["steps"] * tr["train_episodes"],
        "unique_seed_defined_episode_policy": "same 3200 family/seed episodes reused across all conditions and replicates",
        "fresh_audit_consumed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "v838_started": False,
        "gpu_seconds": 0.0,
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


def _historical_anchor() -> dict:
    r = json.loads((ROOT / "experiments/v837_primitive_invention/v837r/results.json").read_text(encoding="utf-8"))
    x = json.loads((ROOT / "experiments/v837_primitive_invention/v837x/results.json").read_text(encoding="utf-8"))
    def med(row): return {f: float(v["validation"]["median"]) for f, v in row["family_results"].items()}
    return {
        "Y0_historical": {"expected_families_passing": 2, "expected_family_medians": med(x["conditions"]["X0_historical_direct"])},
        "Y1_global_control": {"expected_families_passing": 3, "expected_family_medians": med(x["conditions"]["X2_global_scalar_carry"])},
        "Y2_rank4_candidate": {"expected_families_passing": 3, "expected_family_medians": med(r["conditions"]["R3_rank4"])},
    }


def _baseline_compatibility(rows: list[dict]) -> dict:
    historical = _historical_anchor(); threshold = CONFIG["baseline_drift_threshold"]
    output = {"compatible": True, "threshold": threshold, "anchors": {}}
    for condition in ANCHORS:
        observed = _summary(rows, condition); expected = historical[condition]
        deltas = {f: abs(observed["family_validation_medians"][f] - expected["expected_family_medians"][f]) for f in FAMILIES}
        drifted = [f for f, delta in deltas.items() if delta > threshold["absolute_validation_delta"]]
        compatible = len(drifted) < threshold["families_required"]
        output["compatible"] = output["compatible"] and compatible
        output["anchors"][condition] = {
            **expected,
            "observed_families_passing": observed["families_passing"],
            "observed_family_medians": observed["family_validation_medians"],
            "absolute_deltas": deltas,
            "materially_drifted_families": drifted,
            "compatible": compatible,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("anchors", "interaction", "all"), default="all")
    args = parser.parse_args()
    _assert_science_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(exist_ok=True)
    seed_policy = {
        "development_seed_range": CONFIG["training"]["development_seed_range"],
        "validation_seed_range": CONFIG["training"]["validation_seed_range"],
        "unique_seed_defined_episodes": 3200,
        "reuse_policy": "same 3200 family/seed episodes reused across every condition and replicate",
    }
    if args.phase in {"anchors", "all"}:
        rows = _run(ANCHORS)
        write_json(HERE / "raw" / "anchor_runs.json", {"rows": rows, "seed_policy": seed_policy})
        compat = _baseline_compatibility(rows)
        write_json(HERE / "diagnostics" / "baseline_compatibility.json", compat)
        print(json.dumps(compat, indent=2), flush=True)
        if not compat["compatible"]:
            (HERE / "FAILURE.md").write_text("# V837y CANDIDATE_INTERACTION_BASELINE_DRIFT\n\nAt least one anchor drifted by >0.10 validation on two or more families. Interaction interpretation is blocked.\n", encoding="utf-8")
            return 2
    if args.phase in {"interaction", "all"}:
        compat_path = HERE / "diagnostics" / "baseline_compatibility.json"
        if not compat_path.exists() or not json.loads(compat_path.read_text(encoding="utf-8")).get("compatible"):
            raise SystemExit("V837y interaction blocked until anchors reproduce")
        rows = _run(INTERACTION)
        write_json(HERE / "raw" / "interaction_runs.json", {"rows": rows, "seed_policy": seed_policy})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
