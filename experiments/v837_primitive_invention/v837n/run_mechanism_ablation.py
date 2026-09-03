from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.gru_reference_explicit import ExplicitGRUReferenceModel

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
FULL = "full_gru"
ABLATIONS = [name for name in CONFIG["conditions"] if name != FULL]


def _configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _active_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device).view(1, -1) < lengths.view(-1, 1)


def _safe_quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "variance": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "variance": float(np.var(values)),
    }


def _temporal_variance(tensor: torch.Tensor, lengths: torch.Tensor) -> float:
    values: list[float] = []
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 1:
            continue
        per_dim = tensor[batch_index, :length, :].var(dim=0, unbiased=False)
        values.append(float(per_dim.mean().item()))
    return float(np.mean(values)) if values else 0.0


def _binary_entropy(values: torch.Tensor) -> float:
    p = torch.clamp(values, 1e-7, 1.0 - 1e-7)
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return float(entropy.mean().item())


def _lag1_autocorrelation(states: torch.Tensor, lengths: torch.Tensor) -> float:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 1:
            continue
        left.append(states[batch_index, :-1, :][: length - 1].detach().cpu().numpy().reshape(-1))
        right.append(states[batch_index, 1:, :][: length - 1].detach().cpu().numpy().reshape(-1))
    if not left:
        return 0.0
    a = np.concatenate(left)
    b = np.concatenate(right)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _override_episode_mean(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 0:
            continue
        mean = tensor[batch_index, :length, :].mean(dim=0, keepdim=True)
        result[batch_index, :length, :] = mean.expand(length, -1)
    return result


def _override_time_shuffle(tensor: torch.Tensor, lengths: torch.Tensor, seed: int) -> torch.Tensor:
    result = tensor.clone()
    generator = np.random.default_rng(int(seed))
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 1:
            continue
        order = generator.permutation(length)
        index = torch.as_tensor(order, dtype=torch.long, device=tensor.device)
        result[batch_index, :length, :] = tensor[batch_index, :length, :].index_select(0, index)
    return result


def _success_rate(task, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    hits = [task.success(float(pred), float(target)) for pred, target in zip(predictions.tolist(), targets.tolist())]
    return float(np.mean(hits))


def _gate_and_state_diagnostics(model: ExplicitGRUReferenceModel, task, validation_seeds: list[int], initialization_seed: int) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        predictions, trace = model(observations, lengths, return_trace=True)
    mask = _active_mask(lengths, trace.states.shape[1])
    updates = trace.updates[mask]
    resets = trace.resets[mask]
    states = trace.states[mask]
    candidates = trace.candidates[mask]
    update_input = trace.update_input_components[mask]
    update_state = trace.update_state_components[mask]
    reset_input = trace.reset_input_components[mask]
    reset_state = trace.reset_state_components[mask]

    diagnostics = {
        "validation_success_recomputed": _success_rate(task, predictions, targets),
        "update": {
            **_safe_quantiles(updates.detach().cpu().numpy().reshape(-1)),
            "entropy": _binary_entropy(updates),
            "temporal_variance": _temporal_variance(trace.updates, lengths),
            "input_conditioned_component_variance": float(update_input.var(unbiased=False).item()),
            "state_conditioned_component_variance": float(update_state.var(unbiased=False).item()),
        },
        "reset": {
            **_safe_quantiles(resets.detach().cpu().numpy().reshape(-1)),
            "entropy": _binary_entropy(resets),
            "temporal_variance": _temporal_variance(trace.resets, lengths),
            "input_conditioned_component_variance": float(reset_input.var(unbiased=False).item()),
            "state_conditioned_component_variance": float(reset_state.var(unbiased=False).item()),
        },
        "hidden_state_norm": float(torch.linalg.vector_norm(states, dim=-1).mean().item()),
        "hidden_state_autocorrelation": _lag1_autocorrelation(trace.states, lengths),
        "candidate_state_norm": float(torch.linalg.vector_norm(candidates, dim=-1).mean().item()),
        "state_saturation_fraction": float((torch.abs(states) >= 0.95).float().mean().item()),
        "candidate_saturation_fraction": float((torch.abs(candidates) >= 0.95).float().mean().item()),
    }
    forward_calls = 1

    if model.condition == FULL:
        replay = {}
        replay_specs = {
            "update_time_shuffle": {"update_override": _override_time_shuffle(trace.updates, lengths, deterministic_int("v837n-replay-update-shuffle", initialization_seed))},
            "update_episode_mean": {"update_override": _override_episode_mean(trace.updates, lengths)},
            "reset_time_shuffle": {"reset_override": _override_time_shuffle(trace.resets, lengths, deterministic_int("v837n-replay-reset-shuffle", initialization_seed))},
            "reset_episode_mean": {"reset_override": _override_episode_mean(trace.resets, lengths)},
        }
        with torch.no_grad():
            for name, kwargs in replay_specs.items():
                replay_predictions = model(observations, lengths, **kwargs)
                replay[name] = _success_rate(task, replay_predictions, targets)
                forward_calls += 1
        diagnostics["counterfactual_replay"] = replay
    return diagnostics, forward_calls


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    training = CONFIG["training"]
    train_seeds = list(range(int(training["development_seed_range"][0]), int(training["development_seed_range"][1]) + 1))
    validation_seeds = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    initialization_seed = deterministic_int("v837j-primary-init", family, replicate)

    def factory():
        return ExplicitGRUReferenceModel(
            int(CONFIG["models"]["hidden_size"]),
            int(CONFIG["models"]["input_dim"]),
            condition=condition,
        )

    result = train_sequence_model(
        model_factory=factory,
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=initialization_seed,
        steps=int(training["steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        curve_steps=tuple(training["curve_steps"]),
    )
    gate_diag, extra_forward_calls = _gate_and_state_diagnostics(result.model, task, validation_seeds, initialization_seed)
    result.resources.forward_calls += int(extra_forward_calls)
    return {
        "condition": condition,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(initialization_seed),
        "train_seed_first": train_seeds[0],
        "train_seed_last": train_seeds[-1],
        "validation_seed_first": validation_seeds[0],
        "validation_seed_last": validation_seeds[-1],
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "gate_statistics": gate_diag,
        "nominal_parameter_count": int(result.model.parameter_count()),
        "active_parameter_count": int(result.model.active_parameter_count()),
        "parameter_bytes": int(result.model.parameter_bytes()),
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def summarize_condition(condition: str, rows: list[dict]) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    per_family: dict[str, dict] = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
        if not fr:
            continue
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        pass_flags = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837n-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(pass_flags),
            "aggregate_capacity_pass": bool(aggregate),
        }

    gate_keys = [
        ("update_mean", lambda r: r["gate_statistics"]["update"]["mean"]),
        ("update_median", lambda r: r["gate_statistics"]["update"]["median"]),
        ("update_temporal_variance", lambda r: r["gate_statistics"]["update"]["temporal_variance"]),
        ("update_entropy", lambda r: r["gate_statistics"]["update"]["entropy"]),
        ("reset_mean", lambda r: r["gate_statistics"]["reset"]["mean"]),
        ("reset_median", lambda r: r["gate_statistics"]["reset"]["median"]),
        ("reset_temporal_variance", lambda r: r["gate_statistics"]["reset"]["temporal_variance"]),
        ("reset_entropy", lambda r: r["gate_statistics"]["reset"]["entropy"]),
        ("hidden_state_norm", lambda r: r["gate_statistics"]["hidden_state_norm"]),
        ("hidden_state_autocorrelation", lambda r: r["gate_statistics"]["hidden_state_autocorrelation"]),
        ("candidate_state_norm", lambda r: r["gate_statistics"]["candidate_state_norm"]),
        ("state_saturation_fraction", lambda r: r["gate_statistics"]["state_saturation_fraction"]),
    ]
    diagnostics = {}
    for key, getter in gate_keys:
        values = np.asarray([getter(row) for row in selected], dtype=float)
        diagnostics[key] = continuous_summary(values)

    per_family_gate = {}
    for family in FAMILIES:
        fr = [row for row in selected if row["family"] == family]
        per_family_gate[family] = {
            "update_temporal_variance": continuous_summary(np.asarray([r["gate_statistics"]["update"]["temporal_variance"] for r in fr], dtype=float)),
            "reset_temporal_variance": continuous_summary(np.asarray([r["gate_statistics"]["reset"]["temporal_variance"] for r in fr], dtype=float)),
            "update_input_conditioned_variance": continuous_summary(np.asarray([r["gate_statistics"]["update"]["input_conditioned_component_variance"] for r in fr], dtype=float)),
            "update_state_conditioned_variance": continuous_summary(np.asarray([r["gate_statistics"]["update"]["state_conditioned_component_variance"] for r in fr], dtype=float)),
            "reset_input_conditioned_variance": continuous_summary(np.asarray([r["gate_statistics"]["reset"]["input_conditioned_component_variance"] for r in fr], dtype=float)),
            "reset_state_conditioned_variance": continuous_summary(np.asarray([r["gate_statistics"]["reset"]["state_conditioned_component_variance"] for r in fr], dtype=float)),
        }

    replay = {}
    if condition == FULL:
        for family in FAMILIES:
            fr = [row for row in selected if row["family"] == family]
            for replay_name in ("update_time_shuffle", "update_episode_mean", "reset_time_shuffle", "reset_episode_mean"):
                values = np.asarray([r["gate_statistics"]["counterfactual_replay"][replay_name] for r in fr], dtype=float)
                replay.setdefault(replay_name, {})[family] = continuous_summary(values)

    return {
        "nominal_parameter_count": int(selected[0]["nominal_parameter_count"]),
        "active_parameter_count": int(selected[0]["active_parameter_count"]),
        "families_passing": int(families_passing),
        "family_results": per_family,
        "gate_diagnostics": diagnostics,
        "per_family_gate_diagnostics": per_family_gate,
        "counterfactual_replay": replay,
        "resource_accounting": {
            "model_fits": len(selected),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in selected)),
            "examples_processed": int(sum(row["resources"]["examples_processed"] for row in selected)),
            "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in selected)),
            "forward_calls": int(sum(row["resources"]["forward_calls"] for row in selected)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in selected)),
            "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in selected)),
            "gpu_seconds": 0.0,
        },
    }


def _run_conditions(conditions: list[str]) -> list[dict]:
    replicates = int(CONFIG["training"]["replicates"])
    jobs = [(condition, family, replicate) for condition in conditions for family in FAMILIES for replicate in range(replicates)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['condition']} {row['family']} r{row['replicate']}: "
                f"dev={row['development_success']:.3f} val={row['validation_success']:.3f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    return rows


def _positive_control_payload(rows: list[dict]) -> dict:
    summary = summarize_condition(FULL, rows)
    v837l = json.loads((ROOT / "experiments/v837_primitive_invention/v837l/results.json").read_text(encoding="utf-8"))
    reference = v837l["conditions"]["4x"]["gru_reference"]
    drift = {}
    over = 0
    threshold = float(CONFIG["positive_control_gate"]["compatibility_drift_threshold"])
    for family in FAMILIES:
        explicit_median = float(summary["family_results"][family]["validation"]["median"])
        reference_median = float(reference["family_results"][family]["validation"]["median"])
        absolute = abs(explicit_median - reference_median)
        drift[family] = {
            "explicit_validation_median": explicit_median,
            "v837l_validation_median": reference_median,
            "absolute_drift": absolute,
        }
        over += int(absolute > threshold)
    parameter_match = summary["nominal_parameter_count"] == int(CONFIG["models"]["reference_parameter_count"])
    compatible = (
        int(summary["families_passing"]) >= int(CONFIG["positive_control_gate"]["families_required"])
        and over <= int(CONFIG["positive_control_gate"]["max_families_over_drift_threshold"])
        and parameter_match
    )
    return {
        "compatible": bool(compatible),
        "families_passing": int(summary["families_passing"]),
        "parameter_count": int(summary["nominal_parameter_count"]),
        "reference_parameter_count": int(CONFIG["models"]["reference_parameter_count"]),
        "parameter_count_match": bool(parameter_match),
        "families_over_drift_threshold": int(over),
        "drift_threshold": threshold,
        "per_family": drift,
        "summary": summary,
        "fresh_audit_consumed": False,
    }


def _assert_science_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise SystemExit("fresh audit already consumed; V837n cannot proceed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("full", "ablations", "all"), default="all")
    args = parser.parse_args()
    _assert_science_locks()
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "diagnostics").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)

    if args.phase in {"full", "all"}:
        rows = _run_conditions([FULL])
        write_json(HERE / "raw" / "full_gru.json", {"rows": rows, "fresh_audit_consumed": False})
        payload = _positive_control_payload(rows)
        write_json(HERE / "diagnostics" / "full_gru_positive_control.json", payload)
        print(f"full explicit GRU: {payload['families_passing']}/5; compatible={payload['compatible']}", flush=True)
        if not payload["compatible"]:
            (HERE / "FAILURE.md").write_text(
                "# V837n IMPLEMENTATION FAILURE\n\nThe explicit full-GRU positive control did not reproduce the calibrated V837l reference regime. Ablations were not interpreted.\n",
                encoding="utf-8",
            )
            return 2

    if args.phase in {"ablations", "all"}:
        positive_path = HERE / "diagnostics" / "full_gru_positive_control.json"
        if not positive_path.exists():
            raise SystemExit("run --phase full first")
        positive = json.loads(positive_path.read_text(encoding="utf-8"))
        if positive.get("compatible") is not True:
            raise SystemExit("full explicit GRU compatibility failed; ablations blocked")
        rows = _run_conditions(ABLATIONS)
        write_json(HERE / "raw" / "ablations.json", {"rows": rows, "fresh_audit_consumed": False})
        summaries = {condition: summarize_condition(condition, rows) for condition in ABLATIONS}
        write_json(HERE / "diagnostics" / "ablation_summaries.json", summaries)
        print("ablation families passing: " + ", ".join(f"{name}={summaries[name]['families_passing']}/5" for name in ABLATIONS), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
