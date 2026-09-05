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
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _active_mask, _binary_entropy, _configure_torch, _success_rate, _temporal_variance
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import CONDITION_GRANULARITY, DynamicGranularityGRU, scalarize_dynamic_gate

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
ANCHORS = list(CONFIG["positive_control_conditions"])
SCALARIZED = [c for c in CONFIG["conditions"] if c not in ANCHORS]


def _git_blob_sha256(path: str) -> str:
    data = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def _gate_stats(effective: torch.Tensor, raw: torch.Tensor, lengths: torch.Tensor) -> dict:
    mask = _active_mask(lengths, effective.shape[1])
    eff = effective[mask].detach().cpu().numpy().astype(float)
    rv = raw[mask].detach().cpu().numpy().astype(float)
    flat = eff.reshape(-1)
    raw_flat = rv.reshape(-1)
    def eff_rank(x: np.ndarray) -> float:
        if x.shape[0] < 2:
            return 0.0
        cov = np.cov(x, rowvar=False)
        vals = np.linalg.eigvalsh(cov)
        vals = np.clip(vals, 0.0, None)
        total = float(vals.sum())
        if total <= 1e-15:
            return 0.0
        p = vals / total
        p = p[p > 1e-15]
        return float(np.exp(-(p * np.log(p)).sum()))
    corr = np.corrcoef(rv, rowvar=False) if rv.shape[0] > 1 else np.eye(rv.shape[1])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p10": float(np.quantile(flat, 0.10)),
        "p90": float(np.quantile(flat, 0.90)),
        "temporal_variance": float(_temporal_variance(effective, lengths)),
        "interdimension_variance": float(np.mean(np.var(eff, axis=1))),
        "underlying_vector_interdimension_variance": float(np.mean(np.var(rv, axis=1))),
        "near_zero_fraction": float(np.mean(flat <= 0.05)),
        "near_one_fraction": float(np.mean(flat >= 0.95)),
        "entropy": float(_binary_entropy(effective[mask])),
        "underlying_vector_mean": float(np.mean(raw_flat)),
        "underlying_vector_effective_rank": eff_rank(rv),
        "dimension_means": [float(x) for x in np.mean(rv, axis=0).tolist()],
        "dimension_temporal_variances": [float(x) for x in np.var(rv, axis=0).tolist()],
        "dimension_correlation_matrix": [[float(v) for v in row] for row in corr.tolist()],
    }


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return scalarize_dynamic_gate(t)


def _diagnostics(model: DynamicGranularityGRU, task, validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        predictions, trace = model(observations, lengths, return_trace=True)
    original = _success_rate(task, predictions, targets)
    diag = {
        "validation_success_recomputed": original,
        "update": _gate_stats(trace.updates, trace.raw_dynamic_updates, lengths),
        "reset": _gate_stats(trace.resets, trace.raw_dynamic_resets, lengths),
        "state_norm": float(torch.linalg.vector_norm(trace.states[_active_mask(lengths, trace.states.shape[1])], dim=-1).mean().item()),
        "counterfactual_flattening": {},
    }
    calls = 1
    updates_vector = model.update_granularity == "vector"
    resets_vector = model.reset_granularity == "vector"
    if updates_vector:
        with torch.no_grad():
            p = model(observations, lengths, update_override=_flatten(trace.raw_dynamic_updates))
        score = _success_rate(task, p, targets)
        diag["counterfactual_flattening"]["update"] = {"flattened_validation": score, "flattening_delta": original - score}
        calls += 1
    if resets_vector:
        with torch.no_grad():
            p = model(observations, lengths, reset_override=_flatten(trace.raw_dynamic_resets))
        score = _success_rate(task, p, targets)
        diag["counterfactual_flattening"]["reset"] = {"flattened_validation": score, "flattening_delta": original - score}
        calls += 1
    if updates_vector and resets_vector:
        with torch.no_grad():
            p = model(
                observations, lengths,
                update_override=_flatten(trace.raw_dynamic_updates),
                reset_override=_flatten(trace.raw_dynamic_resets),
            )
        score = _success_rate(task, p, targets)
        diag["counterfactual_flattening"]["both"] = {"flattened_validation": score, "flattening_delta": original - score}
        calls += 1
    return diag, calls


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    tr = CONFIG["training"]
    train_seeds = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    val_seeds = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    init_seed = deterministic_int(tr["initialization_namespace"], family, replicate)
    result = train_sequence_model(
        model_factory=lambda: DynamicGranularityGRU(CONFIG["hidden_size"], CONFIG["input_dim"], condition=condition),
        task=task,
        train_seeds=train_seeds,
        validation_seeds=val_seeds,
        initialization_seed=init_seed,
        steps=tr["steps"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        gradient_clip=tr["gradient_clip"],
        curve_steps=tuple(tr["curve_steps"]),
    )
    diagnostics, extra_calls = _diagnostics(result.model, task, val_seeds)
    result.resources.forward_calls += extra_calls
    return {
        "version": "V837t",
        "condition": condition,
        "family": family,
        "replicate_id": replicate,
        "development_seed_range": tr["development_seed_range"],
        "validation_seed_range": tr["validation_seed_range"],
        "model_init_seed": init_seed,
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "loss_curve": result.learning_curve,
        "update_mode": CONDITION_GRANULARITY[condition][0],
        "reset_mode": CONDITION_GRANULARITY[condition][1],
        "scalarization_mode": "post_sigmoid_mean_broadcast" if "scalarized" in CONDITION_GRANULARITY[condition] else "none",
        "gate_diagnostics": diagnostics,
        "nominal_parameter_count": result.model.nominal_parameter_count(),
        "active_parameter_count": result.model.active_parameter_count(),
        "parameter_bytes": result.model.parameter_bytes(),
        "optimizer_steps": result.resources.optimizer_steps,
        "processed_examples": result.resources.examples_processed,
        "unique_seed_defined_episode_policy": "3200 paired family/seed episodes reused across conditions and replicates",
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs = [(c, f, r) for c in conditions for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda x: (x["condition"], x["family"], x["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    out = {"families_passing": 0, "family_validation_medians": {}, "family_development_medians": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        dev = float(np.median([r["development_success"] for r in fr]))
        val = float(np.median([r["validation_success"] for r in fr]))
        out["family_development_medians"][family] = dev
        out["family_validation_medians"][family] = val
        out["families_passing"] += int(capacity_demonstrated(dev, val))
    return out


def _anchor_guard(rows: list[dict]) -> dict:
    historical = json.loads((ROOT / "experiments/v837_primitive_invention/v837o/results.json").read_text(encoding="utf-8"))["conditions"]
    mapping = {
        "T0_full_vector_gru": "G0_full_dynamic",
        "T1_vector_update_no_reset": "G1_dynamic_update_no_reset",
        "T3_no_update_vector_reset": "G2_no_update_dynamic_reset",
    }
    summaries = {}
    drift = {}
    ok = True
    for current, old in mapping.items():
        summaries[current] = _summary(rows, current)
        ok = ok and summaries[current]["families_passing"] >= CONFIG["representation_family_gate"]
        drift[current] = {
            family: abs(summaries[current]["family_validation_medians"][family] - historical[old]["family_results"][family]["validation"]["median"])
            for family in FAMILIES
        }
    return {"positive_controls_pass": bool(ok), "summaries": summaries, "historical_absolute_drift": drift}


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837s/results.json") != CONFIG["v837s_result_sha256"]:
        raise SystemExit("V837s result changed")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837o/results.json") != CONFIG["v837o_result_sha256"]:
        raise SystemExit("V837o reference result changed")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit is not available")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("anchors", "scalarized", "all"), default="all")
    args = parser.parse_args()
    _assert_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(exist_ok=True)
    seed_policy = {
        "development_seed_range": CONFIG["training"]["development_seed_range"],
        "validation_seed_range": CONFIG["training"]["validation_seed_range"],
        "unique_seed_defined_episodes": CONFIG["unique_seed_defined_episodes"],
        "reuse_policy": "same 3200 family/seed episodes reused across every condition and replicate",
    }
    if args.phase in {"anchors", "all"}:
        rows = _run(ANCHORS)
        write_json(HERE / "raw" / "anchor_runs.json", {"rows": rows, "seed_policy": seed_policy})
        guard = _anchor_guard(rows)
        write_json(HERE / "diagnostics" / "positive_control_guard.json", guard)
        print("anchors: " + ", ".join(f"{c}={guard['summaries'][c]['families_passing']}/5" for c in ANCHORS), flush=True)
        if not guard["positive_controls_pass"]:
            (HERE / "FAILURE.md").write_text("# V837t REFERENCE_BASELINE_DRIFT\n\nAt least one required successful reference anchor fell below 4/5. Scalarized conditions were not interpreted.\n", encoding="utf-8")
            return 2
    if args.phase in {"scalarized", "all"}:
        guard_path = HERE / "diagnostics" / "positive_control_guard.json"
        if not guard_path.exists() or not json.loads(guard_path.read_text(encoding="utf-8")).get("positive_controls_pass"):
            raise SystemExit("V837t scalarized conditions blocked until positive anchors pass")
        rows = _run(SCALARIZED)
        write_json(HERE / "raw" / "scalarized_runs.json", {"rows": rows, "seed_policy": seed_policy})
        print("scalarized: " + ", ".join(f"{c}={_summary(rows,c)['families_passing']}/5" for c in SCALARIZED), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
