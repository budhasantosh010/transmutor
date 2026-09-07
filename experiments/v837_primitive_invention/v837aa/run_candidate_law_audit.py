from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837y.candidate_interaction import CandidateInteractionSpec, ControlledCandidateInteractionModel

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GATE = json.loads((HERE / "frozen_candidate_law_gate.json").read_text(encoding="utf-8"))
Y_CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITION = "Y3_global_control_rank4_candidate"


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_hashes(hash_provider=_git_blob_sha256) -> dict:
    observed = {}
    for path, expected in CONFIG["frozen_parent_git_blob_sha256"].items():
        actual = hash_provider(path)
        observed[path] = actual
        if actual != expected:
            raise RuntimeError(f"frozen parent hash changed: {path}: {actual} != {expected}")
    return observed


def assert_fresh_audit_state(payload: dict) -> None:
    if payload.get("episodes_consumed") != 0:
        raise RuntimeError("fresh audit consumption is nonzero")


def assert_science_locks() -> dict:
    observed = verify_frozen_hashes()
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise RuntimeError("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise RuntimeError("capacity criterion changed")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    assert_fresh_audit_state(audit)
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise RuntimeError("V838 exists")
    if (ROOT / "experiments/v837_primitive_invention/v837ab").exists():
        raise RuntimeError("V837ab must not exist during V837aa")
    if CONFIG["rerun_conditions"] != [CONDITION]:
        raise RuntimeError("V837aa may rerun only Y3")
    if Y_CONFIG["training"]["steps"] != CONFIG["training"]["steps"]:
        raise RuntimeError("training budget drift")
    return observed


def _coupling_seed(replicate: int) -> int:
    tr = CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], replicate)


def _model(replicate: int) -> ControlledCandidateInteractionModel:
    spec = CandidateInteractionSpec(
        global_scalar_control=True,
        candidate_coupling_mode="rank4_cross_block",
        coupling_rank=4,
    )
    return ControlledCandidateInteractionModel(
        high_capacity_generic_graph(replicate),
        spec=spec,
        coupling_initialization_seed=_coupling_seed(replicate),
    )


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    validation = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    if len(train) != tr["development_episodes_per_family"] or len(validation) != tr["validation_episodes_per_family"]:
        raise RuntimeError("configured seed ranges do not match frozen episode counts")
    return train, validation


def _tensor_record(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "data": value.tolist(),
    }


def snapshot_model(model: ControlledCandidateInteractionModel, *, family: str, replicate: int, phase: str) -> dict:
    state = {name: _tensor_record(tensor) for name, tensor in sorted(model.state_dict().items())}
    return {
        "version": "V837aa",
        "parent_condition": CONDITION,
        "family": family,
        "replicate_id": replicate,
        "phase": phase,
        "parameter_count": model.parameter_count(),
        "active_recurrent_controller_macs": model.recurrent_controller_macs,
        "parameter_key_map": {
            "cell_ws": [f"base.cell_ws.{i}" for i in range(10)],
            "cell_wm": [f"base.cell_wm.{i}" for i in range(10)],
            "cell_wx": [f"base.cell_wx.{i}" for i in range(10)],
            "cell_b": [f"base.cell_b.{i}" for i in range(10)],
            "cell_wo": [f"base.cell_wo.{i}" for i in range(10)],
            "edge_weights": "base.edge_weights",
            "global_u": "global_u",
            "global_v": "global_v",
            "global_ws": "global_ws",
            "global_wx": "global_wx",
            "global_b": "global_b",
            "readout_weight": "base.readout.weight",
            "readout_bias": "base.readout.bias"
        },
        "state_dict": state,
    }


def _worker(family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _seeds()
    tr = CONFIG["training"]
    initialization_seed = deterministic_int(tr["base_initialization_namespace"], family, replicate)
    holder: dict[str, dict] = {}

    def factory():
        model = _model(replicate)
        holder["initial"] = snapshot_model(model, family=family, replicate=replicate, phase="initial")
        return model

    result = train_sequence_model(
        model_factory=factory,
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=initialization_seed,
        steps=tr["steps"],
        learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"],
        gradient_clip=tr["gradient_clip"],
        curve_steps=(0, 24, 48, 96, 144, 192),
    )
    trained = snapshot_model(result.model, family=family, replicate=replicate, phase="trained")
    resources = result.resources.to_dict()
    return {
        "run": {
            "version": "V837aa",
            "parent_condition": CONDITION,
            "family": family,
            "replicate_id": replicate,
            "initialization_seed": initialization_seed,
            "coupling_initialization_seed": _coupling_seed(replicate),
            "development_success": result.development.success_rate,
            "validation_success": result.validation.success_rate,
            "development_loss": result.development.loss,
            "validation_loss": result.validation.loss,
            "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
            "parameter_count": result.model.parameter_count(),
            "active_parameter_count": result.model.parameter_count(),
            "recurrent_controller_macs": result.model.recurrent_controller_macs,
            "loss_curve": result.learning_curve,
            "training_resources": resources,
            "training_forward_calls": int(resources.get("forward_calls", 0)),
            "training_backward_calls": int(tr["steps"]),
            "processed_training_examples": int(tr["steps"] * tr["development_episodes_per_family"]),
            "fresh_audit_consumed": False,
            "structural_search_allowed": False,
            "primitive_mining_allowed": False,
            "v838_started": False,
        },
        "initial": holder["initial"],
        "trained": trained,
    }


def _parent_rows() -> dict[tuple[str, int], dict]:
    payload = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/raw/interaction_runs.json").read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["condition"] == CONDITION]
    if len(rows) != 25:
        raise RuntimeError("committed V837y parent does not contain 25 Y3 rows")
    return {(row["family"], int(row["replicate_id"])): row for row in rows}


def _family_summary(rows: list[dict]) -> dict:
    output = {"families_passing": 0, "family_medians": {}}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        dev = float(np.median([row["development_success"] for row in selected]))
        val = float(np.median([row["validation_success"] for row in selected]))
        passed = capacity_demonstrated(dev, val)
        output["family_medians"][family] = {"development": dev, "validation": val, "capacity_demonstrated": passed}
        output["families_passing"] += int(passed)
    return output


def parent_reproduction(rows: list[dict]) -> dict:
    parent = _parent_rows()
    tolerance = CONFIG["parent_reproduction_tolerance"]
    per_run = []
    valid = True
    for row in rows:
        key = (row["family"], int(row["replicate_id"]))
        expected = parent[key]
        val_delta = abs(float(row["validation_success"]) - float(expected["validation_success"]))
        dev_delta = abs(float(row["development_success"]) - float(expected["development_success"]))
        row_valid = (
            val_delta <= tolerance["validation_success"] + 1e-12
            and dev_delta <= tolerance["development_success"] + 1e-12
            and int(row["parameter_count"]) == int(expected["parameter_count"]) == int(tolerance["parameter_count"])
            and int(row["recurrent_controller_macs"]) == int(expected["total_recurrent_controller_macs"]) == int(tolerance["recurrent_controller_macs"])
        )
        valid = valid and row_valid
        per_run.append({
            "family": key[0], "replicate_id": key[1],
            "expected_development_success": expected["development_success"],
            "observed_development_success": row["development_success"],
            "development_abs_delta": dev_delta,
            "expected_validation_success": expected["validation_success"],
            "observed_validation_success": row["validation_success"],
            "validation_abs_delta": val_delta,
            "valid": row_valid,
        })
    observed = _family_summary(rows)
    parent_rows = [{
        "family": family,
        "replicate_id": replicate,
        "development_success": payload["development_success"],
        "validation_success": payload["validation_success"],
    } for (family, replicate), payload in parent.items()]
    expected = _family_summary(parent_rows)
    family_deltas = {}
    for family in FAMILIES:
        delta = abs(observed["family_medians"][family]["validation"] - expected["family_medians"][family]["validation"])
        family_deltas[family] = delta
        if delta > tolerance["family_median_validation"] + 1e-12:
            valid = False
    if observed["families_passing"] != expected["families_passing"] or observed["families_passing"] != tolerance["required_family_pass_count"]:
        valid = False
    if sorted(set(row["family"] for row in rows)) != sorted(FAMILIES):
        valid = False
    return {
        "parent_reproduction_valid": valid,
        "condition": CONDITION,
        "data_regime": "4x_unique",
        "unique_task_episodes": 3200,
        "expected_families_passing": expected["families_passing"],
        "observed_families_passing": observed["families_passing"],
        "expected_family_medians": expected["family_medians"],
        "observed_family_medians": observed["family_medians"],
        "family_validation_median_abs_deltas": family_deltas,
        "tolerances": tolerance,
        "per_run": per_run,
    }


def _resource_summary(rows: list[dict], execution_wall_seconds: float) -> dict:
    training_cpu = float(sum(row["training_resources"].get("cpu_seconds", 0.0) for row in rows))
    training_worker_wall = float(sum(row["training_resources"].get("wall_seconds", 0.0) for row in rows))
    training_forward = int(sum(row["training_forward_calls"] for row in rows))
    training_backward = int(sum(row["training_backward_calls"] for row in rows))
    return {
        "version": "V837aa",
        "model_fits": len(rows),
        "optimizer_steps": int(sum(row["training_resources"].get("optimizer_steps", 0) for row in rows)),
        "processed_training_examples": int(sum(row["processed_training_examples"] for row in rows)),
        "unique_seed_defined_task_episodes": 3200,
        "fresh_task_episodes": 0,
        "training_forward_calls": training_forward,
        "diagnostic_forward_calls": 0,
        "training_backward_calls": training_backward,
        "diagnostic_backward_calls": 0,
        "synthetic_probes": 0,
        "empirical_probes": 0,
        "cpu_seconds_training_workers": training_cpu,
        "wall_seconds_training_workers_sum": training_worker_wall,
        "wall_seconds_training_elapsed": execution_wall_seconds,
        "gpu_seconds": 0.0,
        "fresh_audit_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun only the frozen V837y Y3 parent and capture portable parameter snapshots for V837aa.")
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    observed_hashes = assert_science_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    jobs = [(family, replicate) for family in FAMILIES for replicate in range(CONFIG["replicates"])]
    started = time.perf_counter()
    outputs = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        futures = {pool.submit(_worker, family, replicate): (family, replicate) for family, replicate in jobs}
        for future in as_completed(futures):
            payload = future.result()
            row = payload["run"]
            outputs.append(payload)
            print(f"{row['family']} r{row['replicate_id']}: dev={row['development_success']:.6f} val={row['validation_success']:.6f}", flush=True)
    elapsed = time.perf_counter() - started
    outputs.sort(key=lambda payload: (payload["run"]["family"], payload["run"]["replicate_id"]))
    rows = [payload["run"] for payload in outputs]
    initial = [payload["initial"] for payload in outputs]
    trained = [payload["trained"] for payload in outputs]
    seed_policy = {
        "development_seed_range": CONFIG["training"]["development_seed_range"],
        "validation_seed_range": CONFIG["training"]["validation_seed_range"],
        "unique_seed_defined_task_episodes": 3200,
        "reuse_policy": "exact V837y 4x family/seed episodes reused across all 25 Y3 fits",
    }
    write_json(HERE / "raw" / "runs.json", {"rows": rows, "seed_policy": seed_policy})
    write_json(HERE / "raw" / "initial_parameter_snapshots.json", {"snapshots": initial})
    write_json(HERE / "raw" / "trained_parameter_snapshots.json", {"snapshots": trained})
    reproduction = parent_reproduction(rows)
    write_json(HERE / "diagnostics" / "parent_reproduction.json", reproduction)
    resources = _resource_summary(rows, elapsed)
    write_json(HERE / "diagnostics" / "training_resource_accounting.json", resources)
    hashes = {
        "initial_parameter_snapshots_sha256": _sha256_file(HERE / "raw" / "initial_parameter_snapshots.json"),
        "trained_parameter_snapshots_sha256": _sha256_file(HERE / "raw" / "trained_parameter_snapshots.json"),
        "runs_sha256": _sha256_file(HERE / "raw" / "runs.json"),
        "frozen_parent_git_blob_sha256_verified": observed_hashes,
    }
    write_json(HERE / "diagnostics" / "snapshot_hashes.json", hashes)
    if not reproduction["parent_reproduction_valid"]:
        failure = {
            "version": "V837aa",
            "audit_valid": False,
            "parent_reproduction_valid": False,
            "candidate_law_diagnosis": "Y3_PARENT_REPRODUCTION_FAILURE",
            "recommended_next_axis": "DIAGNOSE_DETERMINISTIC_OR_ENVIRONMENT_DRIFT",
            "representation_adequacy": "still_3_of_5_parent",
            "fresh_audit_consumed": False,
            "primitive_count": 0,
            "structural_search_allowed": False,
            "primitive_mining_allowed": False,
            "v838_started": False,
        }
        write_json(HERE / "results.json", failure)
        write_json(HERE / "diagnostics" / "decision_state.json", failure)
        (HERE / "VERDICT.md").write_text("# V837aa — Y3_PARENT_REPRODUCTION_FAILURE\n\nThe regenerated Y3 parent did not satisfy the frozen reproduction gate. Candidate-law alignment is not interpreted.\n", encoding="utf-8")
        return 2
    print(json.dumps({"parent_reproduction_valid": True, "families_passing": reproduction["observed_families_passing"], "snapshot_hashes": hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
