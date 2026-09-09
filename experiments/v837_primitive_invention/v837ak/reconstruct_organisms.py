from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import build_finalization_model, finalization_common_seed
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology
from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED, RANDOM, HERE, organism_id, parameter_hash, sha256_json, state_dict_hash, write_json

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
MODELS = HERE / "models" / "organisms"
RAW = HERE / "raw"
DIAG = HERE / "diagnostics"


def _load_rows() -> list[dict]:
    directed = json.loads((ROOT / CONFIG["source"]["directed_finalized"]).read_text(encoding="utf-8"))["rows"]
    random = json.loads((ROOT / CONFIG["source"]["random_finalized"]).read_text(encoding="utf-8"))["rows"]
    rows = [dict(row) for row in directed + random]
    if len(directed) != 25 or len(random) != 25 or len(rows) != 50:
        raise RuntimeError("V837AK_SOURCE_POPULATION_MISMATCH")
    counts = {
        "directed": sum(r["engine"] == DIRECTED for r in rows),
        "random": sum(r["engine"] == RANDOM for r in rows),
        "competent": sum(bool(r["competent"]) for r in rows),
        "incompetent": sum(not bool(r["competent"]) for r in rows),
    }
    expected = {"directed": 25, "random": 25, "competent": 40, "incompetent": 10}
    if counts != expected:
        raise RuntimeError(f"V837AK_SOURCE_POPULATION_MISMATCH: {counts} != {expected}")
    seen = set()
    for row in rows:
        oid = organism_id(row)
        if oid in seen:
            raise RuntimeError("V837AK_SOURCE_POPULATION_DUPLICATE")
        seen.add(oid)
        row["organism_id"] = oid
        row["source_row_sha256"] = sha256_json({k: v for k, v in row.items() if k not in {"organism_id", "source_row_sha256"}})
    rows.sort(key=lambda r: (r["engine"], r["family"], int(r["run_index"])))
    return rows


def freeze_source_population() -> list[dict]:
    auth = assert_v837ak_authorized()
    rows = _load_rows()
    payload_rows = []
    for row in rows:
        payload_rows.append({
            "organism_id": row["organism_id"],
            "engine": row["engine"],
            "family": row["family"],
            "run_index": int(row["run_index"]),
            "topology_id": row["topology_id"],
            "finalization_seed": int(row["finalization_seed"]),
            "initial_parameter_hash": row["initial_parameter_hash"],
            "development_success": float(row["development_success"]),
            "final_validation_success": float(row["final_validation_success"]),
            "competent": bool(row["competent"]),
            "source_row_sha256": row["source_row_sha256"],
        })
    source = {
        "version": "V837ak",
        "source_version": "V837aj",
        "authorization": auth,
        "count": len(rows),
        "directed": 25,
        "random": 25,
        "competent": 40,
        "incompetent": 10,
        "rows": payload_rows,
    }
    write_json(RAW / "source_population.json", source)
    write_json(DIAG / "source_population_integrity.json", {
        "version": "V837ak",
        "valid": True,
        "counts": {"total": 50, "directed": 25, "random": 25, "competent": 40, "incompetent": 10},
        "source_population_sha256": sha256_json(source),
        "v837aj_decision_sha256": auth["v837aj_decision_sha256"],
        "fresh_audit_consumed": False,
        "v838_started": False,
    })
    return rows


def _success_rate(task, prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(prediction.tolist(), targets.tolist())]))


def _checkpoint_path(row: dict) -> Path:
    short = "directed" if row["engine"] == DIRECTED else "random"
    return MODELS / f"{short}__{row['family']}__r{int(row['run_index'])}.pt"


def _reconstruct_one(row: dict) -> dict:
    _configure_torch()
    family = str(row["family"]); run_index = int(row["run_index"])
    topology = SearchTopology.from_dict(row["topology"])
    if topology.topology_id != row["topology_id"]:
        raise RuntimeError("V837AK_SOURCE_POPULATION_MISMATCH: topology id")
    expected_seed = int(finalization_common_seed(family, run_index))
    if expected_seed != int(row["finalization_seed"]):
        raise RuntimeError("V837AK_SOURCE_POPULATION_MISMATCH: finalization seed")
    task = task_by_name(family)
    model = build_finalization_model(topology, family, run_index)
    initial_hash = parameter_hash(model)
    if initial_hash != row["initial_parameter_hash"]:
        raise RuntimeError("FINALIZED_ORGANISM_RECONSTRUCTION_FAILURE: initial hash")
    train_seeds = list(range(10000, 10512)); validation_seeds = list(range(20000, 20128))
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0001, foreach=True)
    loss_fn = nn.MSELoss()
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    for _ in range(192):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(observations, lengths)
        loss = loss_fn(prediction, targets); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=True)
        optimizer.step()
    development = evaluate_sequence_model(model, task, train_episodes)
    val_obs, val_lengths, val_targets = episodes_to_batch(validation_episodes)
    model.eval()
    with torch.no_grad():
        val_prediction = model(val_obs, val_lengths)
    validation_success = _success_rate(task, val_prediction, val_targets)
    dev_delta = abs(float(development.success_rate) - float(row["development_success"]))
    val_delta = abs(validation_success - float(row["final_validation_success"]))
    competent = float(development.success_rate) >= 0.90 and validation_success >= 0.85
    equivalent = (
        dev_delta <= 1.0 / 512.0 + 1e-12
        and val_delta <= 1.0 / 128.0 + 1e-12
        and competent == bool(row["competent"])
    )
    if not equivalent:
        raise RuntimeError(
            f"FINALIZED_ORGANISM_RECONSTRUCTION_FAILURE {row['organism_id']} "
            f"dev={development.success_rate} old={row['development_success']} val={validation_success} oldval={row['final_validation_success']} competent={competent}/{row['competent']}"
        )
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    final_hash = state_dict_hash(state)
    checkpoint = {
        "version": "V837ak",
        "organism_id": row["organism_id"],
        "engine": row["engine"],
        "family": family,
        "run_index": run_index,
        "topology": row["topology"],
        "topology_id": row["topology_id"],
        "finalization_seed": expected_seed,
        "source_v837aj_row_sha256": row["source_row_sha256"],
        "initial_parameter_hash": initial_hash,
        "final_state_hash": final_hash,
        "development_success": float(development.success_rate),
        "final_validation_success": validation_success,
        "competent": bool(competent),
        "state_dict": state,
    }
    path = _checkpoint_path(row); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return {
        "organism_id": row["organism_id"], "engine": row["engine"], "family": family, "run_index": run_index,
        "topology_id": row["topology_id"], "finalization_seed": expected_seed,
        "initial_parameter_hash": initial_hash, "final_state_hash": final_hash,
        "source_v837aj_row_sha256": row["source_row_sha256"],
        "old_development_success": float(row["development_success"]), "development_success": float(development.success_rate), "development_delta": dev_delta,
        "old_final_validation_success": float(row["final_validation_success"]), "final_validation_success": validation_success, "validation_delta": val_delta,
        "old_competent": bool(row["competent"]), "competent": bool(competent), "equivalent": True,
        "checkpoint": path.relative_to(ROOT).as_posix(),
        "optimizer_steps": 192, "processed_examples": 192 * 512,
        "cpu_seconds": float(time.process_time() - start_cpu), "wall_seconds": float(time.perf_counter() - start_wall), "gpu_seconds": 0.0,
    }


def reconstruct_all() -> dict:
    rows = freeze_source_population()
    existing = {}
    results_path = RAW / "reconstruction_results.json"
    if results_path.is_file():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        existing = {r["organism_id"]: r for r in prior.get("rows", []) if r.get("equivalent") is True and (ROOT / r.get("checkpoint", "")).is_file()}
    jobs = [row for row in rows if row["organism_id"] not in existing]
    results = dict(existing)
    workers = min(int(CONFIG["reconstruction"]["max_workers"]), max(1, os.cpu_count() or 1), max(1, len(jobs)))
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_reconstruct_one, row): row for row in jobs}
            for future in as_completed(futures):
                row = future.result(); results[row["organism_id"]] = row
                print(f"reconstructed {row['engine']} {row['family']} r{row['run_index']}: dev={row['development_success']:.6f} val={row['final_validation_success']:.6f} competent={row['competent']}", flush=True)
                _write_reconstruction_payload(list(results.values()), incomplete=True)
    return _write_reconstruction_payload(list(results.values()), incomplete=False)


def _write_reconstruction_payload(rows: list[dict], *, incomplete: bool) -> dict:
    rows = sorted(rows, key=lambda r: (r["engine"], r["family"], int(r["run_index"])))
    complete = len(rows) == 50 and all(r.get("equivalent") is True for r in rows)
    payload = {
        "version": "V837ak", "stage": "AK0", "complete": complete and not incomplete,
        "organisms_reconstructed": len(rows), "competent": sum(bool(r.get("competent")) for r in rows),
        "incompetent": sum(not bool(r.get("competent")) for r in rows), "rows": rows,
        "resource_accounting": {
            "fits": len(rows), "optimizer_steps": sum(int(r.get("optimizer_steps", 0)) for r in rows),
            "processed_examples": sum(int(r.get("processed_examples", 0)) for r in rows),
            "cpu_seconds": sum(float(r.get("cpu_seconds", 0.0)) for r in rows),
            "wall_seconds_sum": sum(float(r.get("wall_seconds", 0.0)) for r in rows), "gpu_seconds": 0.0,
        },
    }
    write_json(RAW / "reconstruction_results.json", payload)
    if complete and not incomplete:
        write_json(DIAG / "reconstruction_equivalence.json", {
            "version": "V837ak", "pass": True, "organisms": 50,
            "exact_initial_hashes": all(r["initial_parameter_hash"] for r in rows),
            "competence_labels_exact": all(r["old_competent"] == r["competent"] for r in rows),
            "max_development_delta": max(r["development_delta"] for r in rows),
            "max_validation_delta": max(r["validation_delta"] for r in rows),
            "checkpoint_state_hashes": {r["organism_id"]: r["final_state_hash"] for r in rows},
        })
    return payload


def load_reconstructed_model(organism_row: dict):
    source_rows = _load_rows(); source = next(r for r in source_rows if r["organism_id"] == organism_row["organism_id"])
    topology = SearchTopology.from_dict(source["topology"])
    model = build_finalization_model(topology, source["family"], int(source["run_index"]))
    checkpoint = torch.load(ROOT / organism_row["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["source_v837aj_row_sha256"] != source["source_row_sha256"]:
        raise RuntimeError("V837AK_RECONSTRUCTION_SOURCE_HASH_MISMATCH")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if state_dict_hash({k: v.detach().cpu() for k, v in model.state_dict().items()}) != checkpoint["final_state_hash"]:
        raise RuntimeError("V837AK_RECONSTRUCTION_STATE_HASH_MISMATCH")
    model.eval()
    return model, source, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-only", action="store_true"); args = parser.parse_args()
    if args.source_only:
        rows = freeze_source_population(); print(json.dumps({"source_population": len(rows)}, indent=2)); return 0
    payload = reconstruct_all(); print(json.dumps({k: payload[k] for k in ("complete", "organisms_reconstructed", "competent", "incompetent", "resource_accounting")}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
