from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
V837AF_CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/config.json").read_text(encoding="utf-8"))
FAMILIES = [t.name for t in all_tasks()]
CONDITION = "AF1D_deshared_candidate_input_factorization"

PROTECTED_PATHS = {
    "v837af_config": "experiments/v837_primitive_invention/v837af/config.json",
    "v837af_model": "experiments/v837_primitive_invention/v837af/candidate_input_factorization.py",
    "v837af_frozen_gate": "experiments/v837_primitive_invention/v837af/frozen_candidate_input_transfer_gate.json",
    "v837af_results": "experiments/v837_primitive_invention/v837af/results.json",
    "v837af_raw_transfer": "experiments/v837_primitive_invention/v837af/raw/transfer_runs.json",
    "v837af_decision": "experiments/v837_primitive_invention/v837af/diagnostics/decision_state.json",
    "v837y_parent_model": "experiments/v837_primitive_invention/v837y/candidate_interaction.py",
    "v837r_rank4_coupling": "experiments/v837_primitive_invention/v837r/recurrent_coupling.py",
    "historical_graph_gate": "experiments/v837_primitive_invention/frozen_gates.json",
}

V837L_PATHS = {
    "config": "experiments/v837_primitive_invention/v837l/config.json",
    "results": "experiments/v837_primitive_invention/v837l/results.json",
    "raw_2x": "experiments/v837_primitive_invention/v837l/diagnostics/raw_runs_2x.json",
    "raw_4x": "experiments/v837_primitive_invention/v837l/diagnostics/raw_runs_4x.json",
}


def git_blob_sha256(relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def architecture_lock() -> dict:
    expected = CONFIG["architecture_hashes"]
    observed = {key: git_blob_sha256(path) for key, path in PROTECTED_PATHS.items()}
    compatible = observed == expected
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        compatible = False
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        compatible = False
    payload = {
        "compatible": bool(compatible),
        "expected_hashes": expected,
        "observed_hashes": observed,
        "historical_gate_runtime_sha256": gate_sha256(),
        "capacity_criterion_runtime_sha256": v837_capacity_criterion_sha256(),
        "source_sha": CONFIG["source_sha"],
        "source_condition": CONDITION,
        "failure_code": None if compatible else "AF1D_ARCHITECTURE_LOCK_FAILURE",
    }
    return payload


def historical_v837l_lock() -> dict:
    expected = CONFIG["historical_v837l_hashes"]
    observed = {key: git_blob_sha256(path) for key, path in V837L_PATHS.items()}
    return {"compatible": observed == expected, "expected_hashes": expected, "observed_hashes": observed}


def development_seeds(multiplier: int) -> list[int]:
    if multiplier not in (1, 2, 4):
        raise ValueError(f"unsupported V837ai multiplier: {multiplier}")
    return list(range(10000, 10000 + 128 * int(multiplier)))


def validation_seeds() -> list[int]:
    return list(range(20000, 20128))


def data_nesting_diagnostic() -> dict:
    ai1, ai2, ai4 = map(set, (development_seeds(1), development_seeds(2), development_seeds(4)))
    val = set(validation_seeds())
    exact = (
        sorted(ai1) == list(range(10000, 10128))
        and sorted(ai2) == list(range(10000, 10256))
        and sorted(ai4) == list(range(10000, 10512))
        and sorted(val) == list(range(20000, 20128))
    )
    return {
        "exact": bool(exact),
        "strict_nesting": bool(ai1 < ai2 and ai2 < ai4),
        "development_validation_overlap": len(ai4 & val),
        "duplicates": {
            "1x": len(development_seeds(1)) - len(ai1),
            "2x": len(development_seeds(2)) - len(ai2),
            "4x": len(development_seeds(4)) - len(ai4),
            "validation": len(validation_seeds()) - len(val),
        },
        "ranges": {"1x": [10000, 10127], "2x": [10000, 10255], "4x": [10000, 10511], "validation": [20000, 20127]},
        "counts_per_family": {"1x": 128, "2x": 256, "4x": 512, "validation": 128},
        "total_unique_family_seed_episodes": {"1x": 1280, "2x": 1920, "4x": 3200, "union": 3200},
    }


def base_seed(family: str, replicate: int) -> int:
    tr = V837AF_CONFIG["training"]
    return deterministic_int(tr["base_initialization_namespace"], family, int(replicate))


def coupling_seed(replicate: int) -> int:
    tr = V837AF_CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], int(replicate))


def projection_seed(family: str, replicate: int) -> int:
    tr = V837AF_CONFIG["training"]
    return deterministic_int(tr["projection_initialization_namespace"], family, int(replicate))


def build_af1d(family: str, replicate: int) -> CandidateInputFactorizationY3:
    seed = base_seed(family, replicate)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return CandidateInputFactorizationY3(
        high_capacity_generic_graph(int(replicate)),
        condition=CONDITION,
        coupling_initialization_seed=coupling_seed(replicate),
        projection_seed=projection_seed(family, replicate),
    )


def trainable_tensor_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        arr = param.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(str(arr.dtype).encode("ascii")); h.update(b"\0")
        h.update(str(tuple(arr.shape)).encode("ascii")); h.update(b"\0")
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def initialization_pairing_diagnostic(ai4_rows: list[dict]) -> dict:
    by_key = {(r["family"], int(r["replicate_id"])): r for r in ai4_rows}
    rows = []
    all_equal = True
    parameter_count_ok = True
    macs_ok = True
    projection_initial_identical = True
    for family in FAMILIES:
        for replicate in range(5):
            source = by_key[(family, replicate)]
            hashes = []
            snapshots = []
            for multiplier in (1, 2, 4):
                model = build_af1d(family, replicate)
                hashes.append(trainable_tensor_fingerprint(model))
                snapshots.append({n: p.detach().cpu().clone() for n, p in model.named_parameters()})
                parameter_count_ok &= model.parameter_count() == CONFIG["expected_active_parameters"]
                macs_ok &= model.total_recurrent_controller_projection_macs == CONFIG["expected_active_macs_per_timestep"]
                layers = model.cell_candidate_projections
                if layers is None or len(layers) != 10:
                    projection_initial_identical = False
                else:
                    w0, b0 = layers[0].weight.detach(), layers[0].bias.detach()
                    projection_initial_identical &= all(torch.equal(w0, l.weight.detach()) and torch.equal(b0, l.bias.detach()) for l in layers[1:])
            numerical_equal = all(
                all(torch.equal(snapshots[0][name], snapshots[j][name]) for name in snapshots[0])
                for j in (1, 2)
            )
            source_seed_match = (
                int(source["initialization_seed"]) == base_seed(family, replicate)
                and int(source["coupling_initialization_seed"]) == coupling_seed(replicate)
                and int(source["projection_initialization_seed"]) == projection_seed(family, replicate)
            )
            equal = len(set(hashes)) == 1 and numerical_equal and source_seed_match
            all_equal &= equal
            rows.append({
                "family": family, "replicate_id": replicate,
                "ai1_hash": hashes[0], "ai2_hash": hashes[1], "ai4_reconstructed_hash": hashes[2],
                "all_equal": bool(equal), "parameter_by_parameter_equal": bool(numerical_equal),
                "historical_seed_fields_match": bool(source_seed_match),
                "base_seed": base_seed(family, replicate), "coupling_seed": coupling_seed(replicate), "projection_seed": projection_seed(family, replicate),
            })
    return {
        "pairing_exact": bool(all_equal),
        "method": "deterministic reconstruction under frozen V837af source hashes plus parameter-by-parameter equality; historical AI4 seed fields cross-checked",
        "parameter_count_constant": bool(parameter_count_ok),
        "macs_constant": bool(macs_ok),
        "ten_projection_copies_identical_at_step0": bool(projection_initial_identical),
        "rows": rows,
    }


def load_ai4_rows() -> list[dict]:
    raw = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/raw/transfer_runs.json").read_text(encoding="utf-8"))
    rows = [r for r in raw["rows"] if r.get("condition") == CONDITION]
    return sorted(rows, key=lambda r: (r["family"], int(r["replicate_id"])))


def summarize_rows(rows: list[dict]) -> dict:
    out = {"families_passing": 0, "family_results": {}}
    for family in FAMILIES:
        fr = sorted([r for r in rows if r["family"] == family], key=lambda r: int(r["replicate_id"]))
        if len(fr) != 5:
            raise ValueError(f"expected five rows for {family}, found {len(fr)}")
        dev = float(np.median([float(r["development_success"]) for r in fr]))
        val = float(np.median([float(r["validation_success"]) for r in fr]))
        passed = capacity_demonstrated(dev, val)
        out["families_passing"] += int(passed)
        out["family_results"][family] = {"development_median": dev, "validation_median": val, "pass": bool(passed)}
    return out


def ai4_reuse_diagnostic() -> tuple[dict, list[dict]]:
    rows = load_ai4_rows()
    af = V837AF_CONFIG
    source_results = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/results.json").read_text(encoding="utf-8"))
    source_decision = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    conditions_ok = (
        len(rows) == 25
        and set(r["family"] for r in rows) == set(FAMILIES)
        and all(sum(1 for r in rows if r["family"] == f) == 5 for f in FAMILIES)
        and all(r["condition"] == CONDITION for r in rows)
    )
    tr = af["training"]
    training_ok = (
        af["data_regime"] == "4x_unique" and tr["steps"] == 192 and tr["train_episodes"] == 512
        and tr["development_seed_range"] == [10000,10511] and tr["validation_episodes"] == 128
        and tr["validation_seed_range"] == [20000,20127] and tr["learning_rate"] == 0.005
        and tr["weight_decay"] == 0.0001 and tr["gradient_clip"] == 5.0 and tr["replicates"] == 5
    )
    architecture_ok = (
        CONDITION in af["conditions"] and source_decision.get("best_passing_condition") == CONDITION
        and source_decision.get("representation_adequacy_pass") is True and source_decision.get("sample_efficiency_retest_allowed") is True
        and source_results["conditions"][CONDITION]["active_parameter_count"] == CONFIG["expected_active_parameters"]
        and source_results["conditions"][CONDITION]["recurrent_controller_projection_macs"] == CONFIG["expected_active_macs_per_timestep"]
    )
    reanalysis = summarize_rows(rows)
    expected = {
        "conditional_routing":0.9375,
        "delayed_recall":0.984375,
        "iterative_state":1.0,
        "partial_observation":0.8125,
        "variable_composition":0.875,
    }
    exact_reanalysis = reanalysis["families_passing"] == 4 and all(reanalysis["family_results"][f]["validation_median"] == v for f,v in expected.items())
    compatible = conditions_ok and training_ok and architecture_ok and exact_reanalysis
    payload = {
        "compatible": bool(compatible), "source_version":"V837af", "source_condition":CONDITION,
        "source_sha":CONFIG["source_sha"], "rows_reused":len(rows),
        "source_results_sha256":CONFIG["architecture_hashes"]["v837af_results"],
        "source_raw_run_sha256":CONFIG["architecture_hashes"]["v837af_raw_transfer"],
        "source_config_sha256":CONFIG["architecture_hashes"]["v837af_config"],
        "source_model_sha256":CONFIG["architecture_hashes"]["v837af_model"],
        "source_decision_sha256":CONFIG["architecture_hashes"]["v837af_decision"],
        "checks":{"rows_complete":conditions_ok,"training_protocol":training_ok,"architecture":architecture_ok,"exact_reanalysis":exact_reanalysis},
        "reanalyzed":reanalysis,
        "expected_validation_medians":expected,
        "failure_code":None if compatible else "AF1D_4X_ANCHOR_COMPATIBILITY_FAILURE",
    }
    return payload, rows


def historical_v837l_comparison() -> dict:
    results = json.loads((ROOT / "experiments/v837_primitive_invention/v837l/results.json").read_text(encoding="utf-8"))
    counts = {}
    for label in ("1x","2x","4x"):
        c = results["conditions"][label]
        counts[label] = {
            "gru_reference": int(c["gru_reference"]["families_passing"]),
            "neutral_high_capacity": int(c["neutral_high_capacity"]["families_passing"]),
            "residual_rnn_reference": int(c["residual_rnn_reference"]["families_passing"]),
        }
    expected = {
        "1x":{"gru_reference":2,"neutral_high_capacity":1,"residual_rnn_reference":2},
        "2x":{"gru_reference":3,"neutral_high_capacity":1,"residual_rnn_reference":2},
        "4x":{"gru_reference":5,"neutral_high_capacity":2,"residual_rnn_reference":3},
    }
    return {
        "compatible": counts == expected,
        "source":"V837l", "hashes":CONFIG["historical_v837l_hashes"], "families_passing":counts,
        "historical_gru_minimum_tested_multiplier":4,
        "historical_gru_parameters":875,
        "historical_gru_macs_per_timestep":777,
        "historical_gru_macs_derivation":"6x6 input projection = 36 MACs; GRUCell 3 gates * (6*13 input + 13*13 recurrent) = 741 MACs; total matrix MACs = 777. Bias/nonlinearity/readout excluded to match recurrent/controller/projection accounting semantics.",
    }
