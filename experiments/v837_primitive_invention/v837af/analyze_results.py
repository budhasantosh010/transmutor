from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])


def _load_rows(name: str) -> list[dict]:
    path = HERE / "raw" / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _rows() -> list[dict]:
    return _load_rows("af0_runs.json") + _load_rows("transfer_runs.json")


def _summary(rows: list[dict], condition: str) -> dict:
    out = {"families_passing": 0, "family_results": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        if not fr:
            continue
        dev_values = [float(r["development_success"]) for r in fr]
        val_values = [float(r["validation_success"]) for r in fr]
        dev = float(np.median(dev_values))
        val = float(np.median(val_values))
        passed = bool(capacity_demonstrated(dev, val))
        out["families_passing"] += int(passed)
        out["family_results"][family] = {
            "development": {"median": dev, "values": dev_values},
            "validation": {"median": val, "values": val_values},
            "capacity_demonstrated": passed,
        }
    return out


def _median(rows: list[dict], path: tuple[str, ...], default: float = 0.0) -> float:
    values = []
    for row in rows:
        current = row
        try:
            for key in path:
                current = current[key]
            values.append(float(current))
        except (KeyError, TypeError, ValueError):
            pass
    return float(np.median(values)) if values else default


def _projection_dynamics(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        if not cr:
            continue
        steps = {}
        for step in CONFIG["training"]["curve_steps"]:
            snapshots = []
            for row in cr:
                snapshots.extend([x for x in row["projection_trajectory"] if int(x["step"]) == int(step)])
            if not snapshots:
                continue
            modes = [x["projection"].get("mode") for x in snapshots]
            shared_like = [x["projection"] for x in snapshots if "frobenius_norm" in x["projection"]]
            deshared = [x["projection"] for x in snapshots if x["projection"].get("mode") == "deshared"]
            steps[str(step)] = {
                "mode": modes[0] if modes else None,
                "frobenius_norm_median": float(np.median([x.get("frobenius_norm", 0.0) for x in shared_like])) if shared_like else 0.0,
                "condition_number_median": float(np.median([x["condition_number"] for x in shared_like if x.get("condition_number") is not None])) if any(x.get("condition_number") is not None for x in shared_like) else None,
                "projection_weight_drift_median": float(np.median([x.get("weight_drift", 0.0) for x in shared_like])) if shared_like else 0.0,
                "projection_bias_drift_median": float(np.median([x.get("bias_drift", 0.0) for x in shared_like])) if shared_like else 0.0,
                "deshared_pairwise_distance_median": float(np.median([x.get("pairwise_weight_distance_median", 0.0) for x in deshared])) if deshared else 0.0,
                "deshared_pairwise_cosine_median": float(np.median([x.get("pairwise_weight_cosine_median", 1.0) for x in deshared])) if deshared else 1.0,
                "deshared_divergence_from_shared_initialization_mean_median": float(np.median([x.get("projection_divergence_from_shared_initialization_mean", 0.0) for x in deshared])) if deshared else 0.0,
            }
        output[condition] = steps
    return output


def _effective_maps(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        if not cr:
            continue
        output[condition] = {
            "pairwise_effective_input_map_cosine_median": _median(cr, ("diagnostics", "effective_input_maps", "pairwise_cosine_median")),
            "pairwise_effective_input_map_distance_median": _median(cr, ("diagnostics", "effective_input_maps", "pairwise_distance_median")),
            "effective_input_weight_norm_per_cell_median": [
                float(np.median([r["diagnostics"]["effective_input_maps"]["per_cell"][cell]["weight_norm"] for r in cr]))
                for cell in range(10)
            ],
        }
    return output


def _partial_observation(rows: list[dict], summaries: dict[str, dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        if not cr:
            continue
        family_results = summaries[condition]["family_results"]
        output[condition] = {
            "conditional_routing_validation_median": family_results["conditional_routing"]["validation"]["median"],
            "partial_observation_validation_median": family_results["partial_observation"]["validation"]["median"],
            "variable_composition_validation_median": family_results["variable_composition"]["validation"]["median"],
            "candidate_input_term_norm_median": _median(cr, ("diagnostics", "candidate_input_term", "mean_norm")),
            "candidate_input_temporal_variance_median": _median(cr, ("diagnostics", "candidate_input_term", "mean_temporal_variance")),
            "candidate_jacobian_wrt_visible_input_frobenius_median": _median(cr, ("diagnostics", "candidate_input_term", "candidate_jacobian_wrt_visible_input_frobenius_mean")),
        }
    return output


def _message_dependence(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        if not cr:
            continue
        output[condition] = {
            "message_ablation_success_drop_median": _median(cr, ("diagnostics", "message_dependency", "success_drop")),
            "message_ablation_prediction_delta_median": _median(cr, ("diagnostics", "message_dependency", "mean_abs_prediction_delta")),
        }
    return output


def _compute_efficiency(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        if not cr:
            continue
        row = cr[0]
        output[condition] = {
            "parameter_count": int(row["parameter_count"]),
            "projection_parameter_count": int(row["projection_parameter_count"]),
            "projection_macs_per_timestep": int(row["projection_specific_macs"]),
            "recurrent_controller_macs_per_timestep": int(row["recurrent_controller_macs"]),
            "total_recurrent_controller_projection_macs_per_timestep": int(row["total_recurrent_controller_projection_macs"]),
        }
    return output


def _decision(summaries: dict[str, dict]) -> dict:
    af0 = int(summaries["AF0_y3_parent"]["families_passing"])
    af1 = int(summaries["AF1_shared_candidate_input_factorization"]["families_passing"])
    af1f = int(summaries["AF1F_folded_candidate_input_control"]["families_passing"])
    af1d = int(summaries["AF1D_deshared_candidate_input_factorization"]["families_passing"])
    base = {
        "v837af_complete": True,
        "parent_reproduced": af0 == 3,
        "families_passing": {condition: int(summary["families_passing"]) for condition, summary in summaries.items()},
        "diagnosis": "",
        "qualifiers": [],
        "preferred_condition": None,
        "representation_adequacy_pass": False,
        "v837ag_allowed": False,
        "sample_efficiency_retest_allowed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "v837ae_created": False,
        "v838_started": False,
    }
    if af1f >= 4:
        base["diagnosis"] = "CANDIDATE_COMPOSED_EFFECTIVE_INPUT_MAPPING_SUFFICIENT"
        base["preferred_condition"] = "AF1F_folded_candidate_input_control"
        base["representation_adequacy_pass"] = True
    elif af1 >= 4 and af1d < 4:
        base["diagnosis"] = "SHARED_CANDIDATE_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT"
        base["preferred_condition"] = "AF1_shared_candidate_input_factorization"
        base["representation_adequacy_pass"] = True
    elif af1 >= 4 and af1d >= 4:
        base["diagnosis"] = "CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT"
        base["qualifiers"] = ["SHAREDNESS_NOT_ESTABLISHED"]
        base["preferred_condition"] = "AF1_shared_candidate_input_factorization"
        base["representation_adequacy_pass"] = True
    elif af1 < 4 and af1d >= 4:
        base["diagnosis"] = "DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT"
        base["qualifiers"] = ["SHARED_INPUT_BASIS_HARMFUL"]
        base["preferred_condition"] = "AF1D_deshared_candidate_input_factorization"
        base["representation_adequacy_pass"] = True
    elif af1 < 4 and af1f < 4 and af1d < 4:
        base["diagnosis"] = "CANDIDATE_INPUT_FACTORIZATION_TRANSFER_INSUFFICIENT"
        base["v837ag_allowed"] = True
    else:
        base["diagnosis"] = "V837AF_UNCLASSIFIED_RESULT"
        base["v837af_complete"] = False
    base["sample_efficiency_retest_allowed"] = bool(base["representation_adequacy_pass"])
    return base


def _resources(rows: list[dict]) -> dict:
    return {
        "version": "V837af",
        "model_fits": len(rows),
        "optimizer_steps": sum(int(r["resources"]["optimizer_steps"]) for r in rows),
        "processed_training_examples": sum(int(r["resources"]["examples_processed"]) for r in rows),
        "unique_seed_defined_episodes": 3200,
        "environment_interactions": sum(int(r["resources"]["environment_steps"]) for r in rows),
        "forward_calls": sum(int(r["resources"]["forward_calls"]) for r in rows),
        "backward_calls": sum(int(r["resources"]["optimizer_steps"]) for r in rows),
        "cpu_seconds_worker_sum": sum(float(r["resources"]["cpu_seconds"]) for r in rows),
        "wall_seconds_worker_sum": sum(float(r["resources"]["wall_seconds"]) for r in rows),
        "gpu_seconds": 0.0,
        "fresh_audit_consumed": False,
    }


def _plots(summaries: dict[str, dict], projection: dict, effective: dict, partial: dict, compute: dict) -> None:
    import matplotlib.pyplot as plt

    names = [condition for condition in CONDITIONS if condition in summaries]
    labels = [name.replace("AF0_y3_parent", "AF0").replace("AF1_shared_candidate_input_factorization", "AF1 shared").replace("AF1F_folded_candidate_input_control", "AF1F folded").replace("AF1D_deshared_candidate_input_factorization", "AF1D deshared") for name in names]

    fig = plt.figure(figsize=(9, 4)); ax = fig.add_subplot(111)
    for family in FAMILIES:
        ax.plot(range(len(names)), [summaries[n]["family_results"][family]["validation"]["median"] for n in names], marker="o", label=family)
    ax.set_xticks(range(len(names)), labels, rotation=20, ha="right"); ax.set_ylabel("Validation median"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(HERE / "plots" / "candidate_factorization_family_scores.png", dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
    ax.bar(range(len(names)), [summaries[n]["families_passing"] for n in names]); ax.set_xticks(range(len(names)), labels, rotation=20, ha="right"); ax.set_ylim(0, 5); ax.set_ylabel("Families passing"); fig.tight_layout(); fig.savefig(HERE / "plots" / "shared_vs_folded_vs_deshared.png", dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
    ax.bar(range(len(names)), [partial[n]["partial_observation_validation_median"] for n in names]); ax.set_xticks(range(len(names)), labels, rotation=20, ha="right"); ax.set_ylabel("Partial observation median"); fig.tight_layout(); fig.savefig(HERE / "plots" / "partial_observation_candidate_input.png", dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
    drift = []
    for n in names:
        final = projection.get(n, {}).get(str(CONFIG["training"]["steps"]), {})
        drift.append(final.get("projection_weight_drift_median", final.get("deshared_divergence_from_shared_initialization_mean_median", 0.0)))
    ax.bar(range(len(names)), drift); ax.set_xticks(range(len(names)), labels, rotation=20, ha="right"); ax.set_ylabel("Projection drift"); fig.tight_layout(); fig.savefig(HERE / "plots" / "candidate_projection_drift.png", dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
    ax.bar(range(len(names)), [effective[n]["pairwise_effective_input_map_cosine_median"] for n in names]); ax.set_xticks(range(len(names)), labels, rotation=20, ha="right"); ax.set_ylabel("Median pairwise cosine"); fig.tight_layout(); fig.savefig(HERE / "plots" / "effective_input_map_similarity.png", dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
    ax.scatter([compute[n]["projection_macs_per_timestep"] for n in names], [summaries[n]["families_passing"] for n in names]);
    for n, label in zip(names, labels): ax.annotate(label, (compute[n]["projection_macs_per_timestep"], summaries[n]["families_passing"]))
    ax.set_xlabel("Candidate projection MACs/timestep"); ax.set_ylabel("Families passing"); ax.set_ylim(0, 5); fig.tight_layout(); fig.savefig(HERE / "plots" / "capability_vs_candidate_projection_macs.png", dpi=120); plt.close(fig)


def main() -> int:
    for directory in ("diagnostics", "plots"):
        (HERE / directory).mkdir(exist_ok=True)
    rows = _rows()
    if not rows:
        raise SystemExit("V837af raw runs are missing")
    summaries = {condition: _summary(rows, condition) for condition in CONDITIONS if any(r["condition"] == condition for r in rows)}
    if set(summaries) != set(CONDITIONS):
        raise SystemExit("V837af analysis requires AF0/AF1/AF1F/AF1D")
    projection = _projection_dynamics(rows)
    effective = _effective_maps(rows)
    partial = _partial_observation(rows, summaries)
    message = _message_dependence(rows)
    compute = _compute_efficiency(rows)
    decision = _decision(summaries)
    resources = _resources(rows)
    write_json(HERE / "diagnostics" / "projection_dynamics.json", projection)
    write_json(HERE / "diagnostics" / "effective_input_maps.json", effective)
    write_json(HERE / "diagnostics" / "partial_observation_diagnostics.json", partial)
    write_json(HERE / "diagnostics" / "message_dependence.json", message)
    write_json(HERE / "diagnostics" / "compute_efficiency.json", compute)
    write_json(HERE / "diagnostics" / "decision_state.json", decision)
    write_json(HERE.parent / "v837af_resource_accounting.json", resources)
    step0 = json.loads((HERE / "diagnostics" / "step0_equivalence.json").read_text(encoding="utf-8"))
    anchor = json.loads((HERE / "diagnostics" / "anchor_compatibility.json").read_text(encoding="utf-8"))
    results = {
        "version": "V837af",
        "question": CONFIG["question"],
        "conditions": summaries,
        "parent_compatibility": anchor,
        "step0_equivalence": step0,
        "diagnosis": decision["diagnosis"],
        "qualifiers": decision["qualifiers"],
        "representation_adequacy_pass": decision["representation_adequacy_pass"],
        "v837ag_allowed": decision["v837ag_allowed"],
        "sample_efficiency_retest_allowed": decision["sample_efficiency_retest_allowed"],
        "resource_accounting": resources,
        "unique_seed_defined_episodes": 3200,
        "fresh_audit_consumed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "v837ae_created": False,
        "v838_started": False,
    }
    write_json(HERE / "results.json", results)
    _plots(summaries, projection, effective, partial, compute)
    if decision["representation_adequacy_pass"]:
        (HERE / "PASS.md").write_text(f"# V837af PASS\n\nDiagnosis: `{decision['diagnosis']}`. Neutral representation adequacy restored at >=4/5. Architecture localization must stop; sample-efficiency characterization is next.\n", encoding="utf-8")
        failure = HERE / "FAILURE.md"
        if failure.exists(): failure.unlink()
    else:
        (HERE / "FAILURE.md").write_text(f"# V837af candidate-input transfer insufficient\n\nDiagnosis: `{decision['diagnosis']}`. Controller-side and candidate-side input factorization have now both failed neutral transfer. V837ag is authorized.\n", encoding="utf-8")
        passed = HERE / "PASS.md"
        if passed.exists(): passed.unlink()
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
