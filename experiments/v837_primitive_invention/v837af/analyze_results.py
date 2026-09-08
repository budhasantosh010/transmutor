from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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
CONDITIONS = CONFIG["conditions"]
AF0, AF1, AF1F, AF1D = CONDITIONS


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows() -> list[dict]:
    rows = []
    for name in ("af0_runs.json", "transfer_runs.json"):
        rows.extend(_load(HERE / "raw" / name)["rows"])
    return sorted(rows, key=lambda r: (r["condition"], r["family"], r["replicate_id"]))


def _summary(rows: list[dict], condition: str) -> dict:
    result = {"families_passing": 0, "family_results": {}, "mean_family_validation_median": 0.0}
    vals = []
    rr_condition = [r for r in rows if r["condition"] == condition]
    for family in FAMILIES:
        rr = [r for r in rr_condition if r["family"] == family]
        dev = float(np.median([r["development_success"] for r in rr]))
        val = float(np.median([r["validation_success"] for r in rr]))
        passed = capacity_demonstrated(dev, val)
        result["families_passing"] += int(passed)
        vals.append(val)
        result["family_results"][family] = {
            "development": {"median": dev},
            "validation": {"median": val},
            "capacity_demonstrated": bool(passed),
        }
    first = rr_condition[0]
    result["mean_family_validation_median"] = float(np.mean(vals))
    result["parameter_count"] = int(first["parameter_count"])
    result["active_parameter_count"] = int(first["active_parameter_count"])
    result["projection_parameter_count"] = int(first["projection_parameter_count"])
    result["projection_specific_macs"] = int(first["projection_specific_macs"])
    result["recurrent_controller_projection_macs"] = int(first["recurrent_controller_projection_macs"])
    return result


def _projection_dynamics(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in [r for r in rows if r["condition"] == condition]:
            for point in row["projection_trajectory"]:
                grouped[int(point["step"])].append(point["projection"])
        condition_rows = {}
        for step, projections in sorted(grouped.items()):
            mode = projections[0]["mode"]
            entry = {"fits": len(projections), "mode": mode}
            if mode == "shared":
                entry.update({
                    "frobenius_norm_median": float(np.median([p["frobenius_norm"] for p in projections])),
                    "condition_number_median": float(np.median([p["condition_number"] for p in projections])),
                    "weight_drift_median": float(np.median([p["weight_drift"] for p in projections])),
                    "bias_drift_median": float(np.median([p["bias_drift"] for p in projections])),
                })
            elif mode == "deshared":
                entry.update({
                    "pairwise_weight_distance_median": float(np.median([p["pairwise_weight_distance_median"] for p in projections])),
                    "pairwise_weight_cosine_median": float(np.median([p["pairwise_weight_cosine_median"] for p in projections])),
                    "projection_divergence_from_shared_initialization_mean_median": float(np.median([p["projection_divergence_from_shared_initialization_mean"] for p in projections])),
                    "mean_projection_frobenius_median": float(np.median([np.mean([q["frobenius_norm"] for q in p["per_cell"]]) for p in projections])),
                })
            condition_rows[str(step)] = entry
        output[condition] = condition_rows
    return output


def _effective_input_maps(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        rr = [r for r in rows if r["condition"] == condition]
        output[condition] = {
            "pairwise_cosine_median": float(np.median([r["diagnostics"]["effective_input_maps"]["pairwise_cosine_median"] for r in rr])),
            "pairwise_distance_median": float(np.median([r["diagnostics"]["effective_input_maps"]["pairwise_distance_median"] for r in rr])),
            "per_fit": [
                {
                    "family": r["family"],
                    "replicate_id": r["replicate_id"],
                    **r["diagnostics"]["effective_input_maps"],
                }
                for r in rr
            ],
        }
    return output


def _path_diagnostics(rows: list[dict]) -> tuple[dict, dict]:
    partial = {}
    message = {}
    for condition in CONDITIONS:
        rr = [r for r in rows if r["condition"] == condition]
        partial[condition] = {
            "candidate_input_term_norm_median": float(np.median([r["diagnostics"]["candidate_input"]["mean_input_term_norm"] for r in rr])),
            "candidate_input_temporal_variance_median": float(np.median([r["diagnostics"]["candidate_input"]["mean_temporal_variance"] for r in rr])),
            "candidate_visible_input_jacobian_frobenius_median": float(np.median([r["diagnostics"]["candidate_input"]["mean_candidate_visible_input_jacobian_frobenius"] for r in rr])),
        }
        message[condition] = {
            "success_drop_median": float(np.median([r["diagnostics"]["message_dependency"]["success_drop"] for r in rr])),
            "mean_abs_prediction_delta_median": float(np.median([r["diagnostics"]["message_dependency"]["mean_abs_prediction_delta"] for r in rr])),
        }
    return partial, message


def _resource_accounting(rows: list[dict]) -> dict:
    return {
        "version": "V837af",
        "model_fits": len(rows),
        "optimizer_steps": sum(int(r["resources"]["optimizer_steps"]) for r in rows),
        "processed_training_examples": sum(int(r["processed_examples"]) for r in rows),
        "unique_seed_defined_episodes": 3200,
        "environment_interactions": sum(int(r["resources"]["environment_steps"]) for r in rows),
        "forward_calls": sum(int(r["resources"]["forward_calls"]) for r in rows),
        "backward_calls": sum(int(r["resources"]["optimizer_steps"]) for r in rows),
        "cpu_seconds_worker_sum": sum(float(r["resources"]["cpu_seconds"]) for r in rows),
        "wall_seconds_worker_sum": sum(float(r["resources"]["wall_seconds"]) for r in rows),
        "gpu_seconds": 0.0,
        "fresh_audit_consumed": False,
    }


def _decision(summaries: dict) -> tuple[str, list[str], bool, str | None, bool]:
    shared = summaries[AF1]["families_passing"]
    folded = summaries[AF1F]["families_passing"]
    deshared = summaries[AF1D]["families_passing"]
    if folded >= 4:
        return "CANDIDATE_COMPOSED_EFFECTIVE_INPUT_MAPPING_SUFFICIENT", [], True, AF1F, False
    if shared >= 4 and deshared >= 4:
        return "CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT", ["SHAREDNESS_NOT_ESTABLISHED"], True, AF1, False
    if shared >= 4 and deshared < 4:
        return "SHARED_CANDIDATE_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT", [], True, AF1, False
    if shared < 4 and deshared >= 4:
        return "DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT", ["SHARED_INPUT_BASIS_HARMFUL"], True, AF1D, False
    return "CANDIDATE_INPUT_FACTORIZATION_TRANSFER_INSUFFICIENT", [], False, None, True


def _plots(summaries: dict, dynamics: dict, effective: dict) -> None:
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    labels = ["AF0", "AF1", "AF1F", "AF1D"]
    x = np.arange(len(FAMILIES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, condition in enumerate(CONDITIONS):
        ax.bar(x + (j - 1.5) * width, [summaries[condition]["family_results"][f]["validation"]["median"] for f in FAMILIES], width, label=labels[j])
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, rotation=25, ha="right"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "candidate_factorization_family_scores.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels[1:], [summaries[c]["families_passing"] for c in CONDITIONS[1:]])
    ax.axhline(4, linestyle="--"); ax.set_ylim(0, 5.2); ax.set_ylabel("families passing"); fig.tight_layout()
    fig.savefig(plots / "shared_vs_folded_vs_deshared.png", dpi=150); plt.close(fig)

    key_families = ["conditional_routing", "partial_observation", "variable_composition"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for condition in CONDITIONS:
        ax.plot(key_families, [summaries[condition]["family_results"][f]["validation"]["median"] for f in key_families], marker="o", label=labels[CONDITIONS.index(condition)])
    ax.legend(); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(plots / "partial_observation_candidate_input.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for condition in (AF1, AF1D):
        steps = sorted(int(s) for s in dynamics[condition])
        if condition == AF1:
            y = [dynamics[condition][str(s)].get("weight_drift_median", 0.0) for s in steps]
        else:
            y = [dynamics[condition][str(s)].get("projection_divergence_from_shared_initialization_mean_median", 0.0) for s in steps]
        ax.plot(steps, y, marker="o", label=labels[CONDITIONS.index(condition)])
    ax.set_xlabel("optimizer step"); ax.set_ylabel("projection drift"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "candidate_projection_drift.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [effective[c]["pairwise_cosine_median"] for c in CONDITIONS])
    ax.set_ylabel("median pairwise effective input-map cosine"); fig.tight_layout()
    fig.savefig(plots / "effective_input_map_similarity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([summaries[c]["projection_specific_macs"] for c in CONDITIONS], [summaries[c]["families_passing"] for c in CONDITIONS])
    for i, condition in enumerate(CONDITIONS):
        ax.annotate(labels[i], (summaries[condition]["projection_specific_macs"], summaries[condition]["families_passing"]))
    ax.set_xlabel("candidate projection MACs/timestep"); ax.set_ylabel("families passing"); ax.set_ylim(0, 5.2); fig.tight_layout()
    fig.savefig(plots / "capability_vs_candidate_projection_macs.png", dpi=150); plt.close(fig)


def main() -> int:
    rows = _rows()
    if len(rows) != 100:
        raise SystemExit(f"expected 100 V837af fits, found {len(rows)}")
    parent = _load(HERE / "diagnostics" / "anchor_compatibility.json")
    step0 = _load(HERE / "diagnostics" / "step0_equivalence.json")
    if not parent.get("parent_reproduced") or not step0.get("step0_equivalence_proven"):
        raise SystemExit("V837af interpretation blocked by failed parent/equivalence guard")
    summaries = {condition: _summary(rows, condition) for condition in CONDITIONS}
    dynamics = _projection_dynamics(rows)
    effective = _effective_input_maps(rows)
    partial, message = _path_diagnostics(rows)
    resources = _resource_accounting(rows)
    diagnosis, qualifiers, representation_pass, winner, v837ag_allowed = _decision(summaries)

    write_json(HERE / "diagnostics" / "projection_dynamics.json", dynamics)
    write_json(HERE / "diagnostics" / "effective_input_maps.json", effective)
    write_json(HERE / "diagnostics" / "partial_observation_diagnostics.json", partial)
    write_json(HERE / "diagnostics" / "message_dependence.json", message)
    compute = {
        "conditions": {
            condition: {
                "families_passing": summaries[condition]["families_passing"],
                "parameter_count": summaries[condition]["parameter_count"],
                "active_parameter_count": summaries[condition]["active_parameter_count"],
                "projection_parameter_count": summaries[condition]["projection_parameter_count"],
                "projection_specific_macs": summaries[condition]["projection_specific_macs"],
                "recurrent_controller_projection_macs": summaries[condition]["recurrent_controller_projection_macs"],
            }
            for condition in CONDITIONS
        },
        "resources": resources,
    }
    write_json(HERE / "diagnostics" / "compute_efficiency.json", compute)

    decision_state = {
        "v837af_complete": True,
        "parent_reproduced": True,
        "families_passing": {condition: summaries[condition]["families_passing"] for condition in CONDITIONS},
        "diagnosis": diagnosis,
        "diagnosis_qualifiers": qualifiers,
        "best_passing_condition": winner,
        "representation_adequacy_pass": representation_pass,
        "v837ag_allowed": v837ag_allowed,
        "sample_efficiency_retest_allowed": representation_pass,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "v837ae_created": False,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision_state)
    results = {
        "version": "V837af",
        "parent": "Y3_global_control_rank4_candidate",
        "question": CONFIG["question"],
        "conditions": summaries,
        "projection_dynamics": dynamics,
        "effective_input_maps_summary": {c: {k: v for k, v in effective[c].items() if k != "per_fit"} for c in CONDITIONS},
        "partial_observation_diagnostics": partial,
        "message_dependence": message,
        "diagnosis": diagnosis,
        "diagnosis_qualifiers": qualifiers,
        "best_passing_condition": winner,
        "representation_adequacy_pass": representation_pass,
        "v837ag_allowed": v837ag_allowed,
        "sample_efficiency_retest_allowed": representation_pass,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "v837ae_created": False,
        "v838_started": False,
        "resource_accounting": resources,
    }
    write_json(HERE / "results.json", results)
    write_json(ROOT / "experiments/v837_primitive_invention/v837af_resource_accounting.json", resources)
    _plots(summaries, dynamics, effective)
    status_file = "PASS.md" if representation_pass else "FAILURE.md"
    (HERE / status_file).write_text(
        f"# V837af — {diagnosis}\n\n"
        f"AF0 Y3: {summaries[AF0]['families_passing']}/5.\n"
        f"AF1 shared: {summaries[AF1]['families_passing']}/5.\n"
        f"AF1F folded: {summaries[AF1F]['families_passing']}/5.\n"
        f"AF1D de-shared: {summaries[AF1D]['families_passing']}/5.\n\n"
        f"Representation adequacy: {'PASS' if representation_pass else 'FAIL'}.\n"
        f"V837ag allowed: {v837ag_allowed}.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision_state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
