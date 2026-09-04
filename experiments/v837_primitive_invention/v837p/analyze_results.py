from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary, paired_bootstrap_difference
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])
NEW = "dynamic_scalar_state_modulation"
CONTROL = "parameter_matched_dynamic_additive"
HISTORICAL = "historical_direct"


def _summarize(condition: str, rows: list[dict]) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    per_family: dict[str, dict] = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        flags = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837p-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(flags),
            "aggregate_capacity_pass": bool(aggregate),
        }
    mod_rows = [row["diagnostics"]["state_modulator"] for row in selected if row["diagnostics"]["state_modulator"] is not None]
    modulator = None
    if mod_rows:
        modulator = {
            key: continuous_summary(np.asarray([entry[key] for entry in mod_rows], dtype=float))
            for key in ("mean", "median", "std", "p10", "p90", "temporal_variance", "near_zero_fraction", "near_one_fraction")
        }
        per_family_mod = {}
        for family in FAMILIES:
            entries = [row["diagnostics"]["state_modulator"] for row in selected if row["family"] == family]
            per_family_mod[family] = {
                "mean": continuous_summary(np.asarray([entry["mean"] for entry in entries], dtype=float)),
                "temporal_variance": continuous_summary(np.asarray([entry["temporal_variance"] for entry in entries], dtype=float)),
            }
        modulator["per_family"] = per_family_mod
    return {
        "parameter_count": int(selected[0]["parameter_count"]),
        "parameter_bytes": int(selected[0]["parameter_bytes"]),
        "families_passing": int(families_passing),
        "family_results": per_family,
        "state_modulator_diagnostics": modulator,
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


def _paired(left: str, right: str, rows: list[dict]) -> dict:
    output = {}
    for family in FAMILIES:
        a = sorted((row for row in rows if row["condition"] == left and row["family"] == family), key=lambda row: row["replicate"])
        b = sorted((row for row in rows if row["condition"] == right and row["family"] == family), key=lambda row: row["replicate"])
        output[family] = paired_bootstrap_difference(
            np.asarray([row["validation_success"] for row in a], dtype=float),
            np.asarray([row["validation_success"] for row in b], dtype=float),
            seed=deterministic_int("v837p-paired", left, right, family),
        )
    return output


def _plots(summaries: dict, rows: list[dict]) -> None:
    plot_dir = HERE / "plots"
    plot_dir.mkdir(exist_ok=True)
    labels = CONDITIONS
    counts = [summaries[name]["families_passing"] for name in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Families passing / 5")
    ax.set_title("V837p representation adequacy by neutral condition")
    fig.tight_layout()
    fig.savefig(plot_dir / "families_passing_by_condition.png", dpi=160)
    plt.close(fig)

    x = np.arange(len(FAMILIES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, name in enumerate(labels):
        values = [summaries[name]["family_results"][family]["validation"]["median"] for family in FAMILIES]
        ax.bar(x + (index - 1.5) * width, values, width, label=name)
    ax.set_xticks(x, FAMILIES, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median validation success")
    ax.legend(fontsize=8)
    ax.set_title("V837p family scores")
    fig.tight_layout()
    fig.savefig(plot_dir / "family_scores_by_condition.png", dpi=160)
    plt.close(fig)

    dynamic_rows = [row for row in rows if row["condition"] in {NEW, CONTROL}]
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    names = []
    for name in (NEW, CONTROL):
        values = [row["diagnostics"]["state_modulator"]["mean"] for row in dynamic_rows if row["condition"] == name]
        data.append(values)
        names.append(name)
    ax.boxplot(data, tick_labels=names)
    ax.set_ylabel("Mean dynamic scalar coefficient")
    ax.set_title("V837p dynamic coefficient distributions")
    fig.tight_layout()
    fig.savefig(plot_dir / "dynamic_modulator_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in (HISTORICAL, NEW, CONTROL):
        selected = [row for row in rows if row["condition"] == name and row["family"] == "conditional_routing"]
        curve_by_step: dict[int, list[float]] = {}
        for row in selected:
            for point in row["learning_curve"]:
                curve_by_step.setdefault(int(point["step"]), []).append(float(point["validation_success"]))
        steps = sorted(curve_by_step)
        medians = [float(np.median(curve_by_step[step])) for step in steps]
        ax.plot(steps, medians, marker="o", label=name)
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Routing validation success")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("V837p learning curves — conditional routing")
    fig.tight_layout()
    fig.savefig(plot_dir / "learning_curves_routing.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    params = [summaries[name]["parameter_count"] for name in labels]
    ax.scatter(params, counts)
    for name, px, py in zip(labels, params, counts):
        ax.annotate(name, (px, py), fontsize=8)
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Families passing / 5")
    ax.set_title("V837p parameter count vs success")
    fig.tight_layout()
    fig.savefig(plot_dir / "parameter_count_vs_success.png", dpi=160)
    plt.close(fig)


def main() -> int:
    raw = json.loads((HERE / "raw" / "runs.json").read_text(encoding="utf-8"))
    rows = raw["rows"]
    expected_rows = len(CONDITIONS) * len(FAMILIES) * int(CONFIG["training"]["replicates"])
    if len(rows) != expected_rows:
        raise SystemExit(f"incomplete V837p raw run matrix: {len(rows)} != {expected_rows}")
    summaries = {name: _summarize(name, rows) for name in CONDITIONS}
    for name, spec in CONFIG["conditions"].items():
        if summaries[name]["parameter_count"] != int(spec["parameter_count_expected"]):
            raise SystemExit(f"parameter-count mismatch for {name}")
    if summaries[NEW]["parameter_count"] != summaries[CONTROL]["parameter_count"]:
        raise SystemExit("dynamic modulator and matched additive control are not parameter matched")

    dynamic_pass = summaries[NEW]["families_passing"] >= int(CONFIG["representation_gate"]["families_required"])
    control_pass = summaries[CONTROL]["families_passing"] >= int(CONFIG["representation_gate"]["families_required"])
    if dynamic_pass and not control_pass:
        diagnosis = "GENERIC_DYNAMIC_STATE_MODULATION_SUFFICIENT"
        representation_adequacy = True
        failure_classes: list[str] = []
        next_variant = "V837r"
        next_action = "freeze the recovered neutral cell and calibrate sample efficiency at 1x/2x/4x data"
    elif dynamic_pass and control_pass:
        diagnosis = "DYNAMIC_MODULATION_SUFFICIENT_MULTIPLICATIVE_SPECIFICITY_UNRESOLVED"
        representation_adequacy = True
        failure_classes = []
        next_variant = "V837r"
        next_action = "representation adequacy is restored, but multiplicative state-access specificity is unresolved; sample-efficiency calibration may proceed with the simpler successful dynamic formulation explicitly documented"
    else:
        diagnosis = "SHARED_PROPERTY_TRANSFER_FAILURE"
        representation_adequacy = False
        failure_classes = ["SHARED_PROPERTY_TRANSFER_FAILURE"]
        next_variant = ""
        next_action = "stop neutral transfer and create shared-state-control blocker analysis before testing deeper dense-organization or optimization-geometry differences"

    totals = {
        "model_fits": len(rows),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in rows)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in rows)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in rows)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in rows)),
        "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in rows)),
        "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in rows)),
        "gpu_seconds": 0.0,
    }
    payload = {
        "version": "V837p",
        "parent": "V837o",
        "question": CONFIG["question"],
        "single_change": CONFIG["single_change"],
        "selection_basis": CONFIG["selection_basis"],
        "data_regime": CONFIG["data_regime"],
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "conditions": summaries,
        "parameter_matching": {
            "dynamic_scalar_state_modulation": summaries[NEW]["parameter_count"],
            "parameter_matched_dynamic_additive": summaries[CONTROL]["parameter_count"],
            "exact_match": summaries[NEW]["parameter_count"] == summaries[CONTROL]["parameter_count"],
        },
        "paired_effects": {
            "dynamic_minus_historical": _paired(NEW, HISTORICAL, rows),
            "dynamic_minus_matched_additive": _paired(NEW, CONTROL, rows),
        },
        "representation_adequacy_pass": bool(representation_adequacy),
        "sample_efficiency_tested": False,
        "structural_search_allowed": bool(representation_adequacy),
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "diagnosis": diagnosis,
        "failure_classification": failure_classes,
        "next_variant": next_variant,
        "next_action": next_action,
        "resource_accounting": totals,
    }
    write_json(HERE / "results.json", payload)
    write_json(HERE / "diagnostics" / "condition_summaries.json", summaries)
    write_json(HERE / "diagnostics" / "decision_state.json", {
        "v837o_diagnosis": CONFIG["required_v837o_diagnosis"],
        "v837p_complete": True,
        "v837p_diagnosis": diagnosis,
        "representation_adequacy_pass": bool(representation_adequacy),
        "sample_efficiency_followup_allowed": bool(representation_adequacy),
        "structural_search_allowed": bool(representation_adequacy),
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "next_variant": next_variant,
    })
    _plots(summaries, rows)

    document = HERE / ("PASS.md" if representation_adequacy else "FAILURE.md")
    lines = [
        f"# V837p {'REPRESENTATION ADEQUACY PASS' if representation_adequacy else 'FAILURE'}",
        "",
        f"Diagnosis: **{diagnosis}**.",
        "",
    ]
    for name in CONDITIONS:
        lines.append(f"- {name}: {summaries[name]['families_passing']}/5, {summaries[name]['parameter_count']} parameters")
    lines += [
        "",
        next_action,
        "",
        "Fresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining remains blocked.",
    ]
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V837p diagnosis: {diagnosis}; representation_adequacy={representation_adequacy}")
    print("families passing: " + ", ".join(f"{name}={summaries[name]['families_passing']}/5" for name in CONDITIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
