from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.metrics import (
    binary_summary,
    bootstrap_mean_ci,
    continuous_summary,
    paired_bootstrap_difference,
)
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
PRIMARY = list(CONFIG["conditions"])
REFERENCES = list(CONFIG["references"])


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _condition_rows(rows: list[dict], condition: str) -> list[dict]:
    return [row for row in rows if row["condition"] == condition]


def _nested_metric(rows: list[dict], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        current = row
        try:
            for key in path:
                current = current[key]
        except (KeyError, TypeError):
            continue
        if current is not None and isinstance(current, (int, float)) and math.isfinite(float(current)):
            values.append(float(current))
    return values


def _summarize(condition: str, rows: list[dict]) -> dict:
    selected = _condition_rows(rows, condition)
    if not selected:
        raise RuntimeError(f"no raw rows for {condition}")
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
        if len(fr) != int(CONFIG["training"]["replicates"]):
            raise RuntimeError(f"{condition}/{family} has {len(fr)} rows, expected {CONFIG['training']['replicates']}")
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        replicate_pass = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate_pass)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837q-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(replicate_pass),
            "aggregate_capacity_pass": aggregate_pass,
            "state_effective_rank": continuous_summary(_nested_metric(fr, ("diagnostics", "state", "effective_rank"))),
            "state_participation_ratio": continuous_summary(_nested_metric(fr, ("diagnostics", "state", "participation_ratio"))),
            "state_norm": continuous_summary(_nested_metric(fr, ("diagnostics", "state", "state_norm"))),
            "state_correlation": continuous_summary(_nested_metric(fr, ("diagnostics", "state", "mean_abs_pairwise_correlation"))),
            "gradient_norm": continuous_summary(_nested_metric(fr, ("diagnostics", "gradient", "global_gradient_norm"))),
            "gradient_alignment": continuous_summary(_nested_metric(fr, ("diagnostics", "gradient", "within_group_gradient_cosine_mean"))),
            "message_dependency_success_drop": continuous_summary(_nested_metric(fr, ("diagnostics", "message_dependency", "success_drop"))),
            "cross_cell_influence_prediction_delta": continuous_summary(_nested_metric(fr, ("diagnostics", "cross_cell_influence", "mean_abs_prediction_delta"))),
        }
    resources = {
        "model_fits": len(selected),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in selected)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in selected)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in selected)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in selected)),
        "wall_seconds_sum_workers": float(sum(row["resources"].get("wall_seconds", 0.0) for row in selected)),
        "cpu_seconds_sum_workers": float(sum(row["resources"].get("cpu_seconds", 0.0) for row in selected)),
        "gpu_seconds": float(sum(row.get("gpu_seconds", 0.0) for row in selected)),
    }
    first = selected[0]
    projection = first["diagnostics"].get("projection")
    return {
        "parameter_count": int(first["parameter_count"]),
        "parameter_bytes": int(first["parameter_bytes"]),
        "non_trainable_projection_elements": None if projection is None else int(projection["non_trainable_projection_elements"]),
        "layout": None if projection is None else projection["layout"],
        "families_passing": int(families_passing),
        "family_results": per_family,
        "resource_accounting": resources,
    }


def _paired(rows: list[dict], condition: str, baseline: str) -> dict:
    output = {}
    for family in FAMILIES:
        c = sorted((row for row in rows if row["condition"] == condition and row["family"] == family), key=lambda row: row["replicate"])
        b = sorted((row for row in rows if row["condition"] == baseline and row["family"] == family), key=lambda row: row["replicate"])
        a_vals = np.asarray([row["validation_success"] for row in c], dtype=float)
        b_vals = np.asarray([row["validation_success"] for row in b], dtype=float)
        output[family] = paired_bootstrap_difference(a_vals, b_vals, seed=deterministic_int("v837q-paired", condition, baseline, family))
    return output


def _mean_family_validation(summary: dict) -> float:
    return float(np.mean([summary["family_results"][family]["validation"]["median"] for family in FAMILIES]))


def _diagnose(conditions: dict) -> tuple[str, bool, str, str | None]:
    q0 = conditions["Q0_local_10x4"]
    shared_names = ["Q1_group5_5x8", "Q2_group2_2x20", "Q3_shared_1x40"]
    counts = {name: int(conditions[name]["families_passing"]) for name in ["Q0_local_10x4", *shared_names]}
    means = {name: _mean_family_validation(conditions[name]) for name in ["Q0_local_10x4", *shared_names]}
    best_shared = max(shared_names, key=lambda name: (counts[name], means[name]))
    # Intermediate modularity is only called when a partial layout clears the
    # representation gate and strictly beats the fully shared condition.
    partial_passers = [name for name in ("Q1_group5_5x8", "Q2_group2_2x20") if counts[name] >= 4]
    if partial_passers:
        best_partial = max(partial_passers, key=lambda name: (counts[name], means[name]))
        if counts[best_partial] > counts["Q3_shared_1x40"]:
            return (
                "INTERMEDIATE_MODULARITY_OPTIMAL",
                True,
                "A partially shared 40D organization reaches representation adequacy and outperforms the fully shared layout; localize the minimum sufficient sharing scale next.",
                best_partial,
            )
    if any(counts[name] >= 4 for name in shared_names):
        return (
            "STATE_FRAGMENTATION_CRITICAL",
            True,
            "At least one progressively shared 40D state organization reaches the >=4/5 representation gate while the historical local baseline remains below it; localize the minimum sufficient sharing next.",
            best_shared,
        )
    delta = means[best_shared] - means["Q0_local_10x4"]
    if counts[best_shared] > counts["Q0_local_10x4"] or delta >= float(CONFIG["partial_benefit_min_mean_validation_delta"]):
        return (
            "STATE_SHARING_PARTIAL_BENEFIT",
            False,
            "State sharing improves the fixed-topology neutral substrate but does not restore >=4/5 representation adequacy. Do not reopen search; a separately specified sharing×dynamic-control factorial is the next potentially justified interaction test.",
            best_shared,
        )
    return (
        "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED",
        False,
        "Progressively sharing the same 40 recurrent dimensions does not materially improve competence. Close state ownership and test global cross-dimensional recurrent coupling as the next single variable.",
        None,
    )


def _aggregate_resources(records: list[dict]) -> dict:
    return {
        "model_fits": int(sum(record["resource_accounting"]["model_fits"] for record in records)),
        "optimizer_steps": int(sum(record["resource_accounting"]["optimizer_steps"] for record in records)),
        "examples_processed": int(sum(record["resource_accounting"]["examples_processed"] for record in records)),
        "environment_interactions": int(sum(record["resource_accounting"]["environment_interactions"] for record in records)),
        "forward_calls": int(sum(record["resource_accounting"]["forward_calls"] for record in records)),
        "wall_seconds_sum_workers": float(sum(record["resource_accounting"]["wall_seconds_sum_workers"] for record in records)),
        "cpu_seconds_sum_workers": float(sum(record["resource_accounting"]["cpu_seconds_sum_workers"] for record in records)),
        "gpu_seconds": float(sum(record["resource_accounting"]["gpu_seconds"] for record in records)),
        "structural_search_runs": 0,
        "motif_mining_runs": 0,
        "fresh_audit_episodes": 0,
    }


def _projection_sensitivity(primary_rows: list[dict], extra_rows: list[dict]) -> dict | None:
    if not extra_rows:
        return None
    q3_primary = [row for row in primary_rows if row["condition"] == "Q3_shared_1x40"]
    all_rows = q3_primary + extra_rows
    by_seed = {}
    for seed in sorted({int(row["projection_seed"]) for row in all_rows}):
        sr = [row for row in all_rows if int(row["projection_seed"]) == seed]
        summary = _summarize("Q3_shared_1x40", sr)
        by_seed[str(seed)] = {
            "families_passing": summary["families_passing"],
            "mean_family_validation_median": _mean_family_validation(summary),
        }
    counts = [value["families_passing"] for value in by_seed.values()]
    return {
        "projection_seeds": by_seed,
        "mean_families_passing": float(np.mean(counts)),
        "worst_families_passing": int(min(counts)),
        "best_families_passing": int(max(counts)),
        "classification": "STATE_SHARING_WITH_PROJECTION_SENSITIVITY" if max(counts) - min(counts) >= 2 else "PROJECTION_ROBUST_WITHIN_SCREEN",
    }


def _plot(conditions: dict, references: dict, rows: list[dict], conditional_nm: dict | None) -> None:
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    ordered = PRIMARY
    labels = ["10 local", "5 groups", "2 groups", "1 shared"]
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [conditions[name]["families_passing"] for name in ordered], marker="o")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 5.2); ax.set_ylabel("Families passing / 5"); ax.set_xlabel("Recurrent state organization")
    fig.tight_layout(); fig.savefig(plots / "families_passing_by_state_groups.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for family in FAMILIES:
        ax.plot(x, [conditions[name]["family_results"][family]["validation"]["median"] for name in ordered], marker="o", label=family)
    ax.axhline(0.85, linestyle="--"); ax.set_xticks(x, labels); ax.set_ylabel("Median validation success"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(plots / "family_scores_by_state_groups.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [np.mean([conditions[name]["family_results"][f]["state_effective_rank"]["median"] for f in FAMILIES]) for name in ordered], marker="o")
    ax.set_xticks(x, labels); ax.set_ylabel("Mean family median effective rank")
    fig.tight_layout(); fig.savefig(plots / "state_effective_rank_by_condition.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [np.mean([conditions[name]["family_results"][f]["message_dependency_success_drop"]["median"] for f in FAMILIES]) for name in ordered], marker="o")
    ax.set_xticks(x, labels); ax.set_ylabel("Message-ablation success drop")
    fig.tight_layout(); fig.savefig(plots / "message_dependency_by_state_groups.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    align = []
    for name in ordered:
        values = [conditions[name]["family_results"][f]["gradient_alignment"]["median"] for f in FAMILIES]
        finite = [v for v in values if v is not None]
        align.append(float(np.mean(finite)) if finite else 0.0)
    ax.plot(x, align, marker="o")
    ax.set_xticks(x, labels); ax.set_ylabel("Within-group pathway gradient cosine")
    fig.tight_layout(); fig.savefig(plots / "gradient_alignment_by_state_groups.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sharing_fraction = [0.0, 5/9, 8/9, 1.0]
    mean_val = [_mean_family_validation(conditions[name]) for name in ordered]
    ax.plot(sharing_fraction, mean_val, marker="o")
    ax.set_xlabel("Predefined sharing fraction"); ax.set_ylabel("Mean family validation median")
    fig.tight_layout(); fig.savefig(plots / "state_sharing_vs_validation.png", dpi=160); plt.close(fig)

    if conditional_nm is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(["Q3 messages", "Q3 no messages"], [conditions["Q3_shared_1x40"]["families_passing"], conditional_nm["families_passing"]])
        ax.set_ylim(0, 5.2); ax.set_ylabel("Families passing / 5")
        fig.tight_layout(); fig.savefig(plots / "shared_state_messages_vs_no_messages.png", dpi=160); plt.close(fig)


def main() -> int:
    baseline_rows = _load_rows(HERE / "raw" / "baseline_runs.json")
    primary_rows = _load_rows(HERE / "raw" / "primary_runs.json")
    if not baseline_rows or not primary_rows:
        raise SystemExit("run V837q baseline and primary phases before analysis")
    compatibility = json.loads((HERE / "diagnostics" / "baseline_compatibility.json").read_text(encoding="utf-8"))
    if compatibility.get("compatible") is not True:
        raise SystemExit("V837q baseline is incompatible; refuse interpretation")
    all_primary = baseline_rows + primary_rows
    conditions = {name: _summarize(name, all_primary) for name in PRIMARY}
    references = {name: _summarize(name, all_primary) for name in REFERENCES}
    paired = {name: _paired(all_primary, name, "Q0_local_10x4") for name in PRIMARY[1:]}
    diagnosis, adequacy, next_action, winning = _diagnose(conditions)

    q3_nm_rows = _load_rows(HERE / "raw" / "q3_no_message_runs.json")
    q3_nm = _summarize("Q3_shared_1x40_no_messages", q3_nm_rows) if q3_nm_rows else None
    projection_rows = _load_rows(HERE / "raw" / "q3_projection_sensitivity_runs.json")
    projection_sensitivity = _projection_sensitivity(primary_rows, projection_rows)

    state_diagnostics = {
        name: {
            family: {
                "effective_rank": conditions[name]["family_results"][family]["state_effective_rank"],
                "participation_ratio": conditions[name]["family_results"][family]["state_participation_ratio"],
                "state_norm": conditions[name]["family_results"][family]["state_norm"],
                "pairwise_correlation": conditions[name]["family_results"][family]["state_correlation"],
            }
            for family in FAMILIES
        }
        for name in PRIMARY
    }
    gradient_diagnostics = {
        name: {family: conditions[name]["family_results"][family]["gradient_alignment"] for family in FAMILIES}
        for name in PRIMARY
    }
    message_diagnostics = {
        name: {family: conditions[name]["family_results"][family]["message_dependency_success_drop"] for family in FAMILIES}
        for name in PRIMARY
    }
    cross_cell = {
        name: {family: conditions[name]["family_results"][family]["cross_cell_influence_prediction_delta"] for family in FAMILIES}
        for name in PRIMARY
    }
    resource_records = [*conditions.values(), *references.values()]
    if q3_nm is not None:
        resource_records.append(q3_nm)
    resources = _aggregate_resources(resource_records)

    q3_pass = int(conditions["Q3_shared_1x40"]["families_passing"]) >= 4
    if adequacy:
        next_variant_allowed = "V837r"
    elif diagnosis == "STATE_SHARING_PARTIAL_BENEFIT":
        next_variant_allowed = "V837s_FACTORIAL_SPEC_REQUIRED"
    else:
        next_variant_allowed = "GLOBAL_CROSS_DIMENSION_RECURRENT_COUPLING_SPEC_REQUIRED"

    payload = {
        "version": "V837q",
        "parent": "V837p",
        "question": CONFIG["question"],
        "single_change": CONFIG["single_change"],
        "data_regime": CONFIG["data_regime"],
        "total_state_dim": 40,
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "baseline_compatibility": compatibility,
        "conditions": conditions,
        "references": references,
        "state_diagnostics": state_diagnostics,
        "gradient_diagnostics": gradient_diagnostics,
        "message_diagnostics": message_diagnostics,
        "cross_cell_influence": cross_cell,
        "paired_effects": paired,
        "state_sharing_curve": {
            "num_state_groups": [10, 5, 2, 1],
            "sharing_fraction": [0.0, 5/9, 8/9, 1.0],
            "families_passing": [conditions[name]["families_passing"] for name in PRIMARY],
            "mean_family_validation_median": [_mean_family_validation(conditions[name]) for name in PRIMARY],
        },
        "q3_no_message_control": q3_nm,
        "projection_sensitivity": projection_sensitivity,
        "representation_adequacy_pass": bool(adequacy),
        "winning_condition": winning,
        "diagnosis": diagnosis,
        "diagnostic_pass": True,
        "failure_classification": [] if adequacy else [diagnosis],
        "next_action": next_action,
        "next_variant_allowed": next_variant_allowed,
        "q3_representation_adequacy_pass": q3_pass,
        "sample_efficiency_retest_allowed": bool(adequacy),
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "resource_accounting": resources,
    }
    write_json(HERE / "results.json", payload)
    write_json(HERE / "diagnostics" / "state_diagnostics.json", state_diagnostics)
    write_json(HERE / "diagnostics" / "gradient_diagnostics.json", gradient_diagnostics)
    write_json(HERE / "diagnostics" / "message_diagnostics.json", message_diagnostics)
    write_json(HERE / "diagnostics" / "paired_effects.json", paired)
    decision = {
        "v837q_complete": True,
        "diagnosis": diagnosis,
        "representation_adequacy_pass": bool(adequacy),
        "winning_condition": winning,
        "q3_representation_adequacy_pass": q3_pass,
        "conditional_q3_controls_allowed": q3_pass,
        "next_variant_allowed": next_variant_allowed,
        "sample_efficiency_retest_allowed": bool(adequacy),
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision)
    _plot(conditions, references, all_primary, q3_nm)

    doc = HERE / "PASS.md"
    lines = [
        "# V837q DIAGNOSTIC PASS",
        "",
        "V837q completed the frozen shared-state organization diagnostic. This is not a primitive-invention PASS.",
        "",
        f"Diagnosis: **{diagnosis}**",
        "",
        "## Families passing",
        "",
    ]
    for name in PRIMARY:
        lines.append(f"- {name}: {conditions[name]['families_passing']}/5")
    for name in REFERENCES:
        lines.append(f"- {name}: {references[name]['families_passing']}/5 (reference only)")
    lines += [
        "",
        f"Representation adequacy restored: **{'YES' if adequacy else 'NO'}**.",
        f"Next action: {next_action}",
        "",
        "Structural search remains blocked in V837q. Primitive mining remains blocked. Fresh-audit episodes consumed: 0. Primitives promoted: 0.",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosis": diagnosis, "families_passing": {name: conditions[name]["families_passing"] for name in PRIMARY}, "references": {name: references[name]["families_passing"] for name in REFERENCES}, "representation_adequacy_pass": adequacy, "q3_conditional_controls_allowed": q3_pass}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
