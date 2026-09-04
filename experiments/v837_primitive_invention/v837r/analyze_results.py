from __future__ import annotations

import argparse
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
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary, paired_bootstrap_difference
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
PRIMARY_ORDER = ["R0_local", "R1_rank1", "R2_rank2", "R3_rank4", "R4_rank8", "R5_dense_cross_block"]
CONTROL_ORDER = ["C1_rank1_local", "C2_rank2_local", "C3_rank4_local", "C4_rank8_local", "C5_dense_budget_local"]
MATCH_BY_PRIMARY = {row["matches"]: name for name, row in CONFIG["matched_controls"].items()}


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _all_rows() -> list[dict]:
    return _load_rows(HERE / "raw" / "baseline_runs.json") + _load_rows(HERE / "raw" / "screen_runs.json") + _load_rows(HERE / "raw" / "localization_runs.json")


def _condition_rows(rows: list[dict], condition: str) -> list[dict]:
    return [row for row in rows if row["condition"] == condition]


def _nested_values(rows: list[dict], path: tuple[str, ...]) -> list[float]:
    out = []
    for row in rows:
        cur = row
        try:
            for key in path:
                cur = cur[key]
        except (KeyError, TypeError):
            continue
        if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
            out.append(float(cur))
    return out


def _summarize(condition: str, rows: list[dict]) -> dict:
    selected = _condition_rows(rows, condition)
    if not selected:
        raise RuntimeError(f"no rows for {condition}")
    expected = int(CONFIG["training"]["replicates"])
    family_results = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted([r for r in selected if r["family"] == family], key=lambda r: r["replicate"])
        if len(fr) != expected:
            raise RuntimeError(f"{condition}/{family}: {len(fr)} rows, expected {expected}")
        dev = np.asarray([r["development_success"] for r in fr], dtype=float)
        val = np.asarray([r["validation_success"] for r in fr], dtype=float)
        replicate_pass = np.asarray([r["capacity_demonstrated"] for r in fr], dtype=bool)
        aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate_pass)
        family_results[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837r-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(replicate_pass),
            "aggregate_capacity_pass": bool(aggregate_pass),
            "global_to_local_ratio": continuous_summary(_nested_values(fr, ("diagnostics", "utilization", "global_to_local_ratio"))),
            "global_to_message_ratio": continuous_summary(_nested_values(fr, ("diagnostics", "utilization", "global_to_message_ratio"))),
            "global_recurrent_term_norm": continuous_summary(_nested_values(fr, ("diagnostics", "utilization", "global_recurrent_term_norm"))),
            "local_recurrent_term_norm": continuous_summary(_nested_values(fr, ("diagnostics", "utilization", "local_recurrent_term_norm"))),
            "message_term_norm": continuous_summary(_nested_values(fr, ("diagnostics", "utilization", "message_term_norm"))),
            "effective_coupling_rank": continuous_summary(_nested_values(fr, ("diagnostics", "coupling_matrix", "effective_rank"))),
            "coupling_spectral_norm": continuous_summary(_nested_values(fr, ("diagnostics", "coupling_matrix", "spectral_norm"))),
            "coupling_frobenius_norm": continuous_summary(_nested_values(fr, ("diagnostics", "coupling_matrix", "frobenius_norm"))),
            "offdiag_fraction": continuous_summary(_nested_values(fr, ("diagnostics", "coupling_matrix", "offdiag_fraction"))),
            "message_dependency_success_drop": continuous_summary(_nested_values(fr, ("diagnostics", "message_dependency", "success_drop"))),
            "cross_cell_prediction_delta": continuous_summary(_nested_values(fr, ("diagnostics", "cross_cell_influence", "mean_abs_prediction_delta"))),
            "cross_cell_state_delta": continuous_summary(_nested_values(fr, ("diagnostics", "cross_cell_influence", "mean_abs_other_cell_state_delta"))),
            "global_gradient_norm": continuous_summary(_nested_values(fr, ("diagnostics", "gradient", "global_gradient_norm"))),
            "coupling_gradient_norm": continuous_summary(_nested_values(fr, ("diagnostics", "gradient", "coupling_gradient_norm"))),
            "cell_gradient_norm_variance": continuous_summary(_nested_values(fr, ("diagnostics", "gradient", "cell_gradient_norm_variance"))),
            "cell_gradient_cosine_mean": continuous_summary(_nested_values(fr, ("diagnostics", "gradient", "cell_gradient_cosine_mean"))),
        }
    first = selected[0]
    resources = {
        "model_fits": len(selected),
        "optimizer_steps": int(sum(r["resources"]["optimizer_steps"] for r in selected)),
        "examples_processed": int(sum(r["resources"]["examples_processed"] for r in selected)),
        "environment_interactions": int(sum(r["resources"]["environment_steps"] for r in selected)),
        "forward_calls": int(sum(r["resources"]["forward_calls"] for r in selected)),
        "wall_seconds_sum_workers": float(sum(r["resources"].get("wall_seconds", 0.0) for r in selected)),
        "cpu_seconds_sum_workers": float(sum(r["resources"].get("cpu_seconds", 0.0) for r in selected)),
        "gpu_seconds": float(sum(r.get("gpu_seconds", 0.0) for r in selected)),
    }
    compute = first["diagnostics"]["compute"]
    return {
        "coupling_spec": first["coupling_spec"],
        "parameter_count": int(first["parameter_count"]),
        "added_parameter_count": int(first["added_parameter_count"]),
        "parameter_bytes": int(first["parameter_bytes"]),
        "recurrent_macs": compute,
        "families_passing": int(families_passing),
        "family_results": family_results,
        "resource_accounting": resources,
    }


def _mean_family_validation(summary: dict) -> float:
    return float(np.mean([summary["family_results"][f]["validation"]["median"] for f in FAMILIES]))


def _paired(rows: list[dict], condition: str, baseline: str) -> dict:
    output = {}
    for family in FAMILIES:
        a = sorted([r for r in rows if r["condition"] == condition and r["family"] == family], key=lambda r: r["replicate"])
        b = sorted([r for r in rows if r["condition"] == baseline and r["family"] == family], key=lambda r: r["replicate"])
        if not a or not b:
            continue
        output[family] = paired_bootstrap_difference(
            np.asarray([r["validation_success"] for r in a], dtype=float),
            np.asarray([r["validation_success"] for r in b], dtype=float),
            seed=deterministic_int("v837r-paired", condition, baseline, family),
        )
    return output


def _specificity(primary_name: str, summaries: dict) -> dict | None:
    control_name = MATCH_BY_PRIMARY.get(primary_name)
    if control_name is None or control_name not in summaries:
        return None
    p = summaries[primary_name]
    c = summaries[control_name]
    mean_delta = _mean_family_validation(p) - _mean_family_validation(c)
    count_delta = int(p["families_passing"]) - int(c["families_passing"])
    threshold = float(CONFIG["interaction_guard"]["specificity_min_mean_validation_delta"])
    return {
        "matched_control": control_name,
        "global_families_passing": int(p["families_passing"]),
        "control_families_passing": int(c["families_passing"]),
        "families_passing_delta": count_delta,
        "mean_family_validation_delta": mean_delta,
        "specificity_supported": bool(count_delta > 0 or mean_delta >= threshold),
    }


def _best_global(summaries: dict) -> str:
    available = [name for name in PRIMARY_ORDER[1:] if name in summaries]
    if not available:
        raise RuntimeError("no global coupling conditions available")
    return max(available, key=lambda name: (int(summaries[name]["families_passing"]), _mean_family_validation(summaries[name])))


def _diagnose(summaries: dict) -> tuple[str, bool, str | None, dict]:
    available_global = [name for name in PRIMARY_ORDER[1:] if name in summaries]
    low_rank = [name for name in available_global if name != "R5_dense_cross_block"]
    low_passers = [name for name in low_rank if int(summaries[name]["families_passing"]) >= 4]
    if low_passers:
        rank_order = {"R1_rank1": 1, "R2_rank2": 2, "R3_rank4": 4, "R4_rank8": 8}
        winner = min(low_passers, key=lambda name: rank_order[name])
        return "LOW_RANK_GLOBAL_COUPLING_SUFFICIENT", True, winner, {"specificity": _specificity(winner, summaries)}
    if "R5_dense_cross_block" in summaries and int(summaries["R5_dense_cross_block"]["families_passing"]) >= 4:
        return "HIGH_BANDWIDTH_GLOBAL_COUPLING_REQUIRED", True, "R5_dense_cross_block", {"specificity": _specificity("R5_dense_cross_block", summaries)}
    best = _best_global(summaries)
    best_summary = summaries[best]
    spec = _specificity(best, summaries)
    if int(best_summary["families_passing"]) == 3 and spec is not None and spec["specificity_supported"]:
        return "GLOBAL_COUPLING_PARTIAL_BENEFIT", False, best, {"specificity": spec}
    r0_mean = _mean_family_validation(summaries["R0_local"])
    best_delta_vs_r0 = _mean_family_validation(best_summary) - r0_mean
    if (int(best_summary["families_passing"]) > int(summaries["R0_local"]["families_passing"]) or best_delta_vs_r0 >= 0.05) and (spec is None or not spec["specificity_supported"]):
        return "GLOBAL_COUPLING_SPECIFICITY_NOT_ESTABLISHED", False, best, {"specificity": spec, "delta_vs_r0": best_delta_vs_r0}
    return "GLOBAL_RECURRENT_COUPLING_INSUFFICIENT", False, best, {"specificity": spec, "delta_vs_r0": best_delta_vs_r0}


def _interaction_guard(diagnosis: str, summaries: dict, best: str | None) -> tuple[bool, list[str]]:
    reasons = []
    if best is None:
        return False, reasons
    count = int(summaries[best]["families_passing"])
    spec = _specificity(best, summaries)
    if count >= 4:
        reasons.append("global coupling reaches representation adequacy")
    if count == 3 and spec is not None and spec["specificity_supported"]:
        reasons.append("global coupling reaches 3/5 and specifically beats its matched local control")
    # Conservative mechanistic-complement guard: coupling must be actively used,
    # causally cross-cell, and improve a V837p-failing family by >=0.05 over R0.
    family_results = summaries[best]["family_results"]
    ratios = [family_results[f]["global_to_local_ratio"]["median"] for f in FAMILIES]
    ratios = [v for v in ratios if v is not None]
    influences = [family_results[f]["cross_cell_prediction_delta"]["median"] for f in FAMILIES]
    influences = [v for v in influences if v is not None]
    complementary = []
    for family in ("conditional_routing", "variable_composition"):
        delta = family_results[family]["validation"]["median"] - summaries["R0_local"]["family_results"][family]["validation"]["median"]
        if delta >= 0.05:
            complementary.append(family)
    ratio_threshold = float(CONFIG["interaction_guard"]["mechanistic_use_min_global_to_local_ratio"])
    if ratios and float(np.median(ratios)) >= ratio_threshold and influences and float(np.median(influences)) > 0.01 and complementary:
        reasons.append("strong coupling utilization/cross-cell influence with complementary improvement on a V837p-failing family")
    allowed = bool(reasons) and diagnosis not in {"GLOBAL_COUPLING_SPECIFICITY_NOT_ESTABLISHED", "GLOBAL_RECURRENT_COUPLING_INSUFFICIENT"}
    # Outcome A/B should prioritize compression; permission is recorded but V837s is not auto-run.
    return allowed, reasons


def _aggregate_resources(summaries: dict) -> dict:
    records = list(summaries.values())
    return {
        "model_fits": int(sum(r["resource_accounting"]["model_fits"] for r in records)),
        "optimizer_steps": int(sum(r["resource_accounting"]["optimizer_steps"] for r in records)),
        "examples_processed": int(sum(r["resource_accounting"]["examples_processed"] for r in records)),
        "environment_interactions": int(sum(r["resource_accounting"]["environment_interactions"] for r in records)),
        "forward_calls": int(sum(r["resource_accounting"]["forward_calls"] for r in records)),
        "wall_seconds_sum_workers": float(sum(r["resource_accounting"]["wall_seconds_sum_workers"] for r in records)),
        "cpu_seconds_sum_workers": float(sum(r["resource_accounting"]["cpu_seconds_sum_workers"] for r in records)),
        "gpu_seconds": float(sum(r["resource_accounting"]["gpu_seconds"] for r in records)),
        "structural_search_runs": 0,
        "primitive_mining_runs": 0,
        "fresh_audit_episodes": 0,
    }


def _compute_efficiency(summaries: dict) -> dict:
    output = {}
    for name, summary in summaries.items():
        macs = int(summary["recurrent_macs"]["total_recurrent_macs_actual"])
        passes = int(summary["families_passing"])
        params = int(summary["parameter_count"])
        output[name] = {
            "trainable_parameters": params,
            "parameter_bytes": int(summary["parameter_bytes"]),
            "local_recurrent_macs_per_timestep": int(summary["recurrent_macs"]["local_recurrent_macs"]),
            "coupling_core_macs_per_timestep": int(summary["recurrent_macs"]["coupling_core_macs"]),
            "coupling_actual_macs_per_timestep": int(summary["recurrent_macs"]["coupling_actual_macs"]),
            "total_recurrent_macs_per_timestep": macs,
            "approx_recurrent_flops_per_timestep": int(summary["recurrent_macs"]["approx_recurrent_flops_actual"]),
            "coupling_scaling_complexity": summary["recurrent_macs"]["coupling_scaling_complexity"],
            "families_passing": passes,
            "mean_family_validation_median": _mean_family_validation(summary),
            "capability_per_recurrent_mac": passes / max(1, macs),
            "families_passing_per_trainable_parameter": passes / max(1, params),
            "optimizer_steps": summary["resource_accounting"]["optimizer_steps"],
            "examples_processed": summary["resource_accounting"]["examples_processed"],
            "wall_seconds_sum_workers": summary["resource_accounting"]["wall_seconds_sum_workers"],
            "cpu_seconds_sum_workers": summary["resource_accounting"]["cpu_seconds_sum_workers"],
        }
    return output


def _plot(summaries: dict, compute: dict) -> None:
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    global_names = [name for name in PRIMARY_ORDER if name in summaries]
    labels = [name.replace("R0_local", "local").replace("R1_rank1", "rank1").replace("R2_rank2", "rank2").replace("R3_rank4", "rank4").replace("R4_rank8", "rank8").replace("R5_dense_cross_block", "dense") for name in global_names]
    x = np.arange(len(global_names))
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [summaries[n]["families_passing"] for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylim(0, 5.2); ax.set_ylabel("Families passing / 5"); fig.tight_layout(); fig.savefig(plots / "families_passing_by_coupling_rank.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for family in FAMILIES:
        ax.plot(x, [summaries[n]["family_results"][family]["validation"]["median"] for n in global_names], marker="o", label=family)
    ax.axhline(0.85, linestyle="--"); ax.set_xticks(x, labels); ax.set_ylabel("Median validation success"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "family_scores_by_coupling_rank.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [_mean_family_validation(summaries[n]) for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylabel("Mean family validation median"); fig.tight_layout(); fig.savefig(plots / "coupling_rank_vs_validation.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [compute[n]["total_recurrent_macs_per_timestep"] for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylabel("Approx recurrent MACs / timestep"); fig.tight_layout(); fig.savefig(plots / "coupling_rank_vs_compute.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [compute[n]["capability_per_recurrent_mac"] for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylabel("Families passing / recurrent MAC"); fig.tight_layout(); fig.savefig(plots / "capability_per_recurrent_mac.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [np.mean([summaries[n]["family_results"][f]["global_to_local_ratio"]["median"] or 0.0 for f in FAMILIES]) for n in global_names], marker="o", label="global/local"); ax.set_xticks(x, labels); ax.set_ylabel("Global / local recurrent norm"); fig.tight_layout(); fig.savefig(plots / "global_vs_local_recurrent_norm.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [np.mean([summaries[n]["family_results"][f]["message_dependency_success_drop"]["median"] or 0.0 for f in FAMILIES]) for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylabel("Message-ablation success drop"); fig.tight_layout(); fig.savefig(plots / "message_dependency_by_coupling.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(x, [np.mean([summaries[n]["family_results"][f]["effective_coupling_rank"]["median"] or 0.0 for f in FAMILIES]) for n in global_names], marker="o"); ax.set_xticks(x, labels); ax.set_ylabel("Effective masked coupling rank"); fig.tight_layout(); fig.savefig(plots / "effective_coupling_rank.png", dpi=160); plt.close(fig)


def _screen_only(rows: list[dict]) -> int:
    screen_path = HERE / "diagnostics" / "screen_decision.json"
    if not screen_path.exists():
        raise SystemExit("V837r screen decision missing; run --phase screen first")
    decision = json.loads(screen_path.read_text(encoding="utf-8"))
    print(json.dumps(decision, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    rows = _all_rows()
    if not _load_rows(HERE / "raw" / "baseline_runs.json") or not _load_rows(HERE / "raw" / "screen_runs.json"):
        raise SystemExit("run V837r baseline and screen before analysis")
    compatibility = json.loads((HERE / "diagnostics" / "baseline_compatibility.json").read_text(encoding="utf-8"))
    if compatibility.get("compatible") is not True:
        raise SystemExit("V837r baseline incompatible; refuse interpretation")
    if args.screen_only:
        return _screen_only(rows)
    available = sorted({r["condition"] for r in rows}, key=lambda n: (PRIMARY_ORDER + CONTROL_ORDER).index(n))
    summaries = {name: _summarize(name, rows) for name in available}
    diagnosis, adequacy, best, diagnosis_details = _diagnose(summaries)
    specificity = {name: _specificity(name, summaries) for name in PRIMARY_ORDER[1:] if name in summaries}
    paired_vs_r0 = {name: _paired(rows, name, "R0_local") for name in PRIMARY_ORDER[1:] if name in summaries}
    paired_vs_control = {name: _paired(rows, name, MATCH_BY_PRIMARY[name]) for name in PRIMARY_ORDER[1:] if name in summaries and MATCH_BY_PRIMARY[name] in summaries}
    interaction_allowed, interaction_reasons = _interaction_guard(diagnosis, summaries, best)
    screen_decision = json.loads((HERE / "diagnostics" / "screen_decision.json").read_text(encoding="utf-8"))
    localization_ran = bool(_load_rows(HERE / "raw" / "localization_runs.json"))
    if screen_decision.get("localization_allowed") is True and not localization_ran:
        raise SystemExit("screen authorized rank1/rank8 localization; run --phase localization before final analysis")
    compute = _compute_efficiency(summaries)
    resources = _aggregate_resources(summaries)
    coupling_diagnostics = {
        name: {
            family: {
                "global_to_local_ratio": summaries[name]["family_results"][family]["global_to_local_ratio"],
                "global_to_message_ratio": summaries[name]["family_results"][family]["global_to_message_ratio"],
                "effective_coupling_rank": summaries[name]["family_results"][family]["effective_coupling_rank"],
                "spectral_norm": summaries[name]["family_results"][family]["coupling_spectral_norm"],
                "frobenius_norm": summaries[name]["family_results"][family]["coupling_frobenius_norm"],
                "offdiag_fraction": summaries[name]["family_results"][family]["offdiag_fraction"],
            } for family in FAMILIES
        } for name in PRIMARY_ORDER[1:] if name in summaries
    }
    gradient_diagnostics = {name: {family: {
        "global_gradient_norm": summaries[name]["family_results"][family]["global_gradient_norm"],
        "coupling_gradient_norm": summaries[name]["family_results"][family]["coupling_gradient_norm"],
        "cell_gradient_norm_variance": summaries[name]["family_results"][family]["cell_gradient_norm_variance"],
        "cell_gradient_cosine_mean": summaries[name]["family_results"][family]["cell_gradient_cosine_mean"],
    } for family in FAMILIES} for name in available}
    message_diagnostics = {name: {family: summaries[name]["family_results"][family]["message_dependency_success_drop"] for family in FAMILIES} for name in available}
    cross_cell = {name: {family: {
        "prediction_delta": summaries[name]["family_results"][family]["cross_cell_prediction_delta"],
        "other_cell_state_delta": summaries[name]["family_results"][family]["cross_cell_state_delta"],
    } for family in FAMILIES} for name in PRIMARY_ORDER[1:] if name in summaries}
    next_variant = None
    if adequacy:
        next_variant = "V837t_COUPLING_COMPRESSION_OR_LOCALIZATION"
    elif interaction_allowed:
        next_variant = "V837s"
    else:
        next_variant = "DYNAMIC_VECTOR_STATE_MODULATION_SPEC_REQUIRED"
    result = {
        "version": "V837r",
        "parent": "V837q",
        "question": CONFIG["question"],
        "single_change": CONFIG["single_change"],
        "data_regime": CONFIG["data_regime"],
        "state_layout": CONFIG["state_layout"],
        "total_state_dim": 40,
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "baseline_compatibility": compatibility,
        "conditions": {name: summaries[name] for name in PRIMARY_ORDER if name in summaries},
        "matched_controls": {name: summaries[name] for name in CONTROL_ORDER if name in summaries},
        "specificity_vs_matched_control": specificity,
        "paired_effects_vs_r0": paired_vs_r0,
        "paired_effects_vs_matched_control": paired_vs_control,
        "coupling_diagnostics": coupling_diagnostics,
        "gradient_diagnostics": gradient_diagnostics,
        "message_diagnostics": message_diagnostics,
        "cross_cell_influence": cross_cell,
        "compute_estimates": compute,
        "screen_decision": screen_decision,
        "rank1_run": "R1_rank1" in summaries,
        "rank8_run": "R4_rank8" in summaries,
        "full_dense_control_run": False,
        "full_dense_control_reason": "not required by the cross-block primary question; strict stop/compression logic takes precedence",
        "representation_adequacy_pass": bool(adequacy),
        "diagnosis": diagnosis,
        "diagnosis_details": diagnosis_details,
        "best_condition": best,
        "diagnostic_pass": True,
        "failure_classification": [] if adequacy else [diagnosis],
        "interaction_followup_allowed": bool(interaction_allowed),
        "interaction_followup_reasons": interaction_reasons,
        "next_variant": next_variant,
        "sample_efficiency_retest_allowed": bool(adequacy),
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "v838_started": False,
        "resource_accounting": resources,
    }
    write_json(HERE / "results.json", result)
    write_json(HERE / "diagnostics" / "coupling_diagnostics.json", coupling_diagnostics)
    write_json(HERE / "diagnostics" / "gradient_diagnostics.json", gradient_diagnostics)
    write_json(HERE / "diagnostics" / "message_diagnostics.json", message_diagnostics)
    write_json(HERE / "diagnostics" / "cross_cell_influence.json", cross_cell)
    write_json(HERE / "diagnostics" / "paired_effects.json", {"vs_r0": paired_vs_r0, "vs_matched_control": paired_vs_control})
    write_json(HERE / "diagnostics" / "compute_efficiency.json", compute)
    decision = {
        "v837r_complete": True,
        "diagnosis": diagnosis,
        "best_condition": best,
        "families_passing": None if best is None else int(summaries[best]["families_passing"]),
        "specificity_vs_matched_control": None if best is None else _specificity(best, summaries),
        "representation_adequacy_pass": bool(adequacy),
        "interaction_followup_allowed": bool(interaction_allowed),
        "interaction_followup_reasons": interaction_reasons,
        "next_variant": next_variant,
        "strict_stop_triggered": bool(screen_decision.get("strict_stop_triggered")),
        "rank1_run": "R1_rank1" in summaries,
        "rank8_run": "R4_rank8" in summaries,
        "full_dense_control_run": False,
        "sample_efficiency_retest_allowed": bool(adequacy),
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision)
    _plot(summaries, compute)
    lines = [
        "# V837r DIAGNOSTIC PASS", "", "V837r completed the frozen global recurrent coupling localization diagnostic. This is not a primitive-invention PASS.", "",
        f"Diagnosis: **{diagnosis}**", "", "## Families passing", "",
    ]
    for name in PRIMARY_ORDER:
        lines.append(f"- {name}: {summaries[name]['families_passing']}/5" if name in summaries else f"- {name}: NOT RUN (strict staged execution)")
    lines += ["", "## Matched controls", ""]
    for name in CONTROL_ORDER:
        if name in summaries:
            lines.append(f"- {name}: {summaries[name]['families_passing']}/5")
    lines += [
        "", f"Representation adequacy restored: **{'YES' if adequacy else 'NO'}**.",
        f"Interaction follow-up allowed: **{'YES' if interaction_allowed else 'NO'}**.",
        f"Next variant state: {next_variant}.", "",
        "Structural search remains blocked. Primitive mining remains blocked. Fresh-audit episodes consumed: 0. Primitives promoted: 0. V838 not started.",
    ]
    (HERE / "PASS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnosis": diagnosis,
        "families_passing": {name: summaries[name]["families_passing"] for name in PRIMARY_ORDER if name in summaries},
        "matched_controls": {name: summaries[name]["families_passing"] for name in CONTROL_ORDER if name in summaries},
        "representation_adequacy_pass": adequacy,
        "interaction_followup_allowed": interaction_allowed,
        "next_variant": next_variant,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
