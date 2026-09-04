from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.metrics import paired_bootstrap_difference
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks
from experiments.v837_primitive_invention.v837o.run_factorial_localization import summarize_condition

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
FULL = "G0_full_dynamic"
BOTH_OFF = "G9_no_update_no_reset"


def _load_rows() -> list[dict]:
    full = json.loads((HERE / "raw" / "full_gru.json").read_text(encoding="utf-8"))["rows"]
    factorial = json.loads((HERE / "raw" / "factorial_runs.json").read_text(encoding="utf-8"))["rows"]
    return full + factorial


def _paired(rows: list[dict], condition: str, family: str) -> dict:
    a = sorted((r for r in rows if r["condition"] == condition and r["family"] == family), key=lambda r: r["replicate"])
    b = sorted((r for r in rows if r["condition"] == FULL and r["family"] == family), key=lambda r: r["replicate"])
    av = np.asarray([r["validation_success"] for r in a], dtype=float)
    bv = np.asarray([r["validation_success"] for r in b], dtype=float)
    result = paired_bootstrap_difference(av, bv, seed=deterministic_int("v837o-paired", condition, family))
    result["definition"] = "condition_minus_full_gru"
    return result


def _factorial_effects(summaries: dict) -> dict:
    def med(condition: str, family: str) -> float:
        return float(summaries[condition]["family_results"][family]["validation"]["median"])

    effects = {"dynamic_vs_off_2x2": {}, "static_vector_vs_off_2x2": {}, "dynamic_vs_static_substitution": {}}
    for family in FAMILIES:
        g0, g1, g2, g9 = [med(c, family) for c in (FULL, "G1_dynamic_update_no_reset", "G2_no_update_dynamic_reset", BOTH_OFF)]
        effects["dynamic_vs_off_2x2"][family] = {
            "update_main_effect": 0.5 * ((g0 - g2) + (g1 - g9)),
            "reset_main_effect": 0.5 * ((g0 - g1) + (g2 - g9)),
            "interaction": g0 - g1 - g2 + g9,
        }
        g3, g4, g5 = [med(c, family) for c in ("G3_static_update_vector_no_reset", "G4_no_update_static_reset_vector", "G5_static_update_vector_static_reset_vector")]
        effects["static_vector_vs_off_2x2"][family] = {
            "update_main_effect": 0.5 * ((g5 - g4) + (g3 - g9)),
            "reset_main_effect": 0.5 * ((g5 - g3) + (g4 - g9)),
            "interaction": g5 - g3 - g4 + g9,
        }
        effects["dynamic_vs_static_substitution"][family] = {
            "both_dynamic_minus_both_static_vector": g0 - g5,
            "dynamic_update_minus_static_update_with_reset_off": g1 - g3,
            "dynamic_reset_minus_static_reset_with_update_off": g2 - g4,
        }
    return effects


def _recovery_fraction(summaries: dict) -> dict:
    out = {}
    for condition in CONFIG["conditions"]:
        out[condition] = {}
        for family in FAMILIES:
            full = float(summaries[FULL]["family_results"][family]["validation"]["median"])
            floor = float(summaries[BOTH_OFF]["family_results"][family]["validation"]["median"])
            value = float(summaries[condition]["family_results"][family]["validation"]["median"])
            out[condition][family] = float((value - floor) / (full - floor + 1e-12))
    return out


def _diagnose(summaries: dict) -> tuple[str, bool, bool, str | None, str]:
    counts = {c: int(summaries[c]["families_passing"]) for c in CONFIG["conditions"]}
    if counts[FULL] < 4:
        return "IMPLEMENTATION_OR_BASELINE_DRIFT", False, False, None, "stop: positive control failed"
    if counts[BOTH_OFF] >= int(CONFIG["factorial_gate"]["baseline_drift_if_both_off_at_or_above"]):
        return "BASELINE_DRIFT_OR_STATISTICAL_INSTABILITY", False, False, None, "stop: both-off control unexpectedly reached adequacy"
    g3, g4, g5, g8 = [counts[c] for c in (
        "G3_static_update_vector_no_reset",
        "G4_no_update_static_reset_vector",
        "G5_static_update_vector_static_reset_vector",
        "G8_static_update_scalar_static_reset_scalar",
    )]
    g1 = counts["G1_dynamic_update_no_reset"]
    g2 = counts["G2_no_update_dynamic_reset"]
    if g8 >= 4:
        return "MULTI_PATH_RECURRENCE_SUFFICIENT", True, True, "static_dual_scalar", "run one minimal neutral static multi-path transfer"
    if g3 >= 4 and g4 < 4:
        return "STATIC_CARRY_PATH_SUFFICIENT", True, True, "static_vector_carry", "run one minimal neutral vector-persistence transfer"
    if g4 >= 4 and g3 < 4:
        return "STATIC_CANDIDATE_MODULATION_SUFFICIENT", True, True, "static_candidate_modulation", "run one minimal neutral static candidate-modulation transfer"
    if g5 >= 4 and g3 < 4 and g4 < 4:
        return "COMPLEMENTARY_STATIC_PATHWAYS_REQUIRED", True, True, "static_dual_path", "run one minimal neutral dual-static-path transfer"
    if g5 >= 4:
        return "DYNAMIC_GATING_NOT_REQUIRED", True, True, "simplest_successful_static_pathway", "select the simplest successful static pathway for one neutral transfer"
    if g1 >= 4 and g2 >= 4 and g3 < 4 and g4 < 4 and g5 < 4:
        return "DYNAMIC_STATE_MODULATION_REQUIRED", True, True, "single_dynamic_modulator", "run one generic dynamic neutral modulator"
    return "DIAGNOSTIC_INCONCLUSIVE", False, False, None, "do not modify neutral cell"


def _plot_outputs(summaries: dict, effects: dict, rows: list[dict]) -> None:
    plot_dir = HERE / "plots"
    plot_dir.mkdir(exist_ok=True)
    labels = CONFIG["conditions"]
    counts = [summaries[c]["families_passing"] for c in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)), [c.split("_")[0] for c in labels])
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Families passing / 5")
    ax.set_title("V837o factorial families passing")
    fig.tight_layout(); fig.savefig(plot_dir / "families_passing_factorial.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(FAMILIES)); width = 0.08
    for i, condition in enumerate(labels):
        vals = [summaries[condition]["family_results"][f]["validation"]["median"] for f in FAMILIES]
        ax.bar(x + (i - 4.5) * width, vals, width, label=condition.split("_")[0])
    ax.set_xticks(x, FAMILIES, rotation=20, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("Validation median"); ax.legend(ncol=5, fontsize=8)
    ax.set_title("V837o family scores by factorial condition")
    fig.tight_layout(); fig.savefig(plot_dir / "family_scores_factorial.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    dynamic = effects["dynamic_vs_off_2x2"]
    static = effects["static_vector_vs_off_2x2"]
    x = np.arange(len(FAMILIES))
    ax.plot(x, [dynamic[f]["interaction"] for f in FAMILIES], marker="o", label="dynamic/off interaction")
    ax.plot(x, [static[f]["interaction"] for f in FAMILIES], marker="o", label="static-vector/off interaction")
    ax.axhline(0, linewidth=1); ax.set_xticks(x, FAMILIES, rotation=20, ha="right"); ax.set_ylabel("Interaction effect"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "update_reset_interaction.png", dpi=160); plt.close(fig)

    static_conditions = [c for c in labels if summaries[c]["static_parameter_diagnostics"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = []; values = []; names = []
    p = 0
    for condition in static_conditions:
        for factor in ("update", "reset"):
            diag = summaries[condition]["static_parameter_diagnostics"].get(factor)
            if diag:
                values.append(diag["mean"]["median"]); positions.append(p); names.append(f"{condition.split('_')[0]}-{factor[0]}"); p += 1
    ax.bar(positions, values); ax.set_xticks(positions, names, rotation=45, ha="right"); ax.set_ylabel("Median learned coefficient")
    fig.tight_layout(); fig.savefig(plot_dir / "static_vector_distributions.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = []; values = []; names = []; p = 0
    for condition in static_conditions:
        for factor in ("update", "reset"):
            diag = summaries[condition]["static_parameter_diagnostics"].get(factor)
            if diag:
                values.append(diag["inter_dimension_variance"]["median"]); positions.append(p); names.append(f"{condition.split('_')[0]}-{factor[0]}"); p += 1
    ax.bar(positions, values); ax.set_xticks(positions, names, rotation=45, ha="right"); ax.set_ylabel("Median inter-dimension variance")
    fig.tight_layout(); fig.savefig(plot_dir / "static_vector_variance_by_family.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in (FULL, "G5_static_update_vector_static_reset_vector", "G8_static_update_scalar_static_reset_scalar", BOTH_OFF):
        condition_rows = [r for r in rows if r["condition"] == condition]
        by_step = {}
        for r in condition_rows:
            for point in r["learning_curve"]:
                by_step.setdefault(point["step"], []).append(point["validation_success"])
        steps = sorted(by_step); vals = [float(np.median(by_step[s])) for s in steps]
        ax.plot(steps, vals, marker="o", label=condition.split("_")[0])
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Median validation success"); ax.legend(); ax.set_title("Full vs static learning curves")
    fig.tight_layout(); fig.savefig(plot_dir / "full_vs_static_learning_curves.png", dpi=160); plt.close(fig)

    # Optional flattening effect plot for static-vector conditions.
    flatten_rows = []
    for condition in labels:
        for r in [x for x in rows if x["condition"] == condition]:
            for name, score in r["diagnostics"]["counterfactual_flattening"].items():
                flatten_rows.append((condition, name, r["validation_success"] - score))
    if flatten_rows:
        keys = sorted(set((c, n) for c, n, _ in flatten_rows))
        vals = [float(np.median([d for c2, n2, d in flatten_rows if (c2, n2) == key])) for key in keys]
        fig, ax = plt.subplots(figsize=(12, 5)); ax.bar(range(len(keys)), vals)
        ax.set_xticks(range(len(keys)), [f"{c.split('_')[0]}:{n}" for c, n in keys], rotation=45, ha="right"); ax.set_ylabel("Validation loss from flattening")
        fig.tight_layout(); fig.savefig(plot_dir / "counterfactual_flattening_effect.png", dpi=160); plt.close(fig)


def main() -> int:
    positive = json.loads((HERE / "diagnostics" / "full_gru_positive_control.json").read_text(encoding="utf-8"))
    if positive.get("compatible") is not True:
        raise SystemExit("positive control incompatible; V837o analysis blocked")
    rows = _load_rows()
    summaries = {condition: summarize_condition(condition, rows) for condition in CONFIG["conditions"]}
    effects = _factorial_effects(summaries)
    paired = {condition: {family: _paired(rows, condition, family) for family in FAMILIES} for condition in CONFIG["conditions"] if condition != FULL}
    recovery = _recovery_fraction(summaries)
    diagnosis, diagnostic_pass, neutral_allowed, neutral_type, next_action = _diagnose(summaries)
    _plot_outputs(summaries, effects, rows)

    totals = {
        "model_fits": len(rows),
        "optimizer_steps": int(sum(r["resources"]["optimizer_steps"] for r in rows)),
        "examples_processed": int(sum(r["resources"]["examples_processed"] for r in rows)),
        "environment_interactions": int(sum(r["resources"]["environment_steps"] for r in rows)),
        "forward_calls": int(sum(r["resources"]["forward_calls"] for r in rows)),
        "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in rows)),
        "cpu_seconds_sum_workers": float(sum(r["resources"]["cpu_seconds"] for r in rows)),
        "gpu_seconds": 0.0,
    }
    payload = {
        "version": "V837o",
        "parent": "V837n",
        "question": CONFIG["question"],
        "data_regime": "4x_unique",
        "full_gru_reproduced": True,
        "conditions": summaries,
        "factorial_effects": effects,
        "paired_effects": paired,
        "recovery_fraction": recovery,
        "static_parameter_diagnostics": {c: summaries[c]["static_parameter_diagnostics"] for c in CONFIG["conditions"]},
        "mechanism_diagnosis": diagnosis,
        "diagnostic_pass": bool(diagnostic_pass),
        "neutral_followup_allowed": bool(neutral_allowed),
        "neutral_followup_type": neutral_type,
        "next_variant": "V837p" if neutral_allowed else "",
        "next_action": next_action,
        "resource_accounting": totals,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "primitive_mining_allowed": False,
        "structural_search_allowed": False,
    }
    write_json(HERE / "results.json", payload)
    write_json(HERE / "diagnostics" / "factorial_effects.json", effects)
    write_json(HERE / "diagnostics" / "paired_effects.json", paired)
    write_json(HERE / "diagnostics" / "static_parameter_diagnostics.json", payload["static_parameter_diagnostics"])
    write_json(HERE / "diagnostics" / "decision_state.json", {
        "v837o_complete": True,
        "v837o_diagnosis": diagnosis,
        "neutral_followup_allowed": bool(neutral_allowed),
        "neutral_followup_type": neutral_type,
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
    })
    doc = HERE / ("PASS.md" if diagnostic_pass else "FAILURE.md")
    lines = [f"# V837o {'DIAGNOSTIC PASS' if diagnostic_pass else 'DIAGNOSTIC INCONCLUSIVE'}", "", f"Diagnosis: **{diagnosis}**.", "", "Families passing:"]
    for condition in CONFIG["conditions"]:
        lines.append(f"- {condition}: {summaries[condition]['families_passing']}/5")
    lines += ["", f"Next action: {next_action}", "", "Fresh audit consumed: 0. Primitives promoted: 0. Primitive mining and structural search remain blocked."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V837o diagnosis: {diagnosis}; diagnostic_pass={diagnostic_pass}; neutral_followup={neutral_type}")
    print("families passing: " + ", ".join(f"{c}={summaries[c]['families_passing']}/5" for c in CONFIG["conditions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
