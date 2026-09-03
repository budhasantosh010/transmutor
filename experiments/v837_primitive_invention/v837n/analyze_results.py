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
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import ABLATIONS, CONFIG, FAMILIES, FULL, summarize_condition

HERE = Path(__file__).resolve().parent


def _load_rows() -> tuple[list[dict], list[dict]]:
    full = json.loads((HERE / "raw" / "full_gru.json").read_text(encoding="utf-8"))["rows"]
    ablations = json.loads((HERE / "raw" / "ablations.json").read_text(encoding="utf-8"))["rows"]
    return full, ablations


def _paired_effects(full_rows: list[dict], ablation_rows: list[dict]) -> dict:
    effects = {}
    for condition in ABLATIONS:
        effects[condition] = {}
        for family in FAMILIES:
            full = sorted((r for r in full_rows if r["family"] == family), key=lambda r: r["replicate"])
            abl = sorted((r for r in ablation_rows if r["condition"] == condition and r["family"] == family), key=lambda r: r["replicate"])
            if [r["replicate"] for r in full] != [r["replicate"] for r in abl]:
                raise RuntimeError(f"unpaired replicates for {condition}/{family}")
            a = np.asarray([r["validation_success"] for r in abl], dtype=float)
            b = np.asarray([r["validation_success"] for r in full], dtype=float)
            effects[condition][family] = paired_bootstrap_difference(
                a,
                b,
                seed=deterministic_int("v837n-paired", condition, family),
                confidence=float(CONFIG["mechanism_necessity_gate"]["bootstrap_confidence"]),
            )
    return effects


def _large_drop_families(effect: dict) -> list[str]:
    threshold = float(CONFIG["mechanism_necessity_gate"]["robust_large_drop_threshold"])
    return [
        family
        for family, record in effect.items()
        if float(record["mean_difference"]) <= -threshold and float(record["ci"][1]) < 0.0
    ]


def _diagnose(summaries: dict, effects: dict) -> tuple[str, bool, str, list[str]]:
    full_count = int(summaries[FULL]["families_passing"])
    if full_count < int(CONFIG["positive_control_gate"]["families_required"]):
        return "IMPLEMENTATION_FAILURE", False, "stop", ["REFERENCE_MECHANISM_LOCALIZATION_FAILURE"]

    counts = {name: int(summaries[name]["families_passing"]) for name in summaries}
    update_large = _large_drop_families(effects["no_update"])
    reset_large = _large_drop_families(effects["no_reset"])
    update_strong = counts["no_update"] <= 2 or len(update_large) >= 2
    reset_strong = counts["no_reset"] <= 2 or len(reset_large) >= 2

    if counts["static_update_vector"] >= 4 and counts["static_update_scalar"] <= 3:
        return (
            "VECTOR_STATE_PERSISTENCE_SUFFICIENT",
            True,
            "test a learned static vector persistence coefficient in the neutral cell before dynamic gating",
            [],
        )
    if counts["no_update"] >= 4 and counts["no_reset"] <= 2:
        return (
            "CANDIDATE_STATE_CONDITIONING_CRITICAL",
            True,
            "V837q minimal neutral candidate conditioning",
            [],
        )
    if counts["no_update"] >= 4 and counts["no_reset"] >= 4 and counts["no_update_no_reset"] <= 2:
        return (
            "MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY",
            True,
            "design the simplest common adaptive-state property before transferring multiple GRU mechanisms",
            ["MECHANISM_COUPLING_REQUIRED"],
        )
    if update_strong and reset_strong:
        update_drop = np.mean([effects["no_update"][f]["mean_difference"] for f in FAMILIES])
        reset_drop = np.mean([effects["no_reset"][f]["mean_difference"] for f in FAMILIES])
        next_exp = "V837q minimal neutral candidate conditioning" if reset_drop < update_drop else "V837o adaptive scalar neutral update"
        return "COUPLED_ADAPTIVE_STATE_CONTROL", True, next_exp, ["MECHANISM_COUPLING_REQUIRED"]
    if update_strong and counts["no_reset"] >= 4 and counts["static_update_vector"] <= 3:
        return "ADAPTIVE_UPDATE_CONTROL_CRITICAL", True, "V837o adaptive scalar neutral update", []
    if update_strong:
        return "ADAPTIVE_UPDATE_CONTROL_CONTRIBUTION", True, "V837o adaptive scalar neutral update", []
    if reset_strong:
        return "CANDIDATE_STATE_CONDITIONING_CONTRIBUTION", True, "V837q minimal neutral candidate conditioning", []
    if all(counts[name] >= 4 for name in ABLATIONS):
        return (
            "GRU_GATES_NOT_PRIMARY_EXPLANATION",
            True,
            "investigate dense hidden organization and optimization geometry before changing the neutral cell",
            [],
        )
    return (
        "DIAGNOSTIC_INCONCLUSIVE",
        False,
        "do not transfer a GRU mechanism; investigate dense-reference organization/optimization geometry",
        ["REFERENCE_MECHANISM_LOCALIZATION_FAILURE"],
    )


def _resource_totals(rows: list[dict]) -> dict:
    return {
        "model_fits": len(rows),
        "optimizer_steps": int(sum(r["resources"]["optimizer_steps"] for r in rows)),
        "examples_processed": int(sum(r["resources"]["examples_processed"] for r in rows)),
        "environment_interactions": int(sum(r["resources"]["environment_steps"] for r in rows)),
        "forward_calls": int(sum(r["resources"]["forward_calls"] for r in rows)),
        "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in rows)),
        "cpu_seconds_sum_workers": float(sum(r["resources"]["cpu_seconds"] for r in rows)),
        "gpu_seconds": 0.0,
    }


def _plot(summaries: dict, rows: list[dict]) -> None:
    plot_dir = HERE / "plots"
    plot_dir.mkdir(exist_ok=True)
    conditions = [FULL] + ABLATIONS
    labels = [name.replace("_", "\n") for name in conditions]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(conditions)), [summaries[c]["families_passing"] for c in conditions])
    ax.set_xticks(range(len(conditions)), labels, rotation=20, ha="right")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Families passing / 5")
    ax.set_title("V837n competence by explicit-GRU ablation")
    fig.tight_layout()
    fig.savefig(plot_dir / "families_passing_by_gru_ablation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.11
    for index, condition in enumerate(conditions):
        offset = (index - (len(conditions) - 1) / 2.0) * width
        vals = [summaries[condition]["family_results"][f]["validation"]["median"] for f in FAMILIES]
        ax.bar(x + offset, vals, width=width, label=condition)
    ax.axhline(0.85, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f.replace("_", "\n") for f in FAMILIES])
    ax.set_ylabel("Median validation success")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("V837n family scores by ablation")
    fig.tight_layout()
    fig.savefig(plot_dir / "family_scores_by_ablation.png", dpi=160)
    plt.close(fig)

    for gate, filename in (("update", "update_gate_statistics.png"), ("reset", "reset_gate_statistics.png")):
        fig, ax = plt.subplots(figsize=(10, 5))
        means = [summaries[c]["gate_diagnostics"][f"{gate}_mean"]["median"] for c in conditions]
        ax.bar(range(len(conditions)), means)
        ax.set_xticks(range(len(conditions)), labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"Median {gate} coefficient")
        ax.set_title(f"V837n {gate} coefficient statistics")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    update_var = [summaries[c]["gate_diagnostics"]["update_temporal_variance"]["median"] for c in conditions]
    reset_var = [summaries[c]["gate_diagnostics"]["reset_temporal_variance"]["median"] for c in conditions]
    ax.plot(x, update_var, marker="o", label="update")
    ax.plot(x, reset_var, marker="o", label="reset")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Median temporal variance")
    ax.legend()
    ax.set_title("V837n gate temporal variance")
    fig.tight_layout()
    fig.savefig(plot_dir / "gate_temporal_variance.png", dpi=160)
    plt.close(fig)

    # Aggregate validation learning curves by condition.
    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in conditions:
        condition_rows = [r for r in rows if r["condition"] == condition]
        by_step = {}
        for row in condition_rows:
            for point in row["learning_curve"]:
                by_step.setdefault(int(point["step"]), []).append(float(point["validation_success"]))
        steps = sorted(by_step)
        vals = [float(np.mean(by_step[s])) for s in steps]
        ax.plot(steps, vals, marker="o", label=condition)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Mean validation success across families/replicates")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("V837n full vs ablation learning curves")
    fig.tight_layout()
    fig.savefig(plot_dir / "full_vs_ablation_learning_curves.png", dpi=160)
    plt.close(fig)


def main() -> int:
    positive = json.loads((HERE / "diagnostics" / "full_gru_positive_control.json").read_text(encoding="utf-8"))
    if positive.get("compatible") is not True:
        raise SystemExit("explicit full GRU is not a compatible positive control")
    full_rows, ablation_rows = _load_rows()
    all_rows = full_rows + ablation_rows
    summaries = {FULL: summarize_condition(FULL, full_rows)}
    for condition in ABLATIONS:
        summaries[condition] = summarize_condition(condition, ablation_rows)
    effects = _paired_effects(full_rows, ablation_rows)
    diagnosis, diagnostic_pass, next_experiment, classes = _diagnose(summaries, effects)
    _plot(summaries, all_rows)

    payload = {
        "version": "V837n",
        "parent": "V837m",
        "question": CONFIG["question"],
        "single_change": CONFIG["single_change"],
        "reference_regime": CONFIG["reference_regime"],
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "full_gru_reproduced": True,
        "full_gru_positive_control": positive,
        "conditions": summaries,
        "families_passing": {condition: int(summary["families_passing"]) for condition, summary in summaries.items()},
        "paired_deltas": effects,
        "gate_statistics": {condition: {"aggregate": summary["gate_diagnostics"], "per_family": summary["per_family_gate_diagnostics"]} for condition, summary in summaries.items()},
        "counterfactual_gate_replay": summaries[FULL]["counterfactual_replay"],
        "mechanism_diagnosis": diagnosis,
        "diagnostic_pass": bool(diagnostic_pass),
        "failure_classification": classes,
        "resource_accounting": _resource_totals(all_rows),
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "structural_search_allowed": False,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", payload)
    write_json(HERE / "diagnostics" / "mechanism_analysis.json", {
        "mechanism_diagnosis": diagnosis,
        "families_passing": payload["families_passing"],
        "paired_deltas": effects,
        "counterfactual_gate_replay": payload["counterfactual_gate_replay"],
        "next_experiment": next_experiment,
    })

    doc = HERE / ("PASS.md" if diagnostic_pass else "FAILURE.md")
    lines = [
        f"# V837n {'DIAGNOSTIC PASS' if diagnostic_pass else 'DIAGNOSTIC FAILURE'}",
        "",
        "This is a successful-reference mechanism-localization result, not a Transmutor competence PASS.",
        "",
        f"Mechanism diagnosis: **{diagnosis}**.",
        "",
        "## Families passing",
    ]
    for condition in [FULL] + ABLATIONS:
        lines.append(f"- {condition}: {summaries[condition]['families_passing']}/5")
    lines += [
        "",
        f"Next experiment: {next_experiment}",
        "",
        "Fresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining and structural search remain blocked.",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V837n diagnosis: {diagnosis}; diagnostic_pass={diagnostic_pass}")
    print("families passing: " + ", ".join(f"{c}={summaries[c]['families_passing']}/5" for c in [FULL] + ABLATIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
