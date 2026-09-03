from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.v837d.experiment import CONFIG, FAMILIES

HERE = Path(__file__).resolve().parent
DIAGNOSTICS = HERE / "diagnostics"
PLOTS = HERE / "plots"


def _plot_density_metric(sweep: dict, broadcast: dict, field: str, output: str, ylabel: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    densities = sorted(float(value) for value in sweep["by_density"])
    x = densities + [1.0]
    y = []
    for density in densities:
        y.append(float(sweep["by_density"][str(density)]["representation_diagnostics"][field]["median"]))
    y.append(float(broadcast["summary"]["representation_diagnostics"][field]["median"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Effective/raw-input density condition")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by input density")
    fig.tight_layout()
    fig.savefig(PLOTS / output, dpi=160)
    plt.close(fig)


def _write_plots(sweep: dict, broadcast: dict, controls: dict) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    densities = sorted(float(value) for value in sweep["by_density"])
    labels = [str(value) for value in densities] + ["1.0 broadcast"]
    family_counts = [sweep["by_density"][str(value)]["families_passing_aggregate"] for value in densities] + [broadcast["summary"]["families_passing_aggregate"]]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(labels, family_counts); ax.set_ylabel("Families passing capacity criterion"); ax.set_xlabel("Input density"); ax.set_ylim(0, 5); fig.tight_layout(); fig.savefig(PLOTS / "families_passing_by_density.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(FAMILIES)); width = 0.18
    for offset, density in enumerate(densities):
        values = [sweep["by_density"][str(density)]["per_family"][family]["validation"]["median"] for family in FAMILIES]
        ax.bar(positions + (offset - 1.5) * width, values, width, label=str(density))
    broadcast_values = [broadcast["summary"]["per_family"][family]["validation"]["median"] for family in FAMILIES]
    ax.bar(positions + 1.5 * width, broadcast_values, width, label="broadcast")
    ax.set_xticks(positions); ax.set_xticklabels(FAMILIES, rotation=25, ha="right"); ax.set_ylabel("Median validation success"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "capacity_by_density.png", dpi=160); plt.close(fig)

    _plot_density_metric(sweep, broadcast, "message_dependency_ratio", "message_dependency_by_density.png", "Message dependency ratio")
    _plot_density_metric(sweep, broadcast, "mean_pairwise_state_corr", "cell_state_correlation_by_density.png", "Mean pairwise state correlation")
    _plot_density_metric(sweep, broadcast, "state_saturation_fraction", "saturation_by_density.png", "State saturation fraction")

    selected = controls["selected_density"]
    fixed = controls["fixed_sparse"]
    no_message = controls["no_message"]
    fixed_values = [fixed["per_family"][family]["validation"]["median"] for family in FAMILIES]
    no_message_values = [no_message["per_family"][family]["validation"]["median"] for family in FAMILIES]
    fig, ax = plt.subplots(figsize=(8, 4.5)); positions = np.arange(len(FAMILIES)); width = 0.35
    ax.bar(positions - width/2, fixed_values, width, label=f"fixed sparse {selected}")
    ax.bar(positions + width/2, no_message_values, width, label="same mask, messages disabled")
    ax.set_xticks(positions); ax.set_xticklabels(FAMILIES, rotation=25, ha="right"); ax.set_ylabel("Median validation success"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "sparse_vs_no_message.png", dpi=160); plt.close(fig)


def main() -> int:
    broadcast = json.loads((DIAGNOSTICS / "broadcast_capacity.json").read_text(encoding="utf-8"))
    sweep = json.loads((DIAGNOSTICS / "sparse_density_sweep.json").read_text(encoding="utf-8"))
    controls = json.loads((DIAGNOSTICS / "controls.json").read_text(encoding="utf-8"))
    selected = float(sweep["selected_density"])
    selected_summary = sweep["by_density"][str(selected)]
    families_passing = int(selected_summary["families_passing_aggregate"])
    full_pass = families_passing >= 4

    fixed_minus_broadcast = controls["paired_validation_deltas"]["fixed_minus_broadcast"]
    partial_rule = CONFIG["partial_improvement_rule"]
    improved = [
        family for family in partial_rule["historically_failing_families"]
        if float(fixed_minus_broadcast[family]["ci"][0]) > 0.0
    ]
    strong_non_degraded = all(
        float(fixed_minus_broadcast[family]["median_difference"]) >= -0.10
        for family in partial_rule["previously_strong_families"]
    )
    partial = (not full_pass) and len(improved) >= 2 and strong_non_degraded

    no_message_minus_fixed = controls["paired_validation_deltas"]["no_message_minus_fixed"]
    no_message_not_worse = sum(
        float(no_message_minus_fixed[family]["ci"][0]) >= -0.02 for family in FAMILIES
    ) >= 4
    message_dependency_fixed = float(controls["fixed_sparse"]["representation_diagnostics"]["message_dependency_ratio"]["median"])
    message_dependency_broadcast = float(broadcast["summary"]["representation_diagnostics"]["message_dependency_ratio"]["median"])

    if full_pass:
        outcome = "REPRESENTATION_RECOVERY_CANDIDATE"
        failure_classes = []
        next_experiment = "Freeze the selected fixed-sparse representation and rerun full neutral structural search under the original V837 competence gate before motif mining."
        interpretation = "Fixed sparse raw-input access restored >=4/5 high-capacity family competence. This is a capacity recovery candidate only, not a primitive-invention pass."
        v837e_justified = False
    elif partial:
        outcome = "PARTIAL_REPRESENTATION_IMPROVEMENT"
        failure_classes = ["INPUT_ACCESS_FAILURE"]
        next_experiment = "V837e: allow only the raw input-edge topology to evolve under a fixed input-edge penalty and cheap capacity/search budget."
        interpretation = "Fixed sparsity improved at least two historically failing families with paired bootstrap support without materially degrading previously strong families, but did not restore >=4/5 competence."
        v837e_justified = True
    elif no_message_not_worse:
        outcome = "INPUT_RESTRICTION_REGULARIZATION_EFFECT"
        failure_classes = ["INPUT_ACCESS_FAILURE", "REGULARIZATION_ONLY_EFFECT", "MESSAGE_MEDIATION_FAILURE"]
        next_experiment = "Skip input-edge evolution and test the next isolated cell-law property (generic state-update persistence) because message mediation is not supported as the mechanism."
        interpretation = "Sparse access did not restore competence and disabling messages was statistically comparable on most families, so evidence favors generic input restriction/regularization rather than message-mediated computation."
        v837e_justified = False
    elif message_dependency_fixed > message_dependency_broadcast and families_passing < 4:
        outcome = "MESSAGE_MEDIATION_INDUCED_BUT_REPRESENTATION_STILL_INSUFFICIENT"
        failure_classes = ["INPUT_ACCESS_FAILURE", "MESSAGE_MEDIATION_FAILURE"]
        next_experiment = "Skip further sparsity and test the generic state update law as the next single representation variable."
        interpretation = "Sparse access increased generic message dependence but did not restore competence, so more input sparsity is not justified."
        v837e_justified = False
    else:
        outcome = "RAW_BROADCAST_HYPOTHESIS_NOT_SUPPORTED"
        failure_classes = ["INPUT_ACCESS_FAILURE"]
        next_experiment = "Skip input-edge evolution and test the generic state update law as the next single representation variable."
        interpretation = "Fixed sparse access did not provide supported aggregate capacity improvement over broadcast."
        v837e_justified = False

    selected_rows = [row for row in sweep["rows"] if abs(float(row["requested_density"]) - selected) < 1e-12]
    total_resources = {}
    for key in ("candidate_evaluations", "optimizer_steps", "environment_steps", "examples_processed", "forward_calls", "wall_seconds_sum_workers", "cpu_seconds_sum_workers"):
        total_resources[key] = sum(
            float(source["resource_accounting"].get(key, 0))
            for source in [broadcast["summary"], *[sweep["by_density"][str(d)] for d in sorted(float(v) for v in sweep["by_density"])], controls["shuffled_sparse"], controls["no_message"]]
        )
        if key not in {"wall_seconds_sum_workers", "cpu_seconds_sum_workers"}:
            total_resources[key] = int(total_resources[key])
    total_resources.update({
        "parameter_count": int(selected_rows[0]["resources"]["parameter_count"]),
        "historical_parameter_count": int(broadcast["rows"][0]["resources"]["parameter_count"]),
        "parameter_change_percent": 0.0,
        "selected_input_edges_median": float(selected_summary["resource_accounting"]["input_edges"]["median"]),
        "historical_broadcast_input_edges": int(broadcast["rows"][0]["resources"]["input_edges"]),
        "internal_edges": 55,
    })

    result = {
        "version": "V837d",
        "parent": "V837c",
        "single_change": CONFIG["single_change"],
        "representation_change": "raw input accessibility: broadcast -> deterministic fixed sparse graph-level mask",
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "conditions": {
            "broadcast": broadcast["summary"],
            "fixed_sparse_selected_density": selected_summary,
            "degree_preserving_shuffled_sparse": controls["shuffled_sparse"],
            "no_message": controls["no_message"],
        },
        "capacity_results": {family: selected_summary["per_family"][family] for family in FAMILIES},
        "representation_diagnostics": {
            "input_density": selected_summary["effective_density"],
            "message_dependency_ratio": selected_summary["representation_diagnostics"]["message_dependency_ratio"],
            "pairwise_state_correlation": selected_summary["representation_diagnostics"]["mean_pairwise_state_corr"],
            "state_saturation_fraction": selected_summary["representation_diagnostics"]["state_saturation_fraction"],
            "raw_ablation": selected_summary["representation_diagnostics"]["mean_raw_ablation_effect"],
            "message_ablation": selected_summary["representation_diagnostics"]["mean_message_ablation_effect"],
        },
        "selected_density": selected,
        "families_passing": families_passing,
        "baseline_compatibility": {
            "pass": broadcast["baseline_compatibility_pass"],
            "families_exceeding_0_10": broadcast["families_exceeding_0_10"],
            "family_differences": broadcast["family_differences"],
        },
        "paired_validation_deltas": controls["paired_validation_deltas"],
        "resource_accounting": total_resources,
        "pass_gate": {"capacity_family_development": 0.90, "capacity_family_validation": 0.85, "families_required": 4, "total_families": 5},
        "pass": full_pass,
        "outcome_classification": outcome,
        "failure_classification": failure_classes,
        "interpretation": interpretation,
        "v837e_justified": v837e_justified,
        "primitive_mining_allowed": False,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", result)
    _write_plots(sweep, broadcast, controls)

    if full_pass:
        text = "# V837d PASS\n\nFixed sparse raw-input access restored the high-capacity representation screen to >=4/5 families under the unchanged capacity criterion.\n\nThis is **NOT** a primitive-invention PASS. It establishes only that fixed sparse input access restores sufficient capacity to justify rerunning neutral structural search under the original V837 gates. Primitive mining and fresh-audit seeds remain blocked.\n"
        (HERE / "PASS.md").write_text(text, encoding="utf-8")
    else:
        failed = [family for family, value in result["capacity_results"].items() if not value["aggregate_capacity_pass"]]
        diag = result["representation_diagnostics"]
        text = f"""# V837d FAILURE\n\nWHAT: fixed sparse raw-input access did not restore >=4/5 high-capacity competence.\n\nWHERE: failing aggregate families: {failed}.\n\nWHY: {interpretation}\n\nCONTROLS: historical broadcast, degree-preserving shuffled sparse masks, and same-mask no-message training/evaluation were all run with paired task seeds and matched parameter budget.\n\nDIAGNOSTICS: selected density={selected}; message-dependency median={diag['message_dependency_ratio']['median']}; mean pairwise-state-correlation median={diag['pairwise_state_correlation']['median']}; saturation median={diag['state_saturation_fraction']['median']}; raw-ablation median summary={diag['raw_ablation']['median']}; message-ablation median summary={diag['message_ablation']['median']}.\n\nCLASSIFICATION: {failure_classes}; outcome={outcome}.\n\nNEXT: {next_experiment}\n\nFresh-audit seeds consumed: NO. Primitives promoted: 0.\n"""
        (HERE / "FAILURE.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
