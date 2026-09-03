from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODELS = ["neutral_high_capacity", "gru_reference", "residual_rnn_reference"]
FAMILIES = ["conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"]


def main() -> int:
    result_path = HERE / "results.json"
    if not result_path.exists():
        raise SystemExit("run V837k before analysis")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    conditions = data["conditions"]
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)

    labels = list(conditions)
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 5))
    for model in MODELS:
        values = [conditions[label]["models"][model]["families_passing"] for label in labels]
        plt.plot(x, values, marker="o", label=model)
    plt.xticks(x, labels)
    plt.ylim(0, 5.2)
    plt.ylabel("Families passing frozen capacity criterion")
    plt.xlabel("Optimizer-step multiplier")
    plt.title("V837k: capacity versus optimizer-step budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "families_passing_by_step_budget.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(9, 10), sharex=True)
    for axis, model in zip(axes, MODELS):
        for family in FAMILIES:
            values = [conditions[label]["models"][model]["family_results"][family]["validation"]["median"] for label in labels]
            axis.plot(x, values, marker="o", label=family)
        axis.axhline(0.85, linestyle="--", linewidth=1)
        axis.set_ylabel(model)
        axis.set_ylim(0, 1.05)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Optimizer-step multiplier")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("V837k: validation medians by family and optimizer budget")
    fig.tight_layout()
    fig.savefig(plots / "validation_scores_by_step_budget.png", dpi=160)
    plt.close(fig)

    curve_condition = labels[-1]
    curve_data = conditions[curve_condition].get("learning_curve_summary", {})
    if curve_data:
        fig, axes = plt.subplots(len(MODELS), 1, figsize=(9, 10), sharex=True)
        for axis, model in zip(axes, MODELS):
            for family in FAMILIES:
                points = curve_data[model][family]
                axis.plot([p["step"] for p in points], [p["validation_success_median"] for p in points], label=family)
            axis.axhline(0.85, linestyle="--", linewidth=1)
            axis.set_ylabel(model)
            axis.set_ylim(0, 1.05)
        axes[-1].set_xlabel("Optimizer steps")
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"V837k: validation learning curves at {curve_condition}")
        fig.tight_layout()
        fig.savefig(plots / "learning_curves_by_step_budget.png", dpi=160)
        plt.close(fig)

    summary = {
        "diagnosis": data["diagnosis"],
        "executed_multipliers": data["executed_multipliers"],
        "families_passing": {
            model: {label: conditions[label]["models"][model]["families_passing"] for label in labels}
            for model in MODELS
        },
        "fresh_audit_consumed": False,
    }
    (HERE / "diagnostics" / "plot_data.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
