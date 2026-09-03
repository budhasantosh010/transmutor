from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ["neutral_high_capacity", "gru_reference", "residual_rnn_reference"]
FAMILIES = ["conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"]


def main() -> int:
    data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    labels = list(data["conditions"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.24
    for i, model in enumerate(MODELS):
        ax.bar(x + (i - 1) * width, [data["conditions"][label][model]["families_passing"] for label in labels], width, label=model)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Families passing unchanged criterion")
    ax.set_xlabel("Unique development data multiplier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "families_passing_by_data_budget.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        values = []
        for label in labels:
            medians = [data["conditions"][label][model]["family_results"][family]["validation"]["median"] for family in FAMILIES]
            values.append(float(np.mean(medians)))
        ax.plot(labels, values, marker="o", label=model)
    ax.axhline(0.85, linestyle="--", linewidth=1)
    ax.set_ylabel("Mean of family median validation scores")
    ax.set_xlabel("Unique development data multiplier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "validation_by_data_budget.png", dpi=160)
    plt.close(fig)

    summary = {
        "version": "V837l",
        "diagnosis": data["diagnosis"],
        "pass": data["pass"],
        "resolved_at_data_multiplier": data["resolved_at_data_multiplier"],
        "families_passing": {label: {model: data["conditions"][label][model]["families_passing"] for model in MODELS} for label in labels},
        "fresh_audit_consumed": False,
    }
    (HERE / "diagnostics" / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
