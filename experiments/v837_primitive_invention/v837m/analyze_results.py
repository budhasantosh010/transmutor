from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ["historical_direct", "scalar_persistence", "linear_transport", "parameter_matched_additive"]
FAMILIES = ["conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"]


def main() -> int:
    data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(CONDITIONS, [data["conditions"][name]["families_passing"] for name in CONDITIONS])
    ax.set_ylim(0, 5)
    ax.set_ylabel("Families passing unchanged criterion")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "families_passing_by_update_mode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(FAMILIES))
    width = 0.19
    for i, condition in enumerate(CONDITIONS):
        vals = [data["conditions"][condition]["family_results"][family]["validation"]["median"] for family in FAMILIES]
        ax.bar(x + (i - 1.5) * width, vals, width, label=condition)
    ax.axhline(0.85, linestyle="--", linewidth=1)
    ax.set_xticks(x, FAMILIES, rotation=20)
    ax.set_ylabel("Median validation success")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "family_scores_by_update_mode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(CONDITIONS, [data["conditions"][name]["parameter_count"] for name in CONDITIONS])
    ax.set_ylabel("Trainable parameters")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "parameter_count_control.png", dpi=160)
    plt.close(fig)

    summary = {
        "version": "V837m",
        "diagnosis": data["diagnosis"],
        "pass": data["pass"],
        "families_passing": {name: data["conditions"][name]["families_passing"] for name in CONDITIONS},
        "parameter_matching": data["parameter_matching"],
        "fresh_audit_consumed": False,
    }
    (HERE / "diagnostics" / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
