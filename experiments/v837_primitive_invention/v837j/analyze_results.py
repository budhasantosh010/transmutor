from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_ORDER = ["neutral_high_capacity", "gru_reference", "residual_rnn_reference", "vanilla_rnn_reference"]
MODEL_LABELS = {
    "neutral_high_capacity": "Neutral",
    "gru_reference": "GRU",
    "residual_rnn_reference": "Residual RNN",
    "vanilla_rnn_reference": "Vanilla RNN",
}
FAMILIES = ["conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"]


def load() -> tuple[dict, dict]:
    return (
        json.loads((HERE / "results.json").read_text(encoding="utf-8")),
        json.loads((HERE / "diagnostics" / "raw_runs.json").read_text(encoding="utf-8")),
    )


def save_family_scores(results: dict) -> None:
    x = np.arange(len(FAMILIES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, model in enumerate(MODEL_ORDER):
        medians = [results["models"][model]["family_results"][family]["validation"]["median"] for family in FAMILIES]
        ax.bar(x + (index - 1.5) * width, medians, width, label=MODEL_LABELS[model])
    ax.axhline(0.85, linestyle="--", linewidth=1, label="validation gate")
    ax.set_xticks(x, [family.replace("_", "\n") for family in FAMILIES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median validation success")
    ax.set_title("V837j reference family scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "reference_family_scores.png", dpi=160)
    plt.close(fig)


def save_families_passing(results: dict) -> None:
    values = [results["models"][model]["families_passing"] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], values)
    ax.axhline(4, linestyle="--", linewidth=1, label="4/5 learnability gate")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Families passing")
    ax.set_title("Families passing by learned reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "families_passing_by_reference.png", dpi=160)
    plt.close(fig)


def save_training_vs_validation(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODEL_ORDER:
        train = [results["models"][model]["family_results"][family]["development"]["median"] for family in FAMILIES]
        val = [results["models"][model]["family_results"][family]["validation"]["median"] for family in FAMILIES]
        ax.scatter(train, val, label=MODEL_LABELS[model], s=55)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.axvline(0.90, linestyle=":", linewidth=1)
    ax.axhline(0.85, linestyle=":", linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Median training success")
    ax.set_ylabel("Median validation success")
    ax.set_title("Training vs validation behavior")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "training_vs_validation.png", dpi=160)
    plt.close(fig)


def save_learning_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        by_step: dict[int, list[float]] = {}
        for family in FAMILIES:
            for point in results["learning_curve_summary"][model][family]:
                by_step.setdefault(int(point["step"]), []).append(float(point["validation_success_median"]))
        steps = sorted(by_step)
        values = [float(np.mean(by_step[step])) for step in steps]
        ax.plot(steps, values, marker="o", label=MODEL_LABELS[model])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Mean of family median validation success")
    ax.set_ylim(0, 1.02)
    ax.set_title("V837j learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "learning_curves.png", dpi=160)
    plt.close(fig)


def save_parameter_success(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in MODEL_ORDER:
        params = results["models"][model]["parameter_count"]
        passing = results["models"][model]["families_passing"]
        ax.scatter([params], [passing], s=80, label=MODEL_LABELS[model])
        ax.annotate(MODEL_LABELS[model], (params, passing), xytext=(5, 5), textcoords="offset points")
    ax.axhline(4, linestyle="--", linewidth=1)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Families passing")
    ax.set_ylim(-0.2, 5.2)
    ax.set_title("Parameter count vs success")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "parameter_count_vs_success.png", dpi=160)
    plt.close(fig)


def main() -> int:
    results, raw = load()
    if not raw.get("rows"):
        raise SystemExit("raw V837j runs missing")
    (HERE / "plots").mkdir(exist_ok=True)
    save_family_scores(results)
    save_families_passing(results)
    save_training_vs_validation(results)
    save_learning_curves(results)
    save_parameter_success(results)
    summary = {
        "diagnosis": results["diagnosis"],
        "pass": results["pass"],
        "families_passing_by_model": {model: results["models"][model]["families_passing"] for model in MODEL_ORDER},
        "parameter_count_by_model": {model: results["models"][model]["parameter_count"] for model in MODEL_ORDER},
        "plots": [
            "plots/reference_family_scores.png",
            "plots/families_passing_by_reference.png",
            "plots/training_vs_validation.png",
            "plots/learning_curves.png",
            "plots/parameter_count_vs_success.png",
        ],
    }
    (HERE / "diagnostics" / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
