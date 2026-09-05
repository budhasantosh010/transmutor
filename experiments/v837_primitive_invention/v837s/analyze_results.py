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
from experiments.v837_primitive_invention.common.metrics import continuous_summary, paired_bootstrap_difference
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])


def _load_rows() -> list[dict]:
    path = HERE / "raw" / "runs.json"
    if not path.exists():
        raise SystemExit("run V837s interaction before analysis")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _rows(rows: list[dict], condition: str, family: str | None = None) -> list[dict]:
    out = [row for row in rows if row["condition"] == condition]
    if family is not None:
        out = [row for row in out if row["family"] == family]
    return sorted(out, key=lambda row: (row["family"], row["replicate"]))


def _nested(rows: list[dict], *path: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        current = row
        try:
            for key in path:
                current = current[key]
        except (KeyError, TypeError):
            continue
        if current is not None:
            values.append(float(current))
    return values


def _summary(rows: list[dict], condition: str) -> dict:
    selected = _rows(rows, condition)
    if len(selected) != len(FAMILIES) * int(CONFIG["training"]["replicates"]):
        raise RuntimeError(f"{condition} row count mismatch: {len(selected)}")
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = _rows(rows, condition, family)
        dev = np.asarray([r["development_success"] for r in fr], dtype=float)
        val = np.asarray([r["validation_success"] for r in fr], dtype=float)
        passed = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(passed)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "aggregate_capacity_pass": bool(passed),
            "replicate_capacity_successes": int(sum(bool(r["capacity_demonstrated"]) for r in fr)),
            "global_to_local_ratio": continuous_summary(_nested(fr, "diagnostics", "global_to_local_ratio")),
            "global_to_message_ratio": continuous_summary(_nested(fr, "diagnostics", "global_to_message_ratio")),
            "message_dependency_prediction_delta": continuous_summary(_nested(fr, "diagnostics", "message_dependency", "mean_abs_prediction_delta")),
            "modulator_mean": continuous_summary(_nested(fr, "diagnostics", "state_modulator", "mean")),
            "modulator_std": continuous_summary(_nested(fr, "diagnostics", "state_modulator", "std")),
            "modulator_temporal_variance": continuous_summary(_nested(fr, "diagnostics", "state_modulator", "temporal_variance")),
        }
    first = selected[0]
    return {
        "parameter_count": int(first["parameter_count"]),
        "parameter_bytes": int(first["parameter_bytes"]),
        "recurrent_macs_per_timestep": int(first["recurrent_macs_per_timestep"]),
        "coupling_macs_per_timestep": int(first["coupling_macs_per_timestep"]),
        "modulator_macs_per_timestep": int(first["modulator_macs_per_timestep"]),
        "state_modulation_mode": first["state_modulation_mode"],
        "coupling_spec": first["coupling_spec"],
        "families_passing": int(families_passing),
        "family_results": per_family,
        "mean_family_validation_median": float(np.mean([per_family[f]["validation"]["median"] for f in FAMILIES])),
        "resource_accounting": {
            "model_fits": len(selected),
            "optimizer_steps": int(sum(r["resources"]["optimizer_steps"] for r in selected)),
            "examples_processed": int(sum(r["resources"]["examples_processed"] for r in selected)),
            "environment_interactions": int(sum(r["resources"]["environment_steps"] for r in selected)),
            "forward_calls": int(sum(r["resources"]["forward_calls"] for r in selected)),
            "wall_seconds_sum_workers": float(sum(r["resources"].get("wall_seconds", 0.0) for r in selected)),
            "cpu_seconds_sum_workers": float(sum(r["resources"].get("cpu_seconds", 0.0) for r in selected)),
            "gpu_seconds": float(sum(r.get("gpu_seconds", 0.0) for r in selected)),
        },
    }


def _paired_delta(rows: list[dict], a: str, b: str, family: str) -> dict:
    ar = _rows(rows, a, family)
    br = _rows(rows, b, family)
    av = np.asarray([r["validation_success"] for r in ar], dtype=float)
    bv = np.asarray([r["validation_success"] for r in br], dtype=float)
    return paired_bootstrap_difference(av, bv, seed=deterministic_int("v837s-paired", a, b, family))


def _interaction_effects(rows: list[dict]) -> dict:
    output = {}
    for family in FAMILIES:
        vals = {
            condition: np.asarray([r["validation_success"] for r in _rows(rows, condition, family)], dtype=float)
            for condition in CONDITIONS
        }
        did = (vals["S3_rank4_dynamic_scalar"] - vals["S2_rank4_no_modulation"]) - (
            vals["S1_local_dynamic_scalar"] - vals["S0_local_no_modulation"]
        )
        output[family] = {
            "coupling_effect_without_modulation": _paired_delta(rows, "S2_rank4_no_modulation", "S0_local_no_modulation", family),
            "modulation_effect_without_coupling": _paired_delta(rows, "S1_local_dynamic_scalar", "S0_local_no_modulation", family),
            "modulation_effect_with_coupling": _paired_delta(rows, "S3_rank4_dynamic_scalar", "S2_rank4_no_modulation", family),
            "combined_vs_local": _paired_delta(rows, "S3_rank4_dynamic_scalar", "S0_local_no_modulation", family),
            "true_modulation_vs_matched_additive_under_coupling": _paired_delta(rows, "S3_rank4_dynamic_scalar", "S3C_rank4_matched_dynamic_additive", family),
            "difference_in_differences": {
                "mean": float(np.mean(did)),
                "median": float(np.median(did)),
                "std": float(np.std(did)),
                "replicate_values": [float(v) for v in did.tolist()],
            },
        }
    return output


def _reproduction_check(summaries: dict) -> dict:
    r = json.loads((ROOT / "experiments/v837_primitive_invention/v837r/results.json").read_text(encoding="utf-8"))
    mapping = {
        "S0_local_no_modulation": "R0_local",
        "S2_rank4_no_modulation": "R3_rank4",
    }
    out = {}
    for s_name, r_name in mapping.items():
        deltas = {}
        for family in FAMILIES:
            s_val = summaries[s_name]["family_results"][family]["validation"]["median"]
            r_val = r["conditions"][r_name]["family_results"][family]["validation"]["median"]
            deltas[family] = abs(float(s_val) - float(r_val))
        out[s_name] = {
            "reference": r_name,
            "absolute_validation_median_deltas": deltas,
            "max_absolute_delta": max(deltas.values()),
            "compatible": sum(delta > 0.10 for delta in deltas.values()) < 2,
        }
    return out


def _diagnose(summaries: dict) -> tuple[str, bool, str, bool]:
    s0 = int(summaries["S0_local_no_modulation"]["families_passing"])
    s1 = int(summaries["S1_local_dynamic_scalar"]["families_passing"])
    s2 = int(summaries["S2_rank4_no_modulation"]["families_passing"])
    s3 = int(summaries["S3_rank4_dynamic_scalar"]["families_passing"])
    s3c = int(summaries["S3C_rank4_matched_dynamic_additive"]["families_passing"])
    if s3 >= 4 and s1 < 4 and s2 < 4:
        if s3c < 4:
            return (
                "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INTERACTION",
                True,
                "V837t_COUPLING_COMPRESSION_ALLOWED",
                True,
            )
        return (
            "INTERACTION_RECOVERY_WITHOUT_MULTIPLICATIVE_SPECIFICITY",
            True,
            "DYNAMIC_BRANCH_SPECIFICITY_LOCALIZATION_REQUIRED",
            False,
        )
    if s3 >= 4:
        return (
            "REPRESENTATION_RECOVERY_WITHOUT_INTERACTION_NECESSITY",
            True,
            "ISOLATE_SUFFICIENT_SINGLE_FACTOR_BEFORE_COMPRESSION",
            False,
        )
    return (
        "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT",
        False,
        "DYNAMIC_VECTOR_VALUED_MODULATION_IS_NEXT_SINGLE_VARIABLE",
        False,
    )


def _aggregate_resources(summaries: dict) -> dict:
    records = [v["resource_accounting"] for v in summaries.values()]
    return {
        "model_fits": int(sum(r["model_fits"] for r in records)),
        "optimizer_steps": int(sum(r["optimizer_steps"] for r in records)),
        "examples_processed": int(sum(r["examples_processed"] for r in records)),
        "environment_interactions": int(sum(r["environment_interactions"] for r in records)),
        "forward_calls": int(sum(r["forward_calls"] for r in records)),
        "wall_seconds_sum_workers": float(sum(r["wall_seconds_sum_workers"] for r in records)),
        "cpu_seconds_sum_workers": float(sum(r["cpu_seconds_sum_workers"] for r in records)),
        "gpu_seconds": float(sum(r["gpu_seconds"] for r in records)),
        "structural_search_runs": 0,
        "primitive_mining_runs": 0,
        "fresh_audit_episodes": 0,
    }


def _plots(summaries: dict, interaction: dict) -> None:
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    labels = ["local\nno mod", "local\ndynamic", "rank4\nno mod", "rank4\ndynamic", "rank4\nmatched additive"]
    counts = [summaries[c]["families_passing"] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, counts)
    ax.set_ylim(0, 5.2); ax.set_ylabel("Families passing / 5")
    fig.tight_layout(); fig.savefig(plots / "global_coupling_modulation_factorial.png", dpi=160); plt.close(fig)

    x = np.arange(len(FAMILIES))
    did = [interaction[f]["difference_in_differences"]["mean"] for f in FAMILIES]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, did)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, FAMILIES, rotation=20, ha="right")
    ax.set_ylabel("Validation interaction ΔΔ")
    fig.tight_layout(); fig.savefig(plots / "interaction_effect_by_family.png", dpi=160); plt.close(fig)


def main() -> int:
    rows = _load_rows()
    summaries = {condition: _summary(rows, condition) for condition in CONDITIONS}
    reproduction = _reproduction_check(summaries)
    if not all(v["compatible"] for v in reproduction.values()):
        raise SystemExit("V837s parent-factor reproduction drift; refuse interaction interpretation")
    interaction = _interaction_effects(rows)
    diagnosis, adequacy, next_variant, compression_allowed = _diagnose(summaries)
    resources = _aggregate_resources(summaries)
    compute = {
        condition: {
            "trainable_parameters": summaries[condition]["parameter_count"],
            "recurrent_macs_per_timestep": summaries[condition]["recurrent_macs_per_timestep"],
            "families_passing": summaries[condition]["families_passing"],
            "mean_family_validation_median": summaries[condition]["mean_family_validation_median"],
            "capability_per_recurrent_mac": summaries[condition]["families_passing"] / max(1, summaries[condition]["recurrent_macs_per_timestep"]),
        }
        for condition in CONDITIONS
    }
    result = {
        "version": "V837s",
        "parent": "V837r",
        "question": CONFIG["question"],
        "single_change": CONFIG["single_change"],
        "data_regime": CONFIG["data_regime"],
        "state_layout": CONFIG["state_layout"],
        "total_state_dim": CONFIG["total_state_dim"],
        "conditions": summaries,
        "parent_factor_reproduction": reproduction,
        "factorial_interaction_effects": interaction,
        "compute_efficiency": compute,
        "representation_adequacy_pass": bool(adequacy),
        "diagnosis": diagnosis,
        "diagnostic_pass": True,
        "interaction_recovery": bool(adequacy and summaries["S1_local_dynamic_scalar"]["families_passing"] < 4 and summaries["S2_rank4_no_modulation"]["families_passing"] < 4),
        "multiplicative_specificity_established": bool(adequacy and summaries["S3C_rank4_matched_dynamic_additive"]["families_passing"] < 4),
        "coupling_compression_allowed": bool(compression_allowed),
        "next_variant": next_variant,
        "sample_efficiency_retest_allowed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "v838_started": False,
        "resource_accounting": resources,
    }
    write_json(HERE / "results.json", result)
    write_json(HERE / "diagnostics" / "interaction_effects.json", interaction)
    write_json(HERE / "diagnostics" / "compute_efficiency.json", compute)
    decision = {
        "v837s_complete": True,
        "diagnosis": diagnosis,
        "representation_adequacy_pass": bool(adequacy),
        "families_passing": {condition: summaries[condition]["families_passing"] for condition in CONDITIONS},
        "multiplicative_specificity_established": result["multiplicative_specificity_established"],
        "coupling_compression_allowed": bool(compression_allowed),
        "next_variant": next_variant,
        "sample_efficiency_retest_allowed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "fresh_audit_consumed": False,
        "primitives_promoted": 0,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision)
    _plots(summaries, interaction)
    doc_name = "PASS.md" if adequacy else "FAILURE.md"
    lines = [
        f"# V837s {'REPRESENTATION RECOVERY' if adequacy else 'INTERACTION FAILURE'}",
        "",
        f"Diagnosis: **{diagnosis}**",
        "",
        "## Families passing",
        "",
    ]
    for condition in CONDITIONS:
        lines.append(f"- {condition}: {summaries[condition]['families_passing']}/5")
    lines += [
        "",
        f"Representation adequacy: **{'PASS' if adequacy else 'FAIL'}**.",
        f"Next action: `{next_variant}`.",
        "",
        "Fresh-audit episodes consumed: 0. Structural search: blocked. Primitive mining: blocked. Primitives promoted: 0. V838: not started.",
    ]
    (HERE / doc_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnosis": diagnosis,
        "families_passing": {c: summaries[c]["families_passing"] for c in CONDITIONS},
        "representation_adequacy_pass": adequacy,
        "multiplicative_specificity_established": result["multiplicative_specificity_established"],
        "next_variant": next_variant,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
