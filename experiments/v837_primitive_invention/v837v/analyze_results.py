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
from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONDITIONS = list(CONFIG["conditions"])
FAMILIES = [task.name for task in all_tasks()]


def _summary(values) -> dict:
    return continuous_summary(np.asarray(list(values), dtype=float))


def summarize(rows: list[dict], condition: str) -> dict:
    selected = [r for r in rows if r["condition"] == condition]
    family_results = {}
    families_passing = 0
    for family in FAMILIES:
        fr = [r for r in selected if r["family"] == family]
        dev = [r["development_success"] for r in fr]
        val = [r["validation_success"] for r in fr]
        passed = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(passed)
        family_results[family] = {
            "development": _summary(dev),
            "validation": _summary(val),
            "aggregate_capacity_pass": bool(passed),
            "replicate_capacity_success_rate": float(np.mean([bool(r["capacity_pass"]) for r in fr])),
        }
    gate = {key: _summary(r["diagnostics"]["source_gate"][key] for r in selected) for key in (
        "mean", "median", "std", "p10", "p90", "temporal_variance", "near_zero_fraction", "near_one_fraction", "between_domain_gate_disagreement"
    )}
    coherence_keys = (
        "pairwise_state_cosine_similarity", "state_covariance_trace", "effective_state_rank", "mean_state_norm",
        "mean_state_change_magnitude", "cross_cell_state_change_correlation", "cross_cell_realized_update_correlation",
        "mean_candidate_update_magnitude", "mean_realized_update_magnitude",
    )
    coherence = {key: _summary(r["diagnostics"]["state_coherence"][key] for r in selected) for key in coherence_keys}
    message_drop = _summary(r["diagnostics"]["message_ablation"]["performance_drop"] for r in selected)
    first = selected[0]
    resources = {
        "model_fits": len(selected),
        "optimizer_steps": sum(r["resources"]["optimizer_steps"] for r in selected),
        "processed_examples": sum(r["resources"]["examples_processed"] for r in selected),
        "environment_interactions": sum(r["resources"]["environment_steps"] for r in selected),
        "forward_calls": sum(r["resources"]["forward_calls"] for r in selected),
        "cpu_seconds": float(sum(r["resources"]["cpu_seconds"] for r in selected)),
        "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in selected)),
        "gpu_seconds": 0.0,
    }
    return {
        "domain_count": first["domain_count"], "domain_assignment": first["domain_assignment"], "source_cells": first["source_cells"],
        "families_passing": int(families_passing), "family_results": family_results,
        "nominal_controller_count": first["nominal_controller_count"], "active_controller_count": first["active_controller_count"],
        "nominal_controller_parameters": first["nominal_controller_parameters"], "active_controller_parameters": first["active_controller_parameters"],
        "nominal_parameters": first["nominal_parameters"], "active_parameters": first["active_parameters"],
        "base_macs_per_timestep": first["base_macs_per_timestep"], "controller_macs_per_timestep": first["controller_macs_per_timestep"],
        "total_macs_per_timestep": first["total_macs_per_timestep"],
        "source_gate_diagnostics": gate, "state_coherence": coherence, "message_ablation_performance_drop": message_drop,
        "resource_accounting": resources,
        "capability_per_active_mac": float(families_passing / first["total_macs_per_timestep"]),
        "capability_per_active_parameter": float(families_passing / first["active_parameters"]),
    }


def diagnosis(summaries: dict[str, dict]) -> tuple[str, str | None]:
    passing = [c for c in CONDITIONS if summaries[c]["families_passing"] >= CONFIG["representation_family_gate"]]
    if passing:
        preferred = max(passing, key=lambda c: summaries[c]["domain_count"])
        if preferred == "V3_1_domain":
            return "GLOBAL_CONTROL_SCOPE_SUFFICIENT", preferred
        return "INTERMEDIATE_CONTROL_DOMAIN_SCALE_SUFFICIENT", preferred
    counts = [summaries[c]["families_passing"] for c in CONDITIONS]
    if counts[-1] < counts[0] and max(counts[1:]) <= counts[0]:
        return "CONTROL_SCOPE_COARSENING_HARMFUL", None
    if max(counts[1:]) > counts[0]:
        return "CONTROL_SCOPE_PARTIAL_BENEFIT", None
    return "CONTROL_SCOPE_ALONE_INSUFFICIENT", None


def _plots(s: dict[str, dict]) -> None:
    out = HERE / "plots"
    out.mkdir(exist_ok=True)
    x = np.arange(len(CONDITIONS))
    labels = [str(s[c]["domain_count"]) for c in CONDITIONS]

    plt.figure(figsize=(7,4)); plt.plot(x,[s[c]["families_passing"] for c in CONDITIONS],marker="o"); plt.xticks(x,labels); plt.xlabel("independent control domains"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(out/"families_passing_vs_control_domains.png"); plt.close()
    plt.figure(figsize=(9,4))
    for f in FAMILIES: plt.plot(x,[s[c]["family_results"][f]["validation"]["median"] for c in CONDITIONS],marker="o",label=f)
    plt.xticks(x,labels); plt.xlabel("independent control domains"); plt.ylabel("validation median"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/"family_scores_vs_control_domains.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot([s[c]["total_macs_per_timestep"] for c in CONDITIONS],[s[c]["families_passing"] for c in CONDITIONS],marker="o"); plt.xlabel("active recurrent+controller MACs/timestep"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(out/"active_controller_macs_vs_capability.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot([s[c]["active_parameters"] for c in CONDITIONS],[s[c]["families_passing"] for c in CONDITIONS],marker="o"); plt.xlabel("active trainable parameters"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(out/"active_controller_params_vs_capability.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(x,[s[c]["source_gate_diagnostics"]["temporal_variance"]["median"] for c in CONDITIONS],marker="o"); plt.xticks(x,labels); plt.ylabel("source-gate temporal variance"); plt.tight_layout(); plt.savefig(out/"control_signal_temporal_variance.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(x,[s[c]["source_gate_diagnostics"]["between_domain_gate_disagreement"]["median"] for c in CONDITIONS],marker="o"); plt.xticks(x,labels); plt.ylabel("between-domain gate variance"); plt.tight_layout(); plt.savefig(out/"between_domain_gate_disagreement.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(x,[s[c]["state_coherence"]["cross_cell_realized_update_correlation"]["median"] for c in CONDITIONS],marker="o"); plt.xticks(x,labels); plt.ylabel("cross-cell realized-update correlation"); plt.tight_layout(); plt.savefig(out/"state_change_synchrony.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(x,[s[c]["message_ablation_performance_drop"]["median"] for c in CONDITIONS],marker="o"); plt.xticks(x,labels); plt.ylabel("validation drop with messages disabled"); plt.tight_layout(); plt.savefig(out/"message_dependence_vs_control_scope.png"); plt.close()


def main() -> int:
    raw = json.loads((HERE / "raw" / "runs.json").read_text(encoding="utf-8"))
    rows = raw["rows"]
    expected = len(CONDITIONS) * len(FAMILIES) * CONFIG["training"]["replicates"]
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    baseline = json.loads((HERE / "diagnostics" / "baseline_compatibility.json").read_text(encoding="utf-8"))
    if baseline.get("compatible") is not True:
        raise SystemExit("CONTROL_SCOPE_BASELINE_DRIFT")

    summaries = {condition: summarize(rows, condition) for condition in CONDITIONS}
    diag, preferred = diagnosis(summaries)
    adequacy = preferred is not None
    resource = {
        "model_fits": len(rows), "optimizer_steps": sum(r["resources"]["optimizer_steps"] for r in rows),
        "processed_examples": sum(r["resources"]["examples_processed"] for r in rows),
        "environment_interactions": sum(r["resources"]["environment_steps"] for r in rows),
        "forward_calls": sum(r["resources"]["forward_calls"] for r in rows),
        "cpu_seconds": float(sum(r["resources"]["cpu_seconds"] for r in rows)),
        "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in rows)),
        "gpu_seconds": 0.0, "unique_seed_defined_episodes": 3200,
    }
    result = {
        "version": "V837v", "parent": "V837u", "question": CONFIG["question"], "single_change": CONFIG["single_change"],
        "controller_information_scope": "source_cell_local_only", "gate_pooling": False, "baseline_compatibility": baseline,
        "conditions": summaries, "control_domain_curve": {c: summaries[c]["families_passing"] for c in CONDITIONS},
        "diagnosis": diag, "best_passing_condition": preferred, "representation_adequacy_pass": bool(adequacy),
        "sample_efficiency_retest_allowed": bool(adequacy), "v837w_allowed": not adequacy,
        "structural_search_allowed": False, "primitive_mining_allowed": False, "fresh_audit_consumed": False,
        "primitives_promoted": 0, "large_persistent_storage_tested": False, "v838_started": False,
        "resource_accounting": resource,
    }
    write_json(HERE / "results.json", result)
    write_json(HERE / "diagnostics" / "control_scope_summary.json", summaries)
    compute = {c: {
        "families_passing": summaries[c]["families_passing"], "mean_validation": float(np.mean([summaries[c]["family_results"][f]["validation"]["median"] for f in FAMILIES])),
        "nominal_params": summaries[c]["nominal_parameters"], "active_params": summaries[c]["active_parameters"],
        "base_macs": 160, "active_controller_macs": summaries[c]["controller_macs_per_timestep"], "total_recurrent_controller_macs": summaries[c]["total_macs_per_timestep"],
        "cpu_seconds": summaries[c]["resource_accounting"]["cpu_seconds"], "wall_seconds": summaries[c]["resource_accounting"]["wall_seconds_sum_workers"], "gpu_seconds": 0.0,
        "processed_examples": summaries[c]["resource_accounting"]["processed_examples"], "unique_seed_defined_episodes": 3200,
        "capability_per_active_mac": summaries[c]["capability_per_active_mac"], "capability_per_active_parameter": summaries[c]["capability_per_active_parameter"],
    } for c in CONDITIONS}
    write_json(HERE / "diagnostics" / "compute_efficiency.json", compute)
    decision = {
        "v837v_complete": True, "baseline_reproduced": True,
        "families_passing": {c: summaries[c]["families_passing"] for c in CONDITIONS},
        "best_passing_condition": preferred, "diagnosis": diag, "representation_adequacy_pass": bool(adequacy),
        "v837w_allowed": not adequacy, "sample_efficiency_retest_allowed": bool(adequacy),
        "structural_search_allowed": False, "primitive_mining_allowed": False, "fresh_audit_consumed": False,
        "primitives_promoted": 0, "v838_started": False,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision)
    _plots(summaries)
    doc = HERE / ("PASS.md" if adequacy else "FAILURE.md")
    doc.write_text(
        f"# V837v {diag}\n\nRepresentation adequacy: {'PASS' if adequacy else 'FAIL'}.\n\nControl-domain curve: `{result['control_domain_curve']}`.\n\n"
        + ("This is a fixed-topology representation-adequacy result only; it is not primitive-invention evidence.\n" if adequacy else "V837w is authorized to localize the successful T2 controller information source.\n"),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
