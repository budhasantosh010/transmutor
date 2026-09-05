from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [t.name for t in all_tasks()]
CONDITIONS = CONFIG["conditions"]


def _rows() -> list[dict]:
    rows = []
    for p in (HERE / "raw" / "anchor_runs.json", HERE / "raw" / "scalarized_runs.json"):
        rows.extend(json.loads(p.read_text(encoding="utf-8"))["rows"])
    return rows


def _csummary(rows: list[dict], condition: str) -> dict:
    selected = [r for r in rows if r["condition"] == condition]
    per_family = {}
    count = 0
    for family in FAMILIES:
        fr = [r for r in selected if r["family"] == family]
        dev = np.asarray([r["development_success"] for r in fr], dtype=float)
        val = np.asarray([r["validation_success"] for r in fr], dtype=float)
        passed = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        count += int(passed)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "aggregate_capacity_pass": bool(passed),
        }
    gate_diag = {}
    for pathway in ("update", "reset"):
        vals = [r["gate_diagnostics"][pathway] for r in selected]
        gate_diag[pathway] = {
            key: continuous_summary(np.asarray([v[key] for v in vals], dtype=float))
            for key in (
                "mean", "median", "p10", "p90", "temporal_variance", "interdimension_variance",
                "underlying_vector_interdimension_variance", "near_zero_fraction", "near_one_fraction", "entropy",
                "underlying_vector_effective_rank",
            )
        }
    flatten = {}
    for key in ("update", "reset", "both"):
        vals = [r["gate_diagnostics"]["counterfactual_flattening"].get(key) for r in selected]
        vals = [v for v in vals if v is not None]
        if vals:
            flatten[key] = {
                "flattened_validation": continuous_summary(np.asarray([v["flattened_validation"] for v in vals], dtype=float)),
                "flattening_delta": continuous_summary(np.asarray([v["flattening_delta"] for v in vals], dtype=float)),
                "per_family_delta_median": {
                    family: float(np.median([
                        r["gate_diagnostics"]["counterfactual_flattening"][key]["flattening_delta"]
                        for r in selected if r["family"] == family and key in r["gate_diagnostics"]["counterfactual_flattening"]
                    ])) for family in FAMILIES
                },
            }
    first = selected[0]
    return {
        "update_mode": first["update_mode"],
        "reset_mode": first["reset_mode"],
        "nominal_parameter_count": first["nominal_parameter_count"],
        "active_parameter_count": first["active_parameter_count"],
        "families_passing": count,
        "family_results": per_family,
        "gate_diagnostics": gate_diag,
        "counterfactual_flattening": flatten,
        "resource_accounting": {
            "model_fits": len(selected),
            "optimizer_steps": sum(r["resources"]["optimizer_steps"] for r in selected),
            "examples_processed": sum(r["resources"]["examples_processed"] for r in selected),
            "environment_interactions": sum(r["resources"]["environment_steps"] for r in selected),
            "forward_calls": sum(r["resources"]["forward_calls"] for r in selected),
            "cpu_seconds": float(sum(r["resources"]["cpu_seconds"] for r in selected)),
            "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in selected)),
            "gpu_seconds": 0.0,
        },
    }


def _decision(s: dict[str, dict]) -> tuple[str, str | None]:
    anchors = all(s[c]["families_passing"] >= 4 for c in CONFIG["positive_control_conditions"])
    if not anchors:
        return "REFERENCE_BASELINE_DRIFT", None
    if s["T2_scalarized_update_no_reset"]["families_passing"] >= 4:
        return "DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED", "DYNAMIC_SCALAR_CARRY"
    if s["T4_no_update_scalarized_reset"]["families_passing"] >= 4:
        return "DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED", "POST_TRANSFORM_SCALAR_MODULATION"
    if s["T5_dual_scalarized"]["families_passing"] >= 4:
        return "MULTIPLE_DYNAMIC_PATHWAYS_SUFFICIENT", "DUAL_SCALAR_DYNAMIC_PATHWAYS"
    return "DYNAMIC_VECTOR_GRANULARITY_REQUIRED", "DYNAMIC_VECTOR_STATE_MODULATION"


def _plots(s: dict[str, dict]) -> None:
    plotdir = HERE / "plots"
    plotdir.mkdir(exist_ok=True)
    x = np.arange(len(CONDITIONS))
    counts = [s[c]["families_passing"] for c in CONDITIONS]
    plt.figure(figsize=(9, 4)); plt.bar(x, counts); plt.xticks(x, CONDITIONS, rotation=35, ha="right"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(plotdir / "families_passing_by_dynamic_granularity.png"); plt.close()
    for filename, key in [
        ("gate_temporal_variance.png", "temporal_variance"),
        ("gate_interdimensional_variance.png", "underlying_vector_interdimension_variance"),
    ]:
        plt.figure(figsize=(9,4))
        for pathway in ("update","reset"):
            vals=[s[c]["gate_diagnostics"][pathway][key]["median"] for c in CONDITIONS]
            plt.plot(x,vals,marker="o",label=pathway)
        plt.xticks(x,CONDITIONS,rotation=35,ha="right"); plt.legend(); plt.tight_layout(); plt.savefig(plotdir/filename); plt.close()
    plt.figure(figsize=(9,4))
    for family in FAMILIES:
        plt.plot(x,[s[c]["family_results"][family]["validation"]["median"] for c in CONDITIONS],marker="o",label=family)
    plt.xticks(x,CONDITIONS,rotation=35,ha="right"); plt.ylabel("validation median"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plotdir/"family_scores_vector_vs_scalarized.png"); plt.close()
    pairs=[("T1_vector_update_no_reset","T2_scalarized_update_no_reset","update_scalarization_effect.png"),("T3_no_update_vector_reset","T4_no_update_scalarized_reset","reset_scalarization_effect.png"),("T0_full_vector_gru","T5_dual_scalarized","dual_scalarized_effect.png")]
    for a,b,name in pairs:
        vals_a=[s[a]["family_results"][f]["validation"]["median"] for f in FAMILIES]; vals_b=[s[b]["family_results"][f]["validation"]["median"] for f in FAMILIES]
        z=np.arange(len(FAMILIES)); w=.38; plt.figure(figsize=(8,4)); plt.bar(z-w/2,vals_a,w,label=a); plt.bar(z+w/2,vals_b,w,label=b); plt.xticks(z,FAMILIES,rotation=25,ha="right"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plotdir/name); plt.close()
    deltas=[]; labels=[]
    for c in CONDITIONS:
        for k,v in s[c]["counterfactual_flattening"].items():
            labels.append(c+":"+k); deltas.append(v["flattening_delta"]["median"])
    plt.figure(figsize=(9,4)); plt.bar(np.arange(len(deltas)),deltas); plt.xticks(np.arange(len(deltas)),labels,rotation=35,ha="right"); plt.ylabel("original - flattened"); plt.tight_layout(); plt.savefig(plotdir/"counterfactual_flattening_effect.png"); plt.close()
    plt.figure(figsize=(8,4)); plt.plot([0,1],[s["T1_vector_update_no_reset"]["families_passing"],s["T2_scalarized_update_no_reset"]["families_passing"]],marker="o"); plt.xticks([0,1],["vector update","scalarized update"]); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(plotdir/"update_scalarization_effect.png"); plt.close()


def main() -> int:
    rows = _rows()
    expected = 6 * 5 * CONFIG["training"]["replicates"]
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} V837t rows, found {len(rows)}")
    summaries = {c: _csummary(rows,c) for c in CONDITIONS}
    diagnosis, mode = _decision(summaries)
    positive = all(summaries[c]["families_passing"] >= 4 for c in CONFIG["positive_control_conditions"])
    resource = {
        "model_fits": len(rows),
        "optimizer_steps": sum(r["resources"]["optimizer_steps"] for r in rows),
        "processed_examples": sum(r["resources"]["examples_processed"] for r in rows),
        "environment_interactions": sum(r["resources"]["environment_steps"] for r in rows),
        "forward_calls": sum(r["resources"]["forward_calls"] for r in rows),
        "cpu_seconds": float(sum(r["resources"]["cpu_seconds"] for r in rows)),
        "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in rows)),
        "gpu_seconds": 0.0,
        "unique_seed_defined_episodes": CONFIG["unique_seed_defined_episodes"],
    }
    result = {
        "version":"V837t", "parent":"V837s", "question":CONFIG["question"], "single_change":CONFIG["single_change"],
        "data_regime":"4x_unique", "unique_seed_defined_episodes":3200, "conditions":summaries,
        "gate_granularity_diagnostics":{c:summaries[c]["gate_diagnostics"] for c in CONDITIONS},
        "counterfactual_flattening":{c:summaries[c]["counterfactual_flattening"] for c in CONDITIONS},
        "positive_controls_pass":positive, "diagnosis":diagnosis, "authorized_v837u_mode":mode,
        "neutral_followup_allowed": bool(positive and mode is not None), "representation_adequacy":"REFERENCE_ONLY",
        "resource_accounting":resource, "fresh_audit_consumed":False, "primitive_mining_allowed":False,
        "structural_search_allowed":False, "sample_efficiency_retest_allowed":False,
        "large_persistent_storage_tested":False, "primitives_promoted":0, "v838_started":False,
    }
    write_json(HERE/"results.json",result)
    write_json(HERE/"diagnostics"/"condition_summaries.json",summaries)
    decision={
        "v837t_complete":True, "positive_controls_pass":positive,
        "t0_full_vector":summaries["T0_full_vector_gru"]["families_passing"],
        "t1_vector_update_no_reset":summaries["T1_vector_update_no_reset"]["families_passing"],
        "t2_scalarized_update_no_reset":summaries["T2_scalarized_update_no_reset"]["families_passing"],
        "t3_no_update_vector_reset":summaries["T3_no_update_vector_reset"]["families_passing"],
        "t4_no_update_scalarized_reset":summaries["T4_no_update_scalarized_reset"]["families_passing"],
        "t5_dual_scalarized":summaries["T5_dual_scalarized"]["families_passing"],
        "diagnosis":diagnosis, "neutral_followup_allowed":bool(positive and mode), "authorized_v837u_mode":mode,
        "fresh_audit_consumed":False, "structural_search_allowed":False, "primitive_mining_allowed":False, "v838_started":False,
    }
    write_json(HERE/"diagnostics"/"decision_state.json",decision)
    _plots(summaries)
    (HERE/"PASS.md").write_text(f"# V837t diagnostic complete\n\nDiagnosis: `{diagnosis}`\n\nAuthorized V837u mode: `{mode}`\n",encoding="utf-8")
    print(json.dumps(decision,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
