from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
FAMILIES = [t.name for t in all_tasks()]
Z0 = "Z0_historical_candidate_stage"
Z1 = "Z1_synchronous_candidate_stage"


def _rows() -> list[dict]:
    out = []
    for name in ("z0_runs.json", "z1_runs.json"):
        p = HERE / "raw" / name
        if p.exists(): out.extend(json.loads(p.read_text())["rows"])
    return out


def _summary(rows, condition):
    result = {"families_passing": 0, "family_results": {}, "mean_family_validation_median": 0.0}
    vals = []
    for family in FAMILIES:
        rr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        devs = [r["development_success"] for r in rr]; vs = [r["validation_success"] for r in rr]
        dm = float(np.median(devs)); vm = float(np.median(vs)); vals.append(vm)
        passed = bool(capacity_demonstrated(dm, vm)); result["families_passing"] += int(passed)
        result["family_results"][family] = {
            "development": {"median": dm, "values": devs},
            "validation": {"median": vm, "values": vs},
            "capacity_demonstrated": passed,
        }
    result["mean_family_validation_median"] = float(np.mean(vals))
    return result


def _aggregate_scalar(rows, condition, getter):
    values = [float(getter(r)) for r in rows if r["condition"] == condition]
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))} if values else {}


def _resources(rows):
    out = {"model_fits": len(rows), "optimizer_steps": 0, "processed_examples": 0, "unique_seed_defined_episodes": 3200, "environment_interactions": 0, "forward_calls": 0, "cpu_seconds": 0.0, "wall_seconds_sum_workers": 0.0, "gpu_seconds": 0.0}
    for r in rows:
        res = r["resources"]
        out["optimizer_steps"] += int(res.get("optimizer_steps", 0)); out["processed_examples"] += int(r["processed_examples"])
        out["environment_interactions"] += int(res.get("environment_steps", 0)); out["forward_calls"] += int(res.get("forward_calls", 0))
        out["cpu_seconds"] += float(res.get("cpu_seconds", 0.0)); out["wall_seconds_sum_workers"] += float(res.get("wall_seconds", 0.0)); out["gpu_seconds"] += float(r.get("gpu_seconds", 0.0))
    return out


def _plots(summaries, rows):
    plots = HERE / "plots"; plots.mkdir(exist_ok=True)
    labels = [Z0, Z1]; short = ["historical", "synchronous"]
    fig, ax = plt.subplots();
    for i, fam in enumerate(FAMILIES):
        ax.plot(short, [summaries[c]["family_results"][fam]["validation"]["median"] for c in labels], marker="o", label=fam)
    ax.set_ylim(0, 1.05); ax.set_ylabel("validation median"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "synchronous_vs_historical_family_scores.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(short, [summaries[c]["families_passing"] for c in labels]); ax.set_ylim(0,5); ax.set_ylabel("families passing"); fig.tight_layout(); fig.savefig(plots / "families_passing_candidate_stage.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(short, [np.mean([r["diagnostics"]["state_change_synchrony"]["mean_cell0_cross_cell_correlation"] for r in rows if r["condition"]==c]) for c in labels]); ax.set_ylabel("state-change synchrony"); fig.tight_layout(); fig.savefig(plots / "state_change_synchrony.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(short, [np.mean([r["diagnostics"]["message_dependency"]["success_drop"] for r in rows if r["condition"]==c]) for c in labels]); ax.set_ylabel("message ablation success drop"); fig.tight_layout(); fig.savefig(plots / "message_dependency_stage_comparison.png"); plt.close(fig)
    depth0 = rows[0]["diagnostics"]["candidate_effective_depth"]["per_cell"] if rows else [1]*10
    depth1 = [1]*10
    fig, ax = plt.subplots(); xs=np.arange(10); ax.plot(xs, depth0, marker="o", label="historical"); ax.plot(xs, depth1, marker="o", label="synchronous"); ax.set_xlabel("cell"); ax.set_ylabel("effective candidate depth"); ax.legend(); fig.tight_layout(); fig.savefig(plots / "candidate_effective_depth.png"); plt.close(fig)


def main() -> int:
    guard = json.loads((HERE / "diagnostics/parent_compatibility.json").read_text())
    if not guard.get("compatible"): raise SystemExit("Z0 parent baseline drift")
    rows = _rows()
    if len(rows) != 50: raise SystemExit(f"expected 50 V837z rows, found {len(rows)}")
    summaries = {c: _summary(rows, c) for c in (Z0, Z1)}
    z0, z1 = summaries[Z0], summaries[Z1]
    if z1["families_passing"] >= 4:
        diagnosis = "SYNCHRONOUS_CANDIDATE_STAGE_SUFFICIENT"; adequate = True
    elif z1["families_passing"] > z0["families_passing"] or z1["mean_family_validation_median"] > z0["mean_family_validation_median"] + 0.01:
        diagnosis = "SYNCHRONOUS_CANDIDATE_STAGE_PARTIAL_BENEFIT"; adequate = False
    elif z1["mean_family_validation_median"] < z0["mean_family_validation_median"] - 0.01:
        diagnosis = "HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL"; adequate = False
    else:
        diagnosis = "CANDIDATE_STAGE_SYNCHRONY_INSUFFICIENT"; adequate = False
    stage_depth = {
        Z0: rows[0]["diagnostics"]["candidate_effective_depth"],
        Z1: next(r for r in rows if r["condition"] == Z1)["diagnostics"]["candidate_effective_depth"],
    }
    state_sync = {c: _aggregate_scalar(rows, c, lambda r: r["diagnostics"]["state_change_synchrony"]["mean_cell0_cross_cell_correlation"]) for c in (Z0, Z1)}
    message = {c: _aggregate_scalar(rows, c, lambda r: r["diagnostics"]["message_dependency"]["success_drop"]) for c in (Z0, Z1)}
    resources = _resources(rows)
    compute_efficiency = {}
    for c in (Z0, Z1):
        rr = [r for r in rows if r["condition"] == c]
        compute_efficiency[c] = {
            "families_passing": summaries[c]["families_passing"], "mean_family_validation_median": summaries[c]["mean_family_validation_median"],
            "trainable_params": rr[0]["parameter_count"], "active_params": rr[0]["active_parameter_count"], "controller_params": rr[0]["controller_param_count"], "rank4_params": rr[0]["candidate_branch_param_count"],
            "recurrent_controller_macs": rr[0]["total_recurrent_controller_macs"], "processed_examples": sum(r["processed_examples"] for r in rr), "unique_episodes": 3200,
            "cpu_seconds": sum(float(r["resources"].get("cpu_seconds",0)) for r in rr), "wall_seconds_sum_workers": sum(float(r["resources"].get("wall_seconds",0)) for r in rr), "gpu_seconds": 0.0,
        }
    results = {
        "version":"V837z", "parent":"V837y", "selected_parent":CONFIG["selected_parent"], "single_change":"candidate stage synchronization only",
        "conditions": summaries, "parent_compatibility":guard, "candidate_stage_depth":stage_depth, "state_change_synchrony":state_sync, "message_diagnostics":message,
        "compute_efficiency":compute_efficiency, "resource_accounting":resources, "representation_adequacy_pass":adequate, "diagnosis":diagnosis,
        "sample_efficiency_retest_allowed":adequate, "structural_search_allowed":False, "primitive_mining_allowed":False, "fresh_audit_consumed":False, "primitives_promoted":0, "v838_started":False,
    }
    write_json(HERE / "results.json", results)
    write_json(HERE / "diagnostics/stage_depth.json", stage_depth); write_json(HERE / "diagnostics/state_change_synchrony.json", state_sync); write_json(HERE / "diagnostics/message_dependency.json", message); write_json(HERE / "diagnostics/compute_efficiency.json", compute_efficiency)
    decision = {"v837z_complete":True, "selected_parent":CONFIG["selected_parent"], "families_passing":{Z0:z0["families_passing"],Z1:z1["families_passing"]}, "diagnosis":diagnosis, "representation_adequacy_pass":adequate, "sample_efficiency_retest_allowed":adequate, "structural_search_allowed":False, "primitive_mining_allowed":False, "fresh_audit_consumed":False, "primitives_promoted":0, "v838_started":False}
    write_json(HERE / "diagnostics/decision_state.json", decision)
    write_json(HERE.parent / "v837z_resource_accounting.json", resources)
    if adequate:
        (HERE / "PASS.md").write_text(f"# V837z PASS\n\nDiagnosis: `{diagnosis}`. Representation adequacy restored; architecture modification stops and sample-efficiency characterization is authorized.\n")
    else:
        (HERE / "FAILURE.md").write_text(f"# V837z FAIL\n\nDiagnosis: `{diagnosis}`. Representation adequacy remains below 4/5; candidate-organization improvisation stops.\n")
    _plots(summaries, rows)
    print(json.dumps(decision, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
