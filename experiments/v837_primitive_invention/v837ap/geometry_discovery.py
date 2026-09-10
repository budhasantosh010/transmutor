from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import candidate_order, family_gate
from .failure_ledger import add, make_entry
from .phase_atlas import closest_global_reader_candidate, fit_phase_atlas_organism, save_phase_atlas
from .projected_causal_spaces import space_map
from .setpoint_eval import evaluate_geometry, save_setpoint_diagnostics
from .source_folds import freeze_source_folds
from .tangent_field import TANGENT_FAMILIES
from .tangent_program import tangent_geometry
from .utils import HERE, read_json, sha256_json, write_json


def _key(row: dict) -> tuple:
    return (row.get("family"), int(row.get("k", -1)), row.get("chart_family"), bool(row.get("phase_atlas", False)))


def _distance(metrics: dict, binary: bool) -> dict:
    if not metrics:
        return {"missing_metrics": True}
    if binary:
        return {
            "nrmse_excess": max(0.0, float(metrics.get("normalized_rmse", 99.0)) - 0.15),
            "accuracy_shortfall": max(0.0, 0.95 - float(metrics.get("classification_accuracy", 0.0))),
            "balanced_accuracy_shortfall": max(0.0, 0.95 - float(metrics.get("balanced_accuracy", 0.0))),
        }
    return {
        "nrmse_excess": max(0.0, float(metrics.get("normalized_rmse", 99.0)) - 0.10),
        "pearson_shortfall": max(0.0, 0.95 - abs(float(metrics.get("pearson", 0.0)))),
        "spearman_shortfall": max(0.0, 0.95 - float(metrics.get("spearman", 0.0))),
    }


def _log_reader_failure(row: dict) -> None:
    if row.get("reader_pass"):
        return
    fid = "V837ap-AP3-READER-" + sha256_json({k: row.get(k) for k in ("family", "organism_id", "k", "chart_family")})[:16]
    add(make_entry(
        failure_id=fid, stage="AP3_AP4_READER", branch=row.get("branch", "GLOBAL_CHART"), family=row.get("family"),
        organism=row.get("organism_id"), phase=None, carrier_dimension=row.get("k"), chart_family=row.get("chart_family"),
        chart_degree_rank=row.get("chart", {}).get("degree"), writer_family=row.get("writer_family"),
        fit_partition="AP_CHART_FIT", selection_partition="AP_CHART_SELECT", parameter_count=row.get("parameter_count"),
        stored_bytes=row.get("stored_bytes"), mac_estimate=None, metrics=row.get("reader_metrics", {"error": row.get("error")}),
        acceptance_gate={"continuous": {"nrmse": 0.10, "pearson": 0.95, "spearman": 0.95}, "binary": {"nrmse": 0.15, "accuracy": 0.95, "balanced_accuracy": 0.95}, "coverage": 0.95},
        failed_conditions=[row.get("failure_code", "READER_GATE_FAIL")],
        distance_from_threshold=_distance(row.get("reader_metrics", {}), row.get("family") in {"conditional_routing", "delayed_recall"}),
        scientific_interpretation="This frozen compact chart did not establish the required semantic reader on AP_CHART_SELECT.",
        confounds_ruled_out=["heldout organism leakage", "source retraining", "cross-organism state alignment"],
        confounds_remaining=["more complex predeclared geometry", "phase-conditional chart"],
        next_justified_experiment="Continue only in the frozen V837ap complexity order.",
        reproduction_command="python scripts/reproduce_v837_recovery.py --variant v837ap --stage k1-charts --execute" if int(row.get("k", 0)) == 1 else "python scripts/reproduce_v837_recovery.py --variant v837ap --stage projected-charts --execute",
        artifact_paths=["experiments/v837_primitive_invention/v837ap/diagnostics/chart_conditioning.json"],
    ), append_central=False)


def _log_set_failure(reader_row: dict, sp: dict, branch: str) -> None:
    if sp.get("pass"):
        return
    fid = "V837ap-AP7-SET-" + sha256_json({"family": reader_row.get("family"), "organism": reader_row.get("organism_id"), "k": reader_row.get("k"), "chart": reader_row.get("chart_family"), "writer": sp.get("writer_family"), "atlas": sp.get("phase_atlas", False)})[:16]
    m = sp.get("metrics", {})
    add(make_entry(
        failure_id=fid, stage="AP7_SETPOINT", branch=branch, family=reader_row.get("family"), organism=reader_row.get("organism_id"), phase=None,
        carrier_dimension=reader_row.get("k"), chart_family=reader_row.get("chart_family"), chart_degree_rank=reader_row.get("chart", {}).get("degree"),
        writer_family=sp.get("writer_family", reader_row.get("writer_family")), fit_partition="AP_WRITER_FIT", selection_partition="AP_WRITER_SELECT",
        parameter_count=reader_row.get("parameter_count"), stored_bytes=reader_row.get("stored_bytes"), mac_estimate=None, metrics=m,
        matched_control_metrics={k: v for k, v in m.items() if "random" in k or "shuffled" in k or "permutation" in k},
        ood_metrics={"median_ood_ratio": m.get("median_ood_ratio")}, acceptance_gate={"median_recovery": 0.70, "direction": 0.80, "task_success": 0.75, "trajectory_nrmse": 0.15, "ood": 2.0, "control_margin": 0.20, "p": 0.01},
        failed_conditions=[sp.get("failure_code", "ABSOLUTE_SETPOINT_FAIL")],
        distance_from_threshold={"recovery_shortfall": max(0.0, .70-float(m.get("median_counterfactual_recovery", -1e9))) if m else None, "ood_excess": max(0.0, float(m.get("median_ood_ratio", 1e9))-2.0) if m else None},
        scientific_interpretation="Reader evidence did not upgrade to the frozen absolute semantic SET/control/OOD contract for this organism.",
        confounds_ruled_out=["decoder-only success", "unfitted random-axis control", "norm-only perturbation advantage"],
        confounds_remaining=["later predeclared geometry", "phase atlas"],
        next_justified_experiment="Continue only in the frozen V837ap complexity order.",
        reproduction_command="python scripts/reproduce_v837_recovery.py --variant v837ap --stage setpoints --execute",
        artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/setpoint_results.json"],
    ), append_central=False)


def _synthetic_set_fail(row: dict, reason: str = "READER_GATE_FAIL") -> dict:
    return {"organism_id": row["organism_id"], "family": row["family"], "engine": row.get("engine"), "k": row["k"], "chart_family": row["chart_family"], "writer_family": row.get("writer_family"), "phase_atlas": bool(row.get("phase_atlas", False)), "sample_count": 0, "pass": False, "failure_code": reason, "metrics": {}}


def _evaluate_candidate_rows(rows: list[dict], spaces: dict, branch: str) -> tuple[list[dict], dict]:
    out=[]
    for idx, row in enumerate(rows, 1):
        if not row.get("reader_pass"):
            sp=_synthetic_set_fail(row)
        else:
            q=np.asarray(spaces[(row["family"], row["organism_id"], int(row["k"]))]["q"], dtype=np.float64)
            sp=evaluate_geometry(row, q)
        out.append({"reader": row, "setpoint": sp, "pass": bool(row.get("reader_pass") and sp.get("pass")), "engine": row.get("engine"), "organism_id": row.get("organism_id")})
        _log_set_failure(row, sp, branch)
        print(f"V837ap SET {row['family']} k{row['k']} {row['chart_family']} {idx}/{len(rows)} pass={out[-1]['pass']}", flush=True)
    return out, family_gate(out, "pass")


def run_discovery_geometry_selection() -> dict:
    cond=read_json(HERE/"diagnostics/chart_conditioning.json")
    spaces=space_map(); folds=freeze_source_folds(); all_reader=cond["rows"]
    for row in all_reader:
        _log_reader_failure(row)
    by=defaultdict(list)
    for row in all_reader:
        by[_key(row)].append(row)
    setpoint_rows=[]; tangent_records=[]; atlas_rows=[]; atlas_selection={}; evaluated={}; winners={}

    for family in POWERED_FAMILIES:
        order=candidate_order(family); summaries=cond["family_candidate_summaries"][family]; summary_map={(int(s["k"]),s["chart_family"]):s for s in summaries}
        evaluated[family]=[]; winner=None; k1_set_fail_candidates=[]

        # Global K1 charts first. AFFINE remains an anchor but is allowed to prove itself if it unexpectedly satisfies the frozen gate.
        for cand in [c for c in order if int(c["k"])==1]:
            summary=summary_map[(1,cand["chart_family"])]
            evaluated[family].append({**cand,"reader_family_gate":summary["reader_family_gate"]})
            if not summary["reader_family_pass"]:
                continue
            rows=by[(family,1,cand["chart_family"],False)]
            combo,gate=_evaluate_candidate_rows(rows,spaces,"AP_A_K1")
            setpoint_rows.extend([x["setpoint"] for x in combo])
            evaluated[family][-1]["setpoint_family_gate"]=gate
            if gate["pass"]:
                winner={**cand,"family":family,"selection_stage":"AP7","family_gate":gate,"support_organisms":[x["organism_id"] for x in combo if x["pass"]],"support_engines":gate["passing_engines"]};break
            if cand["chart_family"]!="AFFINE":
                k1_set_fail_candidates.append((cand,rows))
        if winner is not None:
            winners[family]=winner;continue

        # AP-C tangent field: only after a K1 reader family passed but its direct setter failed.
        tangent_winner=None
        if k1_set_fail_candidates:
            base_cand, base_rows=k1_set_fail_candidates[0]
            for kind in TANGENT_FAMILIES:
                tcomb=[]
                for idx,row in enumerate(base_rows,1):
                    if not row.get("reader_pass"):
                        sp=_synthetic_set_fail(row,"READER_GATE_FAIL"); geom={**row,"writer_family":f"TANGENT_{kind}","valid":False}
                    else:
                        q=np.asarray(spaces[(family,row["organism_id"],1)]["q"],dtype=np.float64);geom=tangent_geometry(row,q,kind);sp=evaluate_geometry(geom,q) if geom.get("valid") else _synthetic_set_fail(geom,geom.get("tangent_field",{}).get("failure_code","TANGENT_FIELD_INVALID"))
                    rec={"geometry":geom,"setpoint":sp,"pass":bool(row.get("reader_pass") and sp.get("pass")),"engine":row.get("engine"),"organism_id":row.get("organism_id")};tcomb.append(rec);tangent_records.append(rec);setpoint_rows.append(sp);_log_set_failure(row,sp,"AP_C_STATE_DEPENDENT_TANGENT")
                    print(f"V837ap tangent {family} {kind} {idx}/{len(base_rows)} pass={rec['pass']}",flush=True)
                gate=family_gate(tcomb,"pass");evaluated[family].append({"branch":"AP_C_STATE_DEPENDENT_TANGENT","k":1,"chart_family":base_cand["chart_family"],"writer_family":f"TANGENT_{kind}","phase_atlas":False,"setpoint_family_gate":gate})
                if gate["pass"]:
                    tangent_winner={"branch":"AP_C_STATE_DEPENDENT_TANGENT","k":1,"chart_family":base_cand["chart_family"],"writer_family":f"TANGENT_{kind}","phase_atlas":False,"family":family,"selection_stage":"AP7","family_gate":gate,"support_organisms":[x["organism_id"] for x in tcomb if x["pass"]],"support_engines":gate["passing_engines"]};break
        if tangent_winner is not None:
            winners[family]=tangent_winner;continue

        # K2 -> K4 -> K8, exact within-k LINEAR -> QUADRATIC -> CUBIC.
        for cand in [c for c in order if int(c["k"])>1]:
            summary=summary_map[(int(cand["k"]),cand["chart_family"])]
            evaluated[family].append({**cand,"reader_family_gate":summary["reader_family_gate"]})
            if not summary["reader_family_pass"]:
                continue
            rows=by[(family,int(cand["k"]),cand["chart_family"],False)]
            combo,gate=_evaluate_candidate_rows(rows,spaces,"AP_B_PROJECTED")
            setpoint_rows.extend([x["setpoint"] for x in combo]);evaluated[family][-1]["setpoint_family_gate"]=gate
            if gate["pass"]:
                winner={**cand,"family":family,"selection_stage":"AP7","family_gate":gate,"support_organisms":[x["organism_id"] for x in combo if x["pass"]],"support_engines":gate["passing_engines"]};break
        if winner is not None:
            winners[family]=winner;continue

        # AP-D only when no global candidate survives the complete AP7 SET gate.
        chosen=closest_global_reader_candidate(family,summaries);atlas_selection[family]={k:chosen[k] for k in ("k","chart_family","writer_family") if k in chosen}
        rows=[]
        for idx,oid in enumerate(folds["families"][family]["discovery"],1):
            sr=spaces[(family,oid,int(chosen["k"]))];q=np.asarray(sr["q"],dtype=np.float64);ar=fit_phase_atlas_organism(oid,family,sr["engine"],chosen,q);rows.append(ar);atlas_rows.append(ar)
            print(f"V837ap atlas reader {family} k{chosen['k']} {chosen['chart_family']} {idx}/{len(folds['families'][family]['discovery'])} pass={ar.get('reader_pass',False)}",flush=True)
        rgate=family_gate(rows,"reader_pass")
        ae={"branch":"AP_D_PHASE_ATLAS","k":int(chosen["k"]),"chart_family":chosen["chart_family"],"writer_family":chosen.get("writer_family","GRADIENT_NEWTON"),"phase_atlas":True,"reader_family_gate":rgate}
        evaluated[family].append(ae)
        if rgate["pass"]:
            combo,gate=_evaluate_candidate_rows(rows,spaces,"AP_D_PHASE_ATLAS");setpoint_rows.extend([x["setpoint"] for x in combo]);ae["setpoint_family_gate"]=gate
            if gate["pass"]:
                winner={**ae,"family":family,"selection_stage":"AP7","family_gate":gate,"support_organisms":[x["organism_id"] for x in combo if x["pass"]],"support_engines":gate["passing_engines"]}
        winners[family]=winner

    save_setpoint_diagnostics(setpoint_rows)
    save_phase_atlas(atlas_rows,atlas_selection)
    write_json(HERE/"raw/tangent_field_fit.json",{"version":"V837ap","stage":"AP5_STATE_DEPENDENT_TANGENT","rows":tangent_records,"triggered_families":sorted({r["geometry"]["family"] for r in tangent_records})})
    write_json(HERE/"diagnostics/tangent_field.json",{"version":"V837ap","rows":tangent_records})
    payload={"version":"V837ap","stage":"AP10_DISCOVERY_SELECTION_INPUT","selection_rule":"first candidate in frozen complexity order passing AP7 discovery family gate; quotient/dynamics tested after selection with no geometry fallback","evaluated":evaluated,"family_winners":winners,"winner_count":sum(v is not None for v in winners.values()),"heldout_opened":False}
    write_json(HERE/"raw/discovery_family_geometry_winners.json",payload)
    return payload

if __name__=="__main__":
    print(json.dumps(run_discovery_geometry_selection(),indent=2,default=str))
