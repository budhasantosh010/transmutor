from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import candidate_order, family_gate
from .chart_discovery import run_discovery_reader_ladder
from .commutativity import evaluate_dynamics, save_commutativity
from .data_roles import seeds
from .failure_ledger import add, make_entry
from .phase_atlas import closest_global_reader_candidate, fit_phase_atlas_organism, save_phase_atlas
from .projected_causal_spaces import space_map
from .quotient_eval import evaluate_quotient, save_quotient
from .setpoint_eval import evaluate_geometry, save_setpoint_diagnostics
from .source_folds import freeze_source_folds
from .tangent_program import evaluate_tangent_ladder, save_tangent_rows
from .utils import HERE, read_json, sha256_json, write_json


def _cid(row: dict) -> tuple[int, str, bool, str]:
    return (int(row["k"]), str(row["chart_family"]), bool(row.get("phase_atlas", False)), str(row.get("writer_family", "AUTO")))


def _reader_payload() -> dict:
    p = HERE / "diagnostics/chart_conditioning.json"
    return read_json(p) if p.is_file() else run_discovery_reader_ladder()


def _reader_rows_by_candidate(payload: dict) -> dict:
    out=defaultdict(list)
    for r in payload["rows"]:
        out[(r["family"], int(r["k"]), r["chart_family"])].append(r)
    return out


def _summary_map(payload: dict) -> dict:
    out={}
    for family, rows in payload["family_candidate_summaries"].items():
        for r in rows:
            out[(family,int(r["k"]),r["chart_family"])]=r
    return out


def _log_failure(stage:str, family:str, candidate:dict, gate:dict, *, organism:str|None=None, engine:str|None=None, failure_code:str="CANDIDATE_GATE_FAIL", metrics:dict|None=None, next_experiment:str="Continue the frozen V837ap complexity ladder; do not repeat unchanged.") -> None:
    key=sha256_json({"stage":stage,"family":family,"candidate":candidate,"organism":organism,"failure_code":failure_code})[:14]
    add(make_entry(
        failure_id=f"V837ap-{stage}-{key}", stage=stage, branch=candidate.get("branch") or "UNSPECIFIED", family=family,
        organism=organism, phase=None, carrier_dimension=candidate.get("k"),
        chart_family=candidate.get("chart_family"), chart_degree_rank=candidate.get("degree"),
        writer_family=candidate.get("writer_family"), fit_partition="AP_CHART_FIT/AP_WRITER_FIT",
        selection_partition="AP_CHART_SELECT/AP_WRITER_SELECT", parameter_count=candidate.get("parameter_count",0),
        stored_bytes=candidate.get("stored_bytes",0), mac_estimate=candidate.get("mac_estimate"),
        metrics=metrics or gate, matched_control_metrics=(metrics or {}).get("controls") if isinstance(metrics,dict) else None,
        ood_metrics={"median_ood_ratio":(metrics or {}).get("median_ood_ratio")} if isinstance(metrics,dict) else None,
        acceptance_gate="V837ap frozen reader/SET/control/quotient/dynamics gates", failed_conditions=[failure_code],
        distance_from_threshold="see gate/metrics", scientific_interpretation="This frozen geometry did not establish a compact globally closed causal state under the tested gate.",
        confounds_ruled_out=["source retraining","fresh-audit leakage","cross-organism state alignment","adaptive hyperparameter search"],
        confounds_remaining=["higher/non-tested geometry","program-level causal operators"], result_status="DEFINITIVE_WITHIN_FROZEN_SCOPE",
        next_justified_experiment=next_experiment,
        reproduction_command="python scripts/reproduce_v837_recovery.py --variant v837ap --stage setpoints --execute",
        artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/discovery_family_geometry_winners.json"],
        artifact_hashes={"record":sha256_json(metrics or gate)}, failure_type="SCIENTIFIC_FAILURE",
    ))


def _q_for(spaces:dict, family:str, oid:str, k:int)->np.ndarray:
    return np.asarray(spaces[(family,oid,int(k))]["q"],dtype=np.float64)


def _full_eval_candidate(family:str,candidate:dict,reader_rows:list[dict],spaces:dict,setpoint_rows:list,quotient_rows:list,dynamics_rows:list)->dict:
    evaluated=[]
    for rr in reader_rows:
        oid=rr["organism_id"]; q=_q_for(spaces,family,oid,candidate["k"])
        if not rr.get("reader_pass"):
            evaluated.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":False,"setpoint_pass":False,"quotient_pass":False,"dynamics_pass":False,"full_pass":False})
            continue
        sp=evaluate_geometry(rr,q,eval_partition="AP_WRITER_SELECT",random_controls_count=32); setpoint_rows.append(sp)
        if not sp.get("pass"):
            evaluated.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":False,"quotient_pass":False,"dynamics_pass":False,"full_pass":False})
            _log_failure("AP7_SET",family,candidate,{},organism=oid,engine=rr["engine"],failure_code=sp.get("failure_code","ABSOLUTE_SETPOINT_FAIL"),metrics=sp.get("metrics",{}))
            continue
        qr=evaluate_quotient(rr,q,seeds("AP_QUOTIENT"),"AP_QUOTIENT"); quotient_rows.append(qr)
        if not qr.get("pass"):
            evaluated.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":True,"quotient_pass":False,"dynamics_pass":False,"full_pass":False})
            _log_failure("AP8_QUOTIENT",family,candidate,{},organism=oid,engine=rr["engine"],failure_code=qr.get("failure_code","QUOTIENT_RESIDUAL_SENSITIVITY"),metrics=qr.get("metrics",{}))
            continue
        dr=evaluate_dynamics(rr,q,seeds("AP_DYNAMICS"),"AP_DYNAMICS"); dynamics_rows.append(dr)
        full=bool(dr.get("pass")); evaluated.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":True,"quotient_pass":True,"dynamics_pass":full,"full_pass":full})
        if not full:_log_failure("AP9_DYNAMICS",family,candidate,{},organism=oid,engine=rr["engine"],failure_code=dr.get("failure_code","CANONICAL_DYNAMICS_ROLLOUT_FAIL"),metrics=dr.get("interventional",{}).get("metrics",{}))
    sg=family_gate(evaluated,"setpoint_pass"); fg=family_gate(evaluated,"full_pass")
    return {**candidate,"setpoint_family_gate":sg,"full_family_gate":fg,"organism_results":evaluated,"family_pass":bool(fg["pass"])}


def _evaluate_tangent_family(family:str,base_rows:list[dict],spaces:dict,setpoint_rows:list,quotient_rows:list,dynamics_rows:list,tangent_rows:list)->list[dict]:
    per_kind=defaultdict(list)
    geometry_by_key={}
    for rr in base_rows:
        if not rr.get("reader_pass"):continue
        q=_q_for(spaces,family,rr["organism_id"],1)
        rows=evaluate_tangent_ladder(rr,q)
        for item in rows:
            tangent_rows.append(item);kind=item["geometry"]["writer_family"];per_kind[kind].append(item);geometry_by_key[(kind,rr["organism_id"])]=item["geometry"]
    summaries=[]
    for kind in ("TANGENT_CONSTANT","TANGENT_AFFINE_STATE_FIELD","TANGENT_QUADRATIC_STATE_FIELD"):
        evals=[]
        by_oid={x["geometry"]["organism_id"]:x for x in per_kind.get(kind,[])}
        for rr in base_rows:
            oid=rr["organism_id"]
            if oid not in by_oid:
                evals.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":bool(rr.get("reader_pass")),"setpoint_pass":False,"quotient_pass":False,"dynamics_pass":False,"full_pass":False});continue
            item=by_oid[oid];sp=item["setpoint"];setpoint_rows.append(sp)
            if not sp.get("pass"):
                evals.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":False,"quotient_pass":False,"dynamics_pass":False,"full_pass":False});continue
            geom=item["geometry"];q=_q_for(spaces,family,oid,1);qr=evaluate_quotient(geom,q,seeds("AP_QUOTIENT"),"AP_QUOTIENT");quotient_rows.append(qr)
            if not qr.get("pass"):
                evals.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":True,"quotient_pass":False,"dynamics_pass":False,"full_pass":False});continue
            dr=evaluate_dynamics(geom,q,seeds("AP_DYNAMICS"),"AP_DYNAMICS");dynamics_rows.append(dr);full=bool(dr.get("pass"));evals.append({"organism_id":oid,"engine":rr["engine"],"reader_pass":True,"setpoint_pass":True,"quotient_pass":True,"dynamics_pass":full,"full_pass":full})
        fg=family_gate(evals,"full_pass");summaries.append({"family":family,"branch":"AP_C_STATE_DEPENDENT_TANGENT","k":1,"chart_family":base_rows[0]["chart_family"] if base_rows else None,"writer_family":kind,"phase_atlas":False,"organism_results":evals,"full_family_gate":fg,"family_pass":bool(fg["pass"])})
    return summaries


def run_discovery_program()->dict:
    reader=_reader_payload();rows_by=_reader_rows_by_candidate(reader);summary_map=_summary_map(reader);spaces=space_map();folds=freeze_source_folds()
    setpoint_rows=[];quotient_rows=[];dynamics_rows=[];tangent_rows=[];atlas_rows=[];atlas_selections={};family_records={};winners={}
    for family in POWERED_FAMILIES:
        records=[];any_global_full_family_pass=False;k1_reader_candidates=[]
        # Frozen hierarchy: exhaust K1 chart families first. AP-C tangent must be
        # adjudicated before any K2/K4/K8 projected candidate is allowed.
        for cand in [c for c in candidate_order(family) if int(c["k"]) == 1]:
            key=(family,int(cand["k"]),cand["chart_family"]);summary=summary_map[key];rrows=rows_by[key]
            rec={**cand,"reader_family_gate":summary["reader_family_gate"],"reader_family_pass":bool(summary["reader_family_pass"])}
            if not rec["reader_family_pass"]:
                rec["setpoint_family_gate"]={"pass":False,"reason":"reader family gate failed"};rec["full_family_gate"]={"pass":False,"reason":"reader family gate failed"};rec["family_pass"]=False;records.append(rec);_log_failure("AP3_AP4_READER",family,cand,summary["reader_family_gate"],failure_code="READER_FAMILY_GATE_FAIL",metrics=summary);continue
            full=_full_eval_candidate(family,cand,rrows,spaces,setpoint_rows,quotient_rows,dynamics_rows);records.append({**rec,**full});any_global_full_family_pass=bool(any_global_full_family_pass or full["family_pass"])
            if full["family_pass"]:
                break
            if int(cand["k"])==1 and cand["chart_family"]!="AFFINE":k1_reader_candidates.append((cand,rrows,full))
        # AP-C: only if at least one nonlinear K1 reader-family passes but its standard SET family gate fails.
        tangent_summaries=[]
        if not any_global_full_family_pass:
            for cand,rrows,full in k1_reader_candidates:
                if not full["setpoint_family_gate"].get("pass"):
                    tangent_summaries=_evaluate_tangent_family(family,rrows,spaces,setpoint_rows,quotient_rows,dynamics_rows,tangent_rows);records.extend(tangent_summaries)
                    if any(r.get("family_pass") for r in tangent_summaries):
                        any_global_full_family_pass=True
                    break
        # AP-B projected ladder follows AP-C in the frozen complexity order.
        if not any_global_full_family_pass:
            for cand in [c for c in candidate_order(family) if int(c["k"]) > 1]:
                key=(family,int(cand["k"]),cand["chart_family"]);summary=summary_map[key];rrows=rows_by[key]
                rec={**cand,"reader_family_gate":summary["reader_family_gate"],"reader_family_pass":bool(summary["reader_family_pass"])}
                if not rec["reader_family_pass"]:
                    rec["setpoint_family_gate"]={"pass":False,"reason":"reader family gate failed"};rec["full_family_gate"]={"pass":False,"reason":"reader family gate failed"};rec["family_pass"]=False;records.append(rec);_log_failure("AP4_PROJECTED_READER",family,cand,summary["reader_family_gate"],failure_code="READER_FAMILY_GATE_FAIL",metrics=summary);continue
                full=_full_eval_candidate(family,cand,rrows,spaces,setpoint_rows,quotient_rows,dynamics_rows);records.append({**rec,**full})
                if full["family_pass"]:
                    any_global_full_family_pass=True
                    break
        # AP-D: only if no global candidate survives the complete discovery family gate.
        if not any_global_full_family_pass:
            chosen=closest_global_reader_candidate(family,reader["family_candidate_summaries"][family]);atlas_selections[family]={k:v for k,v in chosen.items() if k not in {"organism_results"}}
            arows=[]
            for oid in folds["families"][family]["discovery"]:
                q=_q_for(spaces,family,oid,int(chosen["k"]));engine=spaces[(family,oid,int(chosen["k"]))]["engine"];geom=fit_phase_atlas_organism(oid,family,engine,chosen,q);atlas_rows.append(geom);ar={"geometry":geom}
                if geom.get("reader_pass"):
                    sp=evaluate_geometry(geom,q,"AP_WRITER_SELECT",32);setpoint_rows.append(sp);ar["setpoint"]=sp
                    if sp.get("pass"):
                        qr=evaluate_quotient(geom,q,seeds("AP_QUOTIENT"),"AP_QUOTIENT");quotient_rows.append(qr);ar["quotient"]=qr
                        if qr.get("pass"):
                            dr=evaluate_dynamics(geom,q,seeds("AP_DYNAMICS"),"AP_DYNAMICS");dynamics_rows.append(dr);ar["dynamics"]=dr
                ar["full_pass"]=bool(geom.get("reader_pass") and ar.get("setpoint",{}).get("pass") and ar.get("quotient",{}).get("pass") and ar.get("dynamics",{}).get("pass"));arows.append(ar)
            gate=family_gate([{"engine":x["geometry"]["engine"],"full_pass":x["full_pass"]} for x in arows],"full_pass");atlas_record={"family":family,"branch":"AP_D_PHASE_ATLAS","k":int(chosen["k"]),"chart_family":chosen["chart_family"],"writer_family":chosen.get("writer_family","AUTO"),"phase_atlas":True,"selection_rule":"closest global reader candidate using AP_CHART_SELECT only","organism_results":arows,"full_family_gate":gate,"family_pass":bool(gate["pass"])};records.append(atlas_record)
        winner=next((r for r in records if r.get("family_pass")),None);winners[family]=None if winner is None else {k:v for k,v in winner.items() if k not in {"organism_results"}};family_records[family]=records
        print(f"V837ap discovery family {family} winner={None if winners[family] is None else (winners[family]['k'],winners[family]['chart_family'],winners[family].get('writer_family'),winners[family].get('phase_atlas'))}",flush=True)
    save_setpoint_diagnostics(setpoint_rows);save_quotient(quotient_rows);save_commutativity(dynamics_rows);save_tangent_rows(tangent_rows);save_phase_atlas(atlas_rows,atlas_selections)
    payload={"version":"V837ap","stage":"AP3_AP10_DISCOVERY","complexity_rule":"first passing candidate in frozen hierarchy; no score maximization","family_records":family_records,"family_winners":winners,"winning_families":sum(v is not None for v in winners.values()),"heldout_opened":False}
    write_json(HERE/"raw/discovery_family_geometry_winners.json",payload);return payload


if __name__=="__main__":print(json.dumps(run_discovery_program(),indent=2,default=str))
