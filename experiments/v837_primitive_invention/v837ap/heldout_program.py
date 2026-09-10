from __future__ import annotations

import json
import math
import time

import numpy as np

from .authorization import POWERED_FAMILIES
from .calibration_frontier import CALIBRATION_LADDER, heldout_family_gate, qualifier
from .commutativity import evaluate_dynamics
from .data_roles import seeds
from .failure_ledger import add, make_entry
from .freeze_geometry import freeze_family_geometries
from .geometry_eval import evaluate_reader_geometry
from .heldout_backend import compile_heldout_backend
from .quotient_eval import evaluate_quotient
from .setpoint_eval import evaluate_geometry
from .source_folds import freeze_source_folds
from .utils import HERE, sha256_json, write_json


def _log_budget_failure(family:str,oid:str,n:int,row:dict)->None:
    fid="V837ap-AP14-UNDERDETERMINED-"+sha256_json({"family":family,"oid":oid,"n":n})[:16]
    add(make_entry(failure_id=fid,stage="AP14_HELDOUT_COMPILATION",branch="HELDOUT_CALIBRATION",family=family,organism=oid,phase=None,carrier_dimension=None,chart_family=None,chart_degree_rank=None,writer_family=None,fit_partition="AP_CHART_FIT/AP_WRITER_FIT",selection_partition="REUSED_HISTORICAL_VALIDATION",parameter_count=row.get("deployed_continuous_coefficients"),stored_bytes=None,mac_estimate=None,metrics=row,acceptance_gate={"effective_fit_samples":">=2x deployed continuous coefficients"},failed_conditions=[row.get("failure_code","UNDERDETERMINED_AT_N")],distance_from_threshold={},scientific_interpretation="This calibration budget is mathematically underdetermined for the frozen held-out backend parameterization and is not converted into a regularized pseudo-result.",confounds_ruled_out=["heldout hyperparameter search","degree search","k search"],confounds_remaining=["larger predeclared calibration budget"],result_status="PROVISIONAL_CALIBRATION_BUDGET_FAILURE",next_justified_experiment="Advance to the next frozen calibration budget only.",reproduction_command="python scripts/reproduce_v837_recovery.py --variant v837ap --stage heldout --execute",artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/heldout_calibration_frontier.json"]),append_central=False)


def run_heldout_program()->dict:
    started=time.perf_counter()
    frozen=freeze_family_geometries();folds=freeze_source_folds();families={};all_rows=[];backend_records=[]
    for family in POWERED_FAMILIES:
        fspec=frozen["families"].get(family);holdouts=folds["families"][family]["holdout"];front=[];minimum=None
        if fspec is None:
            families[family]={"candidate":None,"holdout_count":len(holdouts),"frontier":[],"minimum_successful_n":None,"qualifier":None,"pass":False,"reason":"NO_FROZEN_GEOMETRY"};continue
        for n in CALIBRATION_LADDER:
            nrows=[]
            for idx,oid in enumerate(holdouts,1):
                compiled=compile_heldout_backend(oid,family,fspec,n);backend_records.append(compiled)
                if not compiled.get("valid"):
                    row={"organism_id":oid,"family":family,"engine":compiled.get("engine"),"calibration_n":n,"underdetermined":bool(compiled.get("underdetermined")),"compile":compiled,"pass":False,"failure_code":compiled.get("failure_code")};nrows.append(row);all_rows.append(row)
                    if compiled.get("underdetermined"):_log_budget_failure(family,oid,n,compiled)
                    continue
                geom=compiled["geometry"];q=np.asarray(compiled["q"],dtype=np.float64);chart_fit=compiled["chart_fit_seeds"];writer_fit=compiled["writer_fit_seeds"]
                eval_seeds=seeds("REUSED_HISTORICAL_VALIDATION")
                reader=evaluate_reader_geometry(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION")
                setp=evaluate_geometry(geom,q,"REUSED_HISTORICAL_VALIDATION",chart_fit_seeds=chart_fit,writer_fit_seeds=writer_fit) if reader.get("pass") else {"pass":False,"failure_code":"READER_GATE_FAIL","metrics":{}}
                if reader.get("pass") and setp.get("pass"):
                    quot=evaluate_quotient(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION",writer_fit_seeds=writer_fit)
                    dyn=evaluate_dynamics(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION",chart_fit_seeds=chart_fit,writer_fit_seeds=writer_fit)
                else:
                    quot={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"};dyn={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"}
                passed=bool(reader.get("pass") and setp.get("pass") and quot.get("pass") and dyn.get("pass"));row={"organism_id":oid,"family":family,"engine":geom.get("engine"),"calibration_n":n,"underdetermined":False,"compile":{k:v for k,v in compiled.items() if k not in {"geometry","q"}},"reader":reader,"setpoint":setp,"quotient":quot,"dynamics":dyn,"pass":passed};nrows.append(row);all_rows.append(row)
                print(f"V837ap heldout {family} N={n} {idx}/{len(holdouts)} pass={passed}",flush=True)
            gate=heldout_family_gate(nrows);front.append({"n":n,"gate":gate})
            if minimum is None and gate["pass"]:minimum=n
        families[family]={"candidate":fspec,"holdout_count":len(holdouts),"frontier":front,"minimum_successful_n":minimum,"qualifier":qualifier(minimum),"pass":minimum is not None}
    payload={"version":"V837ap","stage":"AP14_AP15_HELDOUT","evaluation_label":"HELDOUT_ORGANISM / REUSED_HISTORICAL_EPISODE_VALIDATION","calibration_ladder":list(CALIBRATION_LADDER),"families":families,"rows":all_rows,"heldout_backend_opened_only_after_freeze":True,"frozen_geometry_sha256":frozen["frozen_sha256"],"historical_v837an_q_loaded":False,"historical_v837an_backend_loaded":False,"historical_v837ao_backend_loaded":False,"geometry_search_on_holdout":False,"branch_search_on_holdout":False,"chart_search_on_holdout":False,"degree_search_on_holdout":False,"k_search_on_holdout":False,"phase_atlas_search_on_holdout":False,"writer_search_on_holdout":False,"fresh_audit_consumed":False,"wall_seconds":time.perf_counter()-started}
    write_json(HERE/"raw/heldout_calibration_frontier.json",payload);write_json(HERE/"raw/calibration_frontier.json",payload);write_json(HERE/"raw/heldout_backend_results.json",{"version":"V837ap","rows":all_rows,"backend_records":backend_records})
    isolation={"version":"V837ap","pass":True,"frozen_geometry_sha256":frozen["frozen_sha256"],"heldout_read_after_freeze":True,"v837an_q_loaded":False,"v837an_backend_loaded":False,"v837ao_backend_loaded":False,"geometry_search":False,"degree_search":False,"k_search":False,"fresh_audit_consumed":False}
    write_json(HERE/"diagnostics/heldout_isolation.json",isolation);write_json(HERE/"diagnostics/heldout_backend_isolation.json",isolation)
    write_json(HERE/"diagnostics/calibration_frontier.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_heldout_program(),indent=2,default=str))
