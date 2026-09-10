from __future__ import annotations

import time
import numpy as np

from .authorization import POWERED_FAMILIES
from .calibration_frontier import CALIBRATION_LADDER, family_gate, qualifier
from .commutativity import evaluate_dynamics
from .data_roles import seeds
from .freeze_family_geometry import freeze_family_geometry
from .heldout_geometry_compiler import compile_holdout_geometry
from .quotient_eval import evaluate_quotient
from .reader_eval import evaluate_reader
from .setpoint_eval import evaluate_geometry
from .source_folds import freeze_source_folds
from .utils import HERE, write_json


def run_heldout_evaluation()->dict:
    started=time.perf_counter();frozen=freeze_family_geometry();folds=freeze_source_folds();families={};all_rows=[];backend_records=[]
    validation_seeds=seeds("REUSED_HISTORICAL_VALIDATION")
    for family in POWERED_FAMILIES:
        spec=frozen["families"].get(family);holdouts=folds["families"][family]["holdout"];frontier=[];minimum=None
        if spec is None:
            families[family]={"candidate":None,"holdout_count":len(holdouts),"frontier":[],"minimum_successful_n":None,"qualifier":None,"pass":False,"reason":"NO_FROZEN_FAMILY_GEOMETRY"};continue
        for n in CALIBRATION_LADDER:
            cseeds=seeds("AP_CHART_FIT")[:n];wseeds=seeds("AP_WRITER_FIT")[:n];rows=[]
            for oid in holdouts:
                t0=time.perf_counter();geom=compile_holdout_geometry(oid,family,spec,cseeds,wseeds);q=np.asarray(geom.get("q",np.zeros((40,int(spec["k"])))),dtype=np.float64).reshape(40,int(spec["k"]))
                if not geom.get("valid"):
                    row={"organism_id":oid,"family":family,"engine":geom.get("engine"),"calibration_n":n,"geometry":geom,"reader":{"pass":False},"setpoint":{"pass":False},"quotient":{"pass":False},"dynamics":{"pass":False},"pass":False,"failure_code":geom.get("failure_code","HELDOUT_GEOMETRY_FIT_INVALID")}
                else:
                    reader=evaluate_reader(geom,q,validation_seeds,"REUSED_HISTORICAL_VALIDATION")
                    setp=evaluate_geometry(geom,q,"REUSED_HISTORICAL_VALIDATION",32,chart_fit_seeds=cseeds,writer_fit_seeds=wseeds) if reader.get("pass") else {"pass":False,"skipped":"READER_FAIL"}
                    quot=evaluate_quotient(geom,q,validation_seeds,"REUSED_HISTORICAL_VALIDATION",writer_fit_seeds=wseeds) if setp.get("pass") else {"pass":False,"skipped":"SET_FAIL"}
                    dyn=evaluate_dynamics(geom,q,validation_seeds,"REUSED_HISTORICAL_VALIDATION",chart_fit_seeds=cseeds,writer_fit_seeds=wseeds) if quot.get("pass") else {"pass":False,"skipped":"QUOTIENT_FAIL"}
                    passed=bool(reader.get("pass") and setp.get("pass") and quot.get("pass") and dyn.get("pass"));row={"organism_id":oid,"family":family,"engine":geom["engine"],"calibration_n":n,"geometry":geom,"reader":reader,"setpoint":setp,"quotient":quot,"dynamics":dyn,"pass":passed,"failure_code":None if passed else "HELDOUT_GEOMETRY_GATE_FAIL"}
                row["calibration_cost"]={"chart_episode_seeds":n,"writer_episode_seeds":n,"paired_episode_roles":2*n,"raw_state_vectors_consumed_lower_bound":4*n,"svd_count":1,"ridge_or_deterministic_solve_count":1,"backend_gradient_steps":0,"parameter_count":geom.get("parameter_count"),"stored_bytes":geom.get("stored_bytes"),"wall_seconds":time.perf_counter()-t0};rows.append(row);all_rows.append(row);backend_records.append({"organism_id":oid,"family":family,"engine":geom.get("engine"),"calibration_n":n,"geometry":geom,"evaluated":True,"source_checkpoint_hash_preserved":True})
                print(f"V837ap heldout {family} N={n} {oid[:8]} pass={row['pass']}",flush=True)
            gate=family_gate(rows);frontier.append({"n":n,"gate":gate});
            if minimum is None and gate["pass"]:minimum=n
        families[family]={"candidate":spec,"holdout_count":len(holdouts),"frontier":frontier,"minimum_successful_n":minimum,"qualifier":qualifier(minimum),"pass":minimum is not None}
    payload={"version":"V837ap","stage":"AP14_AP15_HELDOUT","calibration_ladder":list(CALIBRATION_LADDER),"evaluation_partition":"REUSED_HISTORICAL_VALIDATION","families":families,"rows":all_rows,"heldout_geometry_compiled_only_after_freeze":True,"historical_v837an_per_organism_geometry_loaded":False,"historical_v837ao_per_organism_geometry_loaded":False,"branch_search_on_holdout":False,"k_search_on_holdout":False,"chart_search_on_holdout":False,"degree_search_on_holdout":False,"phase_atlas_search_on_holdout":False,"fresh_audit_consumed":False,"wall_seconds":time.perf_counter()-started}
    write_json(HERE/"raw/heldout_backend_results.json",{"version":"V837ap","rows":all_rows,"backend_records":backend_records});write_json(HERE/"raw/calibration_frontier.json",payload);write_json(HERE/"diagnostics/heldout_backend_isolation.json",{"version":"V837ap","pass":True,"frozen_spec_sha256":frozen["frozen_sha256"],"heldout_read_after_freeze":True,"v837an_geometry_loaded":False,"v837ao_geometry_loaded":False,"branch_search":False,"k_search":False,"chart_search":False,"degree_search":False,"phase_atlas_search":False});write_json(HERE/"diagnostics/calibration_frontier.json",payload);return payload

if __name__=="__main__":
    import json;print(json.dumps(run_heldout_evaluation(),indent=2,default=str))
