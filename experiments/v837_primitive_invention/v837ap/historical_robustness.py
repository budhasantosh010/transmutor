from __future__ import annotations

import json
import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import family_gate
from .commutativity import evaluate_dynamics
from .data_roles import seeds
from .freeze_geometry import freeze_family_geometries
from .geometry_eval import evaluate_reader_geometry
from .geometry_store import load_discovery_geometry
from .projected_causal_spaces import space_map
from .quotient_eval import evaluate_quotient
from .setpoint_eval import evaluate_geometry
from .source_folds import freeze_source_folds
from .utils import HERE, write_json


def run_historical_robustness()->dict:
    frozen=freeze_family_geometries();folds=freeze_source_folds();spaces=space_map();families={};rows=[];eval_seeds=seeds("REUSED_HISTORICAL_VALIDATION")
    for family in POWERED_FAMILIES:
        fspec=frozen["families"].get(family)
        if fspec is None:
            families[family]={"pass":False,"reason":"NO_FROZEN_GEOMETRY"};continue
        winner=fspec["selection_evidence"];org_rows=[]
        for idx,oid in enumerate(folds["families"][family]["discovery"],1):
            geom=load_discovery_geometry(family,oid,winner);q=np.asarray(spaces[(family,oid,int(winner["k"]))]["q"],dtype=np.float64)
            reader=evaluate_reader_geometry(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION")
            setp=evaluate_geometry(geom,q,"REUSED_HISTORICAL_VALIDATION") if reader.get("pass") else {"pass":False,"failure_code":"READER_GATE_FAIL"}
            if reader.get("pass") and setp.get("pass"):
                quot=evaluate_quotient(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION");dyn=evaluate_dynamics(geom,q,eval_seeds,"REUSED_HISTORICAL_VALIDATION")
            else:
                quot={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"};dyn={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"}
            passed=bool(reader.get("pass") and setp.get("pass") and quot.get("pass") and dyn.get("pass"));row={"organism_id":oid,"family":family,"engine":geom["engine"],"reader":reader,"setpoint":setp,"quotient":quot,"dynamics":dyn,"pass":passed};org_rows.append(row);rows.append(row)
            print(f"V837ap historical robustness {family} {idx}/{len(folds['families'][family]['discovery'])} pass={passed}",flush=True)
        gate=family_gate(org_rows,"pass");recoveries=[r.get("setpoint",{}).get("metrics",{}).get("median_counterfactual_recovery") for r in org_rows if r.get("setpoint",{}).get("metrics",{}).get("median_counterfactual_recovery") is not None];families[family]={"gate":gate,"pass":gate["pass"],"median_set_recovery":float(np.median(recoveries)) if recoveries else None}
    payload={"version":"V837ap","stage":"AP16_REUSED_HISTORICAL_VALIDATION","label":"REUSED_HISTORICAL_VALIDATION_DESCRIPTIVE_POST_FREEZE","descriptive_post_freeze_only":True,"cannot_upgrade_or_rescue_family":True,"not_fresh_audit":True,"fresh_audit_consumed":False,"families":families,"rows":rows,"passing_families":sum(1 for v in families.values() if v.get("pass"))}
    write_json(HERE/"raw/historical_validation_robustness.json",payload);write_json(HERE/"diagnostics/historical_validation_robustness.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_historical_robustness(),indent=2,default=str))
