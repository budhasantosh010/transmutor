from __future__ import annotations

import json
import math

import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import family_gate
from .commutativity import evaluate_dynamics, save_commutativity
from .data_roles import seeds
from .failure_ledger import add, make_entry
from .geometry_store import load_discovery_geometry
from .projected_causal_spaces import space_map
from .quotient_eval import evaluate_quotient, save_quotient
from .utils import HERE, read_json, sha256_json, write_json


def _log_failure(stage: str, geometry: dict, result: dict, code: str) -> None:
    m=result.get("metrics",{})
    fid=f"V837ap-{stage}-"+sha256_json({"family":geometry.get("family"),"organism":geometry.get("organism_id"),"k":geometry.get("k"),"chart":geometry.get("chart_family"),"writer":geometry.get("writer_family"),"atlas":geometry.get("phase_atlas")})[:16]
    add(make_entry(
        failure_id=fid,stage=stage,branch=geometry.get("branch","SELECTED_GEOMETRY"),family=geometry.get("family"),organism=geometry.get("organism_id"),phase=None,
        carrier_dimension=geometry.get("k"),chart_family=geometry.get("chart_family"),chart_degree_rank=(geometry.get("chart") or {}).get("degree") if isinstance(geometry.get("chart"),dict) else None,
        writer_family=geometry.get("writer_family"),fit_partition="AP_CHART_FIT/AP_WRITER_FIT",selection_partition="AP_QUOTIENT" if stage=="AP8_QUOTIENT" else "AP_DYNAMICS",
        parameter_count=geometry.get("parameter_count"),stored_bytes=geometry.get("stored_bytes"),mac_estimate=None,metrics=m,
        matched_control_metrics={},ood_metrics={"median_ood_ratio":m.get("median_ood_ratio")},acceptance_gate={"quotient":{"median_RS":.20,"p90_RS":.40,"trajectory":.10,"prediction":.10,"task":.10,"ood":2.0},"dynamics":{"one_step":.10,"multi_step":.15,"output_recovery":.80,"task":.75,"control_margin":.20,"ood":2.0}},
        failed_conditions=[code],distance_from_threshold={},scientific_interpretation=("The selected semantic geometry remained sensitive to tested private residual variation." if stage=="AP8_QUOTIENT" else "The selected semantic geometry did not approximately close under the frozen benchmark transition law."),
        confounds_ruled_out=["geometry reselection after outcome","heldout leakage","source retraining"],confounds_remaining=["higher-dimensional or program-level causal state"],
        next_justified_experiment="Preserve this failure through META; do not fall back to another geometry after selection.",
        reproduction_command="python scripts/reproduce_v837_recovery.py --variant v837ap --stage quotient --execute" if stage=="AP8_QUOTIENT" else "python scripts/reproduce_v837_recovery.py --variant v837ap --stage dynamics --execute",
        artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/quotient_results.json" if stage=="AP8_QUOTIENT" else "experiments/v837_primitive_invention/v837ap/raw/commutativity_results.json"],
    ),append_central=False)


def run_causal_closure() -> dict:
    selection=read_json(HERE/"raw/discovery_family_geometry_winners.json")
    spaces=space_map(); quotient_rows=[]; dynamics_rows=[]; families={}
    for family in POWERED_FAMILIES:
        winner=selection["family_winners"].get(family)
        if winner is None:
            families[family]={"candidate":None,"pass":False,"reason":"NO_AP7_DISCOVERY_WINNER"};continue
        organism_rows=[]
        support=set(winner.get("support_organisms",[]))
        # Only organisms that already pass reader+SET are eligible for causal-closure compute.
        ids=read_json(HERE/"raw/frozen_source_folds.json")["families"][family]["discovery"]
        for idx,oid in enumerate(ids,1):
            geom=load_discovery_geometry(family,oid,winner)
            if oid not in support:
                organism_rows.append({"organism_id":oid,"engine":geom.get("engine"),"ap7_pass":False,"quotient":None,"dynamics":None,"pass":False});continue
            q=np.asarray(spaces[(family,oid,int(winner["k"]))]["q"],dtype=np.float64)
            quot=evaluate_quotient(geom,q,seeds("AP_QUOTIENT"),"AP_QUOTIENT")
            dyn=evaluate_dynamics(geom,q,seeds("AP_DYNAMICS"),"AP_DYNAMICS")
            quotient_rows.append(quot);dynamics_rows.append(dyn)
            passed=bool(quot.get("pass") and dyn.get("pass"));organism_rows.append({"organism_id":oid,"engine":geom.get("engine"),"ap7_pass":True,"quotient":quot,"dynamics":dyn,"pass":passed})
            if not quot.get("pass"):_log_failure("AP8_QUOTIENT",geom,quot,quot.get("failure_code") or "QUOTIENT_RESIDUAL_SENSITIVITY")
            if not dyn.get("pass"):_log_failure("AP9_DYNAMICS",geom,dyn,dyn.get("failure_code") or "CANONICAL_DYNAMICS_ROLLOUT_FAIL")
            print(f"V837ap closure {family} {idx}/{len(ids)} quotient={quot.get('pass')} dynamics={dyn.get('pass')} full={passed}",flush=True)
        gate=family_gate(organism_rows,"pass")
        families[family]={"candidate":winner,"organism_results":organism_rows,"closure_family_gate":gate,"pass":gate["pass"]}
    save_quotient(quotient_rows);save_commutativity(dynamics_rows)
    payload={"version":"V837ap","stage":"AP8_AP10_CAUSAL_CLOSURE_AND_ADJUDICATION","families":families,"closure_passing_families":sum(1 for v in families.values() if v.get("pass")),"geometry_fallback_after_selection":False}
    write_json(HERE/"raw/model_complexity_adjudication.json",payload)
    write_json(HERE/"diagnostics/quotient_sufficiency.json",{"version":"V837ap","rows":quotient_rows,"families":families})
    return payload

if __name__=="__main__":
    print(json.dumps(run_causal_closure(),indent=2,default=str))
