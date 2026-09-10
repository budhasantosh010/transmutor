from __future__ import annotations

import json
import time

from .authorization import PARTITIONS
from .meta_confirm import _evaluate_family
from .failure_ledger import add, make_entry
from .utils import HERE, read_json, sha256_json, write_json


def run_final_validation()->dict:
    start=time.perf_counter();cpu_start=time.process_time();freeze=read_json(HERE/"raw/frozen_family_abstractions.json");expected=freeze["frozen_sha256"];check=dict(freeze);check.pop("frozen_sha256",None)
    if sha256_json(check)!=expected:raise RuntimeError("V837AN_FINAL_FREEZE_HASH_MISMATCH")
    selected={k:v for k,v in freeze["families"].items() if v is not None}
    if not selected:
        payload={"version":"V837an","stage":"FINAL_VALIDATION","run":False,"reason":"NO_FROZEN_FAMILY_ABSTRACTIONS","validation_seeds_accessed":False,"results":[],"validated_families":[],"validated_family_count":0,"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
        write_json(HERE/"raw/final_validation.json",payload);write_json(HERE/"diagnostics/final_validation.json",payload);return payload
    rows=[];validated=[]
    for family,w in selected.items():
        r=_evaluate_family(family,w["candidate"],"AN_FINAL_VALIDATION",.01);rows.append(r);print(f"V837an FINAL_VALIDATION {family} {w['candidate']['branch']} pass={r['family_pass']} powered={r['powered_organisms']} passing={r['organisms_passing']}",flush=True)
        if r["family_pass"]:
            validated.append(family)
        else:
            add(make_entry(failure_id=f"V837an-VALIDATION-{family}-{w['candidate']['branch']}-{w['candidate']['config_id']}",stage="FINAL_VALIDATION",hypothesis=f"The frozen V837an causal abstraction for {family} generalizes unchanged to held-out validation.",why="Final held-out test after family abstraction freeze; validation cannot select a replacement.",implementation={"candidate":w["candidate"],"freeze_sha256":expected,"no_refit":True,"no_second_best_retry":True},data={"validation":PARTITIONS["AN_FINAL_VALIDATION"]},fit_seeds=[],selection_seeds=PARTITIONS["AN_FINAL_VALIDATION"],controls=["frozen branch controls"],parameter_count=0,mac_cost=r.get("intervention_dof"),metrics={k:v for k,v in r.items() if k!="organism_results"},gate={"eligible_pairs_min":32,"median_recovery_min":.60,"counterfactual_success_min":.70,"direction_agreement_min":.75,"control_margin_min":.20,"paired_p_max":.01,"median_ood_ratio_max":2.0,"minimum_cross_organism_support":5,"minimum_pass_fraction":.60},failed=["VALIDATION_GENERALIZATION_FAILURE"],distance={"pass_fraction_shortfall":max(0.0,.60-float(r.get("pass_fraction",0.0)))},result_status="underpowered" if int(r.get("powered_organisms",0))<5 else "definitive held-out validation failure",failure_type="UNDERPOWERED" if int(r.get("powered_organisms",0))<5 else "SCIENTIFIC_FAILURE",confounds_ruled_out=["selection leakage","validation refitting","second-best retry"],confounds_remaining=["other primitive granularity or later program"],meaning="This frozen family abstraction failed held-out generalization and remains a permanent negative result; no alternative winner is substituted.",uncertainty="The family may require another causal abstraction not selectable inside V837an after validation access.",next_experiment="Preserve this failure and follow the machine-selected next program after V837an closes.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage final --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/final_validation.json","experiments/v837_primitive_invention/v837an/raw/frozen_family_abstractions.json"]))
    payload={"version":"V837an","stage":"FINAL_VALIDATION","run":True,"seeds":PARTITIONS["AN_FINAL_VALIDATION"],"freeze_sha256":expected,"no_refit":True,"validation_seeds_accessed":True,"no_second_best_retry":True,"results":rows,"validated_families":validated,"validated_family_count":len(validated),"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
    write_json(HERE/"raw/final_validation.json",payload);write_json(HERE/"diagnostics/final_validation.json",payload);return payload


if __name__=="__main__":print(json.dumps(run_final_validation(),indent=2))
