from __future__ import annotations

import json
import time

from .authorization import PARTITIONS
from .meta_confirm import _evaluate_family
from .failure_ledger import add, make_entry
from .utils import HERE, read_json, sha256_json, write_json


def run_final_dev()->dict:
    start=time.perf_counter();cpu_start=time.process_time();meta=read_json(HERE/"raw/meta_confirmed_family_abstractions.json")["families"];rows=[];frozen={}
    for family,w in meta.items():
        if w is None:frozen[family]=None;continue
        r=_evaluate_family(family,w["candidate"],"AN_FINAL_DEV",.01);rows.append(r);frozen[family]=w if r["family_pass"] else None;print(f"V837an FINAL_DEV {family} {w['candidate']['branch']} pass={r['family_pass']} powered={r['powered_organisms']} passing={r['organisms_passing']}",flush=True)
        if not r["family_pass"]:
            add(make_entry(failure_id=f"V837an-FINAL-DEV-{family}-{w['candidate']['branch']}-{w['candidate']['config_id']}",stage="FINAL_DEV_CONFIRM",hypothesis=f"The meta-confirmed causal abstraction for {family} generalizes unchanged to FINAL_DEV.",why="Independent final development confirmation before validation freeze.",implementation={"candidate":w["candidate"],"no_refit":True},data={"final_dev":PARTITIONS["AN_FINAL_DEV"]},fit_seeds=[],selection_seeds=PARTITIONS["AN_FINAL_DEV"],controls=["frozen branch controls"],parameter_count=0,mac_cost=r.get("intervention_dof"),metrics={k:v for k,v in r.items() if k!="organism_results"},gate={"eligible_pairs_min":24,"recovery_min":.60,"success_min":.70,"direction_min":.75,"control_margin_min":.20,"paired_p_max":.01,"ood_max":2.0,"minimum_pass_fraction":.60},failed=["FAMILY_CAUSAL_ABSTRACTION_DEV_CONFIRM_FAIL"],distance={"pass_fraction_shortfall":max(0.0,.60-float(r.get("pass_fraction",0.0)))},result_status="underpowered" if int(r.get("powered_organisms",0))<5 else "definitive FINAL_DEV failure",failure_type="UNDERPOWERED" if int(r.get("powered_organisms",0))<5 else "SCIENTIFIC_FAILURE",confounds_ruled_out=["META-confirm overfit","second-best replacement after failure"],confounds_remaining=["other primitive level"],meaning="The frozen family abstraction failed FINAL_DEV and is removed; no second-best candidate is substituted.",uncertainty="Family remains unresolved at the tested granularity.",next_experiment="Continue V837an with this family frozen as null.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage dev-confirm --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/final_dev_confirmation.json"]))
    payload={"version":"V837an","stage":"FINAL_DEV_CONFIRM","seeds":PARTITIONS["AN_FINAL_DEV"],"no_refit":True,"results":rows,"family_winners":frozen,"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
    write_json(HERE/"raw/final_dev_confirmation.json",payload)
    freeze={"version":"V837an","frozen_before_validation":True,"families":frozen,"source_final_dev_sha256":sha256_json(payload)};freeze["frozen_sha256"]=sha256_json(freeze)
    write_json(HERE/"raw/frozen_family_abstractions.json",freeze);write_json(HERE/"diagnostics/final_freeze.json",{"version":"V837an","frozen_sha256":freeze["frozen_sha256"],"family_count":sum(v is not None for v in frozen.values()),"validation_unlocked":True})
    return payload


if __name__=="__main__":print(json.dumps(run_final_dev(),indent=2))
