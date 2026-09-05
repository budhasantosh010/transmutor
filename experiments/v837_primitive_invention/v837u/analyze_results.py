from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text(encoding="utf-8")); CONDITIONS=CONFIG["conditions"]; FAMILIES=[t.name for t in all_tasks()]


def summarize(rows,condition):
    selected=[r for r in rows if r["condition"]==condition]; count=0; fam={}
    for f in FAMILIES:
        fr=[r for r in selected if r["family"]==f]; dev=np.asarray([r["development_success"] for r in fr]); val=np.asarray([r["validation_success"] for r in fr]); p=capacity_demonstrated(float(np.median(dev)),float(np.median(val))); count+=int(p)
        fam[f]={"development":continuous_summary(dev),"validation":continuous_summary(val),"aggregate_capacity_pass":bool(p)}
    mods=[r["diagnostics"]["state_modulator"] for r in selected if r["diagnostics"]["state_modulator"] is not None]
    md=None if not mods else {k:continuous_summary(np.asarray([m[k] for m in mods])) for k in ("mean","median","std","p10","p90","temporal_variance","near_zero_fraction","near_one_fraction")}
    first=selected[0]
    return {"families_passing":count,"family_results":fam,"parameter_count":first["parameter_count"],"parameter_bytes":first["parameter_bytes"],"controller_parameter_count":first["controller_parameter_count"],"controller_macs_per_timestep":first["controller_macs_per_timestep"],"base_recurrent_macs_per_timestep":160,"total_recurrent_controller_macs_per_timestep":first["total_recurrent_controller_macs_per_timestep"],"state_modulation_location":first["state_modulation_location"],"state_modulator_diagnostics":md,
    "resource_accounting":{"model_fits":len(selected),"optimizer_steps":sum(r["resources"]["optimizer_steps"] for r in selected),"processed_examples":sum(r["resources"]["examples_processed"] for r in selected),"environment_interactions":sum(r["resources"]["environment_steps"] for r in selected),"forward_calls":sum(r["resources"]["forward_calls"] for r in selected),"cpu_seconds":float(sum(r["resources"]["cpu_seconds"] for r in selected)),"wall_seconds_sum_workers":float(sum(r["resources"]["wall_seconds"] for r in selected)),"gpu_seconds":0.0}}


def plots(s):
    d=HERE/"plots"; d.mkdir(exist_ok=True); x=np.arange(len(CONDITIONS))
    plt.figure(figsize=(8,4)); plt.bar(x,[s[c]["families_passing"] for c in CONDITIONS]); plt.xticks(x,CONDITIONS,rotation=30,ha="right"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(d/"families_passing_neutral_followup.png"); plt.close()
    plt.figure(figsize=(9,4))
    for f in FAMILIES: plt.plot(x,[s[c]["family_results"][f]["validation"]["median"] for c in CONDITIONS],marker="o",label=f)
    plt.xticks(x,CONDITIONS,rotation=30,ha="right"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(d/"neutral_followup_family_scores.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter([s[c]["parameter_count"] for c in CONDITIONS],[s[c]["families_passing"] for c in CONDITIONS]);
    for c in CONDITIONS: plt.annotate(c,(s[c]["parameter_count"],s[c]["families_passing"]),fontsize=7); plt.xlabel("parameters"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(d/"capability_vs_parameter_count.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter([s[c]["total_recurrent_controller_macs_per_timestep"] for c in CONDITIONS],[s[c]["families_passing"] for c in CONDITIONS]);
    for c in CONDITIONS: plt.annotate(c,(s[c]["total_recurrent_controller_macs_per_timestep"],s[c]["families_passing"]),fontsize=7); plt.xlabel("recurrent+controller MACs/timestep"); plt.ylabel("families passing"); plt.tight_layout(); plt.savefig(d/"capability_vs_recurrent_controller_macs.png"); plt.close()


def main():
    rows=json.loads((HERE/"raw/runs.json").read_text(encoding="utf-8"))["rows"]
    expected=len(CONDITIONS)*5*CONFIG["training"]["replicates"]
    if len(rows)!=expected: raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    s={c:summarize(rows,c) for c in CONDITIONS}; carry=s["U2_dynamic_scalar_carry"]["families_passing"]; control=s["U2C_scalar_scale_candidate_control"]["families_passing"]
    adequacy=carry>=CONFIG["representation_family_gate"]
    if adequacy and control<CONFIG["representation_family_gate"]: diagnosis="DYNAMIC_SCALAR_CARRY_SUFFICIENT"
    elif adequacy: diagnosis="DYNAMIC_SCALAR_CARRY_SPECIFICITY_NOT_ESTABLISHED"
    else: diagnosis="DYNAMIC_SCALAR_CARRY_INSUFFICIENT"
    resource={"model_fits":len(rows),"optimizer_steps":sum(r["resources"]["optimizer_steps"] for r in rows),"processed_examples":sum(r["resources"]["examples_processed"] for r in rows),"environment_interactions":sum(r["resources"]["environment_steps"] for r in rows),"forward_calls":sum(r["resources"]["forward_calls"] for r in rows),"cpu_seconds":float(sum(r["resources"]["cpu_seconds"] for r in rows)),"wall_seconds_sum_workers":float(sum(r["resources"]["wall_seconds"] for r in rows)),"gpu_seconds":0.0,"unique_seed_defined_episodes":3200}
    result={"version":"V837u","parent":"V837t","authorized_mode":CONFIG["authorized_mode"],"question":CONFIG["question"],"single_change":CONFIG["single_change"],"conditions":s,"representation_adequacy_pass":adequacy,"diagnosis":diagnosis,"carry_specificity_established":bool(adequacy and control<4),"vector_specificity":"NOT_APPLICABLE","multiplicative_specificity":"NOT_APPLICABLE","sample_efficiency_retest_allowed":bool(adequacy),"structural_search_allowed":False,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"resource_accounting":resource}
    write_json(HERE/"results.json",result); write_json(HERE/"diagnostics/condition_summaries.json",s)
    decision={"v837u_complete":True,"authorized_mode":CONFIG["authorized_mode"],"families_passing":{c:s[c]["families_passing"] for c in CONDITIONS},"diagnosis":diagnosis,"representation_adequacy_pass":adequacy,"sample_efficiency_retest_allowed":bool(adequacy),"structural_search_allowed":False,"primitive_mining_allowed":False,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False}
    write_json(HERE/"diagnostics/decision_state.json",decision); plots(s)
    name="PASS.md" if adequacy else "FAILURE.md"; (HERE/name).write_text(f"# V837u {diagnosis}\n\nAuthorized mode: `{CONFIG['authorized_mode']}`\n\nRepresentation adequacy: {'PASS' if adequacy else 'FAIL'} ({carry}/5).\n",encoding="utf-8")
    print(json.dumps(decision,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
