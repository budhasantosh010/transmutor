from __future__ import annotations
import json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.metrics import continuous_summary
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import all_tasks
HERE=Path(__file__).resolve().parent; CONFIG=json.loads((HERE/'config.json').read_text()); CONDITIONS=CONFIG['conditions']; FAMILIES=[t.name for t in all_tasks()]

def summarize(rows,c):
    sel=[r for r in rows if r['condition']==c]; fam={}; passed=0
    for f in FAMILIES:
        rr=[r for r in sel if r['family']==f]; d=np.asarray([r['development_success'] for r in rr]); v=np.asarray([r['validation_success'] for r in rr]); p=capacity_demonstrated(float(np.median(d)),float(np.median(v))); passed+=int(p); fam[f]={'development':continuous_summary(d),'validation':continuous_summary(v),'aggregate_capacity_pass':bool(p),'replicate_success_rate':float(np.mean([r['capacity_demonstrated'] for r in rr]))}
    keys=['global_scalar_mean','global_scalar_temporal_variance','global_scalar_p10','global_scalar_p90','near_zero_fraction','near_one_fraction','state_norm','candidate_state_norm','mean_state_candidate_distance']; diag={k:continuous_summary(np.asarray([r['diagnostics'][k] for r in sel])) for k in keys}; first=sel[0]
    resources={'model_fits':len(sel),'optimizer_steps':sum(r['resources']['optimizer_steps'] for r in sel),'processed_examples':sum(r['resources']['examples_processed'] for r in sel),'environment_interactions':sum(r['resources']['environment_steps'] for r in sel),'forward_calls':sum(r['resources']['forward_calls'] for r in sel),'cpu_seconds':float(sum(r['resources']['cpu_seconds'] for r in sel)),'wall_seconds_sum_workers':float(sum(r['resources']['wall_seconds'] for r in sel)),'gpu_seconds':0.0}
    return {'families_passing':passed,'family_results':fam,'parameter_count':first['parameter_count'],'controller_param_count':first['controller_param_count'],'controller_macs':first['controller_macs'],'base_macs':160,'total_macs':first['total_macs'],'carry':first['carry'],'control_type':first['control_type'],'diagnostics':diag,'resource_accounting':resources}

def plots(s):
    d=HERE/'plots'; d.mkdir(exist_ok=True); x=np.arange(len(CONDITIONS));
    plt.figure(figsize=(8,4)); plt.bar(x,[s[c]['families_passing'] for c in CONDITIONS]); plt.xticks(x,CONDITIONS,rotation=25,ha='right'); plt.tight_layout(); plt.savefig(d/'global_scalar_control_family_scores.png'); plt.close()
    plt.figure(figsize=(9,4));
    for f in FAMILIES: plt.plot(x,[s[c]['family_results'][f]['validation']['median'] for c in CONDITIONS],marker='o',label=f)
    plt.xticks(x,CONDITIONS,rotation=25,ha='right'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(d/'global_vs_local_scalar_control.png'); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter([s[c]['total_macs'] for c in CONDITIONS],[s[c]['families_passing'] for c in CONDITIONS]); [plt.annotate(c,(s[c]['total_macs'],s[c]['families_passing']),fontsize=7) for c in CONDITIONS]; plt.xlabel('MACs/timestep'); plt.ylabel('families passing'); plt.tight_layout(); plt.savefig(d/'capability_vs_controller_macs.png'); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter([s[c]['parameter_count'] for c in CONDITIONS],[s[c]['families_passing'] for c in CONDITIONS]); [plt.annotate(c,(s[c]['parameter_count'],s[c]['families_passing']),fontsize=7) for c in CONDITIONS]; plt.tight_layout(); plt.savefig(d/'capability_vs_active_params.png'); plt.close()
    plt.figure(figsize=(5,4)); plt.bar([0,1],[s['X2_global_scalar_carry']['families_passing'],s['X2C_global_scale_candidate_control']['families_passing']]); plt.xticks([0,1],['carry','scale control']); plt.tight_layout(); plt.savefig(d/'global_carry_vs_scaling_control.png'); plt.close()
    plt.figure(figsize=(5,4)); plt.bar([0,1],[s['X2_global_scalar_carry']['diagnostics']['global_scalar_temporal_variance']['median'],s['X2C_global_scale_candidate_control']['diagnostics']['global_scalar_temporal_variance']['median']]); plt.xticks([0,1],['X2','X2C']); plt.tight_layout(); plt.savefig(d/'global_control_gate_dynamics.png'); plt.close()

def main():
    rows=json.loads((HERE/'raw/runs.json').read_text())['rows'];
    if len(rows)!=100: raise SystemExit('expected 100 fits')
    guard=json.loads((HERE/'diagnostics/anchor_guard.json').read_text());
    if not guard.get('compatible'): raise SystemExit('X0/X1 baseline drift')
    s={c:summarize(rows,c) for c in CONDITIONS}; x2=s['X2_global_scalar_carry']['families_passing']; x2c=s['X2C_global_scale_candidate_control']['families_passing']; adequate=x2>=4
    if adequate and x2c<4: diagnosis='GLOBAL_TEMPORAL_CONTROL_WITH_ADAPTIVE_CARRY_SUFFICIENT'
    elif adequate: diagnosis='GLOBAL_DYNAMIC_SIGNAL_SUFFICIENT'
    elif x2==3: diagnosis='GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT'
    else: diagnosis='GLOBAL_SCALAR_CONTROLLER_TRANSFER_FAILURE'
    resource={'model_fits':len(rows),'optimizer_steps':sum(r['resources']['optimizer_steps'] for r in rows),'processed_examples':sum(r['resources']['examples_processed'] for r in rows),'environment_interactions':sum(r['resources']['environment_steps'] for r in rows),'forward_calls':sum(r['resources']['forward_calls'] for r in rows),'cpu_seconds':float(sum(r['resources']['cpu_seconds'] for r in rows)),'wall_seconds_sum_workers':float(sum(r['resources']['wall_seconds'] for r in rows)),'gpu_seconds':0.0,'unique_seed_defined_episodes':3200}
    result={'version':'V837x','parent':'V837w','authorized_controller_mode':CONFIG['authorized_controller_mode'],'conditions':s,'diagnosis':diagnosis,'representation_adequacy_pass':adequate,'sample_efficiency_retest_allowed':adequate,'structural_search_allowed':False,'primitive_mining_allowed':False,'fresh_audit_consumed':False,'primitives_promoted':0,'large_persistent_storage_tested':False,'v838_started':False,'resource_accounting':resource}
    decision={'v837x_complete':True,'authorized_controller_mode':CONFIG['authorized_controller_mode'],'families_passing':{c:s[c]['families_passing'] for c in CONDITIONS},'diagnosis':diagnosis,'representation_adequacy_pass':adequate,'sample_efficiency_retest_allowed':adequate,'structural_search_allowed':False,'primitive_mining_allowed':False,'fresh_audit_consumed':False,'primitives_promoted':0,'v838_started':False}
    write_json(HERE/'results.json',result); write_json(HERE/'diagnostics/global_scalar_control_summary.json',s); write_json(HERE/'diagnostics/decision_state.json',decision); write_json(HERE/'diagnostics/compute_efficiency.json',{'conditions':{c:{'families_passing':s[c]['families_passing'],'parameter_count':s[c]['parameter_count'],'controller_param_count':s[c]['controller_param_count'],'controller_macs':s[c]['controller_macs'],'total_macs':s[c]['total_macs'],'capability_per_active_mac':s[c]['families_passing']/s[c]['total_macs'],'capability_per_active_parameter':s[c]['families_passing']/s[c]['parameter_count'],'resource_accounting':s[c]['resource_accounting']} for c in CONDITIONS},'program':resource}); write_json(ROOT/'experiments/v837_primitive_invention/v837x_resource_accounting.json',resource); plots(s)
    name='PASS.md' if adequate else 'FAILURE.md'; (HERE/name).write_text(f'# V837x {diagnosis}\n\nX0/X1/X2/X2C families passing: {decision["families_passing"]}.\nRepresentation adequacy: {"PASS" if adequate else "FAIL"}.\n',encoding='utf-8'); print(json.dumps(decision,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
