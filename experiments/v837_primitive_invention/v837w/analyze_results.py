from __future__ import annotations
import json, sys
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
        rr=[r for r in sel if r['family']==f]; d=np.asarray([r['development_success'] for r in rr]); v=np.asarray([r['validation_success'] for r in rr]); p=capacity_demonstrated(float(np.median(d)),float(np.median(v))); passed+=int(p)
        fam[f]={'development':continuous_summary(d),'validation':continuous_summary(v),'aggregate_capacity_pass':bool(p),'replicate_success_rate':float(np.mean([r['capacity_demonstrated'] for r in rr]))}
    keys=['gate_mean','gate_temporal_variance','gate_p10','gate_p90','near_zero_fraction','near_one_fraction','input_logit_norm','state_logit_norm','bias_logit_norm','input_over_total','state_over_total']
    diag={k:continuous_summary(np.asarray([r['diagnostics'][k] for r in sel])) for k in keys}
    first=sel[0]; resources={'model_fits':len(sel),'optimizer_steps':sum(r['resources']['optimizer_steps'] for r in sel),'processed_examples':sum(r['resources']['examples_processed'] for r in sel),'environment_interactions':sum(r['resources']['environment_steps'] for r in sel),'forward_calls':sum(r['resources']['forward_calls'] for r in sel),'cpu_seconds':float(sum(r['resources']['cpu_seconds'] for r in sel)),'wall_seconds_sum_workers':float(sum(r['resources']['wall_seconds'] for r in sel)),'gpu_seconds':0.0}
    return {'families_passing':passed,'family_results':fam,'controller_information_mode':first['controller_information_mode'],'nominal_parameter_count':first['nominal_parameter_count'],'active_parameter_count':first['active_parameter_count'],'nominal_update_controller_parameters':first['nominal_update_controller_parameters'],'active_update_controller_parameters':first['active_update_controller_parameters'],'diagnostics':diag,'resource_accounting':resources}

def diagnose(s):
    w0=s['W0_joint_input_state']['families_passing']; w1=s['W1_input_only']['families_passing']; w2=s['W2_state_only']['families_passing']; w3=s['W3_bias_only']['families_passing']
    if w0<4: return 'REFERENCE_CONTROLLER_INFORMATION_INCONCLUSIVE',None
    if w3>=4: return 'DYNAMIC_CONTROL_SPECIFICITY_NOT_ESTABLISHED',None
    if w1>=4: return 'GLOBAL_CONTROL_INPUT_ONLY_SUFFICIENT','INPUT_ONLY_GLOBAL_SCALAR'
    if w2>=4: return 'GLOBAL_STATE_ASSESSMENT_REQUIRED','STATE_ONLY_GLOBAL_SCALAR'
    return 'JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED','JOINT_INPUT_STATE_GLOBAL_SCALAR'

def plots(s):
    d=HERE/'plots'; d.mkdir(exist_ok=True); x=np.arange(len(CONDITIONS))
    plt.figure(figsize=(8,4)); plt.bar(x,[s[c]['families_passing'] for c in CONDITIONS]); plt.xticks(x,CONDITIONS,rotation=25,ha='right'); plt.ylabel('families passing'); plt.tight_layout(); plt.savefig(d/'families_passing_input_state_joint.png'); plt.close()
    plt.figure(figsize=(9,4));
    for f in FAMILIES: plt.plot(x,[s[c]['family_results'][f]['validation']['median'] for c in CONDITIONS],marker='o',label=f)
    plt.xticks(x,CONDITIONS,rotation=25,ha='right'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(d/'controller_information_family_scores.png'); plt.close()
    plt.figure(figsize=(8,4)); plt.bar(x,[s[c]['diagnostics']['gate_temporal_variance']['median'] for c in CONDITIONS]); plt.xticks(x,CONDITIONS,rotation=25,ha='right'); plt.tight_layout(); plt.savefig(d/'gate_temporal_variance_by_information_source.png'); plt.close()
    w0=s['W0_joint_input_state']['diagnostics']; plt.figure(figsize=(5,4)); plt.bar([0,1],[w0['input_logit_norm']['median'],w0['state_logit_norm']['median']]); plt.xticks([0,1],['input','state']); plt.ylabel('median logit norm'); plt.tight_layout(); plt.savefig(d/'input_vs_state_logit_contribution.png'); plt.close()
    rows=json.loads((HERE/'raw/runs.json').read_text())['rows']; w0r=[r for r in rows if r['condition']=='W0_joint_input_state']; labels=['input_only','state_only','bias_only']; vals=[]
    for label in labels: vals.append(float(np.mean([r['diagnostics']['counterfactual_source_ablation'][label]['delta'] for r in w0r])))
    plt.figure(figsize=(6,4)); plt.bar(np.arange(3),vals); plt.xticks(np.arange(3),labels); plt.ylabel('mean original - ablated'); plt.tight_layout(); plt.savefig(d/'counterfactual_source_ablation.png'); plt.close()

def main():
    rows=json.loads((HERE/'raw/runs.json').read_text())['rows']; expected=len(CONDITIONS)*5*CONFIG['training']['replicates']
    if len(rows)!=expected: raise SystemExit(f'expected {expected}, got {len(rows)}')
    guard=json.loads((HERE/'diagnostics/positive_control_guard.json').read_text());
    if not guard.get('compatible'): raise SystemExit('W0 baseline drift')
    s={c:summarize(rows,c) for c in CONDITIONS}; diagnosis,mode=diagnose(s); allowed=mode is not None
    resource={'model_fits':len(rows),'optimizer_steps':sum(r['resources']['optimizer_steps'] for r in rows),'processed_examples':sum(r['resources']['examples_processed'] for r in rows),'environment_interactions':sum(r['resources']['environment_steps'] for r in rows),'forward_calls':sum(r['resources']['forward_calls'] for r in rows),'cpu_seconds':float(sum(r['resources']['cpu_seconds'] for r in rows)),'wall_seconds_sum_workers':float(sum(r['resources']['wall_seconds'] for r in rows)),'gpu_seconds':0.0,'unique_seed_defined_episodes':3200}
    result={'version':'V837w','parent':'V837v','question':CONFIG['question'],'reference_only':True,'conditions':s,'diagnosis':diagnosis,'neutral_global_controller_allowed':allowed,'authorized_v837x_mode':mode,'fresh_audit_consumed':False,'structural_search_allowed':False,'primitive_mining_allowed':False,'large_persistent_storage_tested':False,'v838_started':False,'resource_accounting':resource}
    decision={'v837w_complete':True,'w0_joint':s['W0_joint_input_state']['families_passing'],'w1_input_only':s['W1_input_only']['families_passing'],'w2_state_only':s['W2_state_only']['families_passing'],'w3_bias_only':s['W3_bias_only']['families_passing'],'diagnosis':diagnosis,'neutral_global_controller_allowed':allowed,'authorized_v837x_mode':mode,'fresh_audit_consumed':False,'structural_search_allowed':False,'primitive_mining_allowed':False,'v838_started':False}
    write_json(HERE/'results.json',result); write_json(HERE/'diagnostics/controller_information_summary.json',s); write_json(HERE/'diagnostics/decision_state.json',decision); write_json(HERE/'diagnostics/compute_efficiency.json',{'conditions':{c:s[c]['resource_accounting']|{'families_passing':s[c]['families_passing'],'active_params':s[c]['active_parameter_count']} for c in CONDITIONS},'program':resource}); write_json(ROOT/'experiments/v837_primitive_invention/v837w_resource_accounting.json',resource)
    plots(s); name='PASS.md' if allowed else 'FAILURE.md'; (HERE/name).write_text(f'# V837w {diagnosis}\n\nW0/W1/W2/W3 families passing: {decision["w0_joint"]}/{decision["w1_input_only"]}/{decision["w2_state_only"]}/{decision["w3_bias_only"]}.\nAuthorized V837x mode: `{mode}`.\n',encoding='utf-8'); print(json.dumps(decision,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
