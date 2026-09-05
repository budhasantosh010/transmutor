from __future__ import annotations

import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _active_mask, _configure_torch, _success_rate, _temporal_variance
from experiments.v837_primitive_invention.v837w.gru_controller_information import ControllerInformationGRU, INFORMATION_MODES

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
CONDITIONS=list(CONFIG['conditions']); FAMILIES=[t.name for t in all_tasks()]

def _blob(path:str)->str:
    return hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=ROOT)).hexdigest()

def _locks():
    if gate_sha256()!=CONFIG['historical_gate_hash']: raise SystemExit('historical gate changed')
    if v837_capacity_criterion_sha256()!=CONFIG['capacity_criterion_hash']: raise SystemExit('capacity criterion changed')
    for p,k in [('experiments/v837_primitive_invention/v837t/results.json','v837t_result_sha256'),('experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json','v837t_decision_sha256'),('experiments/v837_primitive_invention/v837v/diagnostics/decision_state.json','v837v_decision_sha256')]:
        if _blob(p)!=CONFIG[k]: raise SystemExit(f'{p} changed')
    v=json.loads((ROOT/'experiments/v837_primitive_invention/v837v/diagnostics/decision_state.json').read_text())
    if v.get('representation_adequacy_pass') is not False or v.get('v837w_allowed') is not True: raise SystemExit('V837w not authorized by V837v')
    audit=json.loads((ROOT/'experiments/v837_primitive_invention/audit/audit_results.json').read_text())
    if audit.get('episodes_consumed')!=0: raise SystemExit('fresh audit consumed')

def _stats(trace,lengths):
    mask=_active_mask(lengths,trace.updates.shape[1])
    gate=trace.updates[mask].detach().cpu().numpy().reshape(-1)
    inp=trace.input_logits[mask].detach(); st=trace.state_logits[mask].detach(); bias=trace.bias_logits[mask].detach()
    total=torch.linalg.vector_norm(inp+st+bias,dim=-1)
    ni=torch.linalg.vector_norm(inp,dim=-1); ns=torch.linalg.vector_norm(st,dim=-1); nb=torch.linalg.vector_norm(bias,dim=-1)
    return {
      'gate_mean':float(np.mean(gate)),'gate_temporal_variance':float(_temporal_variance(trace.updates,lengths)),
      'gate_p10':float(np.quantile(gate,.1)),'gate_p90':float(np.quantile(gate,.9)),
      'near_zero_fraction':float(np.mean(gate<=.05)),'near_one_fraction':float(np.mean(gate>=.95)),
      'input_logit_norm':float(ni.mean()),'state_logit_norm':float(ns.mean()),'bias_logit_norm':float(nb.mean()),
      'input_over_total':float((ni/(total+1e-12)).mean()),'state_over_total':float((ns/(total+1e-12)).mean())
    }

def _diagnostics(model,task,val_seeds):
    eps=[task.generate(s,'validation') for s in val_seeds]; obs,lengths,targets=episodes_to_batch(eps)
    model.eval()
    with torch.no_grad(): pred,tr=model(obs,lengths,return_trace=True)
    original=_success_rate(task,pred,targets); out=_stats(tr,lengths); out['validation_success_recomputed']=original; out['counterfactual_source_ablation']={}
    calls=1
    if model.information_mode=='joint':
        for mode in ('input_only','state_only','bias_only'):
            with torch.no_grad(): p=model(obs,lengths,information_mode_override=mode)
            score=_success_rate(task,p,targets); out['counterfactual_source_ablation'][mode]={'validation_success':score,'delta':original-score}; calls+=1
    return out,calls

def _worker(condition,family,rep):
    _configure_torch(); task=task_by_name(family); tr=CONFIG['training']; train=list(range(tr['development_seed_range'][0],tr['development_seed_range'][1]+1)); val=list(range(tr['validation_seed_range'][0],tr['validation_seed_range'][1]+1)); seed=deterministic_int(tr['initialization_namespace'],family,rep)
    res=train_sequence_model(model_factory=lambda:ControllerInformationGRU(CONFIG['hidden_size'],CONFIG['input_dim'],condition=condition),task=task,train_seeds=train,validation_seeds=val,initialization_seed=seed,steps=tr['steps'],learning_rate=tr['learning_rate'],weight_decay=tr['weight_decay'],gradient_clip=tr['gradient_clip'],curve_steps=tuple(tr['curve_steps']))
    diag,calls=_diagnostics(res.model,task,val); res.resources.forward_calls+=calls
    m=res.model
    return {'version':'V837w','condition':condition,'controller_information_mode':m.information_mode,'family':family,'replicate_id':rep,'development_success':res.development.success_rate,'validation_success':res.validation.success_rate,'development_loss':res.development.loss,'validation_loss':res.validation.loss,'capacity_demonstrated':capacity_demonstrated(res.development.success_rate,res.validation.success_rate),'loss_curve':res.learning_curve,'input_dynamic_enabled':m.information_mode in {'joint','input_only'},'state_dynamic_enabled':m.information_mode in {'joint','state_only'},'bias_enabled':True,'nominal_parameter_count':m.nominal_parameter_count(),'active_parameter_count':m.active_parameter_count(),'nominal_update_controller_parameters':m.nominal_update_controller_parameters,'active_update_controller_parameters':m.active_update_controller_parameters,'diagnostics':diag,'resources':res.resources.to_dict(),'optimizer_steps':tr['steps'],'processed_examples':tr['steps']*tr['train_episodes'],'unique_seed_defined_episode_policy':'same 3200 family/seed episodes reused across conditions and replicates','task_family_label_in_model_input':False,'fresh_audit_consumed':False,'gpu_seconds':0.0}

def _summary(rows,c):
    n=0; med={}
    for f in FAMILIES:
        rr=[r for r in rows if r['condition']==c and r['family']==f]; d=float(np.median([r['development_success'] for r in rr])); v=float(np.median([r['validation_success'] for r in rr])); med[f]=v; n+=int(capacity_demonstrated(d,v))
    return {'families_passing':n,'family_validation_medians':med}

def main():
    _locks(); [ (HERE/n).mkdir(exist_ok=True) for n in ('raw','diagnostics','plots') ]
    jobs=[(c,f,r) for c in CONDITIONS for f in FAMILIES for r in range(CONFIG['training']['replicates'])]; rows=[]
    with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as pool:
        futs={pool.submit(_worker,*j):j for j in jobs}
        for fut in as_completed(futs):
            row=fut.result(); rows.append(row); print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}",flush=True)
    rows.sort(key=lambda r:(r['condition'],r['family'],r['replicate_id']))
    write_json(HERE/'raw/runs.json',{'rows':rows,'unique_seed_defined_episodes':3200,'reuse_policy':'paired reuse across conditions and replicates','fresh_audit_consumed':False})
    w0=_summary(rows,'W0_joint_input_state'); t2=json.loads((ROOT/'experiments/v837_primitive_invention/v837t/results.json').read_text())['conditions']['T2_scalarized_update_no_reset']; drift={f:abs(w0['family_validation_medians'][f]-t2['family_results'][f]['validation']['median']) for f in FAMILIES}; guard={'compatible':w0['families_passing']==4 and max(drift.values())<=1e-12,'families_passing':w0['families_passing'],'absolute_family_median_drift':drift}
    write_json(HERE/'diagnostics/positive_control_guard.json',guard)
    print(json.dumps({'W0_guard':guard,'summaries':{c:_summary(rows,c) for c in CONDITIONS}},indent=2)); return 0 if guard['compatible'] else 2

if __name__=='__main__': raise SystemExit(main())
