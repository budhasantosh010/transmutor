from __future__ import annotations
import hashlib,json,os,subprocess,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated,v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int,gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks,task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _active_mask,_configure_torch,_success_rate,_temporal_variance
from experiments.v837_primitive_invention.v837x.global_scalar_control import GlobalScalarNeutralModel
HERE=Path(__file__).resolve().parent; CONFIG=json.loads((HERE/'config.json').read_text()); CONDITIONS=CONFIG['conditions']; FAMILIES=[t.name for t in all_tasks()]

def _blob(p): return hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)).hexdigest()
def _locks():
    if gate_sha256()!=CONFIG['historical_gate_hash'] or v837_capacity_criterion_sha256()!=CONFIG['capacity_criterion_hash']: raise SystemExit('frozen gate changed')
    if _blob('experiments/v837_primitive_invention/v837w/diagnostics/decision_state.json')!=CONFIG['v837w_decision_sha256']: raise SystemExit('V837w decision changed')
    w=json.loads((ROOT/'experiments/v837_primitive_invention/v837w/diagnostics/decision_state.json').read_text())
    if w.get('neutral_global_controller_allowed') is not True or w.get('authorized_v837x_mode')!=CONFIG['authorized_controller_mode']: raise SystemExit('V837x mode not authorized')
    if json.loads((ROOT/'experiments/v837_primitive_invention/audit/audit_results.json').read_text()).get('episodes_consumed')!=0: raise SystemExit('fresh audit consumed')

def _diag(model,task,val_seeds):
    eps=[task.generate(s,'validation') for s in val_seeds]; obs,lengths,targets=episodes_to_batch(eps); model.eval()
    with torch.no_grad(): pred,tr=model(obs,lengths,return_trace=True)
    mask=_active_mask(lengths,tr.states.shape[1]); st=tr.states[mask].reshape(-1,10,4); cand=tr.candidate_states[mask].reshape(-1,10,4); gates=tr.state_modulators[mask].reshape(-1,10).detach().cpu().numpy()
    # Gate columns are identical for X2/X2C by construction; retain all cells for transparent accounting.
    flat=gates.reshape(-1); changes=torch.linalg.vector_norm((tr.states-tr.candidate_states)[mask].reshape(-1,10,4),dim=-1).detach().cpu().numpy()
    return {'validation_success_recomputed':_success_rate(task,pred,targets),'global_scalar_mean':float(np.mean(flat)),'global_scalar_temporal_variance':float(_temporal_variance(tr.state_modulators[:,:,0,:],lengths)),'global_scalar_p10':float(np.quantile(flat,.1)),'global_scalar_p90':float(np.quantile(flat,.9)),'near_zero_fraction':float(np.mean(flat<=.05)),'near_one_fraction':float(np.mean(flat>=.95)),'state_norm':float(torch.linalg.vector_norm(st,dim=-1).mean()),'candidate_state_norm':float(torch.linalg.vector_norm(cand,dim=-1).mean()),'mean_state_candidate_distance':float(np.mean(changes))}

def _worker(c,f,r):
    _configure_torch(); task=task_by_name(f); tr=CONFIG['training']; train=list(range(tr['development_seed_range'][0],tr['development_seed_range'][1]+1)); val=list(range(tr['validation_seed_range'][0],tr['validation_seed_range'][1]+1)); seed=deterministic_int(tr['initialization_namespace'],f,r)
    res=train_sequence_model(model_factory=lambda:GlobalScalarNeutralModel(__import__('experiments.v837_primitive_invention.failures.run_blocker_diagnostic',fromlist=['high_capacity_generic_graph']).high_capacity_generic_graph(r),condition=c,authorized_mode=CONFIG['authorized_controller_mode']),task=task,train_seeds=train,validation_seeds=val,initialization_seed=seed,steps=tr['steps'],learning_rate=tr['learning_rate'],weight_decay=tr['weight_decay'],gradient_clip=tr['gradient_clip'],curve_steps=tuple(tr['curve_steps']))
    d=_diag(res.model,task,val); res.resources.forward_calls+=1; m=res.model
    return {'version':'V837x','condition':c,'authorized_controller_mode':CONFIG['authorized_controller_mode'],'family':f,'replicate_id':r,'development_success':res.development.success_rate,'validation_success':res.validation.success_rate,'development_loss':res.development.loss,'validation_loss':res.validation.loss,'capacity_demonstrated':capacity_demonstrated(res.development.success_rate,res.validation.success_rate),'loss_curve':res.learning_curve,'controller_input_dim':46 if c.startswith('X2') else (14 if c=='X1_local_scalar_carry' else 0),'controller_param_count':m.controller_param_count,'controller_macs':m.controller_macs,'base_macs':160,'total_macs':160+m.controller_macs,'parameter_count':m.parameter_count(),'parameter_bytes':m.parameter_bytes(),'carry':c in {'X1_local_scalar_carry','X2_global_scalar_carry'},'control_type':'global_scalar' if c.startswith('X2') else ('local_scalar' if c=='X1_local_scalar_carry' else 'none'),'diagnostics':d,'resources':res.resources.to_dict(),'processed_examples':tr['steps']*tr['train_episodes'],'unique_seed_defined_episode_policy':'same 3200 family/seed episodes reused across conditions and replicates','fresh_audit_consumed':False,'task_family_label_in_model_input':False,'gpu_seconds':0.0}

def _summary(rows,c):
    out={'families_passing':0,'family_validation_medians':{}}
    for f in FAMILIES:
        rr=[r for r in rows if r['condition']==c and r['family']==f]; d=float(np.median([r['development_success'] for r in rr])); v=float(np.median([r['validation_success'] for r in rr])); out['family_validation_medians'][f]=v; out['families_passing']+=int(capacity_demonstrated(d,v))
    return out

def main():
    _locks(); [(HERE/n).mkdir(exist_ok=True) for n in ('raw','diagnostics','plots')]; jobs=[(c,f,r) for c in CONDITIONS for f in FAMILIES for r in range(5)]; rows=[]
    with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as pool:
        fs={pool.submit(_worker,*j):j for j in jobs}
        for fut in as_completed(fs): row=fut.result(); rows.append(row); print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}",flush=True)
    rows.sort(key=lambda r:(r['condition'],r['family'],r['replicate_id'])); write_json(HERE/'raw/runs.json',{'rows':rows,'unique_seed_defined_episodes':3200,'reuse_policy':'paired reuse across conditions and replicates','fresh_audit_consumed':False})
    u=json.loads((ROOT/'experiments/v837_primitive_invention/v837u/results.json').read_text())['conditions']; anchors={'X0_historical_direct':'U0_historical_direct','X1_local_scalar_carry':'U2_dynamic_scalar_carry'}; drift={}; ok=True
    for x,old in anchors.items():
        s=_summary(rows,x); oldr=u[old]; fd={f:abs(s['family_validation_medians'][f]-oldr['family_results'][f]['validation']['median']) for f in FAMILIES}; drift[x]={'families_passing':s['families_passing'],'expected_families_passing':oldr['families_passing'],'family_median_drift':fd}; ok=ok and s['families_passing']==oldr['families_passing'] and max(fd.values())<=1e-12
    write_json(HERE/'diagnostics/anchor_guard.json',{'compatible':ok,'anchors':drift}); print(json.dumps({'anchor_guard':ok,'summaries':{c:_summary(rows,c) for c in CONDITIONS}},indent=2)); return 0 if ok else 2
if __name__=='__main__': raise SystemExit(main())
