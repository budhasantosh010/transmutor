from __future__ import annotations

from pathlib import Path

from .utils import HERE, read_json, write_json


def _load(name,default):
    p=HERE/name
    return read_json(p) if p.is_file() else default


def compute_resource_accounting() -> dict:
    fits=_load('raw/discovery_backend_fits.json',{'rows':[]}).get('rows',[]);setp=_load('raw/discovery_setpoint_results.json',{'rows':[]}).get('rows',[]);quot=_load('raw/quotient_results.json',{'rows':[]}).get('rows',[]);dyn=_load('raw/commutativity_results.json',{'rows':[]}).get('rows',[]);held=_load('raw/heldout_calibration_frontier.json',{'rows':[]});heldrows=held.get('rows',[])
    svd=sum(1 if r.get('variant')!='B2_PHASE_K1' else 3 for r in fits if r.get('valid'))+sum(int(r.get('calibration_cost',{}).get('svd_count',0)) for r in heldrows)
    ridge=0
    phasefits=0
    for r in fits:
        comps=r.get('components',{});ridge+=2*len(comps);phasefits+=len(comps) if r.get('variant')!='B0_GLOBAL_K1' else 0
    ridge+=sum(int(r.get('calibration_cost',{}).get('ridge_solves',0)) for r in heldrows)
    # Exact low-level call counts are not retroactively inferable from PyTorch module hooks in this run; report conservative structural counts and mark them as derived.
    natural=sum(64 for _ in fits);set_calls=sum(int(r.get('sample_count',0)) for r in setp);random_calls=32*set_calls;quot_calls=sum(int(r.get('metrics',{}).get('pair_count',0)) for r in quot);dyn_calls=sum(int(r.get('select',{}).get('interventional',{}).get('metrics',{}).get('interventions',0)) for r in dyn);held_calls=sum(int(r.get('calibration_n',0)) for r in heldrows)
    payload={'version':'V837ao','new_model_fits':0,'model_optimizer_steps':0,'backend_gradient_steps':0,'model_training_examples':0,'natural_forward_calls_derived':natural,'set_intervention_forward_calls_derived':set_calls,'random_control_calls_derived':random_calls,'quotient_calls_derived':quot_calls,'dynamics_rollout_calls_derived':dyn_calls,'heldout_calibration_pairs_consumed':held_calls,'historical_robustness_calls':0,'svd_count':svd,'ridge_solve_count':ridge,'phase_specific_fits':phasefits,'law_recovery_fits':0,'cpu_seconds':None,'cpu_seconds_exact':False,'wall_seconds':None,'gpu_seconds':0.0,'cache_hits':None,'cache_misses':None,'backend_storage_bytes':sum(int(r.get('calibration_cost',{}).get('backend_stored_bytes',0)) for r in heldrows),'large_persistent_storage_tested':False,'notes':['No model training occurred.','Forward-call counters are structural derived counts where exact hook-level counters were not persisted.','No GPU execution was required.']}
    write_json(HERE/'diagnostics/resource_accounting.json',payload);write_json(HERE/'v837ao_resource_accounting.json',payload);write_json(HERE.parent/'v837ao_resource_accounting.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(compute_resource_accounting(),indent=2))
