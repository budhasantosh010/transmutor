from __future__ import annotations

import time
from collections import defaultdict

from .pairwise_replay import TEST, aggregate, evaluate_pair
from .utils import HERE, read_json, sha256_file, write_json


def _fit_bundles(pair_index:int, family:str)->dict:
    fits=read_json(HERE/'raw/adapter_fits.json')['rows']
    row=next(r for r in fits if int(r['pair_index'])==int(pair_index))
    fam=row['families'][family]
    return {'same':fam['same'],'different':fam['different']}


def _evaluate_track(name:str, selected:dict|None, pairs:list[dict])->dict:
    if selected is None:
        return {'track':name,'run':False,'reason':'NO_SELECT_PASS','classes':[],'pooled':aggregate([],True)}
    rows=[]
    for pi,pair in enumerate(pairs):
        if name=='CAUSAL_TRACK' and not pair.get('primary_causal'): continue
        rows.append(evaluate_pair(pair,_fit_bundles(pi,selected['family']),selected['scope'],TEST,'validation'))
    by=defaultdict(list)
    for row in rows:by[row['class_id']].append(row)
    power={c['class_id']:c for c in read_json(HERE/'diagnostics/statistical_power.json')['classes']}
    classes=[]
    for cid,crows in sorted(by.items()):
        p=power[cid]; gate=aggregate(crows,True,bool(p['strong_claim_powered']))
        if not p['strong_claim_powered'] and gate.get('effect_gate_pass'):
            qualifier='INTERFACE_RESCUE_EFFECT_PRESENT_BUT_UNDERPOWERED'
        elif gate.get('pass'): qualifier='PAIRWISE_BOUNDARY_ALIGNED'
        else: qualifier='PAIRWISE_BOUNDARY_ALIGNMENT_NOT_ESTABLISHED'
        classes.append({'class_id':cid,'N':len(crows),'minimum_possible_p':p['minimum_possible_p'],'strong_claim_powered':p['strong_claim_powered'],'metrics':gate,'qualifier':qualifier,'rows':crows})
    pooled=aggregate(rows,True,True)
    # Pooled/global is descriptive; report stratifications rather than authorizing a primitive.
    fam={};eng={}
    for key in sorted({r['family'] for r in rows}): fam[key]=aggregate([r for r in rows if r['family']==key],False)
    for key in sorted({r['engine'] for r in rows}): eng[key]=aggregate([r for r in rows if r['engine']==key],False)
    return {'track':name,'run':True,'selected_config':selected,'classes':classes,'pooled':pooled,'family_stratified':fam,'engine_stratified':eng,'row_count':len(rows)}


def run_pairwise_test()->dict:
    selected_path=HERE/'raw/selected_interface_configs.json'
    if not selected_path.is_file(): raise RuntimeError('AL6 blocked: selected interface not frozen')
    frozen_sha=sha256_file(selected_path);selected=read_json(selected_path);pairs=read_json(HERE/'raw/frozen_pairs.json')['pairs'];t0=time.perf_counter();cpu0=time.process_time()
    global_result=_evaluate_track('GLOBAL_TRACK',selected.get('global_track'),pairs)
    causal_result=_evaluate_track('CAUSAL_TRACK',selected.get('causal_track'),pairs)
    if sha256_file(selected_path)!=frozen_sha: raise RuntimeError('AL6_SELECTED_CONFIG_MUTATED')
    payload={'version':'V837al','stage':'AL6','align_test_seeds':[20000,20127],'adapter_refit_on_test':False,'selected_interface_sha256':frozen_sha,'global_track':global_result,'causal_track':causal_result,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.perf_counter()-t0}
    write_json(HERE/'raw/pairwise_test_results.json',payload);write_json(HERE/'diagnostics/pairwise_interchangeability.json',payload);return payload

if __name__=='__main__':run_pairwise_test()
