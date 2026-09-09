from __future__ import annotations

import time
import numpy as np

from .adapter_families import fit_gate_map, fit_vector_map
from .interface_ports import collect_interface_trace, port_matrices, trace_integrity
from .pairwise_replay import TEST, aggregate, evaluate_pair
from .utils import HERE, read_json, sha256_file, write_json, seeds

FIT=seeds(10000,10063)


def _identity_vector_map(family:str,d:int)->dict:
    if family=='SIGNED_PERMUTATION':
        return {'kind':family,'perm':list(range(d)),'signs':[1.0]*d,'dimension':d,'valid':True,'canonical_identity':True}
    if family=='DIAGONAL_AFFINE':
        return {'kind':family,'a':[1.0]*d,'b':[0.0]*d,'dimension':d,'valid':True,'canonical_identity':True}
    if family=='RIGID_AFFINE':
        return {'kind':family,'q':np.eye(d).tolist(),'mu_x':[0.0]*d,'mu_y':[0.0]*d,'dimension':d,'valid':True,'canonical_identity':True}
    if family=='FULL_AFFINE':
        return {'kind':family,'A':np.eye(d).tolist(),'b':[0.0]*d,'ridge':1e-6,'dimension':d,'valid':True,'canonical_identity':True}
    raise ValueError(family)


def _identity_bundle(family:str,k:int)->dict:
    ports={
        'state':[_identity_vector_map(family,4) for _ in range(k)],
        'external_messages':[_identity_vector_map(family,4) for _ in range(k)],
        'global_term':[_identity_vector_map(family,4) for _ in range(k)],
        'projected_input':[_identity_vector_map(family,6) for _ in range(k)],
        'output':[_identity_vector_map(family,4) for _ in range(k)],
        'gate':{'kind':'GATE_SIGN' if family=='SIGNED_PERMUTATION' else 'GATE_LOGIT_AFFINE','a':1.0,'b':0.0,'valid':True,'canonical_identity':True},
    }
    return {'family':family,'ports':ports,'valid':True,'canonical_identity':True}


def _affine(m:dict)->tuple[np.ndarray,np.ndarray]:
    k=m['kind'];d=int(m.get('dimension',len(m.get('a',m.get('perm',[]))) or len(m.get('mu_x',[])) or len(m.get('b',[]))))
    if k=='SIGNED_PERMUTATION':
        A=np.zeros((d,d));
        for j,src in enumerate(m['perm']):A[src,j]=float(m['signs'][j])
        return A,np.zeros(d)
    if k=='DIAGONAL_AFFINE':return np.diag(np.asarray(m['a'],dtype=float)),np.asarray(m['b'],dtype=float)
    if k=='RIGID_AFFINE':
        A=np.asarray(m['q'],dtype=float);mx=np.asarray(m['mu_x'],dtype=float);my=np.asarray(m['mu_y'],dtype=float);return A,my-mx@A
    if k=='FULL_AFFINE':return np.asarray(m['A'],dtype=float),np.asarray(m['b'],dtype=float)
    raise ValueError(k)


def _pair_map(rec_to_c:dict, donor_to_c:dict)->dict:
    Ar,br=_affine(rec_to_c);Ad,bd=_affine(donor_to_c);inv=np.linalg.inv(Ad);A=Ar@inv;b=(br-bd)@inv
    return {'kind':'FULL_AFFINE','A':A.tolist(),'b':b.tolist(),'dimension':A.shape[0],'valid':True,'canonical_composition':True}


def _output_map(donor_to_c:dict, rec_to_c:dict)->dict:
    Ad,bd=_affine(donor_to_c);Ar,br=_affine(rec_to_c);inv=np.linalg.inv(Ar);A=Ad@inv;b=(bd-br)@inv
    return {'kind':'FULL_AFFINE','A':A.tolist(),'b':b.tolist(),'dimension':A.shape[0],'valid':True,'canonical_composition':True}


def _gate_pair(rec_to_c:dict, donor_to_c:dict)->dict:
    ad=float(donor_to_c['a']);
    if abs(ad)<1e-4:return {'kind':'GATE_LOGIT_AFFINE','a':0.0,'b':0.0,'valid':False}
    return {'kind':'GATE_LOGIT_AFFINE','a':float(rec_to_c['a'])/ad,'b':(float(rec_to_c['b'])-float(donor_to_c['b']))/ad,'valid':True,'canonical_composition':True}


def _fit_to_anchor(occ:dict, anchor:dict, family:str, fit_seeds=FIT)->dict:
    if occ['occurrence_id']==anchor['occurrence_id']:
        return _identity_bundle(family,len(occ['nodes']))
    ot=collect_interface_trace(occ['organism_id'],occ['nodes'],fit_seeds,'development');ct=collect_interface_trace(anchor['organism_id'],anchor['nodes'],fit_seeds,'development');trace_integrity(ot,ct);op=port_matrices(ot,occ['nodes']);cp=port_matrices(ct,anchor['nodes']);ports={}
    for port in ('state','external_messages','global_term','projected_input','output'):
        ports[port]=[fit_vector_map(x,y,family,require_invertible=(port=='state')) for x,y in zip(op[port],cp[port])]
    ports['gate']=fit_gate_map(op['gate'][0],cp['gate'][0],family)
    return {'family':family,'ports':ports,'valid':all(m.get('valid',True) for p in ports.values() for m in (p if isinstance(p,list) else [p]))}


def _compose(rec_c,don_c)->dict:
    ports={}
    for p in ('state','external_messages','global_term','projected_input'):
        ports[p]=[_pair_map(r,d) for r,d in zip(rec_c['ports'][p],don_c['ports'][p])]
    ports['output']=[_output_map(d,r) for r,d in zip(rec_c['ports']['output'],don_c['ports']['output'])]
    ports['gate']=_gate_pair(rec_c['ports']['gate'],don_c['ports']['gate'])
    return {'family':rec_c['family'],'ports':ports,'valid':rec_c['valid'] and don_c['valid'],'canonical_composed':True}


def _anchors_index():
    a=read_json(HERE/'raw/canonical_anchors.json')['anchors'];return {(x['class_id'],x['family']):x['canonical_occurrence'] for x in a}


def _run_track(name,selected,pairs):
    if selected is None:return {'track':name,'run':False,'reason':'NO_SELECTED_CONFIG','classes':[],'pooled':aggregate([],True),'row_count':0,'direct_pair_fit_used':False,'occurrence_to_canonical_only':True}
    anchors=_anchors_index();cache={};rows=[];bundle_records=[]
    def mapping(occ,cid):
        key=(occ['occurrence_id'],cid,occ['family'],selected['family'])
        if key not in cache:
            anchor=anchors.get((cid,occ['family']))
            if anchor is None:return None
            cache[key]=_fit_to_anchor(occ,anchor,selected['family'])
        return cache[key]
    for pi,pair in enumerate(pairs):
        if name=='CAUSAL_TRACK' and not pair.get('primary_causal'):continue
        cid=pair['class_id'];rec=pair['recipient'];same=pair['same_class_donor'];diff=pair['different_class_donor'];rc=mapping(rec,cid);sc=mapping(same,cid)
        if rc is None or sc is None:continue
        # Different-class control gets the same fitting opportunity to the recipient class canonical anchor.
        dc=mapping(diff,cid)
        if dc is None:
            anchor=anchors.get((cid,rec['family']))
            if anchor is None:continue
            dc=_fit_to_anchor(diff,anchor,selected['family']);cache[(diff['occurrence_id'],cid,diff['family'],selected['family'])]=dc
        bundles={'same':_compose(rc,sc),'different':_compose(rc,dc)}
        bundle_records.append({'pair_index':pi,'track':name,'class_id':cid,'recipient_occurrence_id':rec['occurrence_id'],'same_occurrence_id':same['occurrence_id'],'different_occurrence_id':diff['occurrence_id'],'same':bundles['same'],'different':bundles['different']})
        rows.append(evaluate_pair(pair,bundles,selected['scope'],TEST,'validation'))
    from collections import defaultdict
    by=defaultdict(list)
    for r in rows:by[r['class_id']].append(r)
    power={c['class_id']:c for c in read_json(HERE/'diagnostics/statistical_power.json')['classes']};classes=[]
    for cid,crows in sorted(by.items()):classes.append({'class_id':cid,'N':len(crows),'metrics':aggregate(crows,True,bool(power[cid]['strong_claim_powered'])),'rows':crows})
    return {'track':name,'run':True,'selected_config':selected,'classes':classes,'pooled':aggregate(rows,True,True),'direct_pair_fit_used':False,'occurrence_to_canonical_only':True,'row_count':len(rows),'bundle_records':bundle_records}


def run_canonical_test():
    t0=time.perf_counter();cpu0=time.process_time();selected_path=HERE/'raw/selected_interface_configs.json';frozen=sha256_file(selected_path);sel=read_json(selected_path);pairs=read_json(HERE/'raw/frozen_pairs.json')['pairs']
    global_result=_run_track('GLOBAL_TRACK',sel.get('global_track'),pairs);causal_result=_run_track('CAUSAL_TRACK',sel.get('causal_track'),pairs)
    bundle_records=list(global_result.pop('bundle_records',[]))+list(causal_result.pop('bundle_records',[]))
    payload={'version':'V837al','stage':'AL7','align_test_seeds':[20000,20127],'selected_interface_sha256':frozen,'global_track':global_result,'causal_track':causal_result,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.perf_counter()-t0}
    if sha256_file(selected_path)!=frozen:raise RuntimeError('AL7_SELECTED_CONFIG_MUTATED')
    write_json(HERE/'raw/canonical_pair_bundles.json',{'version':'V837al','fit_data':'ALIGN_FIT_ONLY','direct_pair_fit_used':False,'bundles':bundle_records});write_json(HERE/'raw/canonical_test_results.json',payload);write_json(HERE/'diagnostics/canonical_interface.json',payload);return payload

if __name__=='__main__':run_canonical_test()
