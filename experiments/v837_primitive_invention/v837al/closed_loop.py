from __future__ import annotations

import time
import numpy as np
import torch
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from .aligned_transplant import run_aligned_transplant
from .canonical_interface import _anchors_index, _fit_to_anchor, _compose
from .heldout_pairwise import _fit_bundles
from .interface_ports import load_model
from .utils import HERE, read_json, write_json

TEST=list(range(20000,20128))


def _success(task,pred,target):return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),target.tolist())]))

def _causal_class_pass(payload):
    track=payload.get('causal_track',{})
    for c in track.get('classes',[]):
        if c.get('metrics',{}).get('pass'):return c['class_id']
    return None


def _original(model,obs,lengths):
    model.eval();
    with torch.no_grad():return model(obs,lengths,return_trace=True)


def run_closed_loop():
    t0=time.perf_counter();cpu0=time.process_time();pairwise=read_json(HERE/'raw/pairwise_test_results.json');cid=_causal_class_pass(pairwise)
    if cid is None:
        payload={'version':'V837al','stage':'AL8','run':False,'reason':'NO_POWERED_CAUSAL_PAIRWISE_PASS','classes':[],'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.perf_counter()-t0};write_json(HERE/'raw/closed_loop_results.json',payload);write_json(HERE/'diagnostics/closed_loop_interchangeability.json',payload);return payload
    selected=read_json(HERE/'raw/selected_interface_configs.json')['causal_track'];pairs=read_json(HERE/'raw/frozen_pairs.json')['pairs'];target=[(i,p) for i,p in enumerate(pairs) if p['class_id']==cid and p.get('primary_causal')]
    canonical=read_json(HERE/'raw/canonical_test_results.json') if (HERE/'raw/canonical_test_results.json').is_file() else {};canonical_pass=_causal_class_pass(canonical) is not None
    mode='CANONICAL' if canonical_pass else 'PAIRWISE';rows=[];self_max={'prediction':0.0,'state':0.0,'message':0.0};aligned_self_max={'prediction':0.0,'state':0.0,'message':0.0};anchors=_anchors_index();aligned_self_tolerance=1e-5 if selected['family'] in {'SIGNED_PERMUTATION','RIGID_AFFINE'} else 1e-4
    for pi,pair in target:
        rec=pair['recipient'];same=pair['same_class_donor'];diff=pair['different_class_donor'];model,_=load_model(rec['organism_id']);task=task_by_name(rec['family']);eps=[task.generate(s,'validation') for s in TEST];obs,lengths,targets=episodes_to_batch(eps)
        with torch.no_grad():original_pred,original_trace=model(obs,lengths,return_trace=True)
        rp=PortedPrimitive(model,rec['nodes']);self_pred,self_trace=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=rp,bundle={'ports':{}},scope='000000',return_trace=True)
        idx=list(rec['nodes']);self_max['prediction']=max(self_max['prediction'],float(torch.max(torch.abs(self_pred-original_pred)).item()));self_max['state']=max(self_max['state'],float(torch.max(torch.abs(self_trace.states[:,:,idx,:]-original_trace.states[:,:,idx,:])).item()));self_max['message']=max(self_max['message'],float(torch.max(torch.abs(self_trace.messages[:,:,idx,:]-original_trace.messages[:,:,idx,:])).item()))
        if max(self_max.values())>1e-6:raise RuntimeError('ALIGNED_TRANSPLANT_IMPLEMENTATION_INVALID')
        anchor=anchors.get((cid,rec['family']))
        if anchor is None:raise RuntimeError('ALIGNED_SELF_CANONICAL_ANCHOR_MISSING')
        rec_to_c=_fit_to_anchor(rec,anchor,selected['family']);roundtrip=_compose(rec_to_c,rec_to_c)
        aligned_self_pred,aligned_self_trace=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=rp,bundle=roundtrip,scope=selected['scope'],return_trace=True)
        aligned_self_max['prediction']=max(aligned_self_max['prediction'],float(torch.max(torch.abs(aligned_self_pred-original_pred)).item()));aligned_self_max['state']=max(aligned_self_max['state'],float(torch.max(torch.abs(aligned_self_trace.states[:,:,idx,:]-original_trace.states[:,:,idx,:])).item()));aligned_self_max['message']=max(aligned_self_max['message'],float(torch.max(torch.abs(aligned_self_trace.messages[:,:,idx,:]-original_trace.messages[:,:,idx,:])).item()))
        if max(aligned_self_max.values())>aligned_self_tolerance:raise RuntimeError('ALIGNED_SELF_REALITY_GATE_FAILED')
        # Direct pairwise bundles are always available. Canonical mode uses the exact canonical-composed bundles saved by AL7.
        if mode=='CANONICAL':
            bundles_index=read_json(HERE/'raw/canonical_pair_bundles.json')['bundles'];entry=next(x for x in bundles_index if int(x['pair_index'])==pi and x['track']=='CAUSAL_TRACK');bundles={'same':entry['same'],'different':entry['different']}
        else:bundles=_fit_bundles(pi,selected['family'])
        sm,_=load_model(same['organism_id']);dm,_=load_model(diff['organism_id']);sp=PortedPrimitive(sm,same['nodes']);dp=PortedPrimitive(dm,diff['nodes']);rand=randomized_primitive(sp,int(same['organism_id'][:12],16)%2_000_000_000)
        with torch.no_grad():
            same_pred,_=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=sp,bundle=bundles['same'],scope=selected['scope'],return_trace=True)
            diff_pred,_=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=dp,bundle=bundles['different'],scope=selected['scope'],return_trace=True)
            rand_pred,_=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=rand,bundle=bundles['same'],scope=selected['scope'],return_trace=True)
        base=_success(task,original_pred,targets);ss=_success(task,same_pred,targets);ds=_success(task,diff_pred,targets);rs=_success(task,rand_pred,targets)
        rows.append({'pair_index':pi,'class_id':cid,'recipient':rec,'same_class_donor':same,'different_class_donor':diff,'original_success':base,'same_success':ss,'different_success':ds,'randomized_success':rs,'same_success_drop':base-ss,'same_competent':ss>=0.85})
    drops=np.asarray([r['same_success_drop'] for r in rows]);same=np.asarray([r['same_success'] for r in rows]);diff=np.asarray([r['different_success'] for r in rows]);rand=np.asarray([r['randomized_success'] for r in rows]);ret=float(np.mean([r['same_competent'] for r in rows]));imp=np.minimum(same-diff,same-rand)
    from experiments.v837_primitive_invention.v837ak.utils import one_sided_paired_sign_permutation
    p=one_sided_paired_sign_permutation(imp.tolist());gate=float(np.median(drops))<=0.05 and ret>=0.60 and float(np.median(same))>=float(np.median(diff))+0.10 and float(np.median(same))>=float(np.median(rand))+0.10 and p<=0.01 and len(rows)>=7
    metrics={'pass':bool(gate),'N':len(rows),'median_same_success_drop':float(np.median(drops)),'competence_retention_fraction':ret,'median_same_success':float(np.median(same)),'median_different_success':float(np.median(diff)),'median_randomized_success':float(np.median(rand)),'p':p}
    payload={'version':'V837al','stage':'AL8','run':True,'mode':mode,'class_id':cid,'self_transplant_reality':{**self_max,'tolerance':1e-6,'pass':max(self_max.values())<=1e-6},'aligned_self_reality':{**aligned_self_max,'tolerance':aligned_self_tolerance,'transform_family':selected['family'],'pass':max(aligned_self_max.values())<=aligned_self_tolerance},'metrics':metrics,'rows':rows,'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.perf_counter()-t0}
    write_json(HERE/'raw/closed_loop_results.json',payload);write_json(HERE/'diagnostics/closed_loop_interchangeability.json',payload);return payload

if __name__=='__main__':run_closed_loop()
