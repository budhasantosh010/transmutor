from __future__ import annotations

import numpy as np
import torch
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.utils import one_sided_paired_sign_permutation
from .adapter_fit import _fit_bundle
from .aligned_transplant import run_aligned_transplant
from .canonical_interface import _anchors_index,_compose,_fit_to_anchor
from .interface_ports import collect_interface_trace,load_model
from .pairwise_replay import TEST,aggregate,evaluate_pair
from .utils import HERE,read_json,write_json

BUDGETS=(8,16,32,64)


def _success(task,pred,target):return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),target.tolist())]))


def _direct_bundles(pair,family,fit_seeds):
    rec,same,diff=pair['recipient'],pair['same_class_donor'],pair['different_class_donor']
    rt=collect_interface_trace(rec['organism_id'],rec['nodes'],fit_seeds,'development');st=collect_interface_trace(same['organism_id'],same['nodes'],fit_seeds,'development');dt=collect_interface_trace(diff['organism_id'],diff['nodes'],fit_seeds,'development')
    return {'same':_fit_bundle(rt,st,rec['nodes'],same['nodes'],family),'different':_fit_bundle(rt,dt,rec['nodes'],diff['nodes'],family)}


def _canonical_bundles(pair,family,fit_seeds):
    anchors=_anchors_index();cid=pair['class_id'];rec,same,diff=pair['recipient'],pair['same_class_donor'],pair['different_class_donor'];anchor=anchors[(cid,rec['family'])]
    rc=_fit_to_anchor(rec,anchor,family,fit_seeds);sc=_fit_to_anchor(same,anchor,family,fit_seeds);dc=_fit_to_anchor(diff,anchor,family,fit_seeds)
    return {'same':_compose(rc,sc),'different':_compose(rc,dc)}


def _closed_loop_rows(indexed_pairs,selected,mode,fit_seeds):
    rows=[]
    for _,pair in indexed_pairs:
        rec,same,diff=pair['recipient'],pair['same_class_donor'],pair['different_class_donor'];bundles=_canonical_bundles(pair,selected['family'],fit_seeds) if mode=='CANONICAL' else _direct_bundles(pair,selected['family'],fit_seeds)
        model,_=load_model(rec['organism_id']);task=task_by_name(rec['family']);eps=[task.generate(s,'validation') for s in TEST];obs,lengths,targets=episodes_to_batch(eps)
        with torch.no_grad():orig=model(obs,lengths)
        sm,_=load_model(same['organism_id']);dm,_=load_model(diff['organism_id']);sp=PortedPrimitive(sm,same['nodes']);dp=PortedPrimitive(dm,diff['nodes']);rand=randomized_primitive(sp,int(same['organism_id'][:12],16)%2_000_000_000)
        with torch.no_grad():
            same_p=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=sp,bundle=bundles['same'],scope=selected['scope'])
            diff_p=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=dp,bundle=bundles['different'],scope=selected['scope'])
            rand_p=run_aligned_transplant(model,obs,lengths,recipient_nodes=rec['nodes'],donor=rand,bundle=bundles['same'],scope=selected['scope'])
        base=_success(task,orig,targets);ss=_success(task,same_p,targets);ds=_success(task,diff_p,targets);rs=_success(task,rand_p,targets);rows.append({'original':base,'same':ss,'different':ds,'randomized':rs,'drop':base-ss,'competent':ss>=0.85})
    drops=np.asarray([r['drop'] for r in rows]);same=np.asarray([r['same'] for r in rows]);diff=np.asarray([r['different'] for r in rows]);rand=np.asarray([r['randomized'] for r in rows]);imp=np.minimum(same-diff,same-rand);p=one_sided_paired_sign_permutation(imp.tolist());gate=float(np.median(drops))<=0.05 and float(np.mean([r['competent'] for r in rows]))>=0.60 and float(np.median(same))>=float(np.median(diff))+0.10 and float(np.median(same))>=float(np.median(rand))+0.10 and p<=0.01 and len(rows)>=7
    return {'pass':bool(gate),'N':len(rows),'median_same_success_drop':float(np.median(drops)),'competence_retention_fraction':float(np.mean([r['competent'] for r in rows])),'median_same_success':float(np.median(same)),'median_different_success':float(np.median(diff)),'median_randomized_success':float(np.median(rand)),'p':p}


def run_alignment_data_frontier():
    closed=read_json(HERE/'raw/closed_loop_results.json')
    if not closed.get('run') or not closed.get('metrics',{}).get('pass'):
        payload={'version':'V837al','stage':'AL9','run':False,'reason':'CLOSED_LOOP_PASS_REQUIRED','budgets':[]};write_json(HERE/'raw/alignment_data_frontier.json',payload);return payload
    selected=read_json(HERE/'raw/selected_interface_configs.json')['causal_track'];cid=closed['class_id'];pairs=read_json(HERE/'raw/frozen_pairs.json')['pairs'];indexed=[(i,p) for i,p in enumerate(pairs) if p['class_id']==cid and p.get('primary_causal')];rows=[]
    for n in BUDGETS:
        fit_seeds=list(range(10000,10000+n));boundary=[]
        for _,pair in indexed:
            bundles=_canonical_bundles(pair,selected['family'],fit_seeds) if closed['mode']=='CANONICAL' else _direct_bundles(pair,selected['family'],fit_seeds);boundary.append(evaluate_pair(pair,bundles,selected['scope'],TEST,'validation'))
        bg=aggregate(boundary,True,True);cg=_closed_loop_rows(indexed,selected,closed['mode'],fit_seeds);rows.append({'episodes':n,'boundary':bg,'closed_loop':cg,'preserves_gates':bool(bg['pass'] and cg['pass'])})
    passing=[r['episodes'] for r in rows if r['preserves_gates']];payload={'version':'V837al','stage':'AL9','run':True,'frozen_transform':selected['family'],'frozen_scope':selected['scope'],'mode':closed['mode'],'rows':rows,'minimum_tested_alignment_episodes':min(passing) if passing else None};write_json(HERE/'raw/alignment_data_frontier.json',payload);return payload

if __name__=='__main__':run_alignment_data_frontier()
