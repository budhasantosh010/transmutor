from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency import CONFIG, FAMILIES, summarize_rows

HERE=Path(__file__).resolve().parent
REGIMES=("1x","2x","4x")
DEV_COUNTS={"1x":128,"2x":256,"4x":512}
TOTAL_UNIQUE={"1x":1280,"2x":1920,"4x":3200}
AF1D_MACS=1206
GRU_MACS=777


def load(path:Path)->dict: return json.loads(path.read_text(encoding='utf-8'))

def rows_by_regime()->dict[str,list[dict]]:
    return {
        "1x":load(HERE/'raw/ai1_runs.json')['rows'],
        "2x":load(HERE/'raw/ai2_runs.json')['rows'],
        "4x":load(HERE/'raw/ai4_reused_anchor.json')['rows'],
    }


def summaries(rows)->dict:
    out={}
    for regime in REGIMES:
        s=summarize_rows(rows[regime])
        s['development_episodes_per_family']=DEV_COUNTS[regime]
        s['validation_episodes_per_family']=128
        s['total_unique_family_seed_episodes']=TOTAL_UNIQUE[regime]
        out[regime]=s
    return out


def replicate_stability(rows)->dict:
    out={}
    for regime in REGIMES:
        out[regime]={}
        for family in FAMILIES:
            rr=sorted([r for r in rows[regime] if r['family']==family],key=lambda r:int(r['replicate_id']))
            vals=np.asarray([float(r['validation_success']) for r in rr],dtype=float)
            out[regime][family]={
                'replicate_validation_scores':[float(v) for v in vals], 'median':float(np.median(vals)),
                'minimum':float(np.min(vals)),'maximum':float(np.max(vals)),'std':float(np.std(vals)),
                'replicates_validation_ge_0_85':int(np.sum(vals>=0.85)),
            }
    return out


def family_trajectories(summary)->dict:
    return {family:{regime:{
        'development_median':summary[regime]['family_results'][family]['development_median'],
        'validation_median':summary[regime]['family_results'][family]['validation_median'],
        'pass':summary[regime]['family_results'][family]['pass'],
    } for regime in REGIMES} for family in FAMILIES}


def _stats(values)->dict:
    a=np.asarray(values,dtype=float)
    return {'mean':float(np.mean(a)),'median':float(np.median(a)),'p10':float(np.percentile(a,10)),'p90':float(np.percentile(a,90))}


def paired_effects(rows)->dict:
    maps={reg:{(r['family'],int(r['replicate_id'])):float(r['validation_success']) for r in rows[reg]} for reg in REGIMES}
    out={}
    for a,b,name in (("1x","2x","AI2_minus_AI1"),("2x","4x","AI4_minus_AI2"),("1x","4x","AI4_minus_AI1")):
        values=[]; per_family={}; pairs=[]
        for family in FAMILIES:
            fv=[]
            for rep in range(5):
                delta=maps[b][(family,rep)]-maps[a][(family,rep)]
                values.append(delta); fv.append(delta); pairs.append({'family':family,'replicate_id':rep,'delta':delta})
            per_family[family]=_stats(fv)
        out[name]={'overall':_stats(values),'per_family':per_family,'pairs':pairs}
    return out


def projection_specialization(rows)->dict:
    out={}
    for regime in REGIMES:
        grouped=defaultdict(list)
        for row in rows[regime]:
            for point in row.get('projection_trajectory',[]): grouped[int(point['step'])].append(point['projection'])
        timeline={}
        for step,ps in sorted(grouped.items()):
            des=[p for p in ps if p.get('mode')=='deshared']
            if not des: continue
            timeline[str(step)]={
                'fits':len(des),
                'pairwise_frobenius_distance_median':float(np.median([p['pairwise_weight_distance_median'] for p in des])),
                'pairwise_cosine_median':float(np.median([p['pairwise_weight_cosine_median'] for p in des])),
                'mean_drift_from_shared_initialization_median':float(np.median([p['projection_divergence_from_shared_initialization_mean'] for p in des])),
                'projection_frobenius_norm_median':float(np.median([np.mean([q['frobenius_norm'] for q in p['per_cell']]) for p in des])),
                'projection_condition_number_median':float(np.median([np.median([q['condition_number'] for q in p['per_cell']]) for p in des])),
                'projection_bias_drift_median':float(np.median([np.median([q['bias_drift'] for q in p['per_cell']]) for p in des])),
                'singular_values_median':[float(x) for x in np.median(np.asarray([[q['singular_values'] for q in p['per_cell']] for p in des],dtype=float),axis=(0,1))],
            }
        out[regime]={
            'timeline':timeline,
            'checkpoint_note':'AI4 uses the historical V837af checkpoints actually stored in its accepted rows; no 4x rerun was performed.' if regime=='4x' else 'V837ai checkpoints 0,32,64,96,128,160,192.',
        }
    return out


def effective_input_maps(rows)->dict:
    out={}
    for regime in REGIMES:
        rr=rows[regime]
        e=[r['diagnostics']['effective_input_maps'] for r in rr]
        c=[r['diagnostics']['candidate_input'] for r in rr]
        out[regime]={
            'pairwise_distance_median':float(np.median([x['pairwise_distance_median'] for x in e])),
            'pairwise_cosine_median':float(np.median([x['pairwise_cosine_median'] for x in e])),
            'candidate_input_term_norm_median':float(np.median([x['mean_input_term_norm'] for x in c])),
            'candidate_input_temporal_variance_median':float(np.median([x['mean_temporal_variance'] for x in c])),
            'candidate_visible_input_jacobian_frobenius_median':float(np.median([x['mean_candidate_visible_input_jacobian_frobenius'] for x in c])),
        }
    return out


def message_dependence(rows)->dict:
    out={}
    for regime in REGIMES:
        rr=rows[regime]
        out[regime]={
            'success_drop_median':float(np.median([r['diagnostics']['message_dependency']['success_drop'] for r in rr])),
            'mean_abs_prediction_delta_median':float(np.median([r['diagnostics']['message_dependency']['mean_abs_prediction_delta'] for r in rr])),
        }
    return out


def controller_dynamics(rows)->dict:
    out={}
    for regime in ('1x','2x'):
        rr=[r['diagnostics']['controller'] for r in rows[regime]]
        out[regime]={key:float(np.median([r[key] for r in rr])) for key in ('gate_mean','gate_median','temporal_variance','p10','p90','carry_fraction','rewrite_fraction')}
    out['4x']={
        'available':False,
        'reason':'Accepted V837af AF1D raw rows do not contain global-gate summary statistics. V837ai does not rerun the 4x anchor merely to add this descriptive diagnostic.'
    }
    return out


def active_training_timesteps(multiplier:int)->int:
    seeds=list(range(10000,10000+128*multiplier))
    total=0
    for family in FAMILIES:
        task=task_by_name(family)
        per_fit=sum(len(task.generate(seed,'development').observations) for seed in seeds)
        total += per_fit*5*192
    return int(total)


def compute_efficiency(rows,summary)->dict:
    new_rows=rows['1x']+rows['2x']; reused=rows['4x']
    def resource_sum(rr,key): return sum(float(r['resources'][key]) for r in rr)
    new={
        'model_fits':len(new_rows),'optimizer_steps':sum(int(r['resources']['optimizer_steps']) for r in new_rows),
        'processed_training_examples':sum(int(r['processed_examples']) for r in new_rows),
        'environment_interactions':sum(int(r['resources']['environment_steps']) for r in new_rows),
        'forward_calls':sum(int(r['resources']['forward_calls']) for r in new_rows),
        'backward_calls':sum(int(r['resources']['optimizer_steps']) for r in new_rows),
        'cpu_seconds_worker_sum':resource_sum(new_rows,'cpu_seconds'),'wall_seconds_worker_sum':resource_sum(new_rows,'wall_seconds'),'gpu_seconds':0.0,
        'modeled_active_training_timesteps':sum(int(r['modeled_active_training_timesteps']) for r in new_rows),
        'modeled_training_recurrent_mac_volume':sum(int(r['modeled_training_recurrent_mac_volume']) for r in new_rows),
    }
    historical={
        'model_fits':len(reused),'optimizer_steps':sum(int(r['resources']['optimizer_steps']) for r in reused),
        'processed_training_examples':sum(int(r['processed_examples']) for r in reused),
        'environment_interactions':sum(int(r['resources']['environment_steps']) for r in reused),
        'forward_calls':sum(int(r['resources']['forward_calls']) for r in reused),
        'backward_calls':sum(int(r['resources']['optimizer_steps']) for r in reused),
        'cpu_seconds_worker_sum':resource_sum(reused,'cpu_seconds'),'wall_seconds_worker_sum':resource_sum(reused,'wall_seconds'),'gpu_seconds':0.0,
        'note':'historical evidence only; not newly consumed by V837ai',
    }
    modeled={}
    for regime,m in (("1x",1),("2x",2),("4x",4)):
        timesteps=active_training_timesteps(m)
        modeled[regime]={
            'af1d_active_training_timesteps':timesteps,
            'af1d_macs_per_timestep':AF1D_MACS,
            'af1d_modeled_recurrent_mac_volume':int(timesteps*AF1D_MACS),
            'historical_gru_macs_per_timestep':GRU_MACS,
            'historical_gru_modeled_recurrent_mac_volume_same_episode_protocol':int(timesteps*GRU_MACS),
            'families_passing_af1d':summary[regime]['families_passing'],
        }
    return {
        'new_execution_resources':new,'historical_reused_ai4_resources':historical,
        'new_model_fits':50,'reused_accepted_4x_fits':25,'total_evidence_rows':75,
        'new_optimizer_steps_expected':9600,'historical_referenced_optimizer_steps':4800,
        'new_processed_examples_expected':1843200,'historical_referenced_processed_examples':2457600,
        'union_unique_task_episodes':3200,'active_parameters':1643,'projection_parameters':420,'af1d_macs_per_timestep':1206,
        'historical_gru_parameters':875,'af1d_to_gru_parameter_ratio':1643/875,
        'modeled_recurrent_mac_volume':modeled,
        'modeled_mac_semantics':'matrix MAC volume derived from actual active training sequence timesteps; not measured energy',
    }


def make_decision(summary)->dict:
    counts={reg:int(summary[reg]['families_passing']) for reg in REGIMES}
    monotonic=counts['1x']<=counts['2x']<=counts['4x']
    minimum=None; claim=False; reduction=None
    if not monotonic:
        diagnosis='NONMONOTONIC_AF1D_DATA_SCALING'; rec=4; structural=True
    elif counts['4x']<4:
        raise RuntimeError('AF1D_4X_REANALYSIS_MISMATCH')
    elif counts['1x']>=4 and counts['2x']>=4:
        diagnosis='AF1D_1X_REPRESENTATION_ADEQUATE'; minimum=1; claim=True; reduction=4.0; rec=1; structural=True
    elif counts['1x']<4 and counts['2x']>=4:
        diagnosis='AF1D_2X_REPRESENTATION_ADEQUATE'; minimum=2; claim=True; reduction=2.0; rec=2; structural=True
    else:
        diagnosis='AF1D_REQUIRES_4X_UNIQUE_DATA'; minimum=4; claim=True; reduction=1.0; rec=4; structural=True
    qualifiers=[]
    if diagnosis=='AF1D_1X_REPRESENTATION_ADEQUATE': qualifiers=['LOWER_UNIQUE_DATA_REQUIREMENT_THAN_HISTORICAL_GRU']
    elif diagnosis=='AF1D_2X_REPRESENTATION_ADEQUATE': qualifiers=['TWO_FOLD_UNIQUE_DATA_REDUCTION_VS_HISTORICAL_GRU']
    elif diagnosis=='AF1D_REQUIRES_4X_UNIQUE_DATA': qualifiers=['SAMPLE_EFFICIENCY_THRESHOLD_NOT_IMPROVED_VS_HISTORICAL_GRU']
    total_ratio=None if minimum is None else 3200/TOTAL_UNIQUE[f'{minimum}x']
    return {
        'version':'V837ai','architecture':'AF1D_deshared_candidate_input_factorization','architecture_frozen':True,'ai4_anchor_reused':True,
        'families_passing':counts,'minimum_tested_sufficient_multiplier':minimum,'pass_count_monotonic':bool(monotonic),
        'historical_gru_minimum_tested_multiplier':4,'unique_development_reduction_vs_gru':reduction,
        'total_unique_episode_reduction_vs_gru_4x_threshold':total_ratio,
        'diagnosis':diagnosis,'diagnosis_qualifiers':qualifiers,'sample_efficiency_claim_allowed':bool(claim),
        'representation_adequacy_confirmed':True,'recommended_structural_search_multiplier':rec,
        'structural_search_recovery_allowed':bool(structural),'primitive_mining_allowed':False,'fresh_audit_consumed':False,
        'primitives_promoted':0,'large_persistent_storage_tested':False,'v837ag_run':False,'v837ah_run':False,'v838_started':False,
        'next_program':'V837aj_STRUCTURAL_SEARCH_RECOVERY',
    }


def plots(summary,proj,effective,message,controller,compute,historical)->None:
    d=HERE/'plots'; d.mkdir(exist_ok=True)
    x=np.asarray([128,256,512]); af=np.asarray([summary[r]['families_passing'] for r in REGIMES]); gru=np.asarray([2,3,5]); neutral=np.asarray([1,1,2])
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(x,af,marker='o',label='AF1D'); ax.plot(x,gru,marker='o',label='historical GRU'); ax.plot(x,neutral,marker='o',label='historical neutral'); ax.axhline(4,linestyle='--'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('families passing'); ax.set_ylim(0,5.2); ax.legend(); fig.tight_layout(); fig.savefig(d/'families_passing_vs_unique_data.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));
    for family in FAMILIES: ax.plot(x,[summary[r]['family_results'][family]['validation_median'] for r in REGIMES],marker='o',label=family)
    ax.axhline(.85,linestyle='--'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('validation median'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(d/'family_scores_vs_unique_data.png',dpi=160); plt.close(fig)
    for name,other,label in (("af1d_vs_historical_gru_data_frontier.png",gru,'historical GRU'),("af1d_vs_historical_neutral_frontier.png",neutral,'historical neutral')):
        fig,ax=plt.subplots(figsize=(6.5,4)); ax.plot(x,af,marker='o',label='AF1D'); ax.plot(x,other,marker='o',label=label); ax.axhline(4,linestyle='--'); ax.set_ylim(0,5.2); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('families passing'); ax.legend(); fig.tight_layout(); fig.savefig(d/name,dpi=160); plt.close(fig)
    final_dist=[proj[r]['timeline'].get('192',{}).get('pairwise_frobenius_distance_median',np.nan) for r in REGIMES]
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,final_dist,marker='o'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('projection pairwise Frobenius distance'); fig.tight_layout(); fig.savefig(d/'projection_divergence_vs_unique_data.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,[effective[r]['pairwise_distance_median'] for r in REGIMES],marker='o'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('effective input-map pairwise distance'); fig.tight_layout(); fig.savefig(d/'effective_input_map_divergence_vs_data.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); cx=[128,256]; ax.plot(cx,[controller[r]['carry_fraction'] for r in ('1x','2x')],marker='o',label='carry'); ax.plot(cx,[controller[r]['rewrite_fraction'] for r in ('1x','2x')],marker='o',label='rewrite'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('gate fraction'); ax.legend(); fig.tight_layout(); fig.savefig(d/'controller_dynamics_vs_data.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,[message[r]['success_drop_median'] for r in REGIMES],marker='o'); ax.set_xlabel('unique development episodes/family'); ax.set_ylabel('median validation success drop without messages'); fig.tight_layout(); fig.savefig(d/'message_dependence_vs_data.png',dpi=160); plt.close(fig)
    proc=np.asarray([24576,49152,98304]); fig,ax=plt.subplots(figsize=(6,4)); ax.plot(proc,af,marker='o'); ax.axhline(4,linestyle='--'); ax.set_xlabel('processed examples / fit'); ax.set_ylabel('families passing'); fig.tight_layout(); fig.savefig(d/'processed_examples_vs_capability.png',dpi=160); plt.close(fig)
    mac=[compute['modeled_recurrent_mac_volume'][r]['af1d_modeled_recurrent_mac_volume'] for r in REGIMES]; fig,ax=plt.subplots(figsize=(6,4)); ax.plot(mac,af,marker='o'); ax.axhline(4,linestyle='--'); ax.set_xlabel('modeled training recurrent MAC volume'); ax.set_ylabel('families passing'); fig.tight_layout(); fig.savefig(d/'modeled_mac_volume_vs_capability.png',dpi=160); plt.close(fig)


def report(summary,decision,stability,proj,effective,message,controller,compute,historical)->str:
    f=lambda r,fam: summary[r]['family_results'][fam]
    lines=['# V837 AF1D Sample-Efficiency Report','',
    '## 1. Why architecture localization stopped','',
    'V837af restored neutral representation adequacy with AF1D at 4/5. V837ai therefore freezes architecture completely and changes only unique development data.','',
    '## 2. Frozen AF1D architecture','',
    'Exact imported `CandidateInputFactorizationY3`, condition `AF1D_deshared_candidate_input_factorization`: 1,643 active parameters, 420 candidate-projection parameters, and 1,206 recurrent/controller/projection MACs per timestep.','',
    '## 3. Historical V837l calibration','',
    'Historical GRU families passing: 1x=2/5, 2x=3/5, 4x=5/5. Historical neutral: 1/5, 1/5, 2/5. Historical residual RNN: 2/5, 2/5, 3/5. These rows were imported, not rerun.','',
    '## 4. Exact 1x/2x/4x definitions','',
    '1x=128 development episodes/family, 2x=256, 4x=512; validation is fixed at 128/family. Development seeds are exact nested prefixes beginning at 10000; validation is 20000..20127.','',
    '## 5. Nested-data proof','',
    'AI1 is a strict subset of AI2, AI2 is a strict subset of AI4, with no development/validation overlap or duplicate seeds. Union unique family/seed episodes remain 3,200.','',
    '## 6. Initialization pairing proof','',
    'For every family x replicate, AI1/AI2 and deterministic reconstruction of historical AI4 have identical trainable-tensor SHA-256 fingerprints and exact parameter-by-parameter equality under the frozen V837af seed laws. All ten projections begin bit-identical within a fit and are untied for optimization.','',
    '## 7. Reused 4x anchor provenance','',
    f"Source SHA `{CONFIG['source_sha']}`; exact V837af AF1D rows reused: 25. Reanalysis independently reproduces 4/5 and the accepted medians.",'',]
    for i,reg in enumerate(REGIMES,8):
        lines += [f'## {i}. AI{reg[0]} {reg} results','', '| family | development median | validation median | pass |','|---|---:|---:|---|']
        for fam in FAMILIES:
            x=f(reg,fam); lines.append(f"| {fam} | {x['development_median']:.6f} | {x['validation_median']:.6f} | {'PASS' if x['pass'] else 'FAIL'} |")
        lines += ['',f"Families passing: **{summary[reg]['families_passing']}/5**",'']
    lines += ['## 11. Family-specific scaling trajectories','']
    for fam in FAMILIES: lines.append(f"- {fam}: "+' -> '.join(f"{r} {f(r,fam)['validation_median']:.6f} ({'P' if f(r,fam)['pass'] else 'F'})" for r in REGIMES))
    lines += ['', '## 12. Replicate stability','', 'Five replicate validation scores, min/max/std, and >=0.85 counts are stored in `diagnostics/replicate_stability.json`; the historical median gate remains authoritative.','',
    '## 13. Projection specialization vs data','',
    f"Final median pairwise projection distances: 1x={proj['1x']['timeline']['192']['pairwise_frobenius_distance_median']:.6f}, 2x={proj['2x']['timeline']['192']['pairwise_frobenius_distance_median']:.6f}, 4x={proj['4x']['timeline']['192']['pairwise_frobenius_distance_median']:.6f}.",'',
    '## 14. Message/controller diagnostics','',
    f"Median message-ablation success drops: 1x={message['1x']['success_drop_median']:.6f}, 2x={message['2x']['success_drop_median']:.6f}, 4x={message['4x']['success_drop_median']:.6f}. Controller statistics are recorded for newly executed 1x/2x. The accepted 4x artifact did not store controller-gate summaries, and V837ai does not rerun 4x solely for a descriptive metric.",'',
    '## 15. Minimum tested sufficient multiplier','', f"`{decision['minimum_tested_sufficient_multiplier']}`",'',
    '## 16. Historical GRU comparison','', f"AF1D pass-count curve: {summary['1x']['families_passing']}/5 -> {summary['2x']['families_passing']}/5 -> 4/5. GRU: 2/5 -> 3/5 -> 5/5.",'',
    '## 17. Parameter-count caveat','', f"AF1D has 1,643 active parameters versus 875 for the historical GRU ({compute['af1d_to_gru_parameter_ratio']:.3f}x). V837ai therefore establishes unique-data threshold behavior, not parameter-matched superiority.",'',
    '## 18. Processed-example comparison','', 'The historical protocol uses the entire development set at each of 192 optimizer steps: 24,576 examples/fit at 1x, 49,152 at 2x, and 98,304 at 4x. Increasing unique data therefore also increases example-presentations.','',
    '## 19. Modeled recurrent-MAC comparison','', 'Modeled volumes use actual active training sequence timesteps times matrix MACs/timestep. They are model-derived compute proxies, not measured energy; CPU/wall time is separately measured for the new runs.','',
    '## 20. Sample-efficiency diagnosis','', f"`{decision['diagnosis']}`",'',
    '## 21. Structural-search authorization','', f"Structural-search recovery authorized: {decision['structural_search_recovery_allowed']}. Recommended data multiplier: {decision['recommended_structural_search_multiplier']}x. Next program: `V837aj_STRUCTURAL_SEARCH_RECOVERY`; not implemented here.",'',
    '## 22. Fresh-audit/mining/V838 status','', 'Fresh audit consumed: 0. Primitive mining: BLOCKED. V837ag/V837ah: NOT RUN. V838: NOT STARTED.','',
    '## 23. Strongest scientific claim','']
    if decision['diagnosis']=='AF1D_1X_REPRESENTATION_ADEQUATE':
        lines.append('The frozen AF1D neutral substrate retains representation adequacy at the original 1x regime using 128 unique development episodes per family, while the historical calibrated GRU first crossed the same gate at 512/family. Under these tested architectures and the frozen 192-step protocol, AF1D reaches adequacy with one quarter of the unique development data required by the historical GRU threshold. This is not parameter-matched superiority: AF1D has 1,643 active parameters versus 875 for the GRU.')
    elif decision['diagnosis']=='AF1D_2X_REPRESENTATION_ADEQUATE':
        lines.append('AF1D first reaches representation adequacy at 256 unique development episodes per family, compared with 512 for the historical GRU calibration. This establishes a twofold reduction in the minimum tested unique-development requirement under the tested architectures. AF1D has 1,643 active parameters versus 875 for the GRU.')
    elif decision['diagnosis']=='AF1D_REQUIRES_4X_UNIQUE_DATA':
        lines.append('AF1D solves the neutral representation problem but does not reduce the minimum tested unique-development multiplier relative to the historical GRU calibration. Both first satisfy the representation gate at 4x under their respective tested architectures.')
    else:
        lines.append('AF1D remains competent at the accepted 4x anchor, but its lower-data pass-count curve is nonmonotonic. No lower-data sample-efficiency claim is allowed; structural search conservatively uses 4x.')
    return '\n'.join(lines)+'\n'


def main()->int:
    for req in ('raw/ai1_runs.json','raw/ai2_runs.json','raw/ai4_reused_anchor.json'):
        if not (HERE/req).is_file(): raise SystemExit(f'missing {req}')
    anchor=load(HERE/'diagnostics/ai4_anchor_reuse.json')
    if anchor.get('compatible') is not True or anchor.get('checks',{}).get('exact_reanalysis') is not True: raise SystemExit('AF1D_4X_ANCHOR_COMPATIBILITY_FAILURE')
    rows=rows_by_regime(); summary=summaries(rows)
    if summary['4x']['families_passing'] != 4: raise SystemExit('AF1D_4X_REANALYSIS_MISMATCH')
    trajectories=family_trajectories(summary); paired=paired_effects(rows); stability=replicate_stability(rows)
    proj=projection_specialization(rows); effective=effective_input_maps(rows); message=message_dependence(rows); controller=controller_dynamics(rows)
    historical=load(HERE/'diagnostics/historical_v837l_comparison.json'); compute=compute_efficiency(rows,summary); decision=make_decision(summary)
    write_json(HERE/'diagnostics/family_data_trajectories.json',trajectories)
    write_json(HERE/'diagnostics/paired_data_effects.json',paired)
    write_json(HERE/'diagnostics/replicate_stability.json',stability)
    write_json(HERE/'diagnostics/projection_specialization.json',proj)
    write_json(HERE/'diagnostics/effective_input_maps.json',effective)
    write_json(HERE/'diagnostics/message_dependence.json',message)
    write_json(HERE/'diagnostics/controller_dynamics.json',controller)
    write_json(HERE/'diagnostics/compute_efficiency.json',compute)
    frontier={
        'families_passing':{r:summary[r]['families_passing'] for r in REGIMES},
        'minimum_tested_sufficient_multiplier':decision['minimum_tested_sufficient_multiplier'],
        'pass_count_monotonic':decision['pass_count_monotonic'],
        'historical_gru':{'1x':2,'2x':3,'4x':5},'historical_neutral':{'1x':1,'2x':1,'4x':2},
        'unique_development_reduction_vs_gru':decision['unique_development_reduction_vs_gru'],
    }
    write_json(HERE/'diagnostics/sample_efficiency_frontier.json',frontier); write_json(HERE/'diagnostics/decision_state.json',decision)
    results={
        'version':'V837ai','question':CONFIG['question'],'architecture':decision['architecture'],'architecture_frozen':True,
        'conditions':summary,'family_data_trajectories':trajectories,'sample_efficiency_frontier':frontier,'diagnosis':decision['diagnosis'],
        'diagnosis_qualifiers':decision['diagnosis_qualifiers'],'minimum_tested_sufficient_multiplier':decision['minimum_tested_sufficient_multiplier'],
        'pass_count_monotonic':decision['pass_count_monotonic'],'sample_efficiency_claim_allowed':decision['sample_efficiency_claim_allowed'],
        'representation_adequacy_confirmed':True,'structural_search_recovery_allowed':decision['structural_search_recovery_allowed'],
        'recommended_structural_search_multiplier':decision['recommended_structural_search_multiplier'],'primitive_mining_allowed':False,
        'fresh_audit_consumed':False,'primitives_promoted':0,'large_persistent_storage_tested':False,'v837ag_run':False,'v837ah_run':False,'v838_started':False,
        'resource_accounting':compute,'next_program':'V837aj_STRUCTURAL_SEARCH_RECOVERY',
    }
    write_json(HERE/'results.json',results)
    write_json(ROOT/'experiments/v837_primitive_invention/v837ai_resource_accounting.json',{
        'version':'V837ai','new_execution_resources':compute['new_execution_resources'],'historical_reused_ai4_resources':compute['historical_reused_ai4_resources'],
        'new_model_fits':50,'reused_model_fits':25,'total_evidence_rows':75,'union_unique_task_episodes':3200,'active_parameters':1643,'macs_per_timestep':1206,
    })
    write_json(ROOT/'experiments/v837_primitive_invention/af1d_sample_efficiency_program_resource_accounting.json',compute)
    write_json(ROOT/'experiments/v837_primitive_invention/af1d_sample_efficiency_program_status.json',{
        'v837ai_complete':True,'diagnosis':decision['diagnosis'],'families_passing':decision['families_passing'],'minimum_tested_sufficient_multiplier':decision['minimum_tested_sufficient_multiplier'],
        'sample_efficiency_claim_allowed':decision['sample_efficiency_claim_allowed'],'representation_adequacy':'PASS','architecture_frozen':True,
        'structural_search_recovery_allowed':decision['structural_search_recovery_allowed'],'recommended_structural_search_multiplier':decision['recommended_structural_search_multiplier'],
        'primitive_mining_allowed':False,'fresh_audit_episodes_consumed':0,'primitives_promoted':0,'large_persistent_storage_tested':False,
        'v837ag_run':False,'v837ah_run':False,'v838_started':False,'next_program':'V837aj_STRUCTURAL_SEARCH_RECOVERY',
    })
    plots(summary,proj,effective,message,controller,compute,historical)
    (ROOT/'docs/V837_AF1D_SAMPLE_EFFICIENCY_REPORT.md').write_text(report(summary,decision,stability,proj,effective,message,controller,compute,historical),encoding='utf-8')
    status='PASS.md' if decision['representation_adequacy_confirmed'] else 'FAILURE.md'
    (HERE/status).write_text(f"# V837ai ? {decision['diagnosis']}\n\nPass-count curve: {summary['1x']['families_passing']}/5 -> {summary['2x']['families_passing']}/5 -> {summary['4x']['families_passing']}/5.\n\nMinimum tested sufficient multiplier: {decision['minimum_tested_sufficient_multiplier']}.\nStructural-search recovery allowed: {decision['structural_search_recovery_allowed']}.\nPrimitive mining remains blocked. Fresh audit consumed: 0. V838 not started.\n",encoding='utf-8')
    print(json.dumps(decision,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
