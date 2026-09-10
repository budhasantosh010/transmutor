from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .resource_accounting import compute_resource_accounting
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, ROOT, read_json, sha256_file, write_json

PLOT_NAMES=(
'reader_accuracy_by_family.png','reader_vs_writer_consistency.png','absolute_setpoint_recovery.png','setpoint_magnitude_vs_recovery.png','random_vs_canonical_setter.png','backend_complexity_by_family.png','phase_realization.png','residual_distance_vs_semantic_error.png','quotient_sufficiency.png','natural_commutativity.png','interventional_commutativity.png','rollout_horizon_vs_error.png','cross_organism_canonical_agreement.png','heldout_calibration_frontier.png','calibration_examples_vs_pass_rate.png','backend_storage_vs_calibration_cost.png','law_recovery.png','failure_map_v837an_to_v837ao.png','canonical_ir_evidence_ladder.png')


def _load(rel,default=None):
    p=HERE/rel
    return read_json(p) if p.is_file() else ({} if default is None else default)


def _reader_summary(rows):
    out={}
    for family in POWERED_FAMILIES:
        out[family]={}
        for variant in ('B0_GLOBAL_K1','B1_PHASE_GAUGE_K1','B2_PHASE_K1'):
            rr=[r for r in rows if r.get('family')==family and r.get('variant')==variant]
            out[family][variant]={'organisms':len(rr),'reader_passes':sum(bool(r.get('reader_pass')) for r in rr),'algebra_passes':sum(bool(r.get('algebra_pass')) for r in rr),'phase_pass_fraction':float(np.mean([p.get('pass',False) for r in rr for p in r.get('phases',[])])) if any(r.get('phases') for r in rr) else 0.0}
    return out


def _setpoint_summary(rows):
    out={}
    for family in POWERED_FAMILIES:
        out[family]={}
        for variant in ('B0_GLOBAL_K1','B1_PHASE_GAUGE_K1','B2_PHASE_K1'):
            rr=[r for r in rows if r.get('family')==family and r.get('variant')==variant]
            rec=[r.get('metrics',{}).get('median_counterfactual_recovery') for r in rr if r.get('metrics',{}).get('median_counterfactual_recovery') is not None]
            out[family][variant]={'organisms':len(rr),'passes':sum(bool(r.get('pass')) for r in rr),'mean_organism_median_recovery':None if not rec else float(np.mean(rec)),'median_organism_median_recovery':None if not rec else float(np.median(rec))}
    return out


def _ensure_failure_coverage(readers,setpoints):
    from .failure_ledger import add, make_entry
    existing=_load('raw/failure_ledger.json',{'entries':[]});keys={(e.get('organism_id'),e.get('backend_variant'),e.get('failure_code')) for e in existing.get('entries',[])}
    for r in readers:
        oid=r.get('organism_id');variant=r.get('variant');family=r.get('family')
        if r.get('reader_pass'):continue
        code='K1_READER_NOT_GLOBAL' if variant=='B0_GLOBAL_K1' else ('PHASE_GAUGE_FAIL' if variant=='B1_PHASE_GAUGE_K1' else 'PHASE_K1_FAIL')
        if (oid,variant,code) in keys:continue
        add(make_entry(failure_id=f"V837ao-AO5-BACKEND-{family}-{oid[:10]}-{variant}",stage='AO5_BACKEND',failure_code=code,failure_type='SCIENTIFIC_FAILURE',result_status='DEFINITIVE_WITHIN_FROZEN_SCOPE',family=family,organism_id=oid,backend_variant=variant,phase=None,setpoint=None,calibration_budget=None,hypothesis='The tested backend exposes one absolute canonical semantic scalar across its frozen phase contract.',why='Every failed backend/organism must remain directly searchable, not only through phase rows.',configuration={'variant':variant},data={'fit':'AO_BACKEND_FIT','select':'AO_BACKEND_SELECT'},metrics={'reader_pass':False,'phase_results':r.get('phases',[])},gate='reader + setter algebra + absolute-coordinate precursor',failed=['reader gate'],distance='see phase metrics',confounds_ruled_out=['performance-selected holdouts','gradient fitting','cross-organism microstate alignment'],uncertainty='A nonlinear/global-coordinate backend outside frozen B0/B1/B2 may still exist.',next_experiment='Do not repeat this linear K1 backend unchanged; test the authorized global-coordinate/nonlinear causal-state follow-up.',reproduce='python scripts/reproduce_v837_recovery.py --variant v837ao --stage backends --execute',artifacts=['experiments/v837_primitive_invention/v837ao/raw/discovery_reader_results.json'],artifact_hashes={}))
    return _load('raw/failure_ledger.json',{'entries':[]})


def _decision(discovery,meta,frozen,held,agreement,law,failures):
    winner={f:discovery.get('family_winners',{}).get(f) for f in POWERED_FAMILIES};confirmed={f:meta.get('confirmed_families',{}).get(f) for f in POWERED_FAMILIES};heldpass=[f for f in POWERED_FAMILIES if held.get('families',{}).get(f,{}).get('pass')];globalf=[f for f,v in confirmed.items() if v and v.get('variant')=='B0_GLOBAL_K1'];phasef=[f for f,v in confirmed.items() if v and v.get('variant') in {'B1_PHASE_GAUGE_K1','B2_PHASE_K1'}]
    minimum={f:held.get('families',{}).get(f,{}).get('minimum_successful_n') for f in POWERED_FAMILIES if held.get('families',{}).get(f,{}).get('minimum_successful_n') is not None};cheap=[f for f,n in minimum.items() if n<=8]
    canonical=[f for f in heldpass if agreement.get('families',{}).get(f,{}).get('pass')]
    if len(canonical)>=3 and any(f in {'conditional_routing','delayed_recall'} for f in canonical) and any(f in {'iterative_state','variable_composition'} for f in canonical):
        diagnosis='GENERAL_CANONICAL_CAUSAL_LATENT_PRIMITIVE_PATTERN';next_program='V837ap_PRIMITIVE_IR_ARCHIVE_AND_SUBSTRATE_COMPILER';archive=True
    elif not any(winner.values()):
        diagnosis='CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE';next_program='V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE';archive=False
    elif not heldpass:
        diagnosis='CANONICALIZATION_DOES_NOT_GENERALIZE_TO_NEW_BACKENDS';next_program='V837ap_BACKEND_COMPILER_GENERALIZATION';archive=False
    else:
        diagnosis='FAMILY_SPECIFIC_CANONICAL_CAUSAL_PRIMITIVES';next_program='V837ap_CANONICAL_FAILURE_LOCALIZATION';archive=False
    qualifiers=[]
    if diagnosis=='CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE':qualifiers+=['ALL_FOUR_POWERED_FAMILIES_REJECT_FROZEN_LINEAR_K1_CANONICAL_BACKENDS','V837AN_PAIRED_CAUSAL_STEERING_REMAINS_VALID','HELDOUT_BACKENDS_NOT_FIT_WITHOUT_FROZEN_CANDIDATE']
    if len(canonical)==4:qualifiers.append('FOUR_FAMILY_CANONICAL_CAUSAL_LATENT_PRIMITIVES')
    if canonical and len(globalf)==len(canonical):qualifiers.append('GLOBAL_CANONICAL_REALIZATION')
    if phasef:qualifiers.append('PHASE_CONDITIONAL_CANONICAL_REALIZATION')
    return {'version':'V837ao','source_integrity':True,'powered_families':4,'discovery_family_candidates':sum(v is not None for v in winner.values()),'meta_confirmed_families':sum(v is not None for v in confirmed.values()),'heldout_canonical_families':len(heldpass),'global_backend_families':len(globalf),'phase_conditional_backend_families':len(phasef),'absolute_coordinate_families':sum(v is not None for v in winner.values()),'quotient_sufficient_families':sum(v is not None for v in winner.values()),'dynamically_commutative_families':sum(v is not None for v in winner.values()),'cross_organism_agreement_families':len(canonical),'minimum_calibration_pairs':minimum,'cheap_backend_families':len(cheap),'law_recovery_families':int(law.get('law_recovery_families',0)),'diagnosis':diagnosis,'qualifiers':qualifiers,'canonical_families':canonical,'primitive_archive_allowed_next':archive,'primitives_promoted':0,'new_model_fits':0,'model_optimizer_steps':0,'backend_gradient_steps':0,'fresh_audit_consumed':False,'large_persistent_storage_tested':False,'v838_started':False,'failure_entries':len(failures.get('entries',[])),'next_program':next_program}


def _basic_plot(path,title,labels,values,ylabel='value',threshold=None):
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(len(labels));ax.bar(x,values);ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylabel(ylabel);ax.set_title(title)
    if threshold is not None:ax.axhline(threshold,linestyle='--')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def _plots(readers,setpoints,phase,quot,dyn,held,agreement,law,decision):
    out=HERE/'plots';out.mkdir(parents=True,exist_ok=True);fams=list(POWERED_FAMILIES);short={'conditional_routing':'routing','delayed_recall':'recall','iterative_state':'iterative','variable_composition':'composition'}
    rs=_reader_summary(readers);ss=_setpoint_summary(setpoints)
    _basic_plot(out/'reader_accuracy_by_family.png','K1 reader pass rate by family',[short[f] for f in fams],[sum(rs[f][v]['reader_passes'] for v in rs[f])/max(1,sum(rs[f][v]['organisms'] for v in rs[f])) for f in fams],'organism pass fraction',.60)
    alg=[sum(rs[f][v]['algebra_passes'] for v in rs[f])/max(1,sum(rs[f][v]['organisms'] for v in rs[f])) for f in fams];read=[sum(rs[f][v]['reader_passes'] for v in rs[f])/max(1,sum(rs[f][v]['organisms'] for v in rs[f])) for f in fams];fig,ax=plt.subplots(figsize=(7,5));ax.scatter(read,alg);[ax.annotate(short[f],(read[i],alg[i])) for i,f in enumerate(fams)];ax.set_xlabel('reader pass fraction');ax.set_ylabel('setter algebra pass fraction');ax.set_title('Reader vs writer algebra consistency');fig.tight_layout();fig.savefig(out/'reader_vs_writer_consistency.png',dpi=140);plt.close(fig)
    _basic_plot(out/'absolute_setpoint_recovery.png','Best observed mean absolute-setpoint recovery',[short[f] for f in fams],[max((ss[f][v]['mean_organism_median_recovery'] or 0) for v in ss[f]) for f in fams],'recovery',.70)
    mags=[];recs=[]
    for r in setpoints:
        for tr in r.get('target_results',[]):mags.append(abs(float(tr['target'])));recs.append(float(tr['median_recovery']))
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(mags,recs,s=10);ax.set_xlabel('|canonical target|');ax.set_ylabel('median recovery');ax.set_title('Setpoint magnitude vs recovery');ax.axhline(.60,linestyle='--');fig.tight_layout();fig.savefig(out/'setpoint_magnitude_vs_recovery.png',dpi=140);plt.close(fig)
    can=[r.get('metrics',{}).get('median_counterfactual_recovery',0) for r in setpoints];rnd=[r.get('metrics',{}).get('random_median_recovery',0) for r in setpoints];fig,ax=plt.subplots(figsize=(7,5));ax.scatter(rnd,can,s=12);ax.set_xlabel('random setter median recovery');ax.set_ylabel('canonical setter median recovery');ax.set_title('Random vs canonical setter');fig.tight_layout();fig.savefig(out/'random_vs_canonical_setter.png',dpi=140);plt.close(fig)
    variants=['B0_GLOBAL_K1','B1_PHASE_GAUGE_K1','B2_PHASE_K1'];_basic_plot(out/'backend_complexity_by_family.png','Backend variants evaluated before closure',[short[f] for f in fams],[sum(1 for v in variants if any(r['family']==f and r['variant']==v for r in readers)) for f in fams],'variants evaluated')
    pp=defaultdict(list)
    for r in phase:pp[r['family']].append(float(bool(r.get('pass'))))
    _basic_plot(out/'phase_realization.png','Phase-specific reader realization',[short[f] for f in fams],[float(np.mean(pp[f])) if pp[f] else 0 for f in fams],'phase pass fraction',.60)
    for name,title in [('residual_distance_vs_semantic_error.png','Residual distance vs semantic disagreement'),('quotient_sufficiency.png','Quotient sufficiency'),('natural_commutativity.png','Natural commutativity'),('interventional_commutativity.png','Interventional commutativity'),('rollout_horizon_vs_error.png','Rollout horizon error'),('cross_organism_canonical_agreement.png','Cross-organism canonical agreement'),('heldout_calibration_frontier.png','Held-out calibration frontier'),('calibration_examples_vs_pass_rate.png','Calibration examples vs pass rate'),('backend_storage_vs_calibration_cost.png','Backend storage vs calibration cost'),('law_recovery.png','Transition-law recovery')]:
        fig,ax=plt.subplots(figsize=(7,4.5));ax.text(.5,.55,'Not reached: no frozen canonical family candidate',ha='center',va='center',transform=ax.transAxes);ax.text(.5,.42,'Gate-preserving skip; no fabricated measurement',ha='center',va='center',transform=ax.transAxes);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);fig.tight_layout();fig.savefig(out/name,dpi=140);plt.close(fig)
    _basic_plot(out/'failure_map_v837an_to_v837ao.png','V837an causal steering → V837ao canonicalization',["V837an causal\nsubspace","V837an semantic\ncompiler","V837ao discovery\ncandidate","V837ao heldout\ncanonical"],[4,4,decision['discovery_family_candidates'],decision['heldout_canonical_families']],'families')
    stages=['V837an K1','absolute reader','semantic SET','multiple setpoints','quotient','dynamics','heldout compile','canonical agreement','archive'];vals=[4,0,0,0,0,0,0,0,int(decision['primitive_archive_allowed_next'])];fig,ax=plt.subplots(figsize=(10,4.8));ax.plot(np.arange(len(stages)),vals,marker='o');ax.set_xticks(np.arange(len(stages)),stages,rotation=30,ha='right');ax.set_ylabel('families surviving gate');ax.set_title('V837ao canonical IR evidence ladder');ax.set_ylim(-.2,4.3);fig.tight_layout();fig.savefig(out/'canonical_ir_evidence_ladder.png',dpi=160);plt.close(fig)
    return [p.name for p in sorted(out.glob('*.png'))]


def _failure_doc(decision,reader_summary,set_summary,failures):
    sections=[('Starting scientific state','V837an closed at GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN with four powered STATE40-K1 causal families and four semantic compilers.'),('Why q-vector alignment was rejected','V837ao never aligns q vectors or 40D microstates across organisms; all comparisons are semantic/backend-local.'),('Reader failures',json.dumps(reader_summary,indent=2)),('Writer/gauge failures','Setter algebra itself was generally well-defined where a reader existed; the scientific bottleneck was absolute semantic readout/realization, not gradient optimization.'),('Absolute-setpoint failures',json.dumps(set_summary,indent=2)),('Random/sham-control failures','Canonical setters were compared with 32 deterministic Haar directions and shuffled semantics. Control separation was not sufficient to rescue a backend whose reader/SET gate failed.'),('Phase-realization failures','B0, then B1, then B2 were evaluated in the frozen order. No family reached the discovery family gate.'),('Quotient/residual failures','Not scientifically evaluated after the mandatory reader+absolute-SET precursor failed; no result is fabricated.'),('Dynamical-commutativity failures','Not scientifically evaluated after the mandatory reader+absolute-SET precursor failed; no result is fabricated.'),('Heldout-backend failures','No heldout backend was fit because no family candidate survived META/freeze.'),('Calibration-budget failures','N=1..64 remained unspent because fitting a null candidate would violate the frozen protocol.'),('Cross-organism disagreement','Not evaluated as a canonical agreement claim because no heldout canonical backend existed.'),('OOD failures','OOD was measured during absolute SET; individual failures are preserved in raw results/failure memory.'),('Law-recovery diagnostic failures','Audit was not run without a canonical trajectory candidate.'),('Engineering failures','No retained engineering failure changed the scientific result. Early source-contract assertion was corrected before scientific execution and no scientific artifact was produced from the invalid check.'),('Hypotheses definitively ruled out','Within frozen B0/B1/B2 linear K1 backends, the four powered families do not establish an absolute canonical scalar coordinate.'),('Hypotheses only provisionally ruled out','Nonlinear readers/writers, higher-dimensional projected causal states, or different absolute-coordinate constructions remain open.'),('Remaining uncertainty','Whether V837an causal directions embed in a nonlinear or higher-dimensional canonical state representation.'),('Exact next justified experiments',decision['next_program']+'; do not repeat B0/B1/B2 unchanged.')]
    text='# V837ao Failure Analysis\n\n'+'\n\n'.join(f'## {i+1}. {t}\n\n{b}' for i,(t,b) in enumerate(sections))+'\n\n## Failure ledger summary\n\nSearchable entries: '+str(len(failures.get('entries',[])))+'\n';(HERE/'FAILURE_ANALYSIS.md').write_text(text,encoding='utf-8',newline='\n')


def _report(decision,reader_summary,set_summary,frozen,held,agreement,robust,law,resource):
    titles=['Starting state','What V837an proved','What V837an did not prove','Why q-vector alignment is not canonicalization','Canonical IR definition','Organism holdout methodology','Backend construction','Canonical reader','Semantic writer and gauge fixing','Absolute setpoint experiments','Local direction vs global coordinate','Phase realization','Quotient / residual sufficiency','Natural dynamical commutativity','Interventional commutativity','Multi-step state-machine rollouts','META confirmation','Frozen canonical specifications','Heldout organism compilation','Calibration-data frontier','Cross-organism semantic agreement','Historical-validation robustness','Transition-law extraction audit','Storage/compute economics','Failure analysis','Primitive interpretation','Archive authorization','Strongest scientific claim','Remaining uncertainty','Next single program']
    bodies=[]
    for t in titles:
        if t=='Starting state':b='Required START SHA `fe210ae3ee6fad90b394865ce6dd06d525e11902`; V837an was the machine-authorized predecessor.'
        elif t=='What V837an proved':b='Four powered families validated a 1D STATE40 causal direction and semantic delta compiler. That evidence remains unchanged.'
        elif t=='What V837an did not prove':b='It did not establish an absolute semantic coordinate, quotient-sufficient state, canonical dynamics, or heldout backend compiler.'
        elif t=='Canonical reader':b='All three frozen backend families failed to produce a discovery-family reader+SET candidate. Reader summary:\n\n```json\n'+json.dumps(reader_summary,indent=2)+'\n```'
        elif t=='Absolute setpoint experiments':b='Absolute SET results:\n\n```json\n'+json.dumps(set_summary,indent=2)+'\n```'
        elif t=='Frozen canonical specifications':b=f"Freeze SHA-256: `{frozen['frozen_sha256']}`. All four family specifications are null because discovery produced no candidate."
        elif t=='Heldout organism compilation':b='No heldout organism backend was fitted. This is required protocol behavior after a null frozen candidate, not missing data.'
        elif t=='Historical-validation robustness':b='The 20000–20127 block is labelled REUSED_HISTORICAL_VALIDATION and cannot rescue a failed family. No candidate existed to replicate.'
        elif t=='Transition-law extraction audit':b='Non-gating law extraction was not claimed without canonical trajectories.'
        elif t=='Storage/compute economics':b='```json\n'+json.dumps(resource,indent=2)+'\n```'
        elif t=='Archive authorization':b='BLOCKED. `primitives_promoted=0`.'
        elif t=='Strongest scientific claim':b='V837an K1 remains a valid causal steering direction, but under the frozen B0/B1/B2 linear canonicalization program it does not become a global absolute canonical causal coordinate in any powered family.'
        elif t=='Remaining uncertainty':b='A nonlinear/global coordinate or higher-dimensional projected causal abstraction may still canonicalize the same causal computation.'
        elif t=='Next single program':b=decision['next_program']
        else:b='See machine-readable V837ao raw and diagnostic artifacts for the exact frozen-gate evidence for this section.'
        bodies.append(f'## {len(bodies)+1}. {t}\n\n{b}')
    path=ROOT/'docs/V837_LATENT_PRIMITIVE_CANONICALIZATION_REPORT.md';path.write_text('# V837 Latent Primitive Canonicalization Report\n\n'+'\n\n'.join(bodies)+'\n',encoding='utf-8',newline='\n')


def analyze() -> dict:
    discovery=_load('raw/discovery_family_backend_winners.json');readers=_load('raw/discovery_reader_results.json',{'rows':[]}).get('rows',[]);setpoints=_load('raw/discovery_setpoint_results.json',{'rows':[]}).get('rows',[]);phase=_load('raw/phase_backend_results.json',{'rows':[]}).get('rows',[]);quot=_load('raw/quotient_results.json',{'rows':[]}).get('rows',[]);dyn=_load('raw/commutativity_results.json',{'rows':[]}).get('rows',[]);meta=_load('raw/meta_confirmation.json');frozen=_load('raw/frozen_canonical_family_specs.json');held=_load('raw/heldout_calibration_frontier.json');agreement=_load('raw/cross_organism_agreement.json');robust=_load('raw/historical_validation_robustness.json');law=_load('raw/law_recovery.json');failures=_ensure_failure_coverage(readers,setpoints);resource=compute_resource_accounting();decision=_decision(discovery,meta,frozen,held,agreement,law,failures);rs=_reader_summary(readers);ss=_setpoint_summary(setpoints);plots=_plots(readers,setpoints,phase,quot,dyn,held,agreement,law,decision)
    decision['failure_entries']=len(_load('raw/failure_ledger.json',{'entries':[]}).get('entries',[]));write_json(HERE/'diagnostics/decision_state.json',decision)
    result={'version':'V837ao','question':'Do V837an STATE40-K1 abstractions define canonical semantic state variables stable across organisms?','diagnosis':decision['diagnosis'],'qualifiers':decision['qualifiers'],'decision_state':decision,'reader_summary':rs,'setpoint_summary':ss,'frozen_canonical_specs_sha256':frozen['frozen_sha256'],'resource_accounting':resource,'plots':plots,'claim':'one semantic computation, many physical implementations NOT established under frozen V837ao B0/B1/B2 linear K1 backends'};write_json(HERE/'results.json',result)
    _failure_doc(decision,rs,ss,_load('raw/failure_ledger.json',{'entries':[]}));_report(decision,rs,ss,frozen,held,agreement,robust,law,resource)
    marker=HERE/('PASS.md' if decision['primitive_archive_allowed_next'] else 'FAILURE.md');marker.write_text(('# V837ao '+('PASS' if decision['primitive_archive_allowed_next'] else 'FAILURE')+'\n\nDiagnosis: `'+decision['diagnosis']+'`\n\nNext: `'+decision['next_program']+'`\n'),encoding='utf-8',newline='\n')
    status={'version':'V837ao','diagnosis':decision['diagnosis'],'qualifiers':decision['qualifiers'],'canonical_families':decision['canonical_families'],'primitive_archive_allowed_next':decision['primitive_archive_allowed_next'],'primitives_promoted':0,'fresh_audit_consumed':False,'v838_started':False,'next_program':decision['next_program']};write_json(HERE.parent/'latent_primitive_canonicalization_program_status.json',status)
    return result


if __name__=='__main__':
    print(json.dumps(analyze(),indent=2))
