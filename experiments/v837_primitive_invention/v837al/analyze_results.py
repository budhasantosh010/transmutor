from __future__ import annotations

import json,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .utils import HERE,ROOT,read_json,write_json

DIAG=HERE/'diagnostics';RAW=HERE/'raw';PLOTS=HERE/'plots'
FAMILY_RANK={'SIGNED_PERMUTATION':0,'DIAGONAL_AFFINE':1,'RIGID_AFFINE':2,'FULL_AFFINE':3}


def _r(path,default=None):
    p=HERE/path
    return read_json(p) if p.is_file() else default


def _passing_classes(payload,track='causal_track'):
    if not payload:return []
    return [c for c in payload.get(track,{}).get('classes',[]) if c.get('metrics',{}).get('pass')]


def _decision():
    source=_r('diagnostics/source_integrity.json',{});power=_r('diagnostics/statistical_power.json',{});legacy=_r('diagnostics/legacy_alignment_reproduction.json',{});selected=_r('raw/selected_interface_configs.json',{});pairwise=_r('raw/pairwise_test_results.json',{});canonical=_r('raw/canonical_test_results.json',{});closed=_r('raw/closed_loop_results.json',{});necessity=_r('diagnostics/port_necessity.json',{});frontier=_r('raw/alignment_data_frontier.json',{})
    causal_power=power.get('primary_causal',{});cp=bool(causal_power.get('strong_claim_powered'));pair_pass=bool(_passing_classes(pairwise));canon_pass=bool(_passing_classes(canonical));closed_pass=bool(closed.get('run') and closed.get('metrics',{}).get('pass'))
    causal_sel=selected.get('causal_track');minimum=causal_sel.get('family') if causal_sel and pair_pass else None;required=necessity.get('causal_track',{}).get('required_ports',[]) if pair_pass else []
    if not cp:
        diagnosis='CAUSAL_PRIMITIVE_INTERFACE_EVIDENCE_UNDERPOWERED';next_program='V837am_CAUSAL_INTERFACE_POWER_EXPANSION';archive=False
    elif not pair_pass:
        diagnosis='LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT';next_program='V837am_CONTEXT_CONDITIONED_INTERFACE_OR_PRIMITIVE_REDEFINITION';archive=False
    elif pair_pass and not closed_pass:
        diagnosis='BOUNDARY_ALIGNMENT_SUFFICIENT';next_program='V837am_CONTEXT_DEPENDENCE_LOCALIZATION';archive=False
    elif closed_pass and not canon_pass:
        diagnosis='PAIRWISE_PRIMITIVE_INTERCHANGEABILITY_ESTABLISHED';next_program='V837am_CANONICAL_INTERFACE_STANDARDIZATION';archive=False
    else:
        diagnosis='CANONICAL_FUNCTIONAL_PRIMITIVE_ESTABLISHED';next_program='V837am_PRIMITIVE_COMPRESSION_AND_ARCHIVE';archive=True
    qualifiers=[]
    if pair_pass and not canon_pass:qualifiers.append('PAIRWISE_ALIGNMENT_ONLY')
    if pair_pass and not closed_pass:qualifiers.append('CLOSED_LOOP_CONTEXT_DEPENDENCE_REMAINS')
    qmap={'projected_input':'PRIVATE_INPUT_BASIS_IS_PRIMARY_REUSE_BARRIER','external_messages':'COMMUNICATION_PROTOCOL_MISMATCH','output':'COMMUNICATION_PROTOCOL_MISMATCH','global_term':'GLOBAL_CONTEXT_PROTOCOL_REQUIRED_FOR_REUSE','gate':'GLOBAL_CONTEXT_PROTOCOL_REQUIRED_FOR_REUSE','state':'STATE_BASIS_REQUIRED'}
    for p in required:
        q=qmap.get(p)
        if q and q not in qualifiers:qualifiers.append(q)
    transform_diag={'SIGNED_PERMUTATION':'SIGNED_PERMUTATION_INTERFACE_SUFFICIENT','DIAGONAL_AFFINE':'SCALE_SHIFT_INTERFACE_SUFFICIENT','RIGID_AFFINE':'RIGID_LATENT_BASIS_INTERFACE_SUFFICIENT','FULL_AFFINE':'GENERAL_LINEAR_INTERFACE_ALIGNMENT_SUFFICIENT'}.get(minimum)
    if transform_diag:qualifiers.insert(0,transform_diag)
    legacy_ok=bool(legacy.get('identity_reproduced')) and bool(legacy.get('legacy_orthogonal_reproduced'))
    decision={'version':'V837al','source_integrity':bool(source.get('source_integrity')),'confirmed_classes':6,'causal_classes':1,'statistically_testable_causal_class':cp,'legacy_baseline_reproduced':legacy_ok,'configs_evaluated':int(_r('raw/scope_selection_results.json',{}).get('configs_evaluated',0)),'global_selected_config':selected.get('global_track'),'causal_selected_config':causal_sel,'pairwise_interchangeable_classes':len({c['class_id'] for t in ('global_track','causal_track') for c in pairwise.get(t,{}).get('classes',[]) if c.get('metrics',{}).get('pass')}),'canonical_interchangeable_classes':len({c['class_id'] for t in ('global_track','causal_track') for c in canonical.get(t,{}).get('classes',[]) if c.get('metrics',{}).get('pass')}),'closed_loop_interchangeable_classes':1 if closed_pass else 0,'minimum_successful_transform':minimum,'required_ports':required,'diagnosis':diagnosis,'diagnosis_qualifiers':qualifiers,'primitive_archive_allowed_next':archive,'primitives_promoted':0,'new_model_fits':0,'optimizer_steps':0,'fresh_audit_consumed':False,'large_persistent_storage_tested':False,'v838_started':False,'next_program':next_program,'canonical_primitive_established':diagnosis=='CANONICAL_FUNCTIONAL_PRIMITIVE_ESTABLISHED','alignment_data_frontier_run':bool(frontier.get('run')),'minimum_tested_alignment_episodes':frontier.get('minimum_tested_alignment_episodes')}
    return decision


def _primitive_macs(selected):
    if not selected:return None
    pairs=_r('raw/frozen_pairs.json',{}).get('pairs',[]);causal=next((p for p in pairs if p.get('primary_causal')),None)
    if not causal:return None
    k=int(causal['size']);edges=int(causal['recipient'].get('internal_edge_count',0));base=72*k+4*edges
    if 'projected_input' not in selected.get('ports',[]):base+=36*k
    return base


def _resource(decision):
    fit=_r('raw/adapter_fits.json',{});loc=_r('raw/scope_selection_results.json',{});pair=_r('raw/pairwise_test_results.json',{});can=_r('raw/canonical_test_results.json',{});closed=_r('raw/closed_loop_results.json',{});frontier=_r('raw/alignment_data_frontier.json',{})
    trace_files=list((RAW/'cache/traces').glob('*.pt')) if (RAW/'cache/traces').exists() else []
    selection_replays=sum(int(r.get('valid_pair_count',0))*3 for r in loc.get('results',[]));pair_replays=sum(int(pair.get(t,{}).get('row_count',0))*3 for t in ('global_track','causal_track'));can_replays=sum(int(can.get(t,{}).get('row_count',0))*3 for t in ('global_track','causal_track'));closed_calls=(len(closed.get('rows',[]))*5 if closed.get('run') else 0)
    sel=decision.get('causal_selected_config') or decision.get('global_selected_config');adapter_macs=sel.get('adapter_macs_per_timestep') if sel else None;primitive_macs=_primitive_macs(sel);ratio=(float(adapter_macs)/primitive_macs if adapter_macs is not None and primitive_macs else None)
    cpu=sum(float(x.get('cpu_seconds',0.0) or 0.0) for x in (fit,loc,pair,can,closed));wall=sum(float(x.get('wall_seconds',0.0) or 0.0) for x in (fit,loc,pair,can,closed))
    return {'version':'V837al','new_model_fits':0,'optimizer_steps':0,'processed_training_examples':0,'gradient_steps_used_for_adapters':0,'full_organism_trace_calls':len(trace_files),'adapter_fitting_cpu_seconds':float(fit.get('cpu_seconds',0.0) or 0.0),'pairwise_replay_calls':selection_replays+pair_replays,'canonical_replay_calls':can_replays,'closed_loop_forward_calls':closed_calls,'adapter_macs_per_timestep':adapter_macs,'primitive_internal_macs_per_timestep_estimate':primitive_macs,'adapter_primitive_compute_ratio':ratio,'cpu_seconds':cpu,'wall_seconds':wall,'gpu_seconds':0.0,'fresh_audit_episodes':0,'primitives_promoted':0,'large_persistent_storage_tested':False,'alignment_data_frontier_run':bool(frontier.get('run'))}


def _safe_med(obj,key):
    v=obj.get(key) if obj else None
    return np.nan if v is None else float(v)


def _plots(decision):
    PLOTS.mkdir(parents=True,exist_ok=True);loc=_r('raw/scope_selection_results.json',{});pair=_r('raw/pairwise_test_results.json',{});can=_r('raw/canonical_test_results.json',{});closed=_r('raw/closed_loop_results.json',{});legacy=_r('raw/legacy_baseline_reproduction.json',{});necessity=_r('diagnostics/port_necessity.json',{});cond=_r('diagnostics/adapter_conditioning.json',{});frontier=_r('raw/alignment_data_frontier.json',{})
    # Legacy vs corrected selected causal output error.
    old=[]
    for c in legacy.get('classes',[]):
        if c.get('primary_causal'):old.append(c.get('legacy_aligned',{}).get('median_output'))
    causal_pair=pair.get('causal_track',{}).get('pooled',{});vals=[np.nanmedian(old) if old else np.nan,_safe_med(causal_pair,'median_output')];plt.figure(figsize=(6,4));plt.bar(['legacy orthogonal','selected V837al'],vals);plt.ylabel('output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'legacy_vs_corrected_alignment.png');plt.close()
    # Scope heatmap causal median output.
    fams=['SIGNED_PERMUTATION','DIAGONAL_AFFINE','RIGID_AFFINE','FULL_AFFINE'];mat=np.full((4,64),np.nan)
    for r in loc.get('results',[]):
        if r.get('family') in fams:mat[fams.index(r['family']),int(r['scope'],2)]=_safe_med(r.get('causal_track',{}),'median_output')
    plt.figure(figsize=(14,4));plt.imshow(mat,aspect='auto');plt.yticks(range(4),fams);plt.xlabel('scope integer');plt.colorbar(label='causal output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'interface_scope_heatmap.png');plt.close()
    meds=[]
    for f in fams:
        v=[_safe_med(r.get('causal_track',{}),'median_output') for r in loc.get('results',[]) if r.get('family')==f];meds.append(np.nanmin(v) if v else np.nan)
    plt.figure(figsize=(7,4));plt.bar(fams,meds);plt.xticks(rotation=20);plt.ylabel('best causal output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'transform_family_vs_error.png');plt.close()
    nr=necessity.get('causal_track',{}).get('leave_one_out',[]);plt.figure(figsize=(8,4));plt.bar([r['removed_port'] for r in nr],[r.get('relative_output_nrmse_worsening') or 0 for r in nr]);plt.xticks(rotation=30);plt.ylabel('relative output worsening');plt.tight_layout();plt.savefig(PLOTS/'port_necessity.png');plt.close()
    pc=pair.get('causal_track',{}).get('classes',[]);m=pc[0]['metrics'] if pc else {};plt.figure(figsize=(6,4));plt.bar(['same','different','random'],[_safe_med(m,'median_output'),_safe_med(m,'different_median_output'),_safe_med(m,'randomized_median_output')]);plt.ylabel('held-out output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'same_vs_different_vs_random.png');plt.close()
    cm=can.get('causal_track',{}).get('classes',[]);cm=cm[0]['metrics'] if cm else {};plt.figure(figsize=(6,4));plt.bar(['pairwise','canonical'],[_safe_med(m,'median_output'),_safe_med(cm,'median_output')]);plt.ylabel('output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'pairwise_vs_canonical_alignment.png');plt.close()
    cv=[r.get('condition_number') for r in cond.get('rows',[]) if isinstance(r.get('condition_number'),(int,float)) and np.isfinite(r.get('condition_number'))];plt.figure(figsize=(6,4));plt.hist(cv,bins=30);plt.xlabel('condition number');plt.tight_layout();plt.savefig(PLOTS/'adapter_condition_numbers.png');plt.close()
    xs=[];ys=[]
    for r in loc.get('results',[]):
        y=_safe_med(r.get('causal_track',{}),'median_output')
        if np.isfinite(y):xs.append(r.get('complexity',{}).get('adapter_macs_per_timestep',0));ys.append(y)
    plt.figure(figsize=(6,4));plt.scatter(xs,ys,s=8);plt.xlabel('adapter MACs/timestep');plt.ylabel('causal output NRMSE');plt.tight_layout();plt.savefig(PLOTS/'adapter_macs_vs_replay_error.png');plt.close()
    if closed.get('run'):
        mm=closed['metrics'];v=[1.0-mm['median_same_success_drop'],mm['median_same_success'],mm['median_different_success'],mm['median_randomized_success']]
    else:v=[np.nan]*4
    plt.figure(figsize=(7,4));plt.bar(['recipient baseline','same','different','random'],v);plt.ylabel('success');plt.tight_layout();plt.savefig(PLOTS/'closed_loop_success_by_control.png');plt.close()
    if frontier.get('run'):
        x=[r['episodes'] for r in frontier['rows']];y=[r['closed_loop']['median_same_success'] for r in frontier['rows']];plt.figure(figsize=(6,4));plt.plot(x,y,marker='o');plt.xlabel('ALIGN_FIT episodes');plt.ylabel('same-class closed-loop success');plt.tight_layout();plt.savefig(PLOTS/'alignment_data_frontier.png');plt.close()
    # Main evidence ladder.
    sel=decision.get('causal_selected_config');fam=sel.get('family') if sel else None;best={f:np.nan for f in fams}
    for r in loc.get('results',[]):
        if r.get('family') in best:
            val=_safe_med(r.get('causal_track',{}),'median_output');best[r['family']]=np.nanmin([best[r['family']],val]) if np.isfinite(best[r['family']]) else val
    labels=['identity']+['signed perm','diagonal','rigid','full']+['selected pairwise','canonical','closed-loop'];identity=next((r for r in loc.get('results',[]) if r.get('family')=='IDENTITY'),{});ev=[_safe_med(identity.get('causal_track',{}),'median_output'),best['SIGNED_PERMUTATION'],best['DIAGONAL_AFFINE'],best['RIGID_AFFINE'],best['FULL_AFFINE'],_safe_med(m,'median_output'),_safe_med(cm,'median_output'),(closed.get('metrics',{}).get('median_same_success_drop') if closed.get('run') else np.nan)];plt.figure(figsize=(10,4));plt.bar(labels,ev);plt.xticks(rotation=30);plt.ylabel('error / success drop');plt.tight_layout();plt.savefig(PLOTS/'primitive_interface_evidence.png');plt.close()


def _report(decision,resource):
    pair=_r('raw/pairwise_test_results.json',{});can=_r('raw/canonical_test_results.json',{});closed=_r('raw/closed_loop_results.json',{});power=_r('diagnostics/statistical_power.json',{});legacy=_r('diagnostics/legacy_alignment_reproduction.json',{});sel=_r('raw/selected_interface_configs.json',{});nec=_r('diagnostics/port_necessity.json',{});front=_r('raw/alignment_data_frontier.json',{})
    lines=['# V837 Primitive Interface Alignment Report','','## 1. Why V837al was authorized','V837ak closed at `CONTEXT_BOUND_COMPUTATIONAL_MOTIFS` with six held-out-confirmed classes, one causally specific class, and zero boundary-interchangeable classes.','','## 2. V837ak interface failure','V837al preserves the V837ak motifs and tests only compact explicit boundary-coordinate transformations.','','## 3. Statistical-power preflight',f"Primary causal class: `{power.get('primary_causal',{}).get('class_id')}`; N={power.get('primary_causal',{}).get('independent_recipients')}; p_min={power.get('primary_causal',{}).get('minimum_possible_p')}.",'','## 4. Historical orthogonal diagnostic reproduction',f"Identity reproduced: {legacy.get('identity_reproduced')}; legacy orthogonal reproduced: {legacy.get('legacy_orthogonal_reproduced')}. The legacy rotation-only diagnostic remains historical evidence, not the V837al interface definition.",'','## 5. New explicit boundary ports','State, external messages, global coupling, projected input, scalar gate, and outgoing output are modeled independently.','','## 6. Private projected-input hypothesis','The AF1D de-shared candidate projection is explicitly testable as a boundary port.','','## 7. Transform families','Signed permutation, diagonal affine, centered rigid affine with translation, and ridge full affine were fitted analytically with zero optimizer steps.','','## 8. Exhaustive port-scope localization','Exactly 253 configurations were evaluated per predeclared track.','','## 9. Selected interface configuration',f"GLOBAL: `{sel.get('global_track')}`",f"CAUSAL: `{sel.get('causal_track')}`",'','## 10. Held-out pairwise result',f"CAUSAL: `{pair.get('causal_track',{}).get('pooled')}`",'','## 11. Port necessity',f"CAUSAL required ports: `{nec.get('causal_track',{}).get('required_ports',[])}`",'','## 12. Canonical-interface result',f"CAUSAL: `{can.get('causal_track',{}).get('pooled')}`",'','## 13. Closed-loop substitution',f"`{closed}`",'','## 14. Adapter data requirement',f"`{front}`",'','## 15. Adapter compute overhead',f"Adapter MACs/timestep: {resource.get('adapter_macs_per_timestep')}; primitive estimate: {resource.get('primitive_internal_macs_per_timestep_estimate')}; ratio: {resource.get('adapter_primitive_compute_ratio')}.",'','## 16. Causal-class result',f"Diagnosis: `{decision['diagnosis']}`; qualifiers: `{decision['diagnosis_qualifiers']}`.",'','## 17. Primitive archive authorization',f"Allowed next: **{decision['primitive_archive_allowed_next']}**. Primitives promoted inside V837al: **0**.",'','## 18. Strongest scientific claim',_claim(decision),'','## 19. Red-team alternatives','A boundary replay effect alone is not treated as primitive reuse. Canonicalization and closed-loop causality remain separate gates; validation data cannot alter the selected configuration.','','## 20. Next single program',f"`{decision['next_program']}`",'']
    path=ROOT/'docs/V837_PRIMITIVE_INTERFACE_ALIGNMENT_REPORT.md';path.write_text('\n'.join(lines),encoding='utf-8')


def _claim(d):
    if d['diagnosis']=='CANONICAL_FUNCTIONAL_PRIMITIVE_ESTABLISHED':return 'A causally important independently learned computation became reusable through a compact family-conditional canonical interface with zero organism retraining.'
    if d['diagnosis']=='PAIRWISE_PRIMITIVE_INTERCHANGEABILITY_ESTABLISHED':return 'Direct pairwise closed-loop interchangeability is established, but a scalable canonical interface is not.'
    if d['diagnosis']=='BOUNDARY_ALIGNMENT_SUFFICIENT':return 'Compact boundary alignment is established, but closed-loop context dependence remains.'
    if d['diagnosis']=='LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT':return 'No tested low-complexity linear interface establishes held-out reuse for the powered causal motif.'
    return 'The causal interface evidence remains statistically underpowered for the frozen strong claim.'


def analyze():
    decision=_decision();resource=_resource(decision);write_json(DIAG/'decision_state.json',decision);write_json(DIAG/'alignment_compute.json',resource);write_json(HERE/'v837al_resource_accounting.json',resource);write_json(HERE.parent/'v837al_resource_accounting.json',resource);write_json(HERE.parent/'primitive_interface_alignment_program_status.json',{'version':'V837al','diagnosis':decision['diagnosis'],'next_program':decision['next_program'],'primitive_archive_allowed_next':decision['primitive_archive_allowed_next'],'primitives_promoted':0,'fresh_audit_episodes_consumed':0,'v838_started':False});result={'version':'V837al','question':"What is the smallest explicit transformation of a primitive's boundary interface that makes independently learned instances of the same computational class interoperable?",'decision_state':decision,'diagnosis':decision['diagnosis'],'diagnosis_qualifiers':decision['diagnosis_qualifiers'],'next_program':decision['next_program'],'resource_accounting':resource,'strongest_scientific_claim':_claim(decision)};write_json(HERE/'results.json',result);marker=HERE/('PASS.md' if decision['canonical_primitive_established'] else 'FAILURE.md');other=HERE/('FAILURE.md' if decision['canonical_primitive_established'] else 'PASS.md');
    if other.exists():other.unlink()
    marker.write_text(f"# V837al {decision['diagnosis']}\n\n{_claim(decision)}\n",encoding='utf-8');_plots(decision);_report(decision,resource);return result

if __name__=='__main__':print(json.dumps(analyze(),indent=2))
