from __future__ import annotations

import hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.v837am.authorization import START_SHA,PARTITIONS,assert_authorized
from experiments.v837_primitive_invention.v837am.branch_a_context import configs as a_configs
from experiments.v837_primitive_invention.v837am.branch_b_boundary import configs as b_configs
from experiments.v837_primitive_invention.v837am.branch_c_temporal_interaction import configs as c_configs
from experiments.v837_primitive_invention.v837am.utils import canonical_json,sha256_json

HERE=ROOT/'experiments/v837_primitive_invention/v837am'

def load(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def req(x,msg):
    if not x:raise RuntimeError(msg)
def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def semantic_freeze_sha(f):
    x={k:v for k,v in f.items() if k!='selected_final_hypothesis_sha256'}
    return sha256_json(x)

def main()->int:
    auth=assert_authorized();gate=load('experiments/v837_primitive_invention/v837am/frozen_context_or_redefinition_gate.json')
    req(gate['start_sha']==START_SHA=='7d3b70ee6907b5502e0f229116d74eff4745c71e','V837am start SHA changed')
    req(gate['am_a_config_count']==126 and gate['am_b_config_count']==20 and gate['am_c_config_count']==7,'V837am grid counts changed')
    req(gate['data_partitions']==PARTITIONS,'V837am data partition drift')
    req(gate['ridge_lambda']==1e-6 and gate['state_conditioning_limits']=={'minimum_singular_value':0.05,'condition_number_max':1000.0},'V837am analytic guards changed')
    req(gate['new_model_fits']==0 and gate['organism_optimizer_steps']==0 and gate['adapter_gradient_steps']==0,'V837am zero-training lock changed')
    req(gate['fresh_audit_consumed'] is False and gate['primitive_archive_allowed'] is False and gate['primitives_promoted']==0 and gate['v838_started'] is False,'V837am downstream locks changed')
    req(auth['causal_recipient_count']==38 and auth['minimum_possible_p']==2.0**-38,'V837am causal power drift')
    src=load('experiments/v837_primitive_invention/v837am/diagnostics/source_integrity.json');req(src['source_integrity'] is True and src['causal_recipient_count']==38 and src['new_model_fits']==0 and src['organism_optimizer_steps']==0 and src['adapter_gradient_steps']==0,'V837am source integrity invalid')
    # Permanent failure memory exists before closure and contains historical negatives.
    human=HERE/'FAILURE_ANALYSIS.md';machine=load('experiments/v837_primitive_invention/v837am/raw/failure_ledger.json');central=(ROOT/'docs/V837_FAILURE_LEDGER.md').read_text(encoding='utf-8')
    req(human.is_file() and len(machine.get('entries',[]))>=2,'V837am failure memory missing');ids={e['failure_id'] for e in machine['entries']};req('V837ak-BOUNDARY-INTERCHANGEABILITY-NOT-ESTABLISHED' in ids and 'V837al-STATIC-LINEAR-INTERFACE-EXHAUSTED' in ids,'historical failure backfill missing');req('V837ak-BOUNDARY-INTERCHANGEABILITY-NOT-ESTABLISHED' in central and 'V837al-STATIC-LINEAR-INTERFACE-EXHAUSTED' in central,'central failure ledger missing history')
    for e in machine['entries']:
        for k in ('failure_id','version','stage','source_commit','source_artifact_hashes','hypothesis','exact_implementation','data_used','controls','metrics','acceptance_gate','failed_conditions','distance_from_gate','result_status','failure_type','scientific_interpretation','do_not_repeat_unchanged','uncertainty_remaining','next_justified_experiment','reproduction_command','artifact_paths'):req(k in e,f'incomplete failure entry {e.get("failure_id")}:{k}')
        req(e['failure_type'] in {'SCIENTIFIC_FAILURE','ENGINEERING_FAILURE'},f'failure type invalid {e["failure_id"]}')
    a=load('experiments/v837_primitive_invention/v837am/raw/am_a_selection.json');b=load('experiments/v837_primitive_invention/v837am/raw/am_b_selection.json');c=load('experiments/v837_primitive_invention/v837am/raw/am_c_selection.json')
    req(a.get('science_schema')==2 and a['configs_evaluated']==126,'corrected AM-A grid incomplete');req(b['configs_evaluated']==20,'AM-B grid incomplete');req(c['configs_evaluated']==7,'AM-C grid incomplete')
    req({r['config_id'] for r in a['results']}=={x['config_id'] for x in a_configs()},'hidden/missing AM-A config');req({r['config_id'] for r in b['results']}=={x['config_id'] for x in b_configs()},'hidden/missing AM-B config');req({r['config_id'] for r in c['results']}=={x['config_id'] for x in c_configs()},'hidden/missing AM-C config')
    for prefix,payload in (('V837am-SCHEMA2-',a),('V837am-',b),('V837am-',c)):
        for r in payload['results']:
            if not r.get('pass'):req(prefix+r['config_id'] in ids,f'failed config absent from ledger: {r["config_id"]}')
    # Controls: every evaluated aggregate must expose the randomized-refit anti-cheating gate when rows exist.
    for payload in (a,b,c):
        for r in payload['results']:
            agg=r.get('aggregate',{});rows=int(agg.get('row_count',0) or 0)
            if rows:req('adapter_cheating_gate_pass' in agg and 'random_refit_output' in agg and 'different_output' in agg,f'control metrics missing: {r["config_id"]}')
    rankings=load('experiments/v837_primitive_invention/v837am/raw/frozen_context_cell_rankings.json');req(rankings['fit_seeds']==[10128,10191] and rankings['ranking_frozen_before_selection'] is True,'AM-B ranking not frozen on FIT only')
    meta=load('experiments/v837_primitive_invention/v837am/raw/meta_confirmation.json');freeze=load('experiments/v837_primitive_invention/v837am/raw/selected_final_hypothesis.json');req(meta['seeds']==[10384,10447] and meta['no_refit'] is True,'META_CONFIRM leakage/refit');req(freeze['frozen_before_final_dev_confirm'] is True and freeze['frozen_before_final_validation'] is True,'final hypothesis not frozen');req(freeze['selected_final_hypothesis_sha256']==semantic_freeze_sha(freeze),'final hypothesis semantic hash drift')
    dev=load('experiments/v837_primitive_invention/v837am/raw/final_dev_confirmation.json');val=load('experiments/v837_primitive_invention/v837am/raw/final_validation.json');closed=load('experiments/v837_primitive_invention/v837am/raw/closed_loop_results.json')
    if freeze.get('selected') is None:req(not dev.get('run') and not val.get('run'),'final data used without a frozen candidate')
    if dev.get('run'):req(dev['seeds']==[10448,10511] and dev['no_refit'] is True and dev['selected_hash']==freeze['selected_final_hypothesis_sha256'],'FINAL_DEV_CONFIRM freeze/refit violation')
    if val.get('run'):req(dev.get('pass') and val['seeds']==[20000,20127] and val['no_refit'] is True and val['selected_hash']==freeze['selected_final_hypothesis_sha256'],'FINAL_VALIDATION firewall violation')
    if closed.get('run'):req(val.get('pass') and freeze.get('selected',{}).get('branch')=='AM-A','closed loop authorization violation')
    # No optimizer or organism training in V837am.
    forbidden=('AdamW(','.backward(','optimizer.step(','torch.optim.')
    for p in HERE.glob('*.py'):
        text=p.read_text(encoding='utf-8');req(not any(t in text for t in forbidden),f'optimizer/backprop forbidden in V837am: {p.name}')
    decision=load('experiments/v837_primitive_invention/v837am/diagnostics/decision_state.json');req(decision['new_model_fits']==0 and decision['optimizer_steps']==0 and decision['primitives_promoted']==0 and decision['fresh_audit_consumed'] is False and decision['v838_started'] is False,'V837am final locks violated');req(decision['primitive_archive_allowed_next'] is False,'V837am archive must remain blocked')
    req(decision['diagnosis']=='DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE' and decision['next_program']=='V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED','V837am final diagnosis drift')
    req(decision['am_a_pass_count']==0 and decision['am_b_pass_count']==0 and decision['am_c_pass_count']==0 and decision['meta_confirm_passes']==0,'unexpected V837am branch survivor')
    req(decision['selected_branch'] is None and decision['selected_config'] is None and decision['final_validation_run'] is False and decision['closed_loop_run'] is False,'downstream stage ran without a winner')
    correction=next(e for e in machine['entries'] if e['failure_id']=='V837am-CORRECTION-AM-A-SCHEMA1-INVALIDATED');invalidated=set(correction['exact_implementation']['invalidated_failure_ids']);req(len(invalidated)==126,'invalidated AM-A batch count drift')
    accepted_science=sum(e['failure_type']=='SCIENTIFIC_FAILURE' and e['failure_id'] not in invalidated for e in machine['entries']);engineering=sum(e['failure_type']=='ENGINEERING_FAILURE' for e in machine['entries']);req(decision['failure_entries']==len(machine['entries'])==286 and decision['scientific_failures']==accepted_science==155 and decision['engineering_failures']==engineering==5 and decision['invalidated_measurement_records']==126,'failure accounting drift')
    diag_ledger=load('experiments/v837_primitive_invention/v837am/diagnostics/failure_ledger.json');req(diag_ledger==machine,'raw/diagnostic failure ledger drift')
    human_text=human.read_text(encoding='utf-8');req(all(r['config_id'] in human_text for p in (a,b,c) for r in p['results']),'accepted configuration missing from human failure analysis');invalid_payload=load('experiments/v837_primitive_invention/v837am/raw/invalidated_am_a_selection_schema1.json');req(len(invalid_payload.get('results',[]))==126 and all(r['config_id'] in human_text for r in invalid_payload['results']),'invalidated configuration missing from human failure analysis')
    required_plots={'context_family_selection.png','context_port_bundle_heatmap.png','same_vs_random_refit.png','adapter_capacity_vs_replay.png','message_global_boundary_influence.png','boundary_expansion_vs_error.png','primitive_size_vs_replay.png','temporal_history_vs_error.png','branch_meta_confirmation.png','final_hypothesis_evidence.png','adapter_vs_primitive_compute.png','failure_map.png'};req(required_plots=={p.name for p in (HERE/'plots').glob('*.png')},'V837am required plot set drift')
    # V837al and earlier scientific files are protected.  Only new global evolving docs/manifests may change.
    changed=subprocess.check_output(['git','diff','--name-only',START_SHA,'--','experiments/v837_primitive_invention/v837al','experiments/v837_primitive_invention/v837ak'],cwd=ROOT,text=True).strip();req(changed=='',f'protected historical science changed: {changed}')
    audit=load('experiments/v837_primitive_invention/audit/audit_results.json');req(audit.get('episodes_consumed')==0,'fresh audit consumed')
    print('V837am context interface or primitive redefinition validation: PASS');return 0

if __name__=='__main__':raise SystemExit(main())
