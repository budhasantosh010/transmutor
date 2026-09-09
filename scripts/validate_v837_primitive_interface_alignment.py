from __future__ import annotations

import hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/v837_primitive_invention/v837al';START='802478151a5ab0d4cbbe099e3c18f056e8ce65bd'

def load(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def git_blob_sha(rel):
    data=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT)
    return hashlib.sha256(data).hexdigest()
def req(x,msg):
    if not x:raise RuntimeError(msg)

def main()->int:
    gate=load('experiments/v837_primitive_invention/v837al/frozen_interface_alignment_gate.json');req(gate['starting_sha']==START,'V837al start SHA changed');req(len(gate['confirmed_class_ids'])==6,'V837al six-class lock changed');req(gate['config_count']==253,'V837al config grid changed');req(gate['align_fit_seeds']==[10000,10063] and gate['align_select_seeds']==[10064,10127] and gate['align_test_seeds']==[20000,20127],'V837al data roles changed');req(gate['ridge_lambda']==1e-6 and gate['state_condition_number_cap']==1000.0,'V837al analytic fitting guards changed');req(gate['fresh_audit_consumed'] is False and gate['primitives_promoted']==0 and gate['v838_started'] is False,'V837al frozen science locks changed')
    ak=load('experiments/v837_primitive_invention/v837ak/results.json');req(ak['diagnosis']=='CONTEXT_BOUND_COMPUTATIONAL_MOTIFS' and ak['next_program']=='V837al_PRIMITIVE_INTERFACE_ALIGNMENT','V837ak authorization changed');req(ak['decision_state']['confirmed_classes']==6 and ak['decision_state']['causally_specific_classes']==1 and ak['decision_state']['boundary_interchangeable_classes']==0,'V837ak accepted source counts changed');req(ak['decision_state']['fresh_audit_consumed'] is False and ak['decision_state']['v838_started'] is False,'V837ak source locks changed')
    src=load('experiments/v837_primitive_invention/v837al/diagnostics/source_integrity.json');req(src['source_integrity'] is True and src['checkpoint_count']==50 and src['new_model_fits']==0 and src['optimizer_steps']==0,'V837al source integrity failed')
    source_paths={'decision':'experiments/v837_primitive_invention/v837ak/diagnostics/decision_state.json','results':'experiments/v837_primitive_invention/v837ak/results.json','confirmed':'experiments/v837_primitive_invention/v837ak/raw/confirmed_candidate_classes.json','causal':'experiments/v837_primitive_invention/v837ak/raw/causal_results.json','boundary':'experiments/v837_primitive_invention/v837ak/raw/boundary_substitution_results.json','reconstruction':'experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json'}
    for key,rel in source_paths.items():req(git_blob_sha(rel)==src['source_hashes'][key]==gate['source_hashes'][key],f'V837al source hash drift: {key}')
    recon=load(source_paths['reconstruction']);req(len(recon['rows'])==50,'V837ak reconstruction row count changed')
    for row in recon['rows']:
        oid=row['organism_id'];path=ROOT/row['checkpoint'];req(path.is_file(),f'missing V837ak checkpoint {oid}');req(sha(path)==src['checkpoint_hashes'][oid]==gate['checkpoint_hashes'][oid],f'V837ak checkpoint hash drift {oid}')
    confirmed=load(source_paths['confirmed']);ids=sorted(c['class_id'] for c in confirmed['classes'] if c.get('confirmed'));req(ids==sorted(gate['confirmed_class_ids']),'confirmed class IDs changed');causal=load(source_paths['causal']);cids=[c['class_id'] for c in causal['classes'] if c.get('causal_specificity_pass') is True];req(cids==[gate['primary_causal_class_id']],'causal class changed')
    power=load('experiments/v837_primitive_invention/v837al/diagnostics/statistical_power.json');req(power['primary_causal']['independent_recipients']>=7 and power['primary_causal']['strong_claim_powered'] is True,'primary causal class unexpectedly underpowered')
    legacy=load('experiments/v837_primitive_invention/v837al/diagnostics/legacy_alignment_reproduction.json');req(legacy['identity_reproduced'] is True and legacy['legacy_orthogonal_reproduced'] is True and legacy['max_identity_metric_delta']<=1e-9 and legacy['max_legacy_metric_delta']<=1e-9,'legacy baseline reproduction changed')
    fits=load('experiments/v837_primitive_invention/v837al/raw/adapter_fits.json');req(fits['fit_seeds']==[10000,10063] and fits['fit_split']=='development' and fits['gradient_steps_used_for_adapters']==0,'adapter fit leakage/training detected')
    loc=load('experiments/v837_primitive_invention/v837al/raw/scope_selection_results.json');req(loc['configs_evaluated']==253 and loc['selection_seeds']==[10064,10127],'selection grid/data changed')
    selected_path=HERE/'raw/selected_interface_configs.json';selected=load('experiments/v837_primitive_invention/v837al/raw/selected_interface_configs.json');req(selected['frozen_before_align_test'] is True,'selected config not frozen before test')
    semantic_selected={k:v for k,v in selected.items() if k!='selected_interface_sha256'};semantic_sha=hashlib.sha256(json.dumps(semantic_selected,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest();req(selected['selected_interface_sha256']==semantic_sha,'selected interface semantic hash drift')
    pretty=json.dumps(selected,indent=2,sort_keys=True)+'\n';selected_file_hashes={hashlib.sha256(pretty.encode('utf-8')).hexdigest(),hashlib.sha256(pretty.replace('\n','\r\n').encode('utf-8')).hexdigest()}
    pair=load('experiments/v837_primitive_invention/v837al/raw/pairwise_test_results.json');selected_file_sha=pair['selected_interface_sha256'];req(pair['align_test_seeds']==[20000,20127] and pair['adapter_refit_on_test'] is False and selected_file_sha in selected_file_hashes,'ALIGN_TEST freeze/refit violation')
    anchors=load('experiments/v837_primitive_invention/v837al/raw/canonical_anchors.json');req(all(a['frozen_before_test'] for a in anchors['anchors']),'canonical anchor not frozen');req(all('performance' not in a['selection_rule'].lower() for a in anchors['anchors']),'canonical anchor performance selected')
    canon=load('experiments/v837_primitive_invention/v837al/raw/canonical_test_results.json');req(canon['selected_interface_sha256']==selected_file_sha,'canonical selected config drift');req(canon['global_track'].get('direct_pair_fit_used') is False and canon['causal_track'].get('direct_pair_fit_used') is False,'direct pair fit leaked into canonical mode')
    closed=load('experiments/v837_primitive_invention/v837al/raw/closed_loop_results.json');front=load('experiments/v837_primitive_invention/v837al/raw/alignment_data_frontier.json');req((not front.get('run')) or (closed.get('run') and closed.get('metrics',{}).get('pass')),'alignment data frontier ran without closed-loop pass')
    decision=load('experiments/v837_primitive_invention/v837al/diagnostics/decision_state.json');req(decision['new_model_fits']==0 and decision['optimizer_steps']==0 and decision['primitives_promoted']==0 and decision['fresh_audit_consumed'] is False and decision['v838_started'] is False,'V837al final science locks violated');req(decision['primitive_archive_allowed_next']==(decision['diagnosis']=='CANONICAL_FUNCTIONAL_PRIMITIVE_ESTABLISHED'),'primitive archive authorization inconsistent')
    forbidden=('AdamW(','.backward(','optimizer.step(','torch.optim.')
    for path in HERE.glob('*.py'):
        text=path.read_text(encoding='utf-8');req(not any(tok in text for tok in forbidden),f'optimizer/backprop forbidden in V837al: {path.name}')
    # V837ak scientific package and its report must be byte-identical to the start commit.
    changed=subprocess.check_output(['git','diff','--name-only',START,'--','experiments/v837_primitive_invention/v837ak','docs/V837_FUNCTIONAL_DYNAMICAL_MOTIF_DISCOVERY_REPORT.md'],cwd=ROOT,text=True).strip();req(changed=='',f'protected V837ak files changed: {changed}')
    audit=load('experiments/v837_primitive_invention/audit/audit_results.json');req(audit.get('episodes_consumed')==0,'fresh audit consumed');print('V837al primitive interface alignment validation: PASS');return 0

if __name__=='__main__':raise SystemExit(main())
