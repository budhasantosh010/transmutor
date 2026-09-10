from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/'experiments/v837_primitive_invention/v837ao'
START='fe210ae3ee6fad90b394865ce6dd06d525e11902'
FAMILIES=('conditional_routing','delayed_recall','iterative_state','variable_composition')
PLOTS={
'reader_accuracy_by_family.png','reader_vs_writer_consistency.png','absolute_setpoint_recovery.png','setpoint_magnitude_vs_recovery.png','random_vs_canonical_setter.png','backend_complexity_by_family.png','phase_realization.png','residual_distance_vs_semantic_error.png','quotient_sufficiency.png','natural_commutativity.png','interventional_commutativity.png','rollout_horizon_vs_error.png','cross_organism_canonical_agreement.png','heldout_calibration_frontier.png','calibration_examples_vs_pass_rate.png','backend_storage_vs_calibration_cost.png','law_recovery.png','failure_map_v837an_to_v837ao.png','canonical_ir_evidence_ladder.png'}


def j(rel):return json.loads((HERE/rel).read_text(encoding='utf-8'))

def require(cond,msg):
    if not cond:raise AssertionError(msg)


def validate() -> None:
    required=['RESEARCH_SPEC.md','FAILURE_ANALYSIS.md','frozen_latent_canonicalization_gate.json','raw/failure_ledger.json','diagnostics/failure_ledger.json','raw/frozen_organism_folds.json','raw/data_partition_lock.json','raw/canonical_target_grids.json','raw/discovery_backend_fits.json','raw/discovery_reader_results.json','raw/discovery_setpoint_results.json','raw/phase_backend_results.json','raw/discovery_family_backend_winners.json','raw/quotient_pairs.json','raw/quotient_results.json','raw/commutativity_results.json','raw/meta_confirmation.json','raw/frozen_canonical_family_specs.json','raw/heldout_calibration_frontier.json','raw/heldout_backend_results.json','raw/cross_organism_agreement.json','raw/historical_validation_robustness.json','raw/law_recovery.json','diagnostics/decision_state.json','diagnostics/resource_accounting.json','results.json']
    for rel in required:require((HERE/rel).is_file(),f'missing {rel}')
    gate=j('frozen_latent_canonicalization_gate.json');require(gate['start_sha']==START,'start sha');require(gate['no_cross_organism_state_alignment'] and gate['no_cross_organism_q_alignment'],'alignment lock');require(gate['backend_complexity_order']==['B0_GLOBAL_K1','B1_PHASE_GAUGE_K1','B2_PHASE_K1'],'backend order');require(gate['heldout_calibration_ladder']==[1,2,4,8,16,32,64],'ladder');require(gate['fresh_audit_consumed'] is False and gate['primitives_promoted']==0 and gate['v838_started'] is False,'science locks')
    source=j('diagnostics/source_integrity.json');auth=source['authorization'];require(source['pass'] is True and auth['v837an_diagnosis']=='GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN','source state');require(len(source['powered_families'])==4 and auth['semantic_compiler_families']==4,'source families')
    folds=j('raw/frozen_organism_folds.json');require(folds['performance_sorting'] is False and folds['all_accounted'] is True,'fold lock')
    for f in FAMILIES:
        fam=folds['families'][f];require(not set(fam['discovery'])&set(fam['holdout']),f'{f} overlap');require(len(fam['holdout'])==4,f'{f} holdout count')
        for e,v in fam['engines'].items():require(len(v['holdout'])==2,f'{f} {e} holdout')
    discovery=j('raw/discovery_family_backend_winners.json');require(discovery['evaluated_variants'][FAMILIES[0]]==['B0_GLOBAL_K1','B1_PHASE_GAUGE_K1','B2_PHASE_K1'],'minimum escalation');require(discovery['discovery_family_candidates']==0,'observed candidate count')
    meta=j('raw/meta_confirmation.json');require(meta['no_refit'] is True and meta['no_variant_fallback'] is True and meta['meta_confirmed_families']==0,'meta lock')
    frozen=j('raw/frozen_canonical_family_specs.json');require(frozen['frozen_before_holdout_backend_read'] is True and all(frozen['families'][f] is None for f in FAMILIES),'canonical freeze')
    held=j('raw/heldout_calibration_frontier.json');require(held['calibration_ladder']==[1,2,4,8,16,32,64] and held['historical_v837an_per_organism_backend_loaded'] is False,'holdout isolation');require(held['rows']==[],'no heldout fits after null freeze')
    agree=j('raw/cross_organism_agreement.json');require(agree['microstate_comparison_used'] is False and agree['q_alignment_used'] is False,'canonical agreement only')
    robust=j('raw/historical_validation_robustness.json');require(robust['label']=='REUSED_HISTORICAL_VALIDATION' and robust['may_upgrade_failed_family'] is False,'historical label')
    ledger=j('raw/failure_ledger.json');diag=j('diagnostics/failure_ledger.json');require(ledger==diag and len(ledger['entries'])>=277,'failure ledger')
    decision=j('diagnostics/decision_state.json');require(decision['diagnosis']=='CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE','diagnosis');require(decision['primitive_archive_allowed_next'] is False and decision['primitives_promoted']==0,'archive blocked');require(decision['new_model_fits']==0 and decision['model_optimizer_steps']==0 and decision['backend_gradient_steps']==0,'training locks');require(decision['fresh_audit_consumed'] is False and decision['v838_started'] is False,'audit locks')
    require({p.name for p in (HERE/'plots').glob('*.png')}==PLOTS,'plot set')
    require((ROOT/'docs/V837_LATENT_PRIMITIVE_CANONICALIZATION_REPORT.md').is_file(),'main report')
    protected=subprocess.check_output(['git','diff','--name-only',START,'--','experiments/v837_primitive_invention/v837an','experiments/v837_primitive_invention/v837am','experiments/v837_primitive_invention/v837al','experiments/v837_primitive_invention/v837ak'],cwd=ROOT,text=True).strip();require(protected=='','protected historical diff')
    require(not (ROOT/'experiments/v837_primitive_invention/v838').exists(),'V838 must not exist')
    print('V837ao latent primitive canonicalization validation: PASS')


if __name__=='__main__':validate()
