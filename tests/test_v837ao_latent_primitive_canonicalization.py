from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from experiments.v837_primitive_invention.v837ao.abstract_rollout import next_canonical_state
from experiments.v837_primitive_invention.v837ao.canonical_ir import canonical_ir
from experiments.v837_primitive_invention.v837ao.canonical_reader import fit_k1_reader, read_k1
from experiments.v837_primitive_invention.v837ao.canonical_writer import set_state
from experiments.v837_primitive_invention.v837ao.episode_partitions import PARTITIONS, assert_disjoint_development_partitions
from experiments.v837_primitive_invention.v837ao.gauge_fix import biorthogonal_projector, gauge_fix_writer
from experiments.v837_primitive_invention.v837ao.phase_backends import BACKEND_ORDER
from experiments.v837_primitive_invention.v837ao.residual_projector import residual
from experiments.v837_primitive_invention.v837ao.source_contracts import POWERED_FAMILIES, validate_v837an_source
from experiments.v837_primitive_invention.v837ao.utils import HERE, START_SHA

ROOT=HERE.parents[2]

def j(rel):return json.loads((HERE/rel).read_text(encoding='utf-8'))

def test_start_sha_exact(): assert START_SHA=='fe210ae3ee6fad90b394865ce6dd06d525e11902'
def test_v837an_diagnosis_exact(): assert validate_v837an_source()['decision']['diagnosis']=='GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN'
def test_four_causal_subspace_families(): assert validate_v837an_source()['decision']['causal_subspace_families']==4
def test_four_semantic_compiler_families(): assert validate_v837an_source()['decision']['semantic_compiler_families']==4
def test_partial_observation_unresolved(): assert validate_v837an_source()['frozen']['families']['partial_observation'] is None
def test_v837an_archive_blocked(): assert validate_v837an_source()['decision']['primitive_archive_allowed_next'] is False
def test_v838_false(): assert validate_v837an_source()['decision']['v838_started'] is False

def test_split_before_science(): assert (HERE/'raw/frozen_organism_folds.json').is_file()
def test_split_engine_stratified(): assert all(len(v['engines'])==2 for v in j('raw/frozen_organism_folds.json')['families'].values())
def test_exact_two_holdouts_per_engine_when_available(): assert all(len(e['holdout'])==2 for f in j('raw/frozen_organism_folds.json')['families'].values() for e in f['engines'].values())
def test_no_performance_sorting(): assert j('raw/frozen_organism_folds.json')['performance_sorting'] is False
def test_discovery_holdout_disjoint(): assert all(not(set(f['discovery'])&set(f['holdout'])) for f in j('raw/frozen_organism_folds.json')['families'].values())
def test_all_competent_organisms_accounted_for(): assert j('raw/frozen_organism_folds.json')['all_accounted'] is True

def test_backend_fit_exact(): assert PARTITIONS['AO_BACKEND_FIT']==[10000,10063]
def test_backend_select_exact(): assert PARTITIONS['AO_BACKEND_SELECT']==[10064,10127]
def test_quotient_fit_exact(): assert PARTITIONS['AO_QUOTIENT_FIT']==[10128,10191]
def test_quotient_select_exact(): assert PARTITIONS['AO_QUOTIENT_SELECT']==[10192,10255]
def test_dynamics_fit_exact(): assert PARTITIONS['AO_DYNAMICS_FIT']==[10256,10319]
def test_dynamics_select_exact(): assert PARTITIONS['AO_DYNAMICS_SELECT']==[10320,10383]
def test_meta_exact(): assert PARTITIONS['AO_META_CONFIRM']==[10384,10447]
def test_heldout_eval_exact(): assert PARTITIONS['AO_HELDOUT_ORGANISM_EVAL']==[10448,10511]
def test_fresh_audit_unused(): assert j('diagnostics/decision_state.json')['fresh_audit_consumed'] is False
def test_historical_validation_marked_reused(): assert j('raw/historical_validation_robustness.json')['label']=='REUSED_HISTORICAL_VALIDATION'
def test_development_partitions_disjoint(): assert_disjoint_development_partitions()

def test_routing_ir(): assert canonical_ir('conditional_routing').output_law.startswith('SELECT')
def test_recall_ir(): assert canonical_ir('delayed_recall').semantic_variable=='remembered_value'
def test_iterative_ir_exact(): assert next_canonical_state('iterative_state',.2,{'x':1.0})==.65*.2+.35
def test_composition_ir_exact(): assert np.isclose(next_canonical_state('variable_composition',.2,{'gain':.7,'drive':.1}),np.tanh(.7*.2+.1))
def test_all_ir_dimension_one(): assert all(canonical_ir(f).semantic_dimension==1 for f in POWERED_FAMILIES)

def test_k1_reader_affine():
    q=np.zeros(40);q[0]=1;s=np.zeros((5,40));s[:,0]=np.arange(5);r=fit_k1_reader(s,2*s[:,0]+3,q);assert np.max(np.abs(read_k1(s,r)-(2*s[:,0]+3)))<1e-5
def test_full40_reader_diagnostic_only():
    from experiments.v837_primitive_invention.v837ao.canonical_reader import fit_full40_reader
    assert fit_full40_reader(np.eye(40),np.arange(40.0))['diagnostic_only'] is True
def test_reader_fit_discovery_only(): assert all(r['fit_partition']=='AO_BACKEND_FIT' for r in j('raw/discovery_backend_fits.json')['rows'])
def test_reader_no_heldout_data(): assert j('raw/discovery_backend_fits.json')['heldout_backend_artifacts_read'] is False
def test_semantic_range_frozen(): assert set(j('raw/canonical_target_grids.json')['families'])==set(POWERED_FAMILIES)

def test_reader_writer_gain():
    q=np.zeros(40);q[0]=1;reader={'a':2.0,'b':0.0,'q':q.tolist()};g=gauge_fix_writer(reader,q);assert g['valid'] and np.isclose(g['unit_gain'],1)
def test_unit_gain_writer():
    q=np.zeros(40);q[0]=1;reader={'a':2.0,'b':0.0,'q':q.tolist()};g=gauge_fix_writer(reader,q);assert np.isclose(2*g['writer'][0],1)
def test_read_after_write_exact():
    q=np.zeros(40);q[0]=1;reader={'a':2.0,'b':1.0,'q':q.tolist()};w=np.asarray(gauge_fix_writer(reader,q)['writer']);s=np.zeros((3,40));out=set_state(s,3,reader,w);assert np.max(np.abs(read_k1(out,reader)-3))<1e-12
def test_set_zero_change_identity():
    q=np.zeros(40);q[0]=1;reader={'a':1.0,'b':0.0,'q':q.tolist()};s=np.random.default_rng(1).normal(size=(4,40));assert np.max(np.abs(set_state(s,read_k1(s,reader),reader,q)-s))<1e-12
def test_set_idempotence():
    q=np.zeros(40);q[0]=1;reader={'a':1.0,'b':0.0,'q':q.tolist()};s=np.zeros((2,40));x=set_state(s,2,reader,q);assert np.max(np.abs(set_state(x,2,reader,q)-x))<1e-12
def test_set_overwrite():
    q=np.zeros(40);q[0]=1;reader={'a':1.0,'b':0.0,'q':q.tolist()};s=np.zeros((2,40));x=set_state(set_state(s,-1,reader,q),1,reader,q);assert np.allclose(read_k1(x,reader),1)
def test_no_cross_organism_writer(): assert j('frozen_latent_canonicalization_gate.json')['no_cross_organism_state_alignment'] is True

def test_binary_targets_exact():
    g=j('raw/canonical_target_grids.json')['families'];assert g['conditional_routing']['targets']==[-1.0,1.0] and g['delayed_recall']['targets']==[-1.0,1.0]
def test_continuous_quantiles_fit_only(): assert j('raw/canonical_target_grids.json')['fit_partition']=='AO_BACKEND_FIT'
def test_multiple_magnitudes(): assert len(j('raw/canonical_target_grids.json')['families']['iterative_state']['targets'])==5
def test_local_only_detection(): assert j('diagnostics/decision_state.json')['diagnosis']=='CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE'

def test_b0_global_backend(): assert BACKEND_ORDER[0]=='B0_GLOBAL_K1'
def test_b1_same_direction_phase_gauge(): assert BACKEND_ORDER[1]=='B1_PHASE_GAUGE_K1'
def test_b2_phase_specific_k1(): assert BACKEND_ORDER[2]=='B2_PHASE_K1'
def test_minimum_complexity_rule(): assert j('frozen_latent_canonicalization_gate.json')['backend_complexity_order']==list(BACKEND_ORDER)
def test_all_variants_exhausted_after_failure(): assert all(v==list(BACKEND_ORDER) for v in j('raw/discovery_family_backend_winners.json')['evaluated_variants'].values())

def test_biorthogonal_projector():
    q=np.zeros(40);q[0]=1;reader={'a':2.0,'b':0.0,'q':q.tolist()};w=np.zeros(40);w[0]=.5;P=biorthogonal_projector(reader,w);assert np.allclose(P@P,P)
def test_projector_idempotence():
    q=np.zeros(40);q[0]=1;reader={'a':2.0,'b':0.0,'q':q.tolist()};w=np.zeros(40);w[0]=.5;P=biorthogonal_projector(reader,w);assert np.allclose(P@P,P)
def test_residual_invariant_under_set():
    q=np.zeros(40);q[0]=1;reader={'a':1.0,'b':0.0,'q':q.tolist()};w=np.zeros(40);w[0]=1;component={'reader':reader,'writer':w.tolist()};s=np.random.default_rng(2).normal(size=40);assert np.allclose(residual(s+w*3,component),residual(s,component))
def test_quotient_artifacts_present(): assert (HERE/'raw/quotient_results.json').is_file()
def test_dynamics_artifacts_present(): assert (HERE/'raw/commutativity_results.json').is_file()

def test_32_random_directions(): assert j('frozen_latent_canonicalization_gate.json')['random_direction_controls']==32
def test_control_margin(): assert j('frozen_latent_canonicalization_gate.json')['random_control_margin_min']==.2

def test_family_winner_max_one(): assert all(v is None or isinstance(v,dict) for v in j('raw/discovery_family_backend_winners.json')['family_winners'].values())
def test_meta_no_refit(): assert j('raw/meta_confirmation.json')['no_refit'] is True
def test_meta_no_variant_fallback(): assert j('raw/meta_confirmation.json')['no_variant_fallback'] is True
def test_canonical_spec_frozen_after_meta(): assert j('raw/frozen_canonical_family_specs.json')['frozen_before_holdout_backend_read'] is True
def test_canonical_spec_hash(): assert len(j('raw/frozen_canonical_family_specs.json')['frozen_sha256'])==64

def test_heldout_backend_unread_before_freeze(): assert j('diagnostics/heldout_backend_isolation.json')['heldout_read_after_freeze'] is True
def test_heldout_no_v837an_q_load(): assert j('diagnostics/heldout_backend_isolation.json')['v837an_q_loaded'] is False
def test_calibration_ladder_exact(): assert j('raw/heldout_calibration_frontier.json')['calibration_ladder']==[1,2,4,8,16,32,64]
def test_no_heldout_variant_search(): assert j('raw/heldout_calibration_frontier.json')['variant_search_on_holdout'] is False
def test_no_heldout_k_search(): assert j('raw/heldout_calibration_frontier.json')['k_search_on_holdout'] is False
def test_no_heldout_phase_search(): assert j('raw/heldout_calibration_frontier.json')['phase_search_on_holdout'] is False
def test_no_heldout_fit_when_null_candidate(): assert j('raw/heldout_backend_results.json')['rows']==[]

def test_no_microstate_comparison_gate(): assert j('raw/cross_organism_agreement.json')['microstate_comparison_used'] is False
def test_no_q_alignment_gate(): assert j('raw/cross_organism_agreement.json')['q_alignment_used'] is False

def test_failure_analysis_exists_before_run(): assert (HERE/'FAILURE_ANALYSIS.md').is_file()
def test_every_failed_family_logged(): assert sum(e['stage']=='AO5_FAMILY_PRELIM' for e in j('raw/failure_ledger.json')['entries'])>=12
def test_scientific_vs_engineering_failure(): assert all(e['failure_type'] in {'SCIENTIFIC_FAILURE','ENGINEERING_FAILURE'} for e in j('raw/failure_ledger.json')['entries'])
def test_reproduction_command_present(): assert all(e['reproduction_command'] for e in j('raw/failure_ledger.json')['entries'])
def test_central_failure_ledger_append_only(): assert (ROOT/'docs/V837_FAILURE_LEDGER.md').is_file()

def test_new_model_fits_zero(): assert j('diagnostics/decision_state.json')['new_model_fits']==0
def test_model_optimizer_steps_zero(): assert j('diagnostics/decision_state.json')['model_optimizer_steps']==0
def test_backend_gradient_steps_zero(): assert j('diagnostics/decision_state.json')['backend_gradient_steps']==0
def test_no_new_structure_search(): assert 'B3' not in j('frozen_latent_canonicalization_gate.json')['backend_variants']
def test_no_routing_research_reopen(): assert j('diagnostics/decision_state.json')['discovery_family_candidates']==0
def test_no_primitive_archive_population(): assert j('diagnostics/decision_state.json')['primitive_archive_allowed_next'] is False
def test_primitives_promoted_zero(): assert j('diagnostics/decision_state.json')['primitives_promoted']==0
def test_fresh_audit_zero(): assert j('diagnostics/decision_state.json')['fresh_audit_consumed'] is False
