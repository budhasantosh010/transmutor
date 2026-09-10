from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/v837_primitive_invention/v837ao'
START='fe210ae3ee6fad90b394865ce6dd06d525e11902'
EXPECTED_PLOTS={
'reader_accuracy_by_family.png','reader_vs_writer_consistency.png','absolute_setpoint_recovery.png','setpoint_magnitude_vs_recovery.png','random_vs_canonical_setter.png','backend_complexity_by_family.png','phase_realization.png','residual_distance_vs_semantic_error.png','quotient_sufficiency.png','natural_commutativity.png','interventional_commutativity.png','rollout_horizon_vs_error.png','cross_organism_canonical_agreement.png','heldout_calibration_frontier.png','calibration_examples_vs_pass_rate.png','backend_storage_vs_calibration_cost.png','law_recovery.png','failure_map_v837an_to_v837ao.png','canonical_ir_evidence_ladder.png'}

def j(rel):return json.loads((HERE/rel).read_text(encoding='utf-8'))

def test_required_raw_artifacts_complete():
    names=['source_state.json','v837an_contract_hashes.json','frozen_organism_folds.json','data_partition_lock.json','canonical_target_grids.json','discovery_backend_fits.json','discovery_reader_results.json','discovery_setpoint_results.json','phase_backend_results.json','discovery_family_backend_winners.json','quotient_pairs.json','quotient_results.json','commutativity_results.json','meta_confirmation.json','frozen_canonical_family_specs.json','heldout_calibration_frontier.json','heldout_backend_results.json','cross_organism_agreement.json','historical_validation_robustness.json','law_recovery.json','failure_ledger.json']
    assert all((HERE/'raw'/n).is_file() for n in names)

def test_required_diagnostics_complete():
    names=['authorization.json','source_integrity.json','organism_fold_integrity.json','data_partition_integrity.json','v837an_reproduction.json','reader_fit.json','reader_writer_gain.json','setter_algebra.json','setpoint_generalization.json','random_controls.json','phase_realization.json','residual_projector.json','quotient_sufficiency.json','commutativity_natural.json','commutativity_interventional.json','meta_confirmation.json','canonical_freeze.json','heldout_backend_isolation.json','calibration_frontier.json','cross_organism_agreement.json','historical_robustness.json','law_recovery.json','resource_accounting.json','failure_ledger.json','decision_state.json']
    assert all((HERE/'diagnostics'/n).is_file() for n in names)

def test_required_source_modules_complete():
    names=['authorization.py','source_integrity.py','source_contracts.py','organism_folds.py','episode_partitions.py','canonical_ir.py','abstract_rollout.py','k1_backend.py','canonical_reader.py','canonical_writer.py','gauge_fix.py','phase_backends.py','setpoint_grid.py','setpoint_eval.py','random_controls.py','ood_diagnostics.py','residual_projector.py','quotient_pairs.py','quotient_eval.py','commutativity.py','trajectory_eval.py','meta_confirm.py','freeze_canonical_specs.py','heldout_backend_calibration.py','calibration_frontier.py','cross_organism_agreement.py','historical_robustness.py','law_recovery.py','metrics.py','failure_ledger.py','resource_accounting.py','analyze_results.py','run_pipeline.py']
    assert all((HERE/n).is_file() for n in names)

def test_frozen_gate_hash_present(): assert len(j('frozen_latent_canonicalization_gate.json')['gate_sha256'])==64
def test_fold_hash_present(): assert len(j('raw/frozen_organism_folds.json')['fold_sha256'])==64
def test_frozen_spec_hash_present(): assert len(j('raw/frozen_canonical_family_specs.json')['frozen_sha256'])==64
def test_main_report_exists(): assert (ROOT/'docs/V837_LATENT_PRIMITIVE_CANONICALIZATION_REPORT.md').is_file()
def test_failure_analysis_has_19_sections(): assert (HERE/'FAILURE_ANALYSIS.md').read_text(encoding='utf-8').count('\n## ')>=19
def test_failure_marker_exists(): assert (HERE/'FAILURE.md').is_file() and not (HERE/'PASS.md').exists()
def test_plot_set_exact(): assert {p.name for p in (HERE/'plots').glob('*.png')}==EXPECTED_PLOTS
def test_plot_files_nonempty(): assert all(p.stat().st_size>4000 for p in (HERE/'plots').glob('*.png'))
def test_decision_diagnosis_exact(): assert j('diagnostics/decision_state.json')['diagnosis']=='CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE'
def test_next_program_exact(): assert j('diagnostics/decision_state.json')['next_program']=='V837ap_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE'
def test_archive_blocked(): assert j('diagnostics/decision_state.json')['primitive_archive_allowed_next'] is False
def test_zero_promotions(): assert j('diagnostics/decision_state.json')['primitives_promoted']==0
def test_zero_training():
    d=j('diagnostics/decision_state.json');assert (d['new_model_fits'],d['model_optimizer_steps'],d['backend_gradient_steps'])==(0,0,0)
def test_fresh_audit_false(): assert j('diagnostics/decision_state.json')['fresh_audit_consumed'] is False
def test_v838_not_started(): assert j('diagnostics/decision_state.json')['v838_started'] is False and not (ROOT/'experiments/v837_primitive_invention/v838').exists()
def test_no_heldout_fit_after_null_freeze(): assert j('raw/heldout_backend_results.json')['rows']==[]
def test_historical_robustness_did_not_upgrade(): assert j('raw/historical_validation_robustness.json')['may_upgrade_failed_family'] is False
def test_failure_ledgers_identical(): assert j('raw/failure_ledger.json')==j('diagnostics/failure_ledger.json')
def test_failure_ledger_has_engineering_entries(): assert sum(e['failure_type']=='ENGINEERING_FAILURE' for e in j('raw/failure_ledger.json')['entries'])>=2
def test_protected_historical_tree_unchanged_from_start():
    out=subprocess.check_output(['git','diff','--name-only',f'{START}..HEAD','--','experiments/v837_primitive_invention/v837an','experiments/v837_primitive_invention/v837am','experiments/v837_primitive_invention/v837al','experiments/v837_primitive_invention/v837ak'],cwd=ROOT,text=True).strip();assert out==''
def test_reproduction_dispatcher_has_v837ao():
    out=subprocess.check_output(['python','scripts/reproduce_v837_recovery.py','--variant','v837ao','--stage','meta'],cwd=ROOT,text=True);assert 'experiments.v837_primitive_invention.v837ao.run_pipeline' in out and 'dry run only' in out
def test_no_cross_organism_alignment_in_implementation():
    for p in HERE.glob('*.py'):
        txt=p.read_text(encoding='utf-8')
        assert 'q_A' not in txt and 'q_B' not in txt

def test_resource_accounting_locks():
    r=j('diagnostics/resource_accounting.json');assert r['new_model_fits']==0 and r['model_optimizer_steps']==0 and r['backend_gradient_steps']==0 and r['gpu_seconds']==0.0
def test_results_claim_disciplined(): assert 'NOT established' in j('results.json')['claim']
