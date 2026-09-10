from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"experiments/v837_primitive_invention/v837ap"
START="c47649227baa0800d311ff41128d8f169ddf4f0d"
PLOTS={"canonicalization_ladder.png","v837ao_vs_v837ap_setpoint_recovery.png","global_reader_failure_map.png","nonlinear_k1_chart_family_comparison.png","binary_class_manifold_geometry.png","k2_k4_k8_reader_curve.png","chart_complexity_vs_validation.png","writer_field_conditioning.png","gradient_newton_convergence.png","random_subspace_control_distribution.png","shuffled_semantic_control.png","set_magnitude_generalization.png","projected_dimension_vs_set_recovery.png","quotient_residual_sensitivity.png","natural_commutativity.png","interventional_commutativity.png","phase_atlas_transitions.png","calibration_frontier.png","cross_organism_semantic_agreement.png","historical_robustness.png","law_audit_comparison.png","failure_map_v837ao_to_v837ap.png"}
def j(rel):return json.loads((HERE/rel).read_text(encoding="utf-8"))

def test_required_raw_artifacts():
    names=["source_state.json","frozen_source_folds.json","data_role_lock.json","v837ao_negative_anchor_reproduction.json","causal_projected_spaces.json","projected_subspace_hashes.json","chart_basis.json","chart_reader_fit.json","nonlinear_k1_fit.json","projected_chart_fit.json","inverse_writer_fit.json","gradient_writer_fit.json","tangent_field_fit.json","phase_atlas_fit.json","setpoint_results.json","sham_control_results.json","quotient_results.json","commutativity_results.json","discovery_family_geometry_winners.json","meta_confirmation.json","discovery_final_confirmation.json","frozen_family_geometry.json","heldout_backend_results.json","calibration_frontier.json","cross_organism_agreement.json","historical_validation_robustness.json","law_recovery.json","failure_ledger.json"]
    assert all((HERE/"raw"/n).is_file() for n in names)

def test_required_diagnostics():
    names=["pipeline_stage_index.json","source_integrity.json","v837ao_reproduction.json","model_fit_integrity.json","source_fold_integrity.json","discovery_split_integrity.json","historical_validation_reuse_integrity.json","setpoint_grid.json","chart_conditioning.json","chart_coverage.json","chart_monotonicity.json","writer_conditioning.json","gradient_degeneracy.json","newton_convergence.json","sham_controls.json","setpoint_magnitude_generalization.json","subspace_nesting.json","quotient_sufficiency.json","natural_commutativity.json","interventional_commutativity.json","phase_atlas.json","meta_confirmation.json","discovery_final_confirmation.json","family_geometry_freeze.json","heldout_backend_isolation.json","calibration_frontier.json","cross_organism_agreement.json","historical_validation_robustness.json","law_recovery.json","failure_ledger.json","resource_accounting.json","decision_state.json"]
    assert all((HERE/"diagnostics"/n).is_file() for n in names)

def test_required_modules():
    names=["authorization.py","source_integrity.py","source_folds.py","data_roles.py","baseline_reproduction.py","canonical_ir.py","abstract_rollout.py","projected_causal_spaces.py","chart_features.py","scalar_charts.py","polynomial_charts.py","monotone_charts.py","binary_charts.py","projected_charts.py","chart_reader.py","inverse_writer.py","gradient_writer.py","tangent_field.py","nonlinear_setter.py","phase_atlas.py","geometry_runtime.py","setpoint_grid.py","setpoint_eval.py","sham_controls.py","ood_diagnostics.py","quotient_pairs.py","quotient_eval.py","commutativity.py","candidate_selection.py","chart_discovery.py","discovery_program.py","geometry_store.py","reader_eval.py","meta_confirm.py","discovery_final.py","freeze_family_geometry.py","heldout_geometry_compiler.py","heldout_eval.py","calibration_frontier.py","cross_organism_agreement.py","historical_robustness.py","law_recovery.py","metrics.py","failure_ledger.py","resource_accounting.py","analyze_results.py","run_pipeline.py"]
    assert all((HERE/n).is_file() for n in names)

def test_exact_plot_set():assert {p.name for p in (HERE/"plots").glob("*.png")}==PLOTS
def test_plot_files_nonempty():assert all(p.stat().st_size>3000 for p in (HERE/"plots").glob("*.png"))
def test_report_exists():assert (ROOT/"docs/V837_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE_REPORT.md").is_file()
def test_program_status_exists():assert (ROOT/"experiments/v837_primitive_invention/global_coordinate_or_nonlinear_causal_state_program_status.json").is_file()
def test_freeze_hash_present():assert len(j("raw/frozen_family_geometry.json")["frozen_sha256"])==64
def test_baseline_zero_drift():assert j("raw/v837ao_negative_anchor_reproduction.json")["max_abs_metric_diff"]<=1e-9
def test_source_fold_same_as_v837ao():assert j("raw/frozen_source_folds.json")["same_as_v837ao"] is True
def test_no_archive_population():assert j("diagnostics/decision_state.json")["primitive_archive_allowed_next"] is False
def test_zero_promotions():assert j("diagnostics/decision_state.json")["primitives_promoted"]==0
def test_fresh_audit_false():assert j("diagnostics/decision_state.json")["fresh_audit_consumed"] is False
def test_v838_false():assert j("diagnostics/decision_state.json")["v838_started"] is False and not (ROOT/"experiments/v837_primitive_invention/v838").exists()
def test_zero_source_training():
    d=j("diagnostics/decision_state.json");assert d["new_source_model_fits"]==0 and d["source_optimizer_steps"]==0
def test_failure_ledgers_identical():assert j("raw/failure_ledger.json")==j("diagnostics/failure_ledger.json")
def test_failure_records_reproducible():assert all(e["reproduction_command"] and e["artifact_hashes"] for e in j("raw/failure_ledger.json")["entries"])
def test_protected_history_unchanged_from_start():
    paths=["experiments/v837_primitive_invention/v837ao","experiments/v837_primitive_invention/v837an","experiments/v837_primitive_invention/v837am","experiments/v837_primitive_invention/v837al","experiments/v837_primitive_invention/v837ak"]
    out=subprocess.check_output(["git","diff","--name-only",f"{START}..HEAD","--",*paths],cwd=ROOT,text=True).strip();assert out==""
def test_no_cross_organism_alignment_text():
    for p in HERE.glob("*.py"):
        txt=p.read_text(encoding="utf-8");assert "q_A" not in txt and "q_B" not in txt
def test_resource_locks():
    r=j("diagnostics/resource_accounting.json");assert r["new_source_model_fits"]==0 and r["source_optimizer_steps"]==0 and r["backend_gradient_steps"]==0 and r["gpu_seconds"]==0
def test_historical_validation_postfreeze_only():assert j("diagnostics/historical_validation_reuse_integrity.json")["post_freeze_only"] is True
