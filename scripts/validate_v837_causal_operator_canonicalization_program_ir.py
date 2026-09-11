from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = ROOT / "experiments/v837_primitive_invention/v837ar"
START = "287d37743751554c68742bc275b558d7f9bb1f95"
AQ_CONTRACT = "461faadf9cb9cc1262c9a288236c08875ae5212685bb8dd06e6a4ed92768d9af"
FAMS = ("conditional_routing", "delayed_recall", "iterative_state")
PLOTS = {
    "aq_operator_to_ar_ir_ladder.png", "iterative_recovered_coefficients.png", "iterative_word_length_vs_error.png",
    "routing_granularity_comparison.png", "routing_first_vs_second_order.png", "routing_unseen_compound_error.png",
    "memory_delay_response.png", "memory_decay_parameter.png", "memory_unseen_delay_error.png", "predictive_hankel_spectrum.png",
    "raw_tensor_vs_ir_size.png", "compression_ratio_by_family.png", "shared_vs_individual_ir.png", "control_margin_by_family.png",
    "unseen_word_nrmse.png", "reused_heldout_ir_performance.png", "composition_level_C1_C2_C3.png",
    "primitive_granularity_by_family.png", "variable_composition_negative_control.png", "failure_map_aq_to_ar.png",
    "program_ir_evidence_ladder.png",
}


def j(rel: str):
    return json.loads((HERE / rel).read_text(encoding="utf-8"))


def req(value, message: str):
    if not value:
        raise AssertionError(message)


def require_files(paths):
    for rel in paths:
        req((HERE / rel).is_file(), f"missing {rel}")


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_common() -> dict:
    require_files([
        "README.md", "RESEARCH_SPEC.md", "FAILURE_ANALYSIS.md", "config.json",
        "frozen_causal_operator_canonicalization_gate.json", "raw/source_state.json", "raw/v837aq_contract_import.json",
        "raw/frozen_source_folds.json", "raw/canonical_response_basis.json", "raw/frozen_operator_word_partitions.json",
        "raw/reference_response_tables.json", "raw/failure_ledger.json", "diagnostics/authorization.json",
        "diagnostics/source_integrity.json", "diagnostics/aq_contract_integrity.json", "diagnostics/response_basis_integrity.json",
        "diagnostics/word_partition_integrity.json", "diagnostics/iterative_reality_gate.json", "diagnostics/failure_ledger.json",
    ])
    gate = j("frozen_causal_operator_canonicalization_gate.json")
    req(gate["start_sha"] == START, "start lineage")
    req(gate["v837aq_program_contract_hash"] == AQ_CONTRACT, "AQ program contract hash")
    req(gate["routing_order_ceiling"] == 2, "routing order ceiling")
    req(gate["program_ir_parameter_ceiling"] == 256, "parameter ceiling")
    req(gate["fresh_audit_consumed"] is False, "fresh audit at gate")
    req(gate["primitive_archive_population"] is False and gate["primitives_promoted"] == 0, "archive lock at gate")
    req(gate["v838_started"] is False, "V838 at gate")

    source = j("raw/source_state.json")
    req(source["pass"] is True, "source integrity")
    req(source["start_sha"] == START, "source start")
    req(source["new_source_model_fits"] == 0 and source["source_optimizer_steps"] == 0, "source training changed")
    req(all(v["start_blob_sha256"] == v["head_blob_sha256"] for v in source["protected"].values()), "protected V837aq drift")

    imp = j("raw/v837aq_contract_import.json")
    req(imp["confirmed_families"] == list(FAMS), "AQ families")
    req(imp["operator_orders"] == {"conditional_routing": 2, "delayed_recall": 1, "iterative_state": 1}, "AQ orders")
    req(imp["predictive_rank99"] == {"conditional_routing": 4, "delayed_recall": 1, "iterative_state": 3}, "AQ ranks")

    basis = j("raw/canonical_response_basis.json")
    req(basis["organism_id_in_basis"] is False and basis["semantic_units_only"] is True, "semantic basis")
    req(basis.get("hidden_state_coordinates_in_basis") is False, "hidden coordinate in basis")
    req(basis.get("required_second_order_families") == ["conditional_routing"], "routing second-order basis")

    words = j("raw/frozen_operator_word_partitions.json")
    for family, roles in words["partitions"].items():
        sets = [set(values) for values in roles.values()]
        req(sum(map(len, sets)) == len(set().union(*sets)), f"word leakage {family}")

    ledger = j("raw/failure_ledger.json")
    req(ledger == j("diagnostics/failure_ledger.json"), "failure-ledger mirror")
    req(all(e.get("do_not_retry_unchanged") is True and e.get("reproduction_command") for e in ledger["entries"]), "failure memory incomplete")
    req(all(e.get("failure_type") in {"SCIENTIFIC_FAILURE", "ENGINEERING_FAILURE", "INVALID_EVIDENCE"} for e in ledger["entries"]), "failure type")
    return j("diagnostics/iterative_reality_gate.json")


def _validate_full_path() -> None:
    require_files([
        "raw/iterative_ir_fits.json", "raw/routing_ir_fits.json", "raw/memory_ir_fits.json", "raw/predictive_hankel.json",
        "raw/predictive_ranks.json", "raw/discovery_program_ir_winners.json", "raw/pre_meta_compression.json",
        "raw/meta_confirmation.json", "raw/frozen_canonical_program_irs.json", "raw/final_unseen_word_results.json",
        "raw/reused_aq_heldout_results.json", "raw/cross_organism_agreement.json", "raw/predictive_realization_fallback.json",
        "raw/composition_results.json", "raw/compression_results.json", "raw/variable_composition_negative_control.json",
        "diagnostics/routing_granularity.json", "diagnostics/routing_interaction_order.json", "diagnostics/memory_predictive_state.json",
        "diagnostics/hankel_rank.json", "diagnostics/minimality.json", "diagnostics/shared_vs_individual_fit.json",
        "diagnostics/meta_confirmation.json", "diagnostics/program_ir_freeze.json", "diagnostics/unseen_word_generalization.json",
        "diagnostics/reused_heldout_confirmation.json", "diagnostics/cross_organism_agreement.json",
        "diagnostics/predictive_realization.json", "diagnostics/composition_hierarchy.json", "diagnostics/compression.json",
    ])
    winners = j("raw/discovery_program_ir_winners.json")
    req(winners.get("selection_refit") is False, "selection refit")
    req(winners.get("one_candidate_per_family") is True and winners.get("fallback_after_meta") is False, "selection contract")

    minimal = j("raw/pre_meta_compression.json")
    req(minimal.get("pre_meta") is True and minimal.get("coefficients_refit") is False, "pre-META minimality")

    meta = j("raw/meta_confirmation.json")
    req(meta["no_refit"] is True and meta["no_fallback"] is True, "META isolation")

    frozen = j("raw/frozen_canonical_program_irs.json")
    req(frozen["final_unseen_words_opened"] is False and frozen["reused_aq_heldout_opened"] is False, "freeze ordering")
    forbidden = {"q", "q_vector", "state40", "cell_ids", "message_edge_ids", "checkpoint_parameters", "checkpoint", "state_dict"}
    for family, ir in frozen["families"].items():
        if ir:
            req(not (forbidden & set(_walk_keys(ir))), f"neural state leaked into IR {family}")

    final = j("raw/final_unseen_word_results.json")
    req(final["opened_after_freeze"] is True and final["no_refit"] is True, "final unseen isolation")
    req(final["freeze_sha256"] == frozen["freeze_sha256"], "final freeze hash")

    held = j("raw/reused_aq_heldout_results.json")
    req(held["label"] == "REUSED_AQ_HELDOUT_ORGANISM_CONFIRMATION", "heldout label")
    req(held["no_refit"] and held["no_gain_calibration"] and held["no_bias_calibration"], "heldout calibration leak")
    req(held["freeze_sha256"] == frozen["freeze_sha256"], "heldout freeze hash")

    agreement = j("raw/cross_organism_agreement.json")
    req(agreement["individual_fit_diagnostic_only"] is True and agreement["shared_ir_is_deployed"] is True, "shared IR contract")
    req(agreement["cross_organism_state_alignment"] is False and agreement["cross_organism_q_alignment"] is False, "cross-organism neural alignment")

    pred = j("raw/predictive_realization_fallback.json")
    req(pred["diagnostic_only"] is True and pred["no_refit"] is True and pred["can_rescue_failed_family"] is False, "predictive fallback leak")

    composition = j("raw/composition_results.json")
    req(composition.get("composition_seed_range") == [11320, 11383], "composition seed range")
    req(composition.get("fit_or_selection_rows_used") is False, "composition used fit/select rows")
    for family, data in composition["families"].items():
        req("C1_repeated_operator" in data and "C2_family_program_words" in data and "C3_suboperator_factorization" in data, f"composition levels {family}")

    compression = j("raw/compression_results.json")
    for family, data in compression["families"].items():
        req((not data.get("archiveable")) or data["ir_parameters"] <= 256, f"archive ceiling {family}")
        req(data.get("reference_response_table_archiveable") is False, f"reference table archiveable {family}")


def _validate_closeout(reality_pass: bool) -> None:
    require_files(["diagnostics/resource_accounting.json", "diagnostics/decision_state.json", "results.json"])
    status_path = ROOT / "experiments/v837_primitive_invention/causal_operator_canonicalization_program_status.json"
    req(status_path.is_file(), "missing global V837ar program status")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    req(status.get("version") == "V837ar" and status.get("status") == "COMPLETE", "global V837ar status")
    req(status.get("fresh_audit_consumed") is False and status.get("v838_started") is False, "global V837ar locks")
    req((HERE / "PASS.md").is_file() or (HERE / "FAILURE.md").is_file(), "missing status marker")
    decision = j("diagnostics/decision_state.json")
    req(decision["primitive_archive_population"] is False and decision["primitives_promoted"] == 0, "final archive lock")
    req(decision["fresh_audit_consumed"] is False and decision["v838_started"] is False, "final audit/V838 lock")
    req(decision["new_source_model_fits"] == 0 and decision["source_optimizer_steps"] == 0, "final source training")
    if not reality_pass:
        req(decision["diagnosis"] == "PROGRAM_IR_METHOD_INVALID", "kill-switch diagnosis")
    req({p.name for p in (HERE / "plots").glob("*.png")} == PLOTS, "plot set")
    req((ROOT / "docs/V837_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR_REPORT.md").is_file(), "report")

    protected = subprocess.check_output(
        ["git", "diff", "--name-only", START, "--", "experiments/v837_primitive_invention/v837aq", "experiments/v837_primitive_invention/v837ap", "experiments/v837_primitive_invention/v837ao", "experiments/v837_primitive_invention/v837an", "experiments/v837_primitive_invention/v837ak"],
        cwd=ROOT,
        text=True,
    ).strip()
    req(protected == "", "protected history")
    req(not (ROOT / "experiments/v837_primitive_invention/v838").exists(), "V838")


def validate() -> int:
    reality = _validate_common()
    if reality.get("pass"):
        _validate_full_path()
    _validate_closeout(bool(reality.get("pass")))
    print("V837ar causal operator canonicalization / Program IR validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
