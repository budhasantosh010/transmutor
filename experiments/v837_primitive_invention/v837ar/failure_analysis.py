from __future__ import annotations

from .utils import HERE, read_json, sha256_file, write_json
from .failure_ledger import add, make_entry, sync_central_ledger


def _artifact_hash(rel: str) -> dict:
    path = HERE / rel
    return {"path": rel, "sha256": sha256_file(path)} if path.is_file() else {"path": rel, "missing": True}


def _record(*, failure_id: str, stage: str, family: str | None, failed_gate: str,
            rules_out: str, remains_alive: str, failure_type: str = "SCIENTIFIC_FAILURE",
            candidate_ir=None, candidate_granularity=None, parameter_count=None,
            operator_order=None, predictive_rank=None, fit_evidence=None,
            selection_evidence=None, oracle_metrics=None, organism_metrics=None,
            composition_metrics=None, control_metrics=None, distance=None,
            organism_fold="AQ_DISCOVERY", artifact_rel=""):
    add(make_entry(
        failure_id=failure_id,
        stage=stage,
        family=family,
        organism_fold=organism_fold,
        candidate_ir=None if candidate_ir is None else str(candidate_ir),
        candidate_granularity=None if candidate_granularity is None else str(candidate_granularity),
        parameter_count=parameter_count,
        serialized_bytes=None,
        operator_order=operator_order,
        predictive_rank=predictive_rank,
        fit_evidence=fit_evidence or {},
        selection_evidence=selection_evidence or {},
        oracle_relative_metrics=oracle_metrics or {},
        organism_relative_metrics=organism_metrics or {},
        composition_metrics=composition_metrics or {},
        control_metrics=control_metrics or {},
        failed_gate=failed_gate,
        distance_from_gate=distance or {},
        what_this_rules_out=rules_out,
        what_remains_alive=remains_alive,
        artifact_path_hash=_artifact_hash(artifact_rel) if artifact_rel else {},
        failure_type=failure_type,
    ))


def _log_candidate(stage: str, family: str, row: dict, index: int, artifact_rel: str):
    if row.get("pass"):
        return
    metrics = row.get("metrics", {})
    grammar = row.get("grammar") or row.get("candidate") or f"candidate_{index}"
    _record(
        failure_id=f"V837ar-{stage}-{family}-candidate-{index:03d}",
        stage=stage,
        family=family,
        candidate_ir=grammar,
        candidate_granularity=row.get("granularity", grammar),
        failed_gate="FROZEN_CANDIDATE_GATE",
        fit_evidence=metrics,
        selection_evidence=metrics,
        oracle_metrics=metrics.get("oracle_relative", {}),
        organism_metrics=metrics.get("organism_relative", {}),
        rules_out=f"{grammar} at this exact frozen complexity/granularity and data partition",
        remains_alive="Only later pre-META candidates in the frozen ladder, if any; earlier V837aq operator existence is unchanged.",
        artifact_rel=artifact_rel,
    )


def _log_candidate_ladders():
    specs = (
        ("diagnostics/iterative_reality_gate.json", "AR3_ITERATIVE_REALITY", "iterative_state"),
        ("diagnostics/routing_granularity.json", "AR4_AR5_ROUTING_GRANULARITY", "conditional_routing"),
        ("diagnostics/memory_predictive_state.json", "AR4_MEMORY_GRAMMAR", "delayed_recall"),
    )
    for rel, stage, family in specs:
        path = HERE / rel
        if not path.is_file():
            continue
        payload = read_json(path)
        for index, row in enumerate(payload.get("candidates", [])):
            _log_candidate(stage, family, row, index, rel)


def _log_minimality():
    rel = "raw/pre_meta_compression.json"
    path = HERE / rel
    if not path.is_file():
        return
    payload = read_json(path)
    for family, row in payload.get("families", {}).items():
        if row.get("ir_parameters", 0) and not row.get("archiveable"):
            _record(
                failure_id=f"V837ar-AR7-COMPRESSION-{family}", stage="AR7_COMPRESSION_MINIMALITY", family=family,
                candidate_ir="DISCOVERY_SELECTED_PROGRAM_IR", parameter_count=row.get("ir_parameters"),
                failed_gate="PROGRAM_IR_PARAMETER_OR_REFERENCE_TABLE_GATE",
                fit_evidence=row, rules_out="Archiveable compact Program IR at the selected discovery grammar.",
                remains_alive="The behavioral operator may still exist; only compact archiveability fails.", artifact_rel=rel,
            )


def _log_meta_and_controls():
    rel = "raw/meta_confirmation.json"; path = HERE / rel
    if not path.is_file(): return
    payload = read_json(path)
    for family, row in payload.get("families", {}).items():
        if not row.get("pass"):
            metrics = row.get("metrics", {})
            _record(
                failure_id=f"V837ar-AR8-META-{family}", stage="AR8_META", family=family,
                candidate_ir=(row.get("candidate") or {}).get("grammar") if isinstance(row.get("candidate"), dict) else None,
                failed_gate="META_OPERATOR_WORD_CONFIRMATION", selection_evidence=row,
                oracle_metrics=metrics.get("oracle_relative", {}), organism_metrics=metrics.get("organism_relative", {}),
                control_metrics=row.get("controls", {}), rules_out="META-confirmed Program IR for the frozen discovery winner.",
                remains_alive="V837aq operator existence and the failed discovery-level candidate remain historical facts; no fallback candidate is allowed after META.", artifact_rel=rel,
            )
        candidate_nrmse = row.get("metrics", {}).get("organism_relative", {}).get("response_nrmse")
        for name, control in row.get("controls", {}).items():
            if not control.get("valid"):
                if name == "wrong_family_operator_template":
                    _record(
                        failure_id=f"V837ar-AR8-INVALID-CONTROL-{family}-{name}", stage="AR8_META_CONTROL_INTEGRITY", family=family,
                        candidate_ir=name, failed_gate="SEMANTIC_INTERFACE_COMPATIBILITY", failure_type="INVALID_EVIDENCE",
                        control_metrics=control, rules_out="Coercing a semantically incompatible family template into this family's coordinate-free control set.",
                        remains_alive="Only semantically valid matched controls are admissible.", artifact_rel=rel,
                    )
                continue
            if control.get("control_kind") == "positive_invariance" and not control.get("pass"):
                _record(
                    failure_id=f"V837ar-AR8-INVARIANCE-{family}-{name}", stage="AR8_META_CONTROL_INTEGRITY", family=family,
                    candidate_ir=name, failed_gate="WRONG_PHASE_ZERO_EFFECT", control_metrics=control,
                    rules_out="The claimed semantic IR invariance to inherited wrong-phase/nuisance perturbations.",
                    remains_alive="The operator may remain descriptive, but its semantic-interface abstraction is not cleanly invariant.", artifact_rel=rel,
                )
            if control.get("use_for_margin") and candidate_nrmse is not None and "nrmse" in control:
                margin = float(control["nrmse"] - candidate_nrmse)
                if margin < 0.05:
                    _record(
                        failure_id=f"V837ar-AR8-CONTROL-MARGIN-{family}-{name}", stage="AR8_META_CONTROL_MARGIN", family=family,
                        candidate_ir=name, failed_gate="CONTROL_ERROR_MARGIN_MIN_0.05", control_metrics=control,
                        distance={"observed_margin": margin, "required_margin": 0.05, "shortfall": 0.05 - margin},
                        rules_out="Adequate separation from this matched negative control.",
                        remains_alive="Other controls and earlier operator evidence remain valid; the META Program IR claim fails if any required margin fails.", artifact_rel=rel,
                    )


def _log_generalization(rel: str, stage: str, fold: str):
    path = HERE / rel
    if not path.is_file(): return
    payload = read_json(path)
    for family, row in payload.get("families", {}).items():
        if not row.get("pass"):
            _record(
                failure_id=f"V837ar-{stage}-{family}-family", stage=stage, family=family, organism_fold=fold,
                candidate_ir="FROZEN_PROGRAM_IR", failed_gate=stage, selection_evidence=row,
                rules_out=f"Family-level {stage} generalization under the frozen Program IR.",
                remains_alive="Earlier operator existence and any earlier Program IR evidence remain valid at their original evidence tier.", artifact_rel=rel,
            )
        for organism in row.get("rows", []):
            if not organism.get("pass"):
                _record(
                    failure_id=f"V837ar-{stage}-{family}-{organism.get('organism_id','unknown')[:16]}", stage=stage, family=family, organism_fold=fold,
                    candidate_ir="FROZEN_PROGRAM_IR", failed_gate=f"{stage}_ORGANISM_GATE",
                    oracle_metrics=organism.get("oracle_relative", {}), organism_metrics=organism.get("organism_relative", {}),
                    rules_out="Generalization to this specific frozen organism under the frozen IR.",
                    remains_alive="Family-level success may remain possible if the predeclared aggregate gate still passes.", artifact_rel=rel,
                )


def _log_predictive_fallback():
    rel="raw/predictive_realization_fallback.json"; path=HERE/rel
    if not path.is_file(): return
    for family,row in read_json(path).get("families",{}).items():
        if row.get("required") and not row.get("compact_under_parameter_ceiling"):
            _record(
                failure_id=f"V837ar-AR13-PREDICTIVE-FALLBACK-{family}", stage="AR13_PREDICTIVE_RESPONSE_REALIZATION_FALLBACK", family=family,
                candidate_ir=f"RANK{row.get('rank99')}_PREDICTIVE_RESPONSE_REALIZATION", parameter_count=row.get("rank99_realization",{}).get("factor_parameter_count"),
                predictive_rank=row.get("rank99"), failed_gate="PROGRAM_IR_PARAMETER_CEILING_256", selection_evidence=row,
                rules_out="A compact low-rank response realization under the frozen 256-parameter ceiling.",
                remains_alive="A non-compact response realization may still describe the family but cannot become an archiveable Program IR.", artifact_rel=rel,
            )


def _log_composition():
    rel="raw/composition_results.json"; path=HERE/rel
    if not path.is_file(): return
    payload=read_json(path)
    for family,row in payload.get("families",{}).items():
        for level in ("C1","C2","C3"):
            key={"C1":"C1_repeated_operator","C2":"C2_family_program_words","C3":"C3_suboperator_factorization"}[level]
            metrics=row.get(f"{level}_metrics",{})
            if metrics.get("not_applicable"):
                continue
            if not row.get(key):
                _record(
                    failure_id=f"V837ar-AR14-{family}-{level}", stage="AR14_PROGRAM_ALGEBRA_COMPOSITION_CLOSURE", family=family,
                    candidate_ir="FROZEN_PROGRAM_IR", candidate_granularity=row.get("minimum_closed_granularity") or None,
                    failed_gate=key, composition_metrics=metrics,
                    rules_out=f"{level} composition closure for this family under the frozen Program IR.",
                    remains_alive="Lower evidence tiers and any coarser granularity that separately passes remain valid.", artifact_rel=rel,
                )


def _log_negative_control():
    rel="raw/variable_composition_negative_control.json"; path=HERE/rel
    if not path.is_file(): return
    row=read_json(path)
    if row.get("generic_ir_falsely_universal"):
        _record(
            failure_id="V837ar-NEGATIVE-CONTROL-FALSE-UNIVERSAL", stage="AR15_VARIABLE_COMPOSITION_NEGATIVE_CONTROL", family="variable_composition",
            candidate_ir="ROUTING_IR_SHAPE_COMPATIBLE_PROBE", failed_gate="GENERIC_IR_FALSE_UNIVERSALITY_CONTROL",
            selection_evidence=row, rules_out="Specificity of the routing Program IR against the frozen variable-composition negative family.",
            remains_alive="The routing family result itself remains measured, but cross-family universality must not be claimed.", artifact_rel=rel,
        )


def finalize_failure_analysis():
    _log_candidate_ladders()
    _log_minimality()
    _log_meta_and_controls()
    _log_generalization("raw/final_unseen_word_results.json", "AR10_FINAL_UNSEEN", "AQ_DISCOVERY_FINAL_UNSEEN_WORDS")
    _log_generalization("raw/reused_aq_heldout_results.json", "AR11_REUSED_AQ_HELDOUT", "REUSED_AQ_HELDOUT")
    _log_predictive_fallback()
    _log_composition()
    _log_negative_control()
    sync = sync_central_ledger()
    ledger = read_json(HERE / "raw/failure_ledger.json")
    summary = {
        "version": "V837ar",
        "entries": len(ledger["entries"]),
        "scientific": sum(e["failure_type"] == "SCIENTIFIC_FAILURE" for e in ledger["entries"]),
        "engineering": sum(e["failure_type"] == "ENGINEERING_FAILURE" for e in ledger["entries"]),
        "invalid": sum(e["failure_type"] == "INVALID_EVIDENCE" for e in ledger["entries"]),
        "central_sync": sync,
    }
    write_json(HERE / "diagnostics/failure_ledger.json", ledger)
    return summary
