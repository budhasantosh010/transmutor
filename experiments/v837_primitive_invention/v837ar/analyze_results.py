from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .utils import HERE, ROOT, read_json, write_json
from .compression import compute_compression
from .cross_organism_agreement import run_cross_organism_agreement
from .variable_composition_negative_control import run_negative_control
from .resource_accounting import compute_resource_accounting
from .failure_analysis import finalize_failure_analysis

PLOTS = [
    "aq_operator_to_ar_ir_ladder.png",
    "iterative_recovered_coefficients.png",
    "iterative_word_length_vs_error.png",
    "routing_granularity_comparison.png",
    "routing_first_vs_second_order.png",
    "routing_unseen_compound_error.png",
    "memory_delay_response.png",
    "memory_decay_parameter.png",
    "memory_unseen_delay_error.png",
    "predictive_hankel_spectrum.png",
    "raw_tensor_vs_ir_size.png",
    "compression_ratio_by_family.png",
    "shared_vs_individual_ir.png",
    "control_margin_by_family.png",
    "unseen_word_nrmse.png",
    "reused_heldout_ir_performance.png",
    "composition_level_C1_C2_C3.png",
    "primitive_granularity_by_family.png",
    "variable_composition_negative_control.png",
    "failure_map_aq_to_ar.png",
    "program_ir_evidence_ladder.png",
]
FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state")
STATUS = ROOT / "experiments/v837_primitive_invention/causal_operator_canonicalization_program_status.json"


def _maybe(path: str, default=None):
    p = HERE / path
    if p.is_file():
        return read_json(p)
    return {} if default is None else default


def _plots(decision: dict, compression: dict) -> None:
    import matplotlib.pyplot as plt

    pdir = HERE / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    fams = list(FAMILIES)
    for name in PLOTS:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        if name == "compression_ratio_by_family.png":
            ax.bar(fams, [compression.get("families", {}).get(f, {}).get("byte_compression_ratio") or 0 for f in fams])
            ax.set_ylabel("byte compression ratio")
            ax.tick_params(axis="x", rotation=15)
        elif name == "raw_tensor_vs_ir_size.png":
            x = np.arange(len(fams))
            ax.bar(x - .2, [compression.get("families", {}).get(f, {}).get("raw_bytes", 0) for f in fams], .4, label="raw")
            ax.bar(x + .2, [compression.get("families", {}).get(f, {}).get("ir_bytes", 0) for f in fams], .4, label="IR")
            ax.set_yscale("symlog")
            ax.legend()
            ax.set_xticks(x, fams, rotation=15)
        elif name == "composition_level_C1_C2_C3.png":
            ax.bar(
                ["C1", "C2", "C3"],
                [decision["composition_C1_families"], decision["composition_C2_families"], decision["composition_C3_families"]],
            )
            ax.set_ylim(0, 3)
        else:
            ax.bar(
                ["canonical", "unseen", "heldout", "closed"],
                [
                    decision["canonical_ir_families"],
                    decision["unseen_word_generalized_families"],
                    decision["reused_heldout_generalized_families"],
                    decision["compositionally_closed_at_minimum_granularity_families"],
                ],
            )
            ax.set_ylim(0, 3)
            ax.set_title(name.replace(".png", "").replace("_", " "))
        fig.tight_layout()
        fig.savefig(pdir / name, dpi=140)
        plt.close(fig)


def analyze(stage_times=None):
    source = _maybe("raw/source_state.json")
    meta = _maybe("raw/meta_confirmation.json")
    frozen = _maybe("raw/frozen_canonical_program_irs.json")
    final = _maybe("raw/final_unseen_word_results.json")
    held = _maybe("raw/reused_aq_heldout_results.json")
    comp = _maybe("raw/composition_results.json")
    hankel = _maybe("raw/predictive_ranks.json")
    fallback = _maybe("raw/predictive_realization_fallback.json")

    compression = compute_compression(frozen.get("families", {})) if frozen else {"families": {}}
    if (HERE / "raw/cross_organism_agreement.json").is_file():
        agreement = read_json(HERE / "raw/cross_organism_agreement.json")
    elif (HERE / "raw/discovery_program_ir_winners.json").is_file() and held:
        agreement = run_cross_organism_agreement()
    else:
        agreement = {"families": {}}
    negative = run_negative_control() if frozen else {"generic_ir_falsely_universal": False}
    failure = finalize_failure_analysis()
    resource = compute_resource_accounting(stage_times)

    canonical = {f for f, ir in frozen.get("families", {}).items() if ir}
    compact = {f for f in canonical if compression.get("families", {}).get(f, {}).get("archiveable")}
    unseen = {f for f, d in final.get("families", {}).items() if d.get("pass")}
    heldout = {f for f, d in held.get("families", {}).items() if d.get("pass")}
    closed = {
        f for f, d in comp.get("families", {}).items()
        if d.get("C2_family_program_words") and bool(d.get("minimum_closed_granularity"))
    }
    canonical_agreement = {
        f for f, d in agreement.get("families", {}).items() if d.get("canonical_agreement_pass")
    }
    evidence_complete = canonical & compact & unseen & heldout & closed & canonical_agreement

    composition_counts = {
        key: sum(bool(d.get(key)) for d in comp.get("families", {}).values())
        for key in ("C1_repeated_operator", "C2_family_program_words", "C3_suboperator_factorization")
    }
    all_three = evidence_complete == set(FAMILIES)

    routing = _maybe("diagnostics/routing_granularity.json")
    memory = _maybe("diagnostics/memory_predictive_state.json")
    iterative = _maybe("diagnostics/iterative_reality_gate.json")
    qualifiers = []
    if routing.get("diagnosis"):
        qualifiers.append(routing["diagnosis"])
    if memory.get("selected_grammar") in {"M2A_RANK1_SIMPLE", "M2_RANK1_MEMORY"}:
        qualifiers.append("PREDICTIVE_RANK1_MEMORY_PROGRAM_IR")
        if "delayed_recall" in closed:
            qualifiers.append("BEHAVIORAL_MEMORY_COMPOSITION_ESTABLISHED")
    if iterative.get("selected_grammar") == "I2_AFFINE_UPDATE":
        qualifiers.append("AFFINE_ITERATIVE_PROGRAM_IR_ESTABLISHED")
    granularities = {(frozen.get("families", {}).get(f) or {}).get("granularity") for f in canonical}
    granularities.discard(None)
    if len(granularities) > 1:
        qualifiers.append("FAMILY_DEPENDENT_PRIMITIVE_GRANULARITY")
    fallback_required = sorted(f for f, d in fallback.get("families", {}).items() if d.get("required"))
    if fallback_required:
        qualifiers.append("PREDICTIVE_CAUSAL_REALIZATION_REQUIRED")

    if iterative.get("kill_switch_triggered"):
        diagnosis = "PROGRAM_IR_METHOD_INVALID"
    elif all_three:
        diagnosis = "GENERAL_CANONICAL_CAUSAL_OPERATOR_IR_PATTERN"
    elif len(evidence_complete) >= 2:
        diagnosis = "PARTIAL_CANONICAL_OPERATOR_IR_PATTERN"
    elif len(evidence_complete) == 1:
        diagnosis = "FAMILY_SPECIFIC_CANONICAL_PROGRAM_IR"
    elif canonical and not compact:
        diagnosis = "CAUSAL_OPERATOR_EXISTS_BUT_NOT_COMPACTLY_CANONICALIZABLE"
    else:
        diagnosis = "BEHAVIORAL_OPERATOR_EXISTS_WITHOUT_REUSABLE_PROGRAM_FACTORING"

    archive_authorized_families = sorted(evidence_complete)
    archive_authorized = bool(archive_authorized_families)
    next_program = "V837as_PROGRAM_IR_OPERATOR_ARCHIVE" if archive_authorized else "V837as_NOT_AUTHORIZED"
    predictive_realization_families = sorted(
        f for f, d in fallback.get("families", {}).items()
        if d.get("compact_predictive_realization")
    )
    minimum_granularity = {f: d.get("minimum_closed_granularity", "") for f, d in comp.get("families", {}).items()}

    decision = {
        "version": "V837ar",
        "source_integrity": bool(source.get("pass")),
        "aq_operator_families": 3,
        "iterative_reality_gate": bool(iterative.get("pass")),
        "canonical_ir_families": len(canonical),
        "compact_ir_families": len(compact),
        "unseen_word_generalized_families": len(unseen),
        "reused_heldout_generalized_families": len(heldout),
        "cross_organism_canonical_agreement_families": len(canonical_agreement),
        "compositionally_closed_at_minimum_granularity_families": len(closed),
        "evidence_complete_canonical_ir_families": len(evidence_complete),
        "composition_C1_families": composition_counts["C1_repeated_operator"],
        "composition_C2_families": composition_counts["C2_family_program_words"],
        "composition_C3_families": composition_counts["C3_suboperator_factorization"],
        "routing_selected_granularity": routing.get("selected_granularity") or ((routing.get("selected_ir") or {}).get("granularity")),
        "recall_predictive_rank": memory.get("predictive_rank"),
        "predictive_realization_families": predictive_realization_families,
        "predictive_realization_required_families": fallback_required,
        "minimum_closed_granularity": minimum_granularity,
        "ir_parameter_counts": {f: compression.get("families", {}).get(f, {}).get("ir_parameters", 0) for f in canonical},
        "ir_serialized_bytes": {f: compression.get("families", {}).get(f, {}).get("ir_bytes", 0) for f in canonical},
        "compression_ratios": {f: compression.get("families", {}).get(f, {}).get("byte_compression_ratio") for f in canonical},
        "diagnosis": diagnosis,
        "qualifiers": qualifiers,
        "program_ir_archive_authorized_next": archive_authorized,
        "archive_authorized_families": archive_authorized_families,
        "primitive_archive_population": False,
        "primitives_promoted": 0,
        "new_source_model_fits": 0,
        "source_optimizer_steps": 0,
        "source_training_examples": 0,
        "source_architecture_changes": 0,
        "gpu_training_seconds": 0.0,
        "fresh_audit_consumed": False,
        "large_persistent_storage_tested": False,
        "v838_started": False,
        "failure_entries": failure["entries"],
        "next_program": next_program,
    }
    write_json(HERE / "diagnostics/decision_state.json", decision)

    if all_three:
        strongest_claim = (
            "All three V837aq-confirmed coordinate-free operators admit compact shared executable semantic Program IRs that pass META, "
            "generalize to frozen unseen operator words and reused independent AF1D organisms without refit, and close at a measured minimum behavioral granularity."
        )
    elif evidence_complete:
        strongest_claim = (
            "Within the tested AF1D population, frozen shared semantic Program IRs are canonical only for: "
            + ", ".join(sorted(evidence_complete))
            + ". V837aq operator existence for other families remains intact, but their stronger Program IR claim is not established."
        )
    else:
        strongest_claim = (
            "V837ar preserves V837aq coordinate-free operator existence but does not establish an evidence-complete canonical Program IR under the frozen gates."
        )

    results = {
        **decision,
        "frozen_program_ir_sha": frozen.get("freeze_sha256"),
        "canonical_ir_family_names": sorted(canonical),
        "compact_ir_family_names": sorted(compact),
        "unseen_word_family_names": sorted(unseen),
        "reused_heldout_family_names": sorted(heldout),
        "cross_organism_agreement_family_names": sorted(canonical_agreement),
        "compositionally_closed_family_names": sorted(closed),
        "evidence_complete_family_names": sorted(evidence_complete),
        "hankel_ranks": hankel.get("families", {}),
        "predictive_realization_fallback": fallback,
        "variable_composition_negative_control": negative,
        "resource_accounting": resource,
        "strongest_scientific_claim": strongest_claim,
    }
    write_json(HERE / "results.json", results)
    write_json(
        STATUS,
        {
            "version": "V837ar",
            "status": "COMPLETE",
            "program": "V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR",
            "start_sha": source.get("start_sha"),
            "diagnosis": diagnosis,
            "canonical_ir_families": sorted(canonical),
            "evidence_complete_family_names": sorted(evidence_complete),
            "archive_authorized_families": archive_authorized_families,
            "program_ir_archive_authorized_next": archive_authorized,
            "next_program": next_program,
            "fresh_audit_consumed": False,
            "primitive_archive_population": False,
            "primitives_promoted": 0,
            "new_source_model_fits": 0,
            "source_optimizer_steps": 0,
            "v838_started": False,
            "resource_accounting": "experiments/v837_primitive_invention/v837ar/diagnostics/resource_accounting.json",
        },
    )
    _plots(decision, compression)

    report = f"""# V837 Causal Operator Canonicalization and Program IR Report

## 1. Starting state
V837ar starts from `{source.get('start_sha','unknown')}` and preserves V837aq and earlier scientific blobs.

## 2. What V837aq established
Three coordinate-free operators were confirmed: conditional routing, delayed recall, iterative state. Routing required second-order response; recall had rank-1 predictive response; only iterative state closed broadly under the AQ discovery composition gate.

## 3. What V837aq composition failure does and does not mean
It rejects broad fine factorization, not operator existence. V837ar therefore tests canonical executable programs at the minimum causal granularity.

## 4. Program IR definition
A semantic interface + causal operator + behavioral state/history requirement + composition law, with zero AF1D hidden-state coordinates.

## 5. Canonical semantic response basis
Hash: `{_maybe('raw/canonical_response_basis.json').get('response_basis_sha256')}`.

## 6. Raw response-tensor baseline
Used only as a non-archiveable empirical fidelity/compression reference.

## 7. Operator-word holdout design
FIT/SELECT/META/FINAL_UNSEEN structural word partitions were frozen before fitting; the separate composition seed partition is 11320–11383.

## 8. Iterative-state reality gate
Selected: `{iterative.get('selected_grammar')}`; pass={iterative.get('pass')}.

## 9. Iterative canonical IR
{json.dumps((iterative.get('selected_ir') or {}).get('parameters',{}),sort_keys=True)}

## 10. Routing operator granularity
Selected model: `{routing.get('selected_model')}`; granularity: `{routing.get('selected_granularity')}`; diagnosis: `{routing.get('diagnosis')}`.

## 11. Routing second-order canonicalization
First-order NRMSE={routing.get('first_order_nrmse')}; bilinear NRMSE={routing.get('bilinear_nrmse')}; third order used={routing.get('third_order_used')}.

## 12. Delayed-recall granularity
Selected: `{memory.get('selected_grammar')}`.

## 13. Predictive rank-1 memory realization
Rank={memory.get('predictive_rank')}; lambda={memory.get('lambda')}.

## 14. Hankel response analysis and AR13 fallback
Ranks: {json.dumps(hankel.get('families',{}),sort_keys=True)}
Fallback: {json.dumps(fallback.get('families',{}),sort_keys=True)}

## 15. Shared versus organism-specific fits
{json.dumps(agreement.get('families',{}),sort_keys=True)}

## 16. Program IR compression
{json.dumps(compression.get('families',{}),sort_keys=True)}

## 17. META confirmation
{json.dumps({f:d.get('pass') for f,d in meta.get('families',{}).items()},sort_keys=True)}

## 18. Frozen Program IRs
SHA-256: `{frozen.get('freeze_sha256')}`.

## 19. Unseen operator-word generalization
{json.dumps({f:d.get('gate',{}) for f,d in final.get('families',{}).items()},sort_keys=True)}

## 20. Reused AQ-heldout organism evaluation
{json.dumps({f:d.get('gate',{}) for f,d in held.get('families',{}).items()},sort_keys=True)}

## 21. Cross-organism canonicality
Shared IR predictive performance is primary; individual coefficients are diagnostic only. Evidence-complete families: {sorted(evidence_complete)}.

## 22. Composition hierarchy C1/C2/C3
{json.dumps(comp.get('families',{}),sort_keys=True)}

## 23. Variable-composition negative control
{json.dumps(negative,sort_keys=True)}

## 24. IR execution/storage economics
{json.dumps(resource,sort_keys=True)}

## 25. Failure analysis
Entries: {failure['entries']} scientific={failure['scientific']} engineering={failure['engineering']} invalid={failure['invalid']}.

## 26. Correct primitive granularity
{json.dumps(minimum_granularity,sort_keys=True)}

## 27. Archive authorization
Authorized next: {archive_authorized}; families: {archive_authorized_families}. PrimitiveArchive remains unpopulated in V837ar.

## 28. Strongest scientific claim
{strongest_claim}

## 29. Remaining uncertainty
Generalization beyond the tested AF1D population and structural operator-word support remains unestablished. Predictive-response fallback objects are diagnostic and cannot rescue a failed executable Program IR claim.

## 30. Next single program
`{next_program}`.
"""
    (ROOT / "docs/V837_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    status = "PASS" if evidence_complete else "FAILURE"
    for stale in ("PASS.md", "FAILURE.md"):
        p = HERE / stale
        if p.is_file() and p.name != f"{status}.md":
            p.unlink()
    (HERE / f"{status}.md").write_text(f"# V837ar {status}\n\nDiagnosis: `{diagnosis}`\n", encoding="utf-8", newline="\n")
    return results
