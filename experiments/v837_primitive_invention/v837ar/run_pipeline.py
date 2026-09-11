from __future__ import annotations

import argparse
import time

from .utils import HERE
from .authorization import assert_authorized
from .source_integrity import verify_source_integrity
from .failure_ledger import initialize, add, make_entry
from .word_partitions import freeze_word_partitions, assert_disjoint
from .operator_words import freeze_word_manifest
from .response_basis import build_basis
from .response_tensor import build_reference_tables
from .iterative_ir import run_reality_gate
from .routing_ir import select_routing
from .memory_ir import select_memory
from .predictive_hankel import build_hankel
from .candidate_selection import run_selection
from .compression import compute_pre_meta_minimality
from .meta_confirm import run_meta
from .freeze_program_irs import freeze_program_irs
from .unseen_word_eval import run_unseen_words
from .reused_heldout_eval import run_reused_heldout
from .cross_organism_agreement import run_cross_organism_agreement
from .predictive_realization import run_predictive_realization
from .composition import run_composition
from .analyze_results import analyze

STAGES = (
    "source", "basis", "words", "iterative-reality", "routing", "memory",
    "predictive", "select", "minimality", "meta", "freeze", "unseen",
    "heldout", "agreement", "predictive-realization", "composition", "analyze",
)


def _require(rel: str) -> None:
    if not (HERE / rel).is_file():
        raise RuntimeError(f"V837AR_MISSING_PREDECESSOR:{rel}")


def _iterative_gate_passed() -> bool:
    import json
    return bool(json.loads((HERE / "diagnostics/iterative_reality_gate.json").read_text(encoding="utf-8"))["pass"])


def run_stage(stage: str, stage_times: dict | None = None):
    if stage not in STAGES:
        raise KeyError(stage)
    t0 = time.perf_counter()
    if stage == "source":
        initialize()
        result = assert_authorized()
        verify_source_integrity()
    elif stage == "basis":
        _require("raw/source_state.json")
        result = build_basis()
        build_reference_tables()
    elif stage == "words":
        _require("raw/canonical_response_basis.json")
        result = freeze_word_partitions()
        assert_disjoint(result)
        freeze_word_manifest()
    elif stage == "iterative-reality":
        _require("raw/frozen_operator_word_partitions.json")
        result = run_reality_gate()
    elif stage == "routing":
        _require("diagnostics/iterative_reality_gate.json")
        if not _iterative_gate_passed():
            raise RuntimeError("V837AR_ITERATIVE_KILL_SWITCH")
        result = select_routing()
    elif stage == "memory":
        _require("diagnostics/iterative_reality_gate.json")
        if not _iterative_gate_passed():
            raise RuntimeError("V837AR_ITERATIVE_KILL_SWITCH")
        result = select_memory()
    elif stage == "predictive":
        _require("raw/reference_response_tables.json")
        result = build_hankel()
    elif stage == "select":
        _require("diagnostics/iterative_reality_gate.json")
        _require("diagnostics/routing_granularity.json")
        _require("diagnostics/memory_predictive_state.json")
        result = run_selection()
    elif stage == "minimality":
        _require("raw/discovery_program_ir_winners.json")
        result = compute_pre_meta_minimality()
    elif stage == "meta":
        _require("raw/pre_meta_compression.json")
        result = run_meta()
    elif stage == "freeze":
        _require("raw/meta_confirmation.json")
        result = freeze_program_irs()
    elif stage == "unseen":
        _require("raw/frozen_canonical_program_irs.json")
        result = run_unseen_words()
    elif stage == "heldout":
        _require("raw/final_unseen_word_results.json")
        result = run_reused_heldout()
    elif stage == "agreement":
        _require("raw/reused_aq_heldout_results.json")
        result = run_cross_organism_agreement()
    elif stage == "predictive-realization":
        _require("raw/cross_organism_agreement.json")
        _require("raw/predictive_ranks.json")
        result = run_predictive_realization()
    elif stage == "composition":
        _require("raw/predictive_realization_fallback.json")
        result = run_composition()
    elif stage == "analyze":
        _require("raw/composition_results.json")
        result = analyze(stage_times)
    else:  # guarded by STAGES
        raise KeyError(stage)
    if stage_times is not None:
        stage_times[stage] = time.perf_counter() - t0
    return result


def run_all():
    times = {}
    result = None
    for stage in ("source", "basis", "words", "iterative-reality"):
        result = run_stage(stage, times)
    if not result.get("pass"):
        # The iterative family is the frozen reality gate for the Program IR method.
        # Record the method failure without opening later scientific layers.
        from .failure_analysis import finalize_failure_analysis
        finalize_failure_analysis()
        return result
    for stage in STAGES[4:]:
        result = run_stage(stage, times)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    args = parser.parse_args()
    if args.stage == "all":
        run_all()
    else:
        run_stage(args.stage, {})


if __name__ == "__main__":
    main()
