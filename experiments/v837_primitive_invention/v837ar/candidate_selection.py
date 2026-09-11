from __future__ import annotations

from .utils import HERE, read_json, write_json


def run_selection() -> dict:
    """Freeze one already-evaluated discovery winner per family.

    This stage never refits. The iterative, routing, and memory stages must have
    materialized their fitted coefficients and SELECT-word decisions first.
    """
    required = {
        "iterative_state": HERE / "diagnostics/iterative_reality_gate.json",
        "conditional_routing": HERE / "diagnostics/routing_granularity.json",
        "delayed_recall": HERE / "diagnostics/memory_predictive_state.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V837AR_SELECTION_MISSING_DISCOVERY_ARTIFACTS:{missing}")

    it = read_json(required["iterative_state"])
    rt = read_json(required["conditional_routing"])
    mm = read_json(required["delayed_recall"])
    winners = {"iterative_state": None, "conditional_routing": None, "delayed_recall": None}

    if not it.get("pass"):
        payload = {
            "version": "V837ar",
            "program_ir_method_invalid": True,
            "reason": "ITERATIVE_REALITY_GATE_FAILED",
            "families": winners,
            "selection_refit": False,
            "one_candidate_per_family": True,
            "fallback_after_meta": False,
        }
        write_json(HERE / "raw/discovery_program_ir_winners.json", payload)
        return payload

    winners["iterative_state"] = {"grammar": it["selected_grammar"], "ir": it.get("selected_ir")}
    if rt.get("pass"):
        winners["conditional_routing"] = {"grammar": rt["selected_grammar"], "ir": rt.get("selected_ir")}
    if mm.get("pass"):
        winners["delayed_recall"] = {"grammar": mm["selected_grammar"], "ir": mm.get("selected_ir")}

    payload = {
        "version": "V837ar",
        "program_ir_method_invalid": False,
        "families": winners,
        "selection_refit": False,
        "one_candidate_per_family": True,
        "fallback_after_meta": False,
    }
    write_json(HERE / "raw/discovery_program_ir_winners.json", payload)
    return payload
