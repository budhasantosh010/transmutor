from __future__ import annotations

from .utils import HERE, read_json, write_json


def run_predictive_realization() -> dict:
    """Record whether a frozen Program IR is itself a compact predictive realization.

    This is a diagnostic only. It performs no fitting, no fallback grammar search,
    and cannot rescue a family that failed discovery/META/freeze.
    """
    frozen = read_json(HERE / "raw/frozen_canonical_program_irs.json")
    ranks = read_json(HERE / "raw/predictive_ranks.json")
    out = {
        "version": "V837ar",
        "diagnostic_only": True,
        "no_refit": True,
        "no_fallback_search": True,
        "can_rescue_failed_family": False,
        "families": {},
    }
    for family, ir in frozen.get("families", {}).items():
        rank99 = ranks.get("families", {}).get(family, {}).get("rank99")
        if ir is None:
            out["families"][family] = {
                "frozen_ir_present": False,
                "hankel_rank99": rank99,
                "compact_predictive_realization": False,
                "reason": "NO_FROZEN_PROGRAM_IR",
            }
            continue
        state = ir.get("predictive_state") or {}
        dimension = int(state.get("dimension", 0) or 0)
        out["families"][family] = {
            "frozen_ir_present": True,
            "operator_class": ir.get("operator_class"),
            "granularity": ir.get("granularity"),
            "hankel_rank99": rank99,
            "ir_predictive_state_dimension": dimension,
            "compact_predictive_realization": bool(dimension <= 1 and ir.get("operator_class") in {"LINEAR_STATE_UPDATE_1D", "LINEAR_PHASE_MACHINE_R1"}),
            "rank1_memory_realization": bool(family == "delayed_recall" and rank99 == 1 and ir.get("operator_class") == "LINEAR_PHASE_MACHINE_R1" and dimension == 1),
        }
    write_json(HERE / "raw/predictive_realization_fallback.json", out)
    write_json(HERE / "diagnostics/predictive_realization.json", out)
    return out
