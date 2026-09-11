from __future__ import annotations

import numpy as np
from .program_ir import ir_sha256
from .utils import HERE, read_json, write_json


def _coefficient_stats(shared, individual):
    s = np.asarray(shared, dtype=float)
    deviations = []
    sign_agreement = []
    for coeffs in individual.values():
        x = np.asarray(coeffs, dtype=float)
        if len(x) != len(s):
            continue
        deviations.append(float(np.linalg.norm(x - s) / (np.linalg.norm(s) + 1e-12)))
        sign_agreement.append(float(np.mean(np.sign(x) == np.sign(s))))
    return {
        "organisms": len(deviations),
        "median_normalized_coefficient_deviation": float(np.median(deviations)) if deviations else None,
        "p90_deviation": float(np.quantile(deviations, 0.9)) if deviations else None,
        "sign_agreement": float(np.mean(sign_agreement)) if sign_agreement else None,
    }


def _fit_record(family: str, grammar: str) -> dict:
    if family == "iterative_state":
        return read_json(HERE / "raw/iterative_ir_fits.json")["fits"][grammar]
    if family == "conditional_routing":
        return read_json(HERE / "raw/routing_ir_fits.json")["fits"][grammar]
    return read_json(HERE / "raw/memory_ir_fits.json")["fits"][grammar]


def run_cross_organism_agreement() -> dict:
    winners = read_json(HERE / "raw/discovery_program_ir_winners.json")
    frozen = read_json(HERE / "raw/frozen_canonical_program_irs.json")
    unseen = read_json(HERE / "raw/final_unseen_word_results.json")
    heldout = read_json(HERE / "raw/reused_aq_heldout_results.json")
    out = {
        "version": "V837ar",
        "stage": "AR12_CROSS_ORGANISM_CANONICAL_AGREEMENT",
        "individual_fit_diagnostic_only": True,
        "diagnostic_only": True,
        "shared_ir_is_deployed": True,
        "cross_organism_state_alignment": False,
        "cross_organism_q_alignment": False,
        "families": {},
    }
    for family, winner in winners.get("families", {}).items():
        ir = frozen.get("families", {}).get(family)
        if not winner or ir is None:
            out["families"][family] = {
                "available": False,
                "canonical_agreement_pass": False,
                "reason": "NO_FROZEN_PROGRAM_IR",
            }
            continue
        fit = _fit_record(family, winner["grammar"])
        individual = fit.get("per_organism", {})
        shared = fit.get("coefficients", [])
        coeff = _coefficient_stats(shared, individual) if individual and shared else {
            "organisms": 0,
            "median_normalized_coefficient_deviation": None,
            "p90_deviation": None,
            "sign_agreement": None,
        }
        discovery_generalization = unseen.get("families", {}).get(family, {})
        independent_generalization = heldout.get("families", {}).get(family, {})
        canonical_pass = bool(discovery_generalization.get("pass") and independent_generalization.get("pass"))
        out["families"][family] = {
            "available": True,
            "shared_ir_sha256": ir_sha256(ir),
            "grammar": winner["grammar"],
            "granularity": winner.get("granularity", ir.get("granularity")),
            "discovery_unseen_gate": discovery_generalization.get("gate", {}),
            "reused_heldout_gate": independent_generalization.get("gate", {}),
            "canonical_agreement_pass": canonical_pass,
            "shared_fit_primary": True,
            "individual_coefficients_used_for_prediction": False,
            **coeff,
        }
    write_json(HERE / "raw/cross_organism_agreement.json", out)
    write_json(HERE / "diagnostics/cross_organism_agreement.json", out)
    write_json(HERE / "diagnostics/shared_vs_individual_fit.json", out)
    return out
