from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.v837aq.operator_metrics import response_metrics
from .operator_words import ROLE_SEEDS, iterative_word_cases, routing_word_cases, recall_word_cases
from .program_ir_interpreter import run
from .semantic_execution import discovery_ids, predict_episodes
from .utils import AQ, read_json


def _ir_predict(family, ir, word, variant="identity"):
    if family == "conditional_routing":
        data = {"control": word["control"], "A": word["A"], "B": word["B"]}
        if variant == "label_shuffle": data["A"], data["B"] = data["B"], data["A"]
        if variant == "magnitude_shuffle": data["A"], data["B"] = -data["A"], -data["B"]
        return float(run(ir, data))
    if family == "iterative_state":
        xs = list(word["x"])
        if variant == "time_shuffle":
            xs = list(reversed(xs))
        if variant == "magnitude_shuffle":
            # Shuffle only the magnitudes of the semantic perturbations while
            # preserving their signs and time locations. Reconstruct the
            # unperturbed values from the recorded applied deltas, then rotate
            # perturbation magnitudes deterministically across those locations.
            mods = [(int(j), float(delta)) for j, delta in word.get("actual_mods", [])]
            if len(mods) > 1:
                magnitudes = [abs(delta) for _, delta in mods]
                rotated = magnitudes[1:] + magnitudes[:1]
                for (j, delta), magnitude in zip(mods, rotated):
                    base = float(xs[j]) - delta
                    sign = 1.0 if delta >= 0.0 else -1.0
                    xs[j] = float(np.clip(base + sign * magnitude, -1.0, 1.0))
        return float(run(ir, {"ops": [{"phase": "UPDATE", "x": float(x)} for x in xs]}))
    if family == "delayed_recall":
        delay = int(word["delay"]); value = float(word["value"])
        if variant == "time_shuffle": delay = 16 - delay
        if variant == "magnitude_shuffle": value = -value
        return float(run(ir, {"ops": [{"phase": "WRITE", "value": value}] + [{"phase": "HOLD"} for _ in range(delay)] + [{"phase": "READ"}]}))
    raise KeyError(family)


def _words_and_source(family: str, role: str):
    fn = {"conditional_routing": routing_word_cases, "iterative_state": iterative_word_cases, "delayed_recall": recall_word_cases}[family]
    source = []; words = []
    for oid in discovery_ids(family):
        batch = [w for seed in ROLE_SEEDS[role] for w in fn(seed, role)]
        actual = predict_episodes(oid, [w["episode"] for w in batch])
        source.extend(actual.tolist()); words.extend(batch)
    return words, np.asarray(source, dtype=float)


def _wrong_phase_invariance(family: str) -> dict:
    src = read_json(AQ / f"raw/response_tensor_{family}.json")
    rows = [r for r in src.get("rows", []) if r.get("intervention") == "WRONG_PHASE_NUISANCE"]
    if not rows:
        # Recall uses phase-specific distractor/query nuisance interventions rather than this exact label.
        if family == "delayed_recall":
            rows = [r for r in src.get("rows", []) if r.get("zero_effect_expected")]
    if not rows:
        return {"valid": False, "control_kind": "positive_invariance", "reason": "NO_INHERITED_WRONG_PHASE_ROWS"}
    response = np.asarray([float(r["predicted_response"]) for r in rows], dtype=float)
    nrmse = float(np.sqrt(np.mean(response ** 2)) / 2.0)
    return {
        "valid": True,
        "control_kind": "positive_invariance",
        "use_for_margin": False,
        "n": len(rows),
        "nrmse_to_zero": nrmse,
        "pass": nrmse <= 0.10,
        "source": "V837aq inherited zero-effect semantic/wrong-phase rows",
    }


def _incompetent_control(family: str, ir: dict, role: str) -> dict:
    registry = read_json(AQ / "raw/incompetent_control_registry.json")
    eligible = [r for r in registry.get("rows", []) if r.get("primary_operator_control_eligible") and r.get("family") == family]
    if not eligible:
        return {
            "valid": False,
            "control_kind": "negative_discriminator",
            "use_for_margin": False,
            "reason": "NO_FAMILY_COMPATIBLE_HISTORICAL_INCOMPETENT_ORGANISM",
            "inherited_registry_total": registry.get("historically_incompetent_total", 0),
        }
    fn = {"conditional_routing": routing_word_cases, "iterative_state": iterative_word_cases, "delayed_recall": recall_word_cases}[family]
    words = [w for seed in ROLE_SEEDS[role] for w in fn(seed, role)]
    ir_pred = np.asarray([_ir_predict(family, ir, w) for w in words], dtype=float)
    errors = []
    rows = []
    for record in eligible:
        actual = predict_episodes(record["organism_id"], [w["episode"] for w in words])
        err = float(np.sqrt(np.mean((ir_pred - actual) ** 2)) / 2.0)
        errors.append(err)
        rows.append({"organism_id": record["organism_id"], "engine": record["engine"], "nrmse": err})
    return {
        "valid": True,
        "control_kind": "negative_discriminator",
        "use_for_margin": True,
        "nrmse": float(np.mean(errors)),
        "organisms": rows,
        "source": "V837aq historically incompetent family-compatible organism registry",
    }


def evaluate_controls(family: str, ir: dict, role: str = "META_WORDS") -> dict:
    words, source = _words_and_source(family, role)
    endpoint = float(np.mean(source)) if len(source) else 0.0
    out = {
        "endpoint_only": {
            "nrmse": float(np.sqrt(np.mean((source - endpoint) ** 2)) / 2.0) if len(source) else float("inf"),
            "valid": True,
            "control_kind": "negative_discriminator",
            "use_for_margin": True,
        },
        "response_table_reference": {
            "defined_on_unseen_words": False,
            "archiveable": False,
            "valid": False,
            "control_kind": "reference_upper_bound",
            "use_for_margin": False,
            "reason": "EMPIRICAL_TABLE_NOT_AN_EXECUTABLE_UNSEEN_WORD_MODEL",
        },
        "wrong_phase_invariance": _wrong_phase_invariance(family),
        "historically_incompetent_organisms": _incompetent_control(family, ir, role),
        "wrong_family_operator_template": {
            "valid": False,
            "control_kind": "negative_discriminator",
            "use_for_margin": False,
            "reason": "SEMANTIC_INTERFACE_MISMATCH; not coerced across families",
        },
    }
    variants = {
        "conditional_routing": ["label_shuffle", "magnitude_shuffle"],
        "iterative_state": ["time_shuffle", "magnitude_shuffle"],
        "delayed_recall": ["time_shuffle", "magnitude_shuffle"],
    }[family]
    for variant in variants:
        pred = np.asarray([_ir_predict(family, ir, word, variant) for word in words], dtype=float)
        nrmse = float(np.sqrt(np.mean((pred - source) ** 2)) / 2.0)
        if family == "delayed_recall" and variant == "time_shuffle":
            identity = np.asarray([_ir_predict(family, ir, word, "identity") for word in words], dtype=float)
            invariance_error = float(np.sqrt(np.mean((pred - identity) ** 2)) / 2.0)
            out["delay_shuffle_invariance"] = {
                "nrmse": nrmse,
                "ir_invariance_nrmse": invariance_error,
                "valid": True,
                "control_kind": "positive_invariance",
                "use_for_margin": False,
                "pass": invariance_error <= 0.10,
                "reason": "Delayed recall semantics require remembered value to survive changes in hold duration.",
            }
            continue
        out[variant] = {
            "nrmse": nrmse,
            "valid": True,
            "control_kind": "negative_discriminator",
            "use_for_margin": True,
        }
    return out
