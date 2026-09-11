from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837aq.operator_metrics import family_gate, response_metrics
from .operator_words import ROLE_SEEDS
from .program_ir_interpreter import run
from .semantic_execution import discovery_ids, engine_for, predict_episodes
from .utils import HERE, read_json, write_json

COMPOSITION_SEEDS = tuple(range(11320, 11384))
FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state")


def _assert_seed_partition() -> None:
    reserved = set(COMPOSITION_SEEDS)
    for role, seeds in ROLE_SEEDS.items():
        overlap = reserved.intersection(seeds)
        if overlap:
            raise RuntimeError(f"V837AR_COMPOSITION_SEED_LEAK:{role}:{sorted(overlap)}")


def _max_abs(values) -> float:
    values = [abs(float(x)) for x in values]
    return max(values) if values else 0.0


def _natural_words(family: str) -> list[dict]:
    task = task_by_name(family)
    words = []
    for seed in COMPOSITION_SEEDS:
        episode = task.generate(int(seed), "development")
        word = {
            "seed": int(seed),
            "episode": episode,
            "oracle": float(episode.target),
        }
        if family == "iterative_state":
            word["x"] = np.asarray(episode.observations[1:, 0], dtype=float).tolist()
        elif family == "conditional_routing":
            word["control"] = float(episode.observations[1, 0])
            word["A"] = float(episode.observations[2, 1])
            word["B"] = float(episode.observations[3, 2])
        elif family == "delayed_recall":
            word["value"] = float(episode.target)
            word["delay"] = int(episode.metadata["delay"])
        else:
            raise KeyError(family)
        words.append(word)
    return words


def _ir_predict(family: str, ir: dict, word: dict) -> float:
    if family == "iterative_state":
        return float(run(ir, {"ops": [{"phase": "UPDATE", "x": float(x)} for x in word["x"]]}))
    if family == "conditional_routing":
        return float(run(ir, {"control": word["control"], "A": word["A"], "B": word["B"]}))
    if family == "delayed_recall":
        return float(
            run(
                ir,
                {
                    "ops": [{"phase": "WRITE", "value": word["value"]}]
                    + [{"phase": "HOLD"} for _ in range(word["delay"])]
                    + [{"phase": "READ"}]
                },
            )
        )
    raise KeyError(family)


def _evaluate_c2(family: str, ir: dict, words: list[dict]) -> dict:
    task = task_by_name(family)
    per_organism = []
    pooled_source = []
    pooled_oracle = []
    pooled_pred = []
    pooled_success = []
    ir_pred = np.asarray([_ir_predict(family, ir, word) for word in words], dtype=float)
    oracle = np.asarray([float(word["oracle"]) for word in words], dtype=float)
    success = [bool(task.success(float(p), float(y))) for p, y in zip(ir_pred, oracle)]

    for organism_id in discovery_ids(family):
        source = predict_episodes(organism_id, [word["episode"] for word in words])
        organism_metrics = response_metrics(source, ir_pred, success)
        oracle_metrics = response_metrics(oracle, ir_pred, success)
        passed = bool(
            organism_metrics["response_nrmse"] <= 0.10
            and oracle_metrics["response_nrmse"] <= 0.10
            and oracle_metrics["pearson"] >= 0.85
            and oracle_metrics["direction_agreement"] >= 0.85
            and oracle_metrics["perturbed_task_success"] >= 0.80
        )
        per_organism.append(
            {
                "organism_id": organism_id,
                "engine": engine_for(family, organism_id),
                "pass": passed,
                "organism_relative": organism_metrics,
                "oracle_relative": oracle_metrics,
            }
        )
        pooled_source.extend(source.tolist())
        pooled_oracle.extend(oracle.tolist())
        pooled_pred.extend(ir_pred.tolist())
        pooled_success.extend(success)

    gate = family_gate(per_organism)
    return {
        "word_count": len(words),
        "source_prediction_rows": len(words) * len(per_organism),
        "organism_relative": response_metrics(pooled_source, pooled_pred, pooled_success),
        "oracle_relative": response_metrics(pooled_oracle, pooled_pred, pooled_success),
        "per_organism": per_organism,
        "gate": gate,
        "pass": bool(gate["pass"]),
    }


def _iterative_algebra(ir: dict, words: list[dict]) -> dict:
    p = ir["parameters"]
    a, b, c = float(p["a"]), float(p["b"]), float(p.get("c", 0.0))
    c1_residuals = []
    c3_residuals = []
    for word in words:
        xs = [float(x) for x in word["x"]]
        sequential = _ir_predict("iterative_state", ir, word)
        analytic = 0.0
        n = len(xs)
        for k, x in enumerate(xs):
            analytic += (a ** (n - 1 - k)) * (b * x + c)
        c1_residuals.append(sequential - analytic)
        cut = max(1, n // 2)
        left = float(run(ir, {"ops": [{"phase": "UPDATE", "x": x} for x in xs[:cut]]}))
        right = (
            float(run(ir, {"initial": {"z": left}, "ops": [{"phase": "UPDATE", "x": x} for x in xs[cut:]]}))
            if cut < n
            else left
        )
        c3_residuals.append(sequential - right)
    return {
        "C1_exact_residual_max_abs": _max_abs(c1_residuals),
        "C3_factorization_residual_max_abs": _max_abs(c3_residuals),
        "C1_pass": _max_abs(c1_residuals) <= 1e-10,
        "C3_pass": _max_abs(c3_residuals) <= 1e-10,
        "tested_words": len(words),
    }


def _memory_algebra(ir: dict, words: list[dict]) -> dict:
    p = ir["parameters"]
    w = float(p["w"])
    lam = float(p["lambda"])
    bw = float(p.get("bw", 0.0))
    bh = float(p.get("bh", 0.0))
    br = float(p.get("br", 0.0))
    c1_residuals = []
    c3_residuals = []
    for word in words:
        value = float(word["value"])
        n = int(word["delay"])
        sequential = _ir_predict("delayed_recall", ir, word)
        z0 = w * value + bw
        if n == 0:
            closed = z0 + br
        elif abs(1.0 - lam) <= 1e-14:
            closed = z0 + n * bh + br
        else:
            closed = (lam ** n) * z0 + bh * (1.0 - lam ** n) / (1.0 - lam) + br
        c1_residuals.append(sequential - closed)

        write_state = float(run(ir, {"ops": [{"phase": "WRITE", "value": value}]}))
        hold_state = (
            float(run(ir, {"initial": {"p": write_state}, "ops": [{"phase": "HOLD"} for _ in range(n)]}))
            if n
            else write_state
        )
        read_out = float(run(ir, {"initial": {"p": hold_state}, "ops": [{"phase": "READ"}]}))
        c3_residuals.append(sequential - read_out)
    return {
        "C1_exact_residual_max_abs": _max_abs(c1_residuals),
        "C3_factorization_residual_max_abs": _max_abs(c3_residuals),
        "C1_pass": _max_abs(c1_residuals) <= 1e-10,
        "C3_pass": _max_abs(c3_residuals) <= 1e-10,
        "tested_words": len(words),
    }


def run_composition() -> dict:
    _assert_seed_partition()
    frozen = read_json(HERE / "raw/frozen_canonical_program_irs.json")
    out = {
        "version": "V837ar",
        "measured_not_inferred": True,
        "composition_seed_range": [COMPOSITION_SEEDS[0], COMPOSITION_SEEDS[-1]],
        "fit_or_selection_rows_used": False,
        "families": {},
    }
    for family in FAMILIES:
        ir = frozen["families"].get(family)
        if not ir:
            out["families"][family] = {
                "C1_repeated_operator": False,
                "C2_family_program_words": False,
                "C3_suboperator_factorization": False,
                "minimum_closed_granularity": "",
                "reason": "NULL_AT_PROGRAM_IR_FREEZE",
            }
            continue

        words = _natural_words(family)
        c2_metrics = _evaluate_c2(family, ir, words)
        c2 = bool(c2_metrics["pass"])

        if family == "iterative_state":
            algebra = _iterative_algebra(ir, words)
            c1 = bool(algebra["C1_pass"])
            c3 = bool(c2 and algebra["C3_pass"])
            minimum = "STATE_UPDATE" if c1 and c2 and c3 else ""
        elif family == "delayed_recall":
            algebra = _memory_algebra(ir, words)
            c1 = bool(algebra["C1_pass"])
            c3 = bool(c2 and algebra["C3_pass"])
            minimum = "PREDICTIVE_RANK1_MEMORY" if c1 and c2 and c3 else ""
        elif family == "conditional_routing":
            algebra = {
                "C1_applicable": False,
                "C1_reason": "ROUTE is an atomic static semantic operator, not a repeated temporal update",
                "C3_historical_anchor": "V837aq fine CONTROL/LOAD/SELECT composition failed 2/5",
            }
            c1 = False
            c3 = False
            minimum = ir.get("granularity", "") if c2 else ""
        else:
            raise KeyError(family)

        out["families"][family] = {
            "C1_repeated_operator": c1,
            "C2_family_program_words": c2,
            "C3_suboperator_factorization": c3,
            "minimum_closed_granularity": minimum,
            "metrics": algebra,
            "C2_metrics": c2_metrics,
        }

    write_json(HERE / "raw/composition_results.json", out)
    write_json(HERE / "diagnostics/composition_hierarchy.json", out)
    return out
