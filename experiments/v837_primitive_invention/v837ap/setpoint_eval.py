from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan, many_patched_predictions
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.k1_backend import _phase_arrays
from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES, representative_phase_timesteps
from experiments.v837_primitive_invention.v837ao.trajectory_eval import phase_at_timestep, semantic_timesteps

from .abstract_rollout import rollout_from_setpoint, next_state
from .chart_data import phase_dataset, pooled_phase_dataset
from .chart_reader import read_chart
from .data_roles import seeds
from .geometry_runtime import chart_for_phase, read_state, set_state
from .metrics import causal_recovery, direction_agreement
from .ood_diagnostics import compare as ood_compare, fit_reference
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .setpoint_grid import family_grid
from .sham_controls import matched_p, random_feature_rotation, random_subspaces
from .utils import HERE, deterministic_seed, write_json

BINARY = {"conditional_routing", "delayed_recall"}


def d90_for(organism_id: str, family: str, q: np.ndarray, fit_seeds: list[int] | None = None) -> float:
    data = pair_traces(organism_id, family, fit_seeds if fit_seeds is not None else seeds("AP_WRITER_FIT"))
    mask, _ = _eligibility(data, family)
    values = []
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    for phase in PHASES[family]:
        b, c, _, _ = _phase_arrays(data, mask, family, phase)
        if len(b):
            values.extend(np.linalg.norm((c - b) @ Q, axis=1).tolist())
    return float(np.quantile(values, 0.90)) if values else 1.0


def _refs(organism_id: str, family: str, fit_seeds: list[int] | None = None) -> dict:
    data = pair_traces(organism_id, family, fit_seeds if fit_seeds is not None else seeds("AP_WRITER_FIT"))
    mask, _ = _eligibility(data, family)
    out = {}
    for phase in PHASES[family]:
        b, c, z0, z1 = _phase_arrays(data, mask, family, phase)
        states = np.concatenate([b, c]) if len(b) else np.empty((0, 40))
        semantic = np.concatenate([z0, z1]) if len(b) else np.empty(0)
        if len(states):
            out[phase] = {"reference": fit_reference(states), "states": states, "semantic": semantic}
    return out


def _trajectory_error(geometry: dict, ep, start_t: int, target: float, trace_states: np.ndarray, q: np.ndarray, semantic_range: float) -> float:
    z = float(target)
    errs = []
    for t in semantic_timesteps(ep):
        if t <= start_t:
            continue
        if ep.family in BINARY:
            z = float(z)
        elif ep.family == "iterative_state":
            z = next_state(ep.family, z, {"x": float(np.asarray(ep.causal_inputs["x_t"])[t - 1])})
        else:
            z = next_state(ep.family, z, {"gain": float(np.asarray(ep.causal_inputs["gain"])[t - 1]), "drive": float(np.asarray(ep.causal_inputs["drive"])[t - 1])})
        phase = phase_at_timestep(ep, t)
        if phase is None:
            continue
        got = read_state(geometry, q, trace_states[t], phase)
        errs.append((got - z) / max(float(semantic_range), 1e-12))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0


def _add_binary_prototypes(chart: dict, h: np.ndarray, z: np.ndarray) -> dict:
    h = np.asarray(h, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if h.ndim == 2 and h.shape[1] == 1:
        x = h.reshape(-1)
        chart["prototype_minus"] = float(np.median(x[z < 0])) if np.any(z < 0) else None
        chart["prototype_plus"] = float(np.median(x[z > 0])) if np.any(z > 0) else None
        chart["prototype_separation"] = None if chart["prototype_minus"] is None or chart["prototype_plus"] is None else abs(chart["prototype_plus"] - chart["prototype_minus"])
    return chart


def _fit_chart(h: np.ndarray, z: np.ndarray, chart_family: str, k: int, binary: bool) -> dict:
    if int(k) == 1:
        chart = fit_scalar_chart(np.asarray(h).reshape(-1), z, chart_family, binary)
    else:
        chart = fit_projected_chart(h, z, chart_family, binary)
    if binary:
        chart = _add_binary_prototypes(chart, h, z)
    return chart


def _fit_control_geometry(organism_id: str, family: str, base_geometry: dict, q: np.ndarray, *, shuffle_semantics: bool = False, feature_rotation: np.ndarray | None = None, seed: int = 0, chart_fit_seeds: list[int] | None = None) -> tuple[dict, np.ndarray]:
    """Fit the exact same chart family/capacity for matched sham controls.

    Semantic shuffling is performed separately inside each frozen phase, even for
    a global chart, matching the V837ap contract.
    """
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    if feature_rotation is not None:
        Q = Q @ np.asarray(feature_rotation, dtype=np.float64)
    k = int(base_geometry["k"])
    binary = family in BINARY
    rng = np.random.default_rng(int(seed))
    chart_seeds = chart_fit_seeds if chart_fit_seeds is not None else seeds("AP_CHART_FIT")
    if base_geometry.get("phase_atlas"):
        charts = {}
        for phase in PHASES[family]:
            ch = phase_dataset(organism_id, family, chart_seeds, Q, phase)
            z = np.asarray(ch["semantic"], dtype=np.float64).copy()
            if shuffle_semantics and len(z):
                z = z[rng.permutation(len(z))]
            charts[phase] = _fit_chart(ch["h"], z, base_geometry["chart_family"], k, binary)
        return ({
            "version": "V837ap", "organism_id": organism_id, "family": family,
            "engine": base_geometry["engine"], "k": k, "chart_family": base_geometry["chart_family"],
            "writer_family": "AUTO", "phase_atlas": True, "charts_by_phase": charts,
        }, Q)
    chunks_h = []
    chunks_z = []
    for phase in PHASES[family]:
        ch = phase_dataset(organism_id, family, chart_seeds, Q, phase)
        z = np.asarray(ch["semantic"], dtype=np.float64).copy()
        if shuffle_semantics and len(z):
            z = z[rng.permutation(len(z))]
        chunks_h.append(ch["h"])
        chunks_z.append(z)
    h = np.concatenate(chunks_h) if chunks_h else np.empty((0, k))
    z = np.concatenate(chunks_z) if chunks_z else np.empty(0)
    chart = _fit_chart(h, z, base_geometry["chart_family"], k, binary)
    return ({
        "version": "V837ap", "organism_id": organism_id, "family": family,
        "engine": base_geometry["engine"], "k": k, "chart_family": base_geometry["chart_family"],
        "writer_family": "AUTO", "phase_atlas": False, "chart": chart,
    }, Q)


def evaluate_geometry(geometry: dict, q: np.ndarray, eval_partition: str = "AP_WRITER_SELECT", random_controls_count: int = 32, *, chart_fit_seeds: list[int] | None = None, writer_fit_seeds: list[int] | None = None) -> dict:
    oid = geometry["organism_id"]
    family = geometry["family"]
    k = int(geometry["k"])
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    binary = family in BINARY
    grid = family_grid(family)
    R = float(grid["semantic_range"])
    task = task_by_name(family)
    d90 = d90_for(oid, family, Q, writer_fit_seeds)
    refs = _refs(oid, family, writer_fit_seeds)
    data = pair_traces(oid, family, seeds(eval_partition))
    mask, _ = _eligibility(data, family)
    eligible = np.flatnonzero(mask)
    trace_np = data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]), data["base_trace"].states.shape[1], 40)
    samples = []
    skipped = 0
    invalid = 0
    failure_counts = defaultdict(int)

    for i in eligible:
        p = data["pairs"][i]
        ep = p.base_episode
        for phase, t in representative_phase_timesteps(ep).items():
            if phase not in refs:
                continue
            state = trace_np[i, t].astype(np.float64)
            z = read_state(geometry, Q, state, phase)
            for target in grid["targets"]:
                target = float(target)
                if abs(target - z) < 0.10 * R:
                    skipped += 1
                    continue
                setres = set_state(geometry, Q, state, phase, target, d90)
                if not setres.get("valid"):
                    invalid += 1
                    failure_counts[setres.get("failure_code", "SET_INVALID")] += 1
                    continue
                patched = np.asarray(setres["state"], dtype=np.float64)
                expected = float(rollout_from_setpoint(family, ep, t, target)["final_target"])
                corpus = refs[phase]
                j = int(np.argmin(np.abs(corpus["semantic"] - target)))
                natural = corpus["states"][j]
                ood = float(ood_compare(patched[None, :], natural[None, :], corpus["reference"])["median_ood_ratio"])
                samples.append({
                    "episode_index": int(i), "seed": int(p.base_seed), "phase": phase, "timestep": int(t),
                    "target": target, "state": state, "patched": patched, "z_before": z,
                    "expected_target": expected, "base_prediction": float(data["base_prediction"][i].item()),
                    "ood_ratio": ood, "iterations": int(setres.get("iterations", 0)),
                    "gradient_norm": setres.get("gradient_norm"),
                    "intervention_norm": float(setres.get("intervention_norm", np.linalg.norm(patched - state))),
                })

    if not samples:
        return {
            "organism_id": oid, "family": family, "engine": geometry["engine"], "k": k,
            "chart_family": geometry["chart_family"], "writer_family": geometry.get("writer_family", "AUTO"),
            "phase_atlas": bool(geometry.get("phase_atlas")), "partition": eval_partition,
            "sample_count": 0, "skipped_small_delta": skipped, "invalid_setpoints": invalid,
            "invalid_failure_counts": dict(failure_counts), "pass": False,
            "failure_code": "ABSOLUTE_SETPOINT_FAIL",
        }

    idx = np.asarray([s["episode_index"] for s in samples], dtype=np.int64)
    times = np.asarray([s["timestep"] for s in samples], dtype=np.int64)
    values = np.stack([s["patched"] for s in samples])
    obs = data["base_obs"][torch.as_tensor(idx, dtype=torch.long)]
    lengths = data["base_lengths"][torch.as_tensor(idx, dtype=torch.long)]
    plan = build_patch_plan("STATE40", values, times)
    with torch.no_grad():
        pred, trace = run_instrumented(data["model"], obs, lengths, patch_plan=plan, return_trace=True)
    pred = np.asarray(pred.detach().cpu(), dtype=np.float64)
    tracep = trace.states.detach().cpu().numpy().reshape(len(samples), trace.states.shape[1], 40)
    base = np.asarray([s["base_prediction"] for s in samples], dtype=np.float64)
    expected = np.asarray([s["expected_target"] for s in samples], dtype=np.float64)
    rec = causal_recovery(base, expected, pred)
    direction = direction_agreement(base, expected, pred)
    success = np.asarray([task.success(float(p), float(t)) for p, t in zip(pred, expected)], dtype=float)
    traj = np.asarray([
        _trajectory_error(geometry, data["pairs"][s["episode_index"]].base_episode, s["timestep"], s["target"], tracep[j], Q, R)
        for j, s in enumerate(samples)
    ])
    read_after = np.asarray([
        abs(read_state(geometry, Q, s["patched"], s["phase"]) - s["target"]) / max(R, 1e-12)
        for s in samples
    ])

    # 32 norm-matched state-space setter controls.
    base_states = np.stack([s["state"] for s in samples])
    updates = values - base_states
    norms = np.linalg.norm(updates, axis=1)
    rng = np.random.default_rng(deterministic_seed("v837ap-norm-random", oid, k, geometry["chart_family"], geometry.get("phase_atlas", False)))
    dirs = rng.normal(size=(random_controls_count, 40))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    sets = np.stack([base_states + norms[:, None] * d[None, :] for d in dirs])
    random_preds = many_patched_predictions(data["model"], obs, lengths, "STATE40", sets, times)
    norm_rec = np.stack([causal_recovery(base, expected, p) for p in random_preds])
    norm_pair = np.median(norm_rec, axis=0)

    # Equal-capacity within-phase shuffled-semantic control.
    try:
        shgeom, shq = _fit_control_geometry(
            oid, family, geometry, Q, shuffle_semantics=True,
            seed=deterministic_seed("v837ap-shuffle", oid, k, geometry["chart_family"], geometry.get("phase_atlas", False)),
        )
        shvals = []
        shvalid = True
        for s in samples:
            sr = set_state(shgeom, shq, s["state"], s["phase"], s["target"], d90_for(oid, family, shq, writer_fit_seeds))
            if not sr.get("valid"):
                shvalid = False
                break
            shvals.append(sr["state"])
        if shvalid:
            shp = build_patch_plan("STATE40", np.stack(shvals), times)
            with torch.no_grad():
                shpred = run_instrumented(data["model"], obs, lengths, patch_plan=shp, return_trace=False).detach().cpu().numpy().astype(np.float64)
            shrec = causal_recovery(base, expected, shpred)
        else:
            shrec = np.full(len(samples), -np.inf)
    except Exception:
        shrec = np.full(len(samples), -np.inf)

    # 32 Haar random k-dimensional STATE40 subspaces, same chart capacity.
    # Construct all valid sham setters first, then evaluate them in one batched model call.
    rqs = random_subspaces(40, k, random_controls_count, deterministic_seed("v837ap-random-subspace", oid, k, geometry["chart_family"], geometry.get("phase_atlas", False)))
    random_value_sets = []
    for ri, rq in enumerate(rqs):
        try:
            rgeom, rq2 = _fit_control_geometry(oid, family, geometry, rq, seed=deterministic_seed("v837ap-random-fit", oid, ri), chart_fit_seeds=chart_fit_seeds)
            rd90 = d90_for(oid, family, rq2, writer_fit_seeds)
            rvals = []
            good = True
            for s in samples:
                rr = set_state(rgeom, rq2, s["state"], s["phase"], s["target"], rd90)
                if not rr.get("valid"):
                    good = False
                    break
                rvals.append(rr["state"])
            if good:
                random_value_sets.append(np.stack(rvals))
        except Exception:
            continue
    if random_value_sets:
        rpreds = many_patched_predictions(data["model"], obs, lengths, "STATE40", np.stack(random_value_sets), times)
        rs_recs = [causal_recovery(base, expected, p) for p in rpreds]
    else:
        rs_recs = []
    rspair = np.median(np.stack(rs_recs), axis=0) if rs_recs else np.full(len(samples), -np.inf)

    # Orthogonal feature mixing is diagnostic-only; with a complete total-degree
    # polynomial basis it should retain equivalent capacity.
    feature_diag = None
    if k > 1:
        try:
            rot = random_feature_rotation(k, deterministic_seed("v837ap-feature-rotation", oid, k, geometry["chart_family"]))
            fgeom, fq = _fit_control_geometry(oid, family, geometry, Q, feature_rotation=rot, chart_fit_seeds=chart_fit_seeds)
            fvals = []
            fd90 = d90_for(oid, family, fq, writer_fit_seeds)
            for s in samples:
                fr = set_state(fgeom, fq, s["state"], s["phase"], s["target"], fd90)
                if not fr.get("valid"):
                    raise RuntimeError("feature rotation setter invalid")
                fvals.append(fr["state"])
            fp = build_patch_plan("STATE40", np.stack(fvals), times)
            with torch.no_grad():
                fpred = run_instrumented(data["model"], obs, lengths, patch_plan=fp, return_trace=False).detach().cpu().numpy().astype(np.float64)
            frec = causal_recovery(base, expected, fpred)
            feature_diag = {"median_recovery": float(np.median(frec)), "same_feature_count": True, "orthogonal_mixing": True}
        except Exception as exc:
            feature_diag = {"valid": False, "error": f"{type(exc).__name__}:{exc}", "same_feature_count": True, "orthogonal_mixing": True}

    shuffled_control_valid = bool(len(shrec) == len(samples) and np.all(np.isfinite(shrec)))
    random_subspace_controls_valid = bool(len(rs_recs) == random_controls_count)
    controls_valid = bool(shuffled_control_valid and random_subspace_controls_valid)
    strongest = np.maximum.reduce([norm_pair, shrec, rspair])
    pval = matched_p(rec, strongest, deterministic_seed("v837ap-control-p", oid, k, geometry["chart_family"], geometry.get("phase_atlas", False)))
    by_target = []
    for target in grid["targets"]:
        m = np.asarray([abs(s["target"] - float(target)) < 1e-12 for s in samples])
        if np.any(m):
            by_target.append({
                "target": float(target), "n": int(m.sum()),
                "median_recovery": float(np.median(rec[m])),
                "pass_recovery_060": bool(np.median(rec[m]) >= 0.60),
                "median_read_after_set_nrmse": float(np.median(read_after[m])),
                "median_intervention_norm": float(np.median([samples[j]["intervention_norm"] for j in np.flatnonzero(m)])),
            })
    required_targets = 2 if binary else 6
    targets_pass = sum(r["pass_recovery_060"] for r in by_target) >= required_targets
    shmed = float(np.median(shrec)) if np.all(np.isfinite(shrec)) else float("-inf")
    rsmed = float(np.median(np.stack(rs_recs))) if rs_recs else float("-inf")
    metrics = {
        "median_read_after_set_nrmse": float(np.median(read_after)),
        "median_counterfactual_recovery": float(np.median(rec)),
        "direction_agreement": float(np.mean(direction)),
        "counterfactual_task_success": float(np.mean(success)),
        "trajectory_nrmse": float(np.median(traj)),
        "median_ood_ratio": float(np.median([s["ood_ratio"] for s in samples])),
        "targets_passing_060": int(sum(r["pass_recovery_060"] for r in by_target)),
        "targets_required": required_targets,
        "norm_random_median_recovery": float(np.median(norm_rec)),
        "shuffled_semantic_median_recovery": shmed,
        "random_subspace_median_recovery": rsmed,
        "norm_random_margin": float(np.median(rec) - np.median(norm_rec)),
        "shuffled_margin": float(np.median(rec) - shmed) if np.isfinite(shmed) else float("inf"),
        "random_subspace_margin": float(np.median(rec) - rsmed) if np.isfinite(rsmed) else float("inf"),
        "paired_one_sided_permutation_p": pval,
        "shuffled_control_valid": shuffled_control_valid,
        "random_subspace_controls_valid": random_subspace_controls_valid,
        "controls_valid": controls_valid,
        "median_intervention_norm": float(np.median([s["intervention_norm"] for s in samples])),
        "median_newton_iterations": float(np.median([s["iterations"] for s in samples])),
        "d90": d90,
    }
    read_limit = 0.10 if binary else 0.05
    passed = bool(
        controls_valid
        and metrics["median_read_after_set_nrmse"] <= read_limit
        and metrics["median_counterfactual_recovery"] >= 0.70
        and metrics["direction_agreement"] >= 0.80
        and metrics["counterfactual_task_success"] >= 0.75
        and metrics["trajectory_nrmse"] <= 0.15
        and targets_pass
        and metrics["median_ood_ratio"] <= 2.0
        and metrics["norm_random_margin"] >= 0.20
        and metrics["shuffled_margin"] >= 0.20
        and metrics["random_subspace_margin"] >= 0.20
        and pval <= 0.01
    )
    return {
        "organism_id": oid, "family": family, "engine": geometry["engine"], "k": k,
        "chart_family": geometry["chart_family"], "writer_family": geometry.get("writer_family", "AUTO"),
        "phase_atlas": bool(geometry.get("phase_atlas")), "partition": eval_partition,
        "sample_count": len(samples), "skipped_small_delta": skipped, "invalid_setpoints": invalid,
        "invalid_failure_counts": dict(failure_counts), "metrics": metrics, "target_results": by_target,
        "pass": passed, "failure_code": None if passed else "ABSOLUTE_SETPOINT_FAIL",
        "control_count": {"random_subspaces": len(rs_recs), "norm_random_setters": random_controls_count, "shuffled_semantic": 1},
        "random_feature_mixing_diagnostic": feature_diag,
    }


def save_setpoint_diagnostics(rows: list[dict]) -> None:
    write_json(HERE / "raw/setpoint_results.json", {"version": "V837ap", "rows": rows})
    write_json(HERE / "raw/sham_control_results.json", {"version": "V837ap", "rows": [{
        "organism_id": r["organism_id"], "family": r["family"], "k": r["k"], "chart_family": r["chart_family"],
        "writer_family": r.get("writer_family"), "phase_atlas": r.get("phase_atlas", False),
        "metrics": {k: v for k, v in r.get("metrics", {}).items() if "random" in k or "shuffled" in k or "permutation" in k},
        "control_count": r.get("control_count"), "random_feature_mixing_diagnostic": r.get("random_feature_mixing_diagnostic"),
    } for r in rows]})
    write_json(HERE / "diagnostics/sham_controls.json", {"version": "V837ap", "rows": rows, "wrong_family_diagnostic": {"diagnostic_only": True, "gating": False, "interpretation": "Wrong-family semantics are retained as a non-gating specificity diagnostic; primary inference uses shuffled semantics, Haar random subspaces, and norm-matched setters."}})
    write_json(HERE / "diagnostics/setpoint_magnitude_generalization.json", {"version": "V837ap", "rows": [{
        "organism_id": r["organism_id"], "family": r["family"], "k": r["k"], "chart_family": r["chart_family"],
        "target_results": r.get("target_results", []), "pass": r.get("pass", False),
    } for r in rows]})
    write_json(HERE / "diagnostics/newton_convergence.json", {"version": "V837ap", "rows": [{
        "organism_id": r["organism_id"], "family": r["family"], "k": r["k"], "chart_family": r["chart_family"],
        "median_iterations": r.get("metrics", {}).get("median_newton_iterations"), "invalid_setpoints": r.get("invalid_setpoints", 0),
        "failure_counts": r.get("invalid_failure_counts", {}),
    } for r in rows]})
    write_json(HERE / "diagnostics/gradient_degeneracy.json", {"version": "V837ap", "rows": [{
        "organism_id": r["organism_id"], "family": r["family"], "k": r["k"], "chart_family": r["chart_family"],
        "gradient_degenerate_count": int(r.get("invalid_failure_counts", {}).get("NONLINEAR_READER_GRADIENT_DEGENERATE", 0)),
        "tangent_gain_degenerate_count": int(r.get("invalid_failure_counts", {}).get("TANGENT_SEMANTIC_GAIN_DEGENERATE", 0)),
    } for r in rows]})
    write_json(HERE / "diagnostics/ood.json", {"version": "V837ap", "rows": [{
        "organism_id": r["organism_id"], "family": r["family"], "k": r["k"], "chart_family": r["chart_family"],
        "median_ood_ratio": r.get("metrics", {}).get("median_ood_ratio"), "pass": bool(r.get("metrics", {}).get("median_ood_ratio", math.inf) <= 2.0),
    } for r in rows]})
