from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.trajectory_eval import phase_at_timestep, semantic_timesteps

from .abstract_rollout import rollout_from_setpoint
from .geometry_runtime import read_state, set_state
from .ood_diagnostics import compare as ood_compare
from .quotient_pairs import select_quotient_pairs
from .setpoint_eval import d90_for, _refs
from .setpoint_grid import family_grid
from .utils import HERE, write_json


def _trajectory_disagreement(geometry: dict, q: np.ndarray, ep, start_t: int, ta: np.ndarray, tb: np.ndarray, rz: float) -> float:
    values = []
    for t in semantic_timesteps(ep):
        if t <= start_t:
            continue
        phase = phase_at_timestep(ep, t)
        if phase is None:
            continue
        za = read_state(geometry, q, ta[t], phase)
        zb = read_state(geometry, q, tb[t], phase)
        values.append((za - zb) / max(float(rz), 1e-12))
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def evaluate_quotient(geometry: dict, q: np.ndarray, eval_seeds: list[int], partition_name: str = "AP_QUOTIENT", *, writer_fit_seeds: list[int] | None = None) -> dict:
    oid = geometry["organism_id"]
    family = geometry["family"]
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    rz = float(family_grid(family)["semantic_range"])
    d90 = d90_for(oid, family, Q, writer_fit_seeds)
    refs = _refs(oid, family, writer_fit_seeds)
    pair_spec = select_quotient_pairs(geometry, Q, eval_seeds, partition_name)
    data = pair_traces(oid, family, eval_seeds)
    mask, _ = _eligibility(data, family)
    states = data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]), data["base_trace"].states.shape[1], 40)
    task = task_by_name(family)
    samples = []
    invalid_alignment = 0
    for row in pair_spec["rows"]:
        phase = row["phase"]
        ai = int(row["anchor_episode_index"])
        di = int(row["donor_episode_index"])
        at = int(row["anchor_timestep"])
        dt = int(row["donor_timestep"])
        anchor = states[ai, at].astype(np.float64)
        donor = states[di, dt].astype(np.float64)
        za = read_state(geometry, Q, anchor, phase)
        align = set_state(geometry, Q, donor, phase, za, d90)
        if not align.get("valid"):
            invalid_alignment += 1
            continue
        donor_aligned = np.asarray(align["state"], dtype=np.float64)
        aligned_z = read_state(geometry, Q, donor_aligned, phase)
        if abs(aligned_z - za) > 0.05 * rz:
            invalid_alignment += 1
            continue
        targets = [float(x) for x in family_grid(family)["targets"]]
        target = max(targets, key=lambda x: abs(x - za))
        aset = set_state(geometry, Q, anchor, phase, target, d90)
        dset = set_state(geometry, Q, donor_aligned, phase, target, d90)
        if not aset.get("valid") or not dset.get("valid"):
            continue
        ep = data["pairs"][ai].base_episode
        abstract_target = float(rollout_from_setpoint(family, ep, at, target)["final_target"])
        # The denominator is the high-level effect of setting the anchor semantic state.
        base_expected = float(data["base_prediction"][ai].item())
        ref = refs.get(phase)
        if ref is None:
            ood_ratio = float("inf")
        else:
            iza = int(np.argmin(np.abs(np.asarray(ref["semantic"]) - za)))
            izt = int(np.argmin(np.abs(np.asarray(ref["semantic"]) - target)))
            oods = [
                float(ood_compare(donor_aligned[None, :], np.asarray(ref["states"])[iza][None, :], ref["reference"])["median_ood_ratio"]),
                float(ood_compare(np.asarray(aset["state"])[None, :], np.asarray(ref["states"])[izt][None, :], ref["reference"])["median_ood_ratio"]),
                float(ood_compare(np.asarray(dset["state"])[None, :], np.asarray(ref["states"])[izt][None, :], ref["reference"])["median_ood_ratio"]),
            ]
            ood_ratio = max(oods)
        # For quotient comparison both states receive the exact same future input sequence (anchor episode).
        samples.append({**row, "anchor_state": anchor, "donor_aligned": donor_aligned,
                        "anchor_set": np.asarray(aset["state"]), "donor_set": np.asarray(dset["state"]),
                        "target": float(target), "abstract_target": abstract_target, "base_prediction": base_expected, "anchor_episode_index": ai,
                        "ood_ratio": ood_ratio})
    if not samples:
        return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],
                "k":int(geometry["k"]),"chart_family":geometry["chart_family"],"partition":partition_name,
                "pair_count":0,"invalid_alignment":invalid_alignment,"pass":False,"failure_code":"QUOTIENT_RESIDUAL_SENSITIVITY"}
    idx = np.asarray([s["anchor_episode_index"] for s in samples], dtype=np.int64)
    times = np.asarray([s["anchor_timestep"] for s in samples], dtype=np.int64)
    obs = data["base_obs"][torch.as_tensor(idx, dtype=torch.long)]
    lengths = data["base_lengths"][torch.as_tensor(idx, dtype=torch.long)]
    pa = build_patch_plan("STATE40", np.stack([s["anchor_set"] for s in samples]), times)
    pb = build_patch_plan("STATE40", np.stack([s["donor_set"] for s in samples]), times)
    with torch.no_grad():
        ya, ta = run_instrumented(data["model"], obs, lengths, patch_plan=pa, return_trace=True)
        yb, tb = run_instrumented(data["model"], obs, lengths, patch_plan=pb, return_trace=True)
    ya = ya.detach().cpu().numpy().astype(np.float64)
    yb = yb.detach().cpu().numpy().astype(np.float64)
    ta_np = ta.states.detach().cpu().numpy().reshape(len(samples), ta.states.shape[1], 40)
    tb_np = tb.states.detach().cpu().numpy().reshape(len(samples), tb.states.shape[1], 40)
    rs = []
    traj = []
    final_norm = []
    success_disagreement = []
    set_response = []
    for j, sample in enumerate(samples):
        denom = abs(float(sample["abstract_target"]) - float(sample["base_prediction"])) + 1e-8
        rs.append(abs(float(ya[j]) - float(yb[j])) / denom)
        final_norm.append(abs(float(ya[j]) - float(yb[j])) / denom)
        ep = data["pairs"][sample["anchor_episode_index"]].base_episode
        traj.append(_trajectory_disagreement(geometry, Q, ep, int(sample["anchor_timestep"]), ta_np[j], tb_np[j], rz))
        # A semantic set-response difference is measured by immediate readback after alignment + common SET.
        za = read_state(geometry, Q, sample["anchor_set"], sample["phase"])
        zb = read_state(geometry, Q, sample["donor_set"], sample["phase"])
        set_response.append(abs(za - zb) / max(rz, 1e-12))
        target = float(sample["abstract_target"])
        sa = task.success(float(ya[j]), target)
        sb = task.success(float(yb[j]), target)
        success_disagreement.append(float(sa != sb))
    metrics = {
        "median_residual_sensitivity": float(np.median(rs)),
        "p90_residual_sensitivity": float(np.quantile(rs, 0.90)),
        "canonical_trajectory_disagreement": float(np.median(traj)),
        "final_prediction_disagreement": float(np.median(final_norm)),
        "task_success_disagreement": float(np.mean(success_disagreement)),
        "set_response_disagreement": float(np.median(set_response)),
        "median_ood_ratio": float(np.median([s["ood_ratio"] for s in samples])),
        "pair_count": len(samples),
        "median_state_distance": float(np.median([s["state_distance"] for s in samples])),
        "median_projected_state_distance": float(np.median([s["projected_state_distance"] for s in samples])),
    }
    passed = bool(metrics["median_residual_sensitivity"] <= 0.20 and metrics["p90_residual_sensitivity"] <= 0.40
                  and metrics["canonical_trajectory_disagreement"] <= 0.10 and metrics["final_prediction_disagreement"] <= 0.10
                  and metrics["task_success_disagreement"] <= 0.10 and metrics["median_ood_ratio"] <= 2.0)
    return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],"k":int(geometry["k"]),
            "chart_family":geometry["chart_family"],"writer_family":geometry.get("writer_family","AUTO"),
            "phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"pair_count":len(samples),
            "invalid_alignment":invalid_alignment,"metrics":metrics,"pass":passed,
            "failure_code":None if passed else "QUOTIENT_RESIDUAL_SENSITIVITY"}


def save_quotient(rows: list[dict]) -> None:
    payload={"version":"V837ap","rows":rows}
    write_json(HERE/"raw/quotient_results.json",payload)
    write_json(HERE/"diagnostics/quotient_sufficiency.json",payload)
