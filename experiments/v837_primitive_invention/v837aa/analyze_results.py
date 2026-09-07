from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837aa.candidate_law_alignment import (
    EPS,
    NUM_CELLS,
    SIGNED_PERMUTATIONS,
    VIEWS,
    align_laws,
    block_diagonal_transform,
    classify_candidate_law,
    classify_gradient_compatibility,
    clustering_quality,
    complete_linkage_assignments,
    functional_metric_matrices,
    functional_outputs,
    global_matrix_diagnostics,
    gradient_pair_metrics,
    pairwise_parameter_metrics,
    pairwise_same_matrix,
    relative_transform_index,
    transform_global_matrix,
    transform_law,
)
from experiments.v837_primitive_invention.v837aa.run_candidate_law_audit import (
    CONFIG,
    CONDITION,
    GATE,
    _coupling_seed,
    _model,
    _seeds,
    assert_science_locks,
)

HERE = Path(__file__).resolve().parent
FAMILIES = [task.name for task in all_tasks()]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_array(snapshot: dict, key: str) -> np.ndarray:
    return np.asarray(snapshot["state_dict"][key]["data"], dtype=np.float64)


def extract_laws(snapshot: dict) -> list[dict]:
    laws = []
    for i in range(NUM_CELLS):
        laws.append({
            "Ws": _state_array(snapshot, f"base.cell_ws.{i}"),
            "Wm": _state_array(snapshot, f"base.cell_wm.{i}"),
            "Wx": _state_array(snapshot, f"base.cell_wx.{i}"),
            "b": _state_array(snapshot, f"base.cell_b.{i}"),
            "Wo": _state_array(snapshot, f"base.cell_wo.{i}"),
        })
    return laws


def _torch_dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    if name not in mapping:
        raise ValueError(f"unsupported snapshot dtype: {name}")
    return mapping[name]


def restore_model(snapshot: dict):
    model = _model(int(snapshot["replicate_id"]))
    state = {}
    for name, record in snapshot["state_dict"].items():
        state[name] = torch.tensor(record["data"], dtype=_torch_dtype(record["dtype"]))
    model.load_state_dict(state, strict=True)
    return model


def _fit_key(snapshot: dict) -> str:
    return f"{snapshot['family']}::r{int(snapshot['replicate_id'])}"


def _aggregate_scalars(rows: list[dict], fields: list[str]) -> dict:
    output = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        output[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return output


def _parameter_audit(snapshot: dict, alignment: dict) -> tuple[dict, dict]:
    laws = extract_laws(snapshot)
    aligned = [transform_law(law, SIGNED_PERMUTATIONS[index].matrix) for law, index in zip(laws, alignment["assignments"])]
    raw_views = {view: pairwise_parameter_metrics(laws, view=view) for view in VIEWS}
    aligned_views = {view: pairwise_parameter_metrics(aligned, view=view) for view in VIEWS}
    raw_views["matrix_singular_values"] = {}
    for name in ("Ws", "Wm", "Wx", "Wo"):
        raw_views["matrix_singular_values"][name] = [
            [float(value) for value in np.linalg.svd(np.asarray(law[name]), compute_uv=False).tolist()]
            for law in laws
        ]
    return raw_views, aligned_views


def _synthetic_probes(family: str, replicate: int) -> dict:
    seed = deterministic_int(CONFIG["synthetic_probe_namespace"], family, replicate)
    rng = np.random.default_rng(seed)
    count = int(CONFIG["synthetic_probes_per_fit"])
    return {
        "s": rng.uniform(-1.0, 1.0, size=(count, 4)).astype(np.float64),
        "m": rng.uniform(-1.0, 1.0, size=(count, 4)).astype(np.float64),
        "x": rng.uniform(-1.0, 1.0, size=(count, 6)).astype(np.float64),
    }


def _canonicalize_pool(raw_pool: dict, assignments: list[int]) -> dict:
    state = np.asarray(raw_pool["s"], dtype=np.float64).copy()
    origin = np.asarray(raw_pool["origin_cell"], dtype=np.int64)
    for cell in range(NUM_CELLS):
        mask = origin == cell
        if np.any(mask):
            p = SIGNED_PERMUTATIONS[int(assignments[cell])].matrix.astype(np.float64)
            state[mask] = state[mask] @ p.T
    return {"s": state, "m": np.asarray(raw_pool["m"], dtype=np.float64), "x": np.asarray(raw_pool["x"], dtype=np.float64)}


def _empirical_pool(model, family: str, replicate: int) -> tuple[dict, int]:
    task = task_by_name(family)
    train_seeds, _ = _seeds()
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, _targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _prediction, trace = model(observations, lengths, return_trace=True)
    states = trace.states.detach().cpu().numpy().astype(np.float64)
    messages = trace.messages.detach().cpu().numpy().astype(np.float64)
    obs = observations.detach().cpu()
    lengths_np = lengths.detach().cpu().numpy()
    previous = np.zeros_like(states)
    previous[:, 1:, :, :] = states[:, :-1, :, :]
    b, t, _, _ = states.shape
    active = np.arange(t)[None, :] < lengths_np[:, None]
    s_rows, m_rows, x_rows, origins = [], [], [], []
    flat_obs = obs.reshape(b * t, -1)
    for cell in range(NUM_CELLS):
        visible = model._visible_input(flat_obs, cell).reshape(b, t, -1).detach().cpu().numpy().astype(np.float64)
        s_rows.append(previous[:, :, cell, :][active])
        m_rows.append(messages[:, :, cell, :][active])
        x_rows.append(visible[active])
        origins.append(np.full(int(np.sum(active)), cell, dtype=np.int64))
    s_all = np.concatenate(s_rows, axis=0)
    m_all = np.concatenate(m_rows, axis=0)
    x_all = np.concatenate(x_rows, axis=0)
    origin_all = np.concatenate(origins, axis=0)
    count = int(CONFIG["empirical_probes_per_fit"])
    rng = np.random.default_rng(deterministic_int(CONFIG["empirical_probe_namespace"], family, replicate))
    indices = rng.choice(len(s_all), size=count, replace=len(s_all) < count)
    return {
        "s": s_all[indices],
        "m": m_all[indices],
        "x": x_all[indices],
        "origin_cell": origin_all[indices],
        "pool_source_count": int(len(s_all)),
    }, 1


def _functional_views(laws: list[dict], probes: dict) -> dict:
    return {view: functional_metric_matrices(functional_outputs(laws, probes, view=view)) for view in VIEWS}


def _functional_fit(snapshot: dict, alignment: dict, synthetic: dict, raw_empirical: dict) -> tuple[dict, dict]:
    raw_laws = extract_laws(snapshot)
    aligned_laws = [transform_law(law, SIGNED_PERMUTATIONS[index].matrix) for law, index in zip(raw_laws, alignment["assignments"])]
    canonical_empirical = _canonicalize_pool(raw_empirical, alignment["assignments"])
    syn = {
        "raw": _functional_views(raw_laws, synthetic),
        "aligned": _functional_views(aligned_laws, synthetic),
    }
    raw_emp = {"s": raw_empirical["s"], "m": raw_empirical["m"], "x": raw_empirical["x"]}
    emp = {
        "raw": _functional_views(raw_laws, raw_emp),
        "aligned": _functional_views(aligned_laws, canonical_empirical),
    }
    return syn, emp


def _metric_summary(metrics: dict) -> dict:
    keys = [
        "median_pairwise_cosine",
        "median_pairwise_normalized_rmse",
        "median_pairwise_pearson",
        "median_pairwise_mean_absolute_delta",
        "mean_candidate_output_variance",
        "mean_saturation_fraction",
    ]
    return {key: float(metrics[key]) for key in keys}


def _aggregate_functional(fits: list[dict]) -> dict:
    output = {}
    for view in VIEWS:
        output[view] = {}
        for mode in ("raw", "aligned"):
            summaries = [_metric_summary(fit["views"][view][mode]) for fit in fits]
            output[view][mode] = _aggregate_scalars(summaries, list(summaries[0]))
    return output


def _gradient_laws(model) -> list[dict]:
    laws = []
    for i in range(NUM_CELLS):
        params = {
            "Ws": model.base.cell_ws[i],
            "Wm": model.base.cell_wm[i],
            "Wx": model.base.cell_wx[i],
            "b": model.base.cell_b[i],
            "Wo": model.base.cell_wo[i],
        }
        law = {}
        for name, parameter in params.items():
            if parameter.grad is None:
                law[name] = np.zeros(tuple(parameter.shape), dtype=np.float64)
            else:
                law[name] = parameter.grad.detach().cpu().numpy().astype(np.float64)
        laws.append(law)
    return laws


def _gradient_audit(model, family: str, alignment: dict) -> tuple[dict, int, int]:
    task = task_by_name(family)
    train_seeds, _ = _seeds()
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.train()
    model.zero_grad(set_to_none=True)
    predictions = model(observations, lengths)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()
    raw_laws = _gradient_laws(model)
    aligned_laws = [transform_law(law, SIGNED_PERMUTATIONS[index].matrix) for law, index in zip(raw_laws, alignment["assignments"])]
    output = {
        view: {
            "raw": gradient_pair_metrics(raw_laws, view=view),
            "aligned": gradient_pair_metrics(aligned_laws, view=view),
        }
        for view in VIEWS
    }
    model.zero_grad(set_to_none=True)
    return output, 1, 1


def _global_coupling(snapshot: dict, assignments: list[int], family: str, replicate: int) -> dict:
    u = _state_array(snapshot, "global_u")
    v = _state_array(snapshot, "global_v")
    mask = _state_array(snapshot, "cross_block_mask")
    raw = (u @ v.T) * mask
    canonical = transform_global_matrix(raw, assignments)
    pblock = block_diagonal_transform(assignments)
    rng = np.random.default_rng(deterministic_int("v837aa-global-equivariance", family, replicate))
    s = rng.normal(size=(40,))
    z = pblock @ s
    error = float(np.max(np.abs(canonical @ z - pblock @ (raw @ s))))
    return {
        "raw": global_matrix_diagnostics(raw),
        "canonical": global_matrix_diagnostics(canonical),
        "equivariance_max_abs_error": error,
    }


def _entropy(indices: list[int]) -> float:
    counts = np.asarray(list(Counter(indices).values()), dtype=np.float64)
    probabilities = counts / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _relative_basis(assignments_by_fit: list[dict]) -> dict:
    pair_records = []
    family_groups = defaultdict(list)
    replicate_groups = defaultdict(list)
    for fit in assignments_by_fit:
        family_groups[fit["family"]].append(fit)
        replicate_groups[int(fit["replicate_id"])].append(fit)
    for i in range(NUM_CELLS):
        for j in range(i + 1, NUM_CELLS):
            indices = [relative_transform_index(fit["assignments"][i], fit["assignments"][j]) for fit in assignments_by_fit]
            counts = Counter(indices)
            mode_index, mode_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            family = {}
            for name, fits in sorted(family_groups.items()):
                values = [relative_transform_index(fit["assignments"][i], fit["assignments"][j]) for fit in fits]
                mode = max(Counter(values).values()) / len(values)
                family[name] = {"exact_agreement_frequency": float(mode), "entropy_bits": _entropy(values)}
            replicate = {}
            for rep, fits in sorted(replicate_groups.items()):
                values = [relative_transform_index(fit["assignments"][i], fit["assignments"][j]) for fit in fits]
                mode = max(Counter(values).values()) / len(values)
                replicate[str(rep)] = {"exact_agreement_frequency": float(mode), "entropy_bits": _entropy(values)}
            pair_records.append({
                "cell_i": i,
                "cell_j": j,
                "mode_relative_transform_index": int(mode_index),
                "exact_transform_agreement_frequency": float(mode_count / len(indices)),
                "entropy_bits": _entropy(indices),
                "family_specific": family,
                "replicate_specific": replicate,
            })
    threshold = GATE["relative_basis_stability"]
    frequencies = np.asarray([row["exact_transform_agreement_frequency"] for row in pair_records])
    pair_fraction = float(np.mean(frequencies >= threshold["pair_exact_agreement_min"]))
    family_medians = {
        family: float(np.median([row["family_specific"][family]["exact_agreement_frequency"] for row in pair_records]))
        for family in FAMILIES
    }
    replicate_medians = {
        str(rep): float(np.median([row["replicate_specific"][str(rep)]["exact_agreement_frequency"] for row in pair_records]))
        for rep in range(5)
    }
    stable = bool(
        pair_fraction >= threshold["required_pair_fraction"]
        and min(family_medians.values()) >= threshold["family_min_median_agreement"]
    )
    matrix = np.eye(NUM_CELLS, dtype=np.float64)
    for row in pair_records:
        matrix[row["cell_i"], row["cell_j"]] = matrix[row["cell_j"], row["cell_i"]] = row["exact_transform_agreement_frequency"]
    return {
        "relative_basis_stable": stable,
        "basis_stability_subdiagnosis": threshold["stable_label"] if stable else threshold["variable_label"],
        "pair_fraction_meeting_exact_agreement_threshold": pair_fraction,
        "median_pair_exact_agreement": float(np.median(frequencies)),
        "median_pair_entropy_bits": float(np.median([row["entropy_bits"] for row in pair_records])),
        "family_median_exact_agreement": family_medians,
        "replicate_median_exact_agreement": replicate_medians,
        "pair_exact_agreement_matrix": matrix.tolist(),
        "pairs": pair_records,
    }


def _pair_agreement(a: list[int], b: list[int]) -> float:
    aa, bb = pairwise_same_matrix(a), pairwise_same_matrix(b)
    pair = np.triu_indices(len(a), 1)
    return float(np.mean(aa[pair] == bb[pair]))


def _cluster_audit(functional_syn_fits: list[dict], functional_emp_fits: list[dict]) -> dict:
    syn_by_key = {fit["fit_key"]: fit for fit in functional_syn_fits}
    emp_by_key = {fit["fit_key"]: fit for fit in functional_emp_fits}
    output = {"k_results": {}}
    for k in CONFIG["clustering_k"]:
        fit_rows = []
        same_matrices = []
        for key in sorted(syn_by_key):
            syn = syn_by_key[key]
            emp = emp_by_key[key]
            sm = syn["views"]["core"]["aligned"]
            em = emp["views"]["core"]["aligned"]
            distance = (np.asarray(sm["normalized_rmse"]) + np.asarray(em["normalized_rmse"])) / 2.0
            assignments = complete_linkage_assignments(distance, int(k))
            quality = clustering_quality(assignments, sm, em)
            row = {
                "fit_key": key,
                "family": syn["family"],
                "replicate_id": syn["replicate_id"],
                "assignments": assignments,
                **quality,
            }
            fit_rows.append(row)
            same_matrices.append(pairwise_same_matrix(assignments))
        consensus = np.mean(np.stack(same_matrices, axis=0), axis=0)
        consensus_assignments = complete_linkage_assignments(1.0 - consensus, int(k))
        agreements = [_pair_agreement(row["assignments"], consensus_assignments) for row in fit_rows]
        for row, agreement in zip(fit_rows, agreements):
            row["consensus_pairwise_agreement"] = agreement
        gate = GATE["small_type_vocabulary"]
        quality_count = sum(int(row["passes_within_type_threshold"]) for row in fit_rows)
        agreement_count = sum(int(value >= gate["consensus_pairwise_agreement_min"]) for value in agreements)
        family_quality = {
            family: sum(int(row["passes_within_type_threshold"]) for row in fit_rows if row["family"] == family)
            for family in FAMILIES
        }
        family_agreement = {
            family: float(np.median([row["consensus_pairwise_agreement"] for row in fit_rows if row["family"] == family]))
            for family in FAMILIES
        }
        passes = bool(
            quality_count >= gate["required_fit_count"]
            and min(family_quality.values()) >= gate["minimum_family_quality_fit_count"]
            and agreement_count >= gate["required_consensus_fit_count"]
            and min(family_agreement.values()) >= gate["minimum_family_median_consensus"]
        )
        pair = np.triu_indices(NUM_CELLS, 1)
        depth_distance = np.asarray([abs(i - j) for i, j in zip(pair[0], pair[1])], dtype=np.float64)
        cocluster = consensus[pair]
        depth_assoc = 0.0 if np.std(depth_distance) <= EPS or np.std(cocluster) <= EPS else float(np.corrcoef(depth_distance, cocluster)[0, 1])
        output["k_results"][str(k)] = {
            "passes": passes,
            "quality_fit_count": quality_count,
            "consensus_agreement_fit_count": agreement_count,
            "family_quality_fit_count": family_quality,
            "family_median_consensus_agreement": family_agreement,
            "consensus_assignments": consensus_assignments,
            "consensus_matrix": consensus.tolist(),
            "median_consensus_pairwise_agreement": float(np.median(agreements)),
            "mean_silhouette_like": float(np.mean([row["silhouette_like"] for row in fit_rows])),
            "candidate_stage_depth_distance_cocluster_correlation": depth_assoc,
            "fits": fit_rows,
        }
    passing = [int(k) for k, row in output["k_results"].items() if row["passes"]]
    output["lowest_passing_k"] = min(passing) if passing else None
    return output


def _view_evidence(functional_syn: list[dict], functional_emp: list[dict]) -> dict:
    gate = GATE["classification"]
    syn = {fit["fit_key"]: fit for fit in functional_syn}
    emp = {fit["fit_key"]: fit for fit in functional_emp}
    output = {}
    for view in VIEWS:
        aligned_count = 0
        raw_count = 0
        for key in syn:
            sr = syn[key]["views"][view]["raw"]
            sa = syn[key]["views"][view]["aligned"]
            er = emp[key]["views"][view]["raw"]
            ea = emp[key]["views"][view]["aligned"]
            raw_count += int(sr["median_pairwise_cosine"] >= gate["cosine_min"] and sr["median_pairwise_normalized_rmse"] <= gate["nrmse_max"] and er["median_pairwise_cosine"] >= gate["cosine_min"] and er["median_pairwise_normalized_rmse"] <= gate["nrmse_max"])
            aligned_count += int(sa["median_pairwise_cosine"] >= gate["cosine_min"] and sa["median_pairwise_normalized_rmse"] <= gate["nrmse_max"] and ea["median_pairwise_cosine"] >= gate["cosine_min"] and ea["median_pairwise_normalized_rmse"] <= gate["nrmse_max"])
        output[view] = {"raw_common_threshold_fit_count": raw_count, "aligned_common_threshold_fit_count": aligned_count}
    return output


def _plot_results(results: dict, functional_syn: list[dict], relative_basis: dict, gradient_rows: list[dict], clustering: dict) -> None:
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    syn_raw = [fit["views"]["core"]["raw"]["median_pairwise_normalized_rmse"] for fit in functional_syn]
    syn_aligned = [fit["views"]["core"]["aligned"]["median_pairwise_normalized_rmse"] for fit in functional_syn]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(syn_raw, syn_aligned)
    lo, hi = min(syn_raw + syn_aligned), max(syn_raw + syn_aligned)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("raw core functional NRMSE")
    ax.set_ylabel("aligned core functional NRMSE")
    ax.set_title("V837aa raw vs signed-permutation aligned similarity")
    fig.tight_layout(); fig.savefig(plots / "raw_vs_aligned_similarity.png", dpi=150); plt.close(fig)

    gain = [(r - a) / (r + EPS) for r, a in zip(syn_raw, syn_aligned)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(gain)), gain)
    ax.axhline(GATE["classification"]["alignment_relative_nrmse_reduction_min"], linestyle="--")
    ax.set_xlabel("family×replicate fit")
    ax.set_ylabel("relative NRMSE reduction")
    ax.set_title("V837aa functional alignment gain")
    fig.tight_layout(); fig.savefig(plots / "functional_alignment_gain.png", dpi=150); plt.close(fig)

    mean_grad = np.mean(np.stack([np.asarray(row["core"]["aligned"]["cosine"]) for row in gradient_rows], axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(mean_grad, vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_title("Aligned core gradient cosine")
    fig.tight_layout(); fig.savefig(plots / "gradient_cosine_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(np.asarray(relative_basis["pair_exact_agreement_matrix"]), vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_title("Relative signed-basis stability")
    fig.tight_layout(); fig.savefig(plots / "relative_basis_stability.png", dpi=150); plt.close(fig)

    selected_k = clustering.get("lowest_passing_k") or 2
    matrix = np.asarray(clustering["k_results"][str(selected_k)]["consensus_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Cell-type co-clustering consensus (k={selected_k})")
    fig.tight_layout(); fig.savefig(plots / "cell_type_consensus.png", dpi=150); plt.close(fig)


def main() -> int:
    assert_science_locks()
    reproduction = _load(HERE / "diagnostics" / "parent_reproduction.json")
    if not reproduction.get("parent_reproduction_valid"):
        raise SystemExit("V837aa alignment audit blocked by Y3 parent reproduction failure")
    initial_payload = _load(HERE / "raw" / "initial_parameter_snapshots.json")["snapshots"]
    trained_payload = _load(HERE / "raw" / "trained_parameter_snapshots.json")["snapshots"]
    if len(initial_payload) != 25 or len(trained_payload) != 25:
        raise RuntimeError("V837aa requires exactly 25 initial and 25 trained snapshots")
    initial_by_key = {_fit_key(row): row for row in initial_payload}
    trained_by_key = {_fit_key(row): row for row in trained_payload}
    if set(initial_by_key) != set(trained_by_key):
        raise RuntimeError("initial/trained snapshot keys differ")

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    trained_assignments, initial_assignments = [], []
    raw_parameter_fits, aligned_parameter_fits = [], []
    functional_syn_fits, functional_emp_fits = [], []
    null_fits = []
    gradient_rows = []
    global_rows = []
    diagnostic_forward_calls = 0
    diagnostic_backward_calls = 0

    for key in sorted(trained_by_key):
        trained = trained_by_key[key]
        initial = initial_by_key[key]
        family = trained["family"]
        replicate = int(trained["replicate_id"])
        trained_laws = extract_laws(trained)
        initial_laws = extract_laws(initial)
        trained_alignment = align_laws(trained_laws, view="core", max_iterations=int(CONFIG["alignment_max_iterations"]))
        initial_alignment = align_laws(initial_laws, view="core", max_iterations=int(CONFIG["alignment_max_iterations"]))
        trained_assignments.append({"fit_key": key, "family": family, "replicate_id": replicate, **trained_alignment})
        initial_assignments.append({"fit_key": key, "family": family, "replicate_id": replicate, **initial_alignment})

        raw_param, aligned_param = _parameter_audit(trained, trained_alignment)
        initial_raw_param, initial_aligned_param = _parameter_audit(initial, initial_alignment)
        raw_parameter_fits.append({"fit_key": key, "family": family, "replicate_id": replicate, "views": raw_param})
        aligned_parameter_fits.append({"fit_key": key, "family": family, "replicate_id": replicate, "views": aligned_param})

        model = restore_model(trained)
        empirical_raw, forward_calls = _empirical_pool(model, family, replicate)
        diagnostic_forward_calls += forward_calls
        synthetic = _synthetic_probes(family, replicate)
        trained_syn, trained_emp = _functional_fit(trained, trained_alignment, synthetic, empirical_raw)
        initial_syn, initial_emp = _functional_fit(initial, initial_alignment, synthetic, empirical_raw)
        functional_syn_fits.append({"fit_key": key, "family": family, "replicate_id": replicate, "views": {view: {"raw": trained_syn["raw"][view], "aligned": trained_syn["aligned"][view]} for view in VIEWS}})
        functional_emp_fits.append({"fit_key": key, "family": family, "replicate_id": replicate, "views": {view: {"raw": trained_emp["raw"][view], "aligned": trained_emp["aligned"][view]} for view in VIEWS}, "empirical_source_tuple_count": empirical_raw["pool_source_count"]})

        raw_core = raw_param["core"]
        aligned_core = aligned_param["core"]
        init_raw_core = initial_raw_param["core"]
        init_aligned_core = initial_aligned_param["core"]
        null_fits.append({
            "fit_key": key, "family": family, "replicate_id": replicate,
            "trained_parameter_core": {"raw": raw_core, "aligned": aligned_core},
            "initial_parameter_core": {"raw": init_raw_core, "aligned": init_aligned_core},
            "trained_functional_synthetic": {"raw": _metric_summary(trained_syn["raw"]["core"]), "aligned": _metric_summary(trained_syn["aligned"]["core"])},
            "initial_functional_synthetic": {"raw": _metric_summary(initial_syn["raw"]["core"]), "aligned": _metric_summary(initial_syn["aligned"]["core"])},
            "trained_functional_empirical": {"raw": _metric_summary(trained_emp["raw"]["core"]), "aligned": _metric_summary(trained_emp["aligned"]["core"])},
            "initial_functional_empirical": {"raw": _metric_summary(initial_emp["raw"]["core"]), "aligned": _metric_summary(initial_emp["aligned"]["core"])},
        })

        gradients, fwd, bwd = _gradient_audit(model, family, trained_alignment)
        diagnostic_forward_calls += fwd
        diagnostic_backward_calls += bwd
        gradient_rows.append({"fit_key": key, "family": family, "replicate_id": replicate, **gradients})
        global_rows.append({"fit_key": key, "family": family, "replicate_id": replicate, **_global_coupling(trained, trained_alignment["assignments"], family, replicate)})

    relative_basis = _relative_basis(trained_assignments)
    clustering = _cluster_audit(functional_syn_fits, functional_emp_fits)
    classification_clustering = {str(k): row for k, row in clustering["k_results"].items()}
    classification_clustering["relative_basis_stable"] = relative_basis["relative_basis_stable"]

    syn_by_key = {fit["fit_key"]: fit for fit in functional_syn_fits}
    emp_by_key = {fit["fit_key"]: fit for fit in functional_emp_fits}
    null_by_key = {fit["fit_key"]: fit for fit in null_fits}
    classification_fits = []
    for key in sorted(syn_by_key):
        syn = syn_by_key[key]["views"]["core"]
        emp = emp_by_key[key]["views"]["core"]
        null = null_by_key[key]
        classification_fits.append({
            "trained_raw_synthetic": syn["raw"],
            "trained_aligned_synthetic": syn["aligned"],
            "trained_raw_empirical": emp["raw"],
            "trained_aligned_empirical": emp["aligned"],
            "initial_raw_synthetic": null["initial_functional_synthetic"]["raw"],
            "initial_aligned_synthetic": null["initial_functional_synthetic"]["aligned"],
            "initial_raw_empirical": null["initial_functional_empirical"]["raw"],
            "initial_aligned_empirical": null["initial_functional_empirical"]["aligned"],
        })
    classification = classify_candidate_law(classification_fits, classification_clustering, GATE["classification"])

    aligned_core_gradients = [row["core"]["aligned"] for row in gradient_rows]
    gradient_compatibility = classify_gradient_compatibility(aligned_core_gradients, GATE["gradient_compatibility"])
    gradient_family = {
        family: _aggregate_scalars([row["core"]["aligned"] for row in gradient_rows if row["family"] == family], ["median_cosine", "mean_cosine", "minimum_cosine", "fraction_negative", "fraction_strongly_negative"])
        for family in FAMILIES
    }

    view_evidence = _view_evidence(functional_syn_fits, functional_emp_fits)
    syn_aggregate = _aggregate_functional(functional_syn_fits)
    emp_aggregate = _aggregate_functional(functional_emp_fits)
    trained_raw_syn = syn_aggregate["core"]["raw"]
    trained_aligned_syn = syn_aggregate["core"]["aligned"]
    trained_raw_emp = emp_aggregate["core"]["raw"]
    trained_aligned_emp = emp_aggregate["core"]["aligned"]

    null_summary_rows = []
    for row in null_fits:
        null_summary_rows.append({
            "initial_raw_synthetic_cosine": row["initial_functional_synthetic"]["raw"]["median_pairwise_cosine"],
            "initial_raw_synthetic_nrmse": row["initial_functional_synthetic"]["raw"]["median_pairwise_normalized_rmse"],
            "initial_aligned_synthetic_cosine": row["initial_functional_synthetic"]["aligned"]["median_pairwise_cosine"],
            "initial_aligned_synthetic_nrmse": row["initial_functional_synthetic"]["aligned"]["median_pairwise_normalized_rmse"],
            "initial_raw_empirical_cosine": row["initial_functional_empirical"]["raw"]["median_pairwise_cosine"],
            "initial_raw_empirical_nrmse": row["initial_functional_empirical"]["raw"]["median_pairwise_normalized_rmse"],
            "initial_aligned_empirical_cosine": row["initial_functional_empirical"]["aligned"]["median_pairwise_cosine"],
            "initial_aligned_empirical_nrmse": row["initial_functional_empirical"]["aligned"]["median_pairwise_normalized_rmse"],
        })
    null_aggregate = _aggregate_scalars(null_summary_rows, list(null_summary_rows[0]))

    write_json(HERE / "diagnostics" / "raw_parameter_similarity.json", {"fits": raw_parameter_fits})
    write_json(HERE / "diagnostics" / "aligned_parameter_similarity.json", {"fits": aligned_parameter_fits})
    write_json(HERE / "diagnostics" / "functional_similarity_synthetic.json", {"probe_count_per_fit": 4096, "fits": functional_syn_fits, "aggregate": syn_aggregate})
    write_json(HERE / "diagnostics" / "functional_similarity_empirical.json", {"probe_count_per_fit": 4096, "target_free_pool_construction": True, "fits": functional_emp_fits, "aggregate": emp_aggregate})
    write_json(HERE / "diagnostics" / "initialization_null_control.json", {"fits": null_fits, "aggregate": null_aggregate})
    write_json(HERE / "diagnostics" / "signed_permutation_assignments.json", {"trained": trained_assignments, "initial": initial_assignments})
    write_json(HERE / "diagnostics" / "relative_basis_stability.json", relative_basis)
    write_json(HERE / "diagnostics" / "gradient_alignment.json", {"fits": gradient_rows, "aligned_core_aggregate": _aggregate_scalars(aligned_core_gradients, ["median_cosine", "mean_cosine", "minimum_cosine", "fraction_negative", "fraction_strongly_negative"]), "per_family": gradient_family, "gradient_compatibility": gradient_compatibility})
    write_json(HERE / "diagnostics" / "global_coupling_alignment.json", {"fits": global_rows})
    write_json(HERE / "diagnostics" / "cell_type_clustering.json", clustering)

    training_resources = _load(HERE / "diagnostics" / "training_resource_accounting.json")
    audit_cpu = time.process_time() - cpu_started
    audit_wall = time.perf_counter() - wall_started
    resource = dict(training_resources)
    resource.update({
        "diagnostic_forward_calls": diagnostic_forward_calls,
        "diagnostic_backward_calls": diagnostic_backward_calls,
        "total_backward_calls": int(training_resources["training_backward_calls"] + diagnostic_backward_calls),
        "synthetic_probes": int(25 * CONFIG["synthetic_probes_per_fit"]),
        "empirical_probes": int(25 * CONFIG["empirical_probes_per_fit"]),
        "cpu_seconds_audit_process": float(audit_cpu),
        "wall_seconds_audit_elapsed": float(audit_wall),
        "gpu_seconds": 0.0,
        "peak_memory_bytes": None,
        "peak_memory_instrumented": False,
    })

    decision = {
        "version": "V837aa",
        "audit_valid": True,
        "parent_reproduction_valid": True,
        "direct_common_basis": classification["direct_common_basis"],
        "common_law_after_signed_permutation": classification["common_law_after_signed_permutation"],
        "relative_basis_stable": relative_basis["relative_basis_stable"],
        "basis_stability_subdiagnosis": relative_basis["basis_stability_subdiagnosis"] if classification["common_law_after_signed_permutation"] else "NOT_APPLICABLE",
        "stable_small_type_vocabulary": classification["stable_small_type_vocabulary"],
        "stable_cell_type_count": classification["selected_type_count"],
        "gradient_compatibility": gradient_compatibility,
        "candidate_law_diagnosis": classification["candidate_law_diagnosis"],
        "recommended_next_axis": classification["recommended_next_axis"],
        "representation_adequacy": "still_3_of_5_parent",
        "fresh_audit_consumed": False,
        "primitive_count": 0,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "v837ab_implemented": False,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics" / "decision_state.json", decision)

    results = {
        **decision,
        "parent": CONDITION,
        "parent_reproduction": reproduction,
        "classification_gate_counts": classification["counts"],
        "raw_common_law_evidence": {
            "synthetic_cosine": trained_raw_syn["median_pairwise_cosine"],
            "synthetic_nrmse": trained_raw_syn["median_pairwise_normalized_rmse"],
            "empirical_cosine": trained_raw_emp["median_pairwise_cosine"],
            "empirical_nrmse": trained_raw_emp["median_pairwise_normalized_rmse"],
        },
        "aligned_common_law_evidence": {
            "synthetic_cosine": trained_aligned_syn["median_pairwise_cosine"],
            "synthetic_nrmse": trained_aligned_syn["median_pairwise_normalized_rmse"],
            "empirical_cosine": trained_aligned_emp["median_pairwise_cosine"],
            "empirical_nrmse": trained_aligned_emp["median_pairwise_normalized_rmse"],
        },
        "initialization_null_aggregate": null_aggregate,
        "relative_basis_stability_summary": {
            "relative_basis_stable": relative_basis["relative_basis_stable"],
            "basis_stability_subdiagnosis": relative_basis["basis_stability_subdiagnosis"],
            "median_pair_exact_agreement": relative_basis["median_pair_exact_agreement"],
            "median_pair_entropy_bits": relative_basis["median_pair_entropy_bits"],
        },
        "gradient_alignment_summary": {
            "gradient_compatibility": gradient_compatibility,
            "aligned_core": _aggregate_scalars(aligned_core_gradients, ["median_cosine", "mean_cosine", "minimum_cosine", "fraction_negative", "fraction_strongly_negative"]),
        },
        "cell_type_summary": {
            "lowest_passing_k": clustering["lowest_passing_k"],
            "k_passes": {k: row["passes"] for k, row in clustering["k_results"].items()},
        },
        "parameter_interface_view_evidence": view_evidence,
        "resource_accounting": resource,
    }
    write_json(HERE / "results.json", results)
    _plot_results(results, functional_syn_fits, relative_basis, gradient_rows, clustering)

    artifact_bytes = sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())
    resource["artifact_size_bytes_before_final_resource_write"] = int(artifact_bytes)
    write_json(HERE / "diagnostics" / "compute_efficiency.json", resource)
    write_json(ROOT / "experiments/v837_primitive_invention/v837aa_resource_accounting.json", resource)
    results["resource_accounting"] = resource
    write_json(HERE / "results.json", results)

    diagnosis = decision["candidate_law_diagnosis"]
    recommendation = decision["recommended_next_axis"]
    verdict = f"""# V837aa — {diagnosis}\n\nY3 parent reproduction: PASS (3/5 retained).\n\nCandidate-law diagnosis: `{diagnosis}`.\n\nRecommended next axis: `{recommendation}`.\n\nGradient compatibility: `{gradient_compatibility}`.\n\nRelative basis subdiagnosis: `{decision['basis_stability_subdiagnosis']}`.\n\nV837aa is diagnosis-only: representation adequacy remains the frozen 3/5 Y3 parent; structural search and primitive mining remain blocked; fresh-audit consumption is zero; no V837ab architecture and no V838 experiment were started.\n"""
    (HERE / "VERDICT.md").write_text(verdict, encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
