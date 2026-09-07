from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


CELL_DIM = 4
NUM_CELLS = 10
VIEWS = {
    "core": ("Ws", "Wm", "b"),
    "core_input": ("Ws", "Wm", "Wx", "b"),
    "core_output": ("Ws", "Wm", "b", "Wo"),
    "full": ("Ws", "Wm", "Wx", "b", "Wo"),
}
EPS = 1e-12


@dataclass(frozen=True)
class SignedPermutation:
    index: int
    permutation: tuple[int, int, int, int]
    signs: tuple[int, int, int, int]
    matrix: np.ndarray


def enumerate_signed_permutations() -> list[SignedPermutation]:
    rows: list[SignedPermutation] = []
    index = 0
    for permutation in itertools.permutations(range(CELL_DIM)):
        for signs in itertools.product((-1, 1), repeat=CELL_DIM):
            matrix = np.zeros((CELL_DIM, CELL_DIM), dtype=np.float32)
            for canonical_axis, source_axis in enumerate(permutation):
                matrix[canonical_axis, source_axis] = float(signs[canonical_axis])
            rows.append(SignedPermutation(index, tuple(permutation), tuple(signs), matrix))
            index += 1
    return rows


SIGNED_PERMUTATIONS = enumerate_signed_permutations()
PERMUTATION_INDEX = {
    tuple(int(v) for v in row.matrix.reshape(-1).tolist()): row.index
    for row in SIGNED_PERMUTATIONS
}
IDENTITY_PERMUTATION_INDEX = next(
    row.index for row in SIGNED_PERMUTATIONS if np.array_equal(row.matrix, np.eye(CELL_DIM, dtype=np.float32))
)


def validate_signed_permutations(rows: Iterable[SignedPermutation] = SIGNED_PERMUTATIONS) -> None:
    rows = list(rows)
    if len(rows) != 384:
        raise ValueError(f"expected 384 signed permutations, got {len(rows)}")
    seen: set[tuple[int, ...]] = set()
    identity = np.eye(CELL_DIM, dtype=np.float32)
    for row in rows:
        matrix = np.asarray(row.matrix, dtype=np.float32)
        key = tuple(int(v) for v in matrix.reshape(-1).tolist())
        if key in seen:
            raise ValueError("duplicate signed permutation")
        seen.add(key)
        if not np.array_equal(matrix.T @ matrix, identity):
            raise ValueError("signed permutation is not exactly orthogonal")
        if not np.all(np.isin(matrix, (-1.0, 0.0, 1.0))):
            raise ValueError("signed permutation has invalid entry")
        if not np.all(np.sum(np.abs(matrix), axis=0) == 1) or not np.all(np.sum(np.abs(matrix), axis=1) == 1):
            raise ValueError("signed permutation does not have one signed unit per row/column")


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def transform_law(law: dict, matrix: np.ndarray) -> dict:
    p = _array(matrix)
    return {
        "Ws": p @ _array(law["Ws"]) @ p.T,
        "Wm": p @ _array(law["Wm"]),
        "Wx": p @ _array(law["Wx"]),
        "b": p @ _array(law["b"]),
        "Wo": _array(law["Wo"]) @ p.T,
    }


def transform_gradient_law(law: dict, matrix: np.ndarray) -> dict:
    return transform_law(law, matrix)


def component_distance(a, b) -> float:
    aa, bb = _array(a), _array(b)
    return float(np.linalg.norm(aa - bb) / (np.linalg.norm(aa) + np.linalg.norm(bb) + EPS))


def bundle_distance(a: dict, b: dict, view: str = "core") -> float:
    components = VIEWS[view]
    return float(np.mean([component_distance(a[name], b[name]) for name in components]))


def _component_unit_vector(value) -> np.ndarray:
    array = _array(value).reshape(-1)
    norm = np.linalg.norm(array)
    if norm <= EPS:
        return np.zeros_like(array)
    return array / norm


def balanced_bundle_vector(law: dict, view: str = "core") -> np.ndarray:
    components = VIEWS[view]
    scale = 1.0 / math.sqrt(len(components))
    return np.concatenate([_component_unit_vector(law[name]) * scale for name in components])


def bundle_cosine(a: dict, b: dict, view: str = "core") -> float:
    aa, bb = balanced_bundle_vector(a, view), balanced_bundle_vector(b, view)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return 0.0 if denom <= EPS else float(np.dot(aa, bb) / denom)


def bundle_norm_ratios(a: dict, b: dict, view: str = "core") -> dict[str, float]:
    out = {}
    for name in VIEWS[view]:
        na, nb = float(np.linalg.norm(_array(a[name]))), float(np.linalg.norm(_array(b[name])))
        out[name] = float(min(na, nb) / (max(na, nb) + EPS))
    return out


def _transformed_candidates(law: dict, permutations: list[SignedPermutation] = SIGNED_PERMUTATIONS) -> dict[str, np.ndarray]:
    p = np.stack([row.matrix for row in permutations], axis=0).astype(np.float64)
    ws = _array(law["Ws"])
    wm = _array(law["Wm"])
    wx = _array(law["Wx"])
    b = _array(law["b"])
    wo = _array(law["Wo"])
    return {
        "Ws": np.einsum("pij,jk,plk->pil", p, ws, p),
        "Wm": np.einsum("pij,jk->pik", p, wm),
        "Wx": np.einsum("pij,jk->pik", p, wx),
        "b": np.einsum("pij,j->pi", p, b),
        "Wo": np.einsum("ij,pkj->pik", wo, p),
    }


def _candidate_distances(candidates: dict[str, np.ndarray], target: dict, view: str = "core") -> np.ndarray:
    pieces = []
    for name in VIEWS[view]:
        candidate = candidates[name]
        reference = _array(target[name])
        axes = tuple(range(1, candidate.ndim))
        numerator = np.linalg.norm(candidate - reference, axis=axes)
        denominator = np.linalg.norm(candidate, axis=axes) + np.linalg.norm(reference) + EPS
        pieces.append(numerator / denominator)
    return np.mean(np.stack(pieces, axis=0), axis=0)


def best_alignment_index(law: dict, target: dict, view: str = "core", candidates: dict[str, np.ndarray] | None = None) -> tuple[int, float]:
    if candidates is None:
        candidates = _transformed_candidates(law)
    distances = _candidate_distances(candidates, target, view)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _centroid(laws: list[dict]) -> dict:
    return {name: np.mean(np.stack([_array(law[name]) for law in laws], axis=0), axis=0) for name in ("Ws", "Wm", "Wx", "b", "Wo")}


def align_laws(laws: list[dict], *, view: str = "core", max_iterations: int = 10) -> dict:
    if len(laws) != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} laws")
    caches = [_transformed_candidates(law) for law in laws]
    pairwise_best = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.float64)
    pairwise_index = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.int64)
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            if i == j:
                pairwise_index[i, j] = IDENTITY_PERMUTATION_INDEX
                continue
            idx, distance = best_alignment_index(laws[i], laws[j], view, caches[i])
            pairwise_best[i, j] = distance
            pairwise_index[i, j] = idx
    medoid_scores = pairwise_best.sum(axis=1) + pairwise_best.sum(axis=0)
    medoid = int(np.argmin(medoid_scores))
    assignments = []
    for i in range(NUM_CELLS):
        if i == medoid:
            assignments.append(IDENTITY_PERMUTATION_INDEX)
        else:
            assignments.append(best_alignment_index(laws[i], laws[medoid], view, caches[i])[0])
    converged = False
    convergence_iteration = 0
    centroid = None
    for iteration in range(1, max_iterations + 1):
        aligned = [transform_law(law, SIGNED_PERMUTATIONS[index].matrix) for law, index in zip(laws, assignments)]
        centroid = _centroid(aligned)
        new_assignments = [best_alignment_index(law, centroid, view, cache)[0] for law, cache in zip(laws, caches)]
        convergence_iteration = iteration
        if new_assignments == assignments:
            converged = True
            break
        assignments = new_assignments
    aligned = [transform_law(law, SIGNED_PERMUTATIONS[index].matrix) for law, index in zip(laws, assignments)]
    centroid = _centroid(aligned)
    records = []
    for cell_index, (law, perm_index, aligned_law) in enumerate(zip(laws, assignments, aligned)):
        raw_distance = bundle_distance(law, centroid, view)
        aligned_distance = bundle_distance(aligned_law, centroid, view)
        row = SIGNED_PERMUTATIONS[perm_index]
        records.append({
            "cell_index": cell_index,
            "permutation_index": int(perm_index),
            "permutation": list(row.permutation),
            "signs": list(row.signs),
            "raw_distance_to_centroid": raw_distance,
            "aligned_distance_to_centroid": aligned_distance,
            "absolute_improvement": raw_distance - aligned_distance,
            "relative_improvement": (raw_distance - aligned_distance) / (raw_distance + EPS),
        })
    return {
        "view": view,
        "medoid_cell": medoid,
        "medoid_scores": medoid_scores.tolist(),
        "pairwise_best_distance": pairwise_best.tolist(),
        "pairwise_best_permutation_index": pairwise_index.tolist(),
        "assignments": assignments,
        "records": records,
        "centroid": {name: value.tolist() for name, value in centroid.items()},
        "converged": converged,
        "convergence_iteration": convergence_iteration,
        "max_iterations": max_iterations,
    }


def relative_transform_index(pi_index: int, pj_index: int) -> int:
    pi = SIGNED_PERMUTATIONS[int(pi_index)].matrix.astype(np.int64)
    pj = SIGNED_PERMUTATIONS[int(pj_index)].matrix.astype(np.int64)
    relative = pi.T @ pj
    key = tuple(int(v) for v in relative.reshape(-1).tolist())
    return int(PERMUTATION_INDEX[key])


def pairwise_parameter_metrics(laws: list[dict], *, view: str = "core") -> dict:
    n = len(laws)
    distance = np.zeros((n, n), dtype=np.float64)
    cosine = np.eye(n, dtype=np.float64)
    norm_ratio = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance[i, j] = distance[j, i] = bundle_distance(laws[i], laws[j], view)
            cosine[i, j] = cosine[j, i] = bundle_cosine(laws[i], laws[j], view)
            ratios = bundle_norm_ratios(laws[i], laws[j], view)
            norm_ratio[i, j] = norm_ratio[j, i] = float(np.mean(list(ratios.values())))
    values = np.triu_indices(n, 1)
    return {
        "normalized_frobenius_distance": distance.tolist(),
        "cosine": cosine.tolist(),
        "mean_component_norm_ratio": norm_ratio.tolist(),
        "median_pair_distance": float(np.median(distance[values])),
        "median_pair_cosine": float(np.median(cosine[values])),
        "median_pair_norm_ratio": float(np.median(norm_ratio[values])),
    }


def matrix_singular_summary(laws: list[dict]) -> dict:
    output = {}
    for name in ("Ws", "Wm", "Wx", "Wo"):
        rows = []
        for cell_index, law in enumerate(laws):
            singular = np.linalg.svd(_array(law[name]), compute_uv=False)
            rows.append({
                "cell_index": cell_index,
                "singular_values": [float(v) for v in singular.tolist()],
                "spectral_norm": float(singular.max()) if singular.size else 0.0,
                "frobenius_norm": float(np.linalg.norm(_array(law[name]))),
            })
        output[name] = rows
    return output


def functional_outputs(laws: list[dict], probes: dict, *, view: str = "core") -> np.ndarray:
    s = _array(probes["s"])
    m = _array(probes["m"])
    x = _array(probes["x"])
    outputs = []
    for law in laws:
        pre = s @ _array(law["Ws"]).T + m @ _array(law["Wm"]).T + _array(law["b"])
        if view in {"core_input", "full"}:
            pre = pre + x @ _array(law["Wx"]).T
        candidate = np.tanh(pre)
        if view in {"core_output", "full"}:
            candidate = candidate @ _array(law["Wo"]).T
        outputs.append(candidate)
    return np.stack(outputs, axis=0)


def _safe_corr_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, EPS)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def functional_metric_matrices(outputs: np.ndarray) -> dict:
    values = np.asarray(outputs, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("functional outputs must be [cells,probes,dim]")
    n, _, d = values.shape
    flat = values.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1)
    cosine = (flat @ flat.T) / np.maximum(norms[:, None] * norms[None, :], EPS)
    rms = np.sqrt(np.mean(flat * flat, axis=1))
    sq = np.mean((flat[:, None, :] - flat[None, :, :]) ** 2, axis=2)
    nrmse = np.sqrt(sq) / np.maximum(rms[:, None] + rms[None, :], EPS)
    pearson = _safe_corr_rows(flat)
    per_dim = []
    for dim in range(d):
        per_dim.append(_safe_corr_rows(values[:, :, dim]))
    per_dim_corr = np.stack(per_dim, axis=2)
    mean_abs = np.mean(np.abs(values[:, None, :, :] - values[None, :, :, :]), axis=(2, 3))
    pair = np.triu_indices(n, 1)
    return {
        "cosine": cosine.tolist(),
        "normalized_rmse": nrmse.tolist(),
        "pearson": pearson.tolist(),
        "per_dimension_correlation": per_dim_corr.tolist(),
        "mean_absolute_delta": mean_abs.tolist(),
        "candidate_output_variance": np.var(values, axis=(1, 2)).tolist(),
        "saturation_fraction": np.mean(np.abs(values) >= 0.95, axis=(1, 2)).tolist(),
        "median_pairwise_cosine": float(np.median(cosine[pair])),
        "median_pairwise_normalized_rmse": float(np.median(nrmse[pair])),
        "median_pairwise_pearson": float(np.median(pearson[pair])),
        "median_pairwise_mean_absolute_delta": float(np.median(mean_abs[pair])),
        "mean_candidate_output_variance": float(np.mean(np.var(values, axis=(1, 2)))),
        "mean_saturation_fraction": float(np.mean(np.abs(values) >= 0.95)),
    }


def pairwise_cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    matrix = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors], axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    return (matrix @ matrix.T) / np.maximum(norms[:, None] * norms[None, :], EPS)


def gradient_bundle_vector(law: dict, view: str = "core") -> np.ndarray:
    return np.concatenate([_array(law[name]).reshape(-1) for name in VIEWS[view]])


def gradient_pair_metrics(laws: list[dict], view: str = "core") -> dict:
    cosine = pairwise_cosine_matrix([gradient_bundle_vector(law, view) for law in laws])
    pair = np.triu_indices(len(laws), 1)
    values = cosine[pair]
    norms = [float(np.linalg.norm(gradient_bundle_vector(law, view))) for law in laws]
    return {
        "cosine": cosine.tolist(),
        "median_cosine": float(np.median(values)),
        "mean_cosine": float(np.mean(values)),
        "minimum_cosine": float(np.min(values)),
        "fraction_negative": float(np.mean(values < 0.0)),
        "fraction_strongly_negative": float(np.mean(values < -0.25)),
        "per_cell_gradient_norm": norms,
    }


def complete_linkage_assignments(distance_matrix: np.ndarray, k: int) -> list[int]:
    distance = np.asarray(distance_matrix, dtype=np.float64)
    n = distance.shape[0]
    if distance.shape != (n, n) or not (1 <= k <= n):
        raise ValueError("invalid clustering input")
    clusters: list[tuple[int, ...]] = [(i,) for i in range(n)]
    while len(clusters) > k:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                linkage = max(float(distance[i, j]) for i in clusters[a] for j in clusters[b])
                key = (linkage, clusters[a], clusters[b])
                if best is None or key < best[0]:
                    best = (key, a, b)
        assert best is not None
        _, a, b = best
        merged = tuple(sorted(clusters[a] + clusters[b]))
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in {a, b}]
        clusters.append(merged)
        clusters.sort()
    assignments = [-1] * n
    for label, cluster in enumerate(sorted(clusters)):
        for cell in cluster:
            assignments[cell] = label
    return assignments


def silhouette_like(distance_matrix: np.ndarray, assignments: list[int]) -> float:
    distance = np.asarray(distance_matrix, dtype=np.float64)
    labels = sorted(set(assignments))
    scores = []
    for i, label in enumerate(assignments):
        same = [j for j, other in enumerate(assignments) if other == label and j != i]
        a = float(np.mean([distance[i, j] for j in same])) if same else 0.0
        other_means = []
        for other_label in labels:
            if other_label == label:
                continue
            members = [j for j, other in enumerate(assignments) if other == other_label]
            other_means.append(float(np.mean([distance[i, j] for j in members])))
        b = min(other_means) if other_means else 0.0
        scores.append(0.0 if max(a, b) <= EPS else (b - a) / max(a, b))
    return float(np.mean(scores))


def pairwise_same_matrix(assignments: list[int]) -> np.ndarray:
    labels = np.asarray(assignments)
    return (labels[:, None] == labels[None, :]).astype(np.float64)


def clustering_quality(assignments: list[int], synthetic_metrics: dict, empirical_metrics: dict) -> dict:
    labels = sorted(set(assignments))
    pairs = [(i, j) for i in range(len(assignments)) for j in range(i + 1, len(assignments)) if assignments[i] == assignments[j]]
    def summarize(metrics: dict) -> dict:
        if not pairs:
            return {"within_median_cosine": 1.0, "within_median_nrmse": 0.0}
        cos = [metrics["cosine"][i][j] for i, j in pairs]
        nrmse = [metrics["normalized_rmse"][i][j] for i, j in pairs]
        return {"within_median_cosine": float(np.median(cos)), "within_median_nrmse": float(np.median(nrmse))}
    combined_distance = (np.asarray(synthetic_metrics["normalized_rmse"]) + np.asarray(empirical_metrics["normalized_rmse"])) / 2.0
    syn = summarize(synthetic_metrics)
    emp = summarize(empirical_metrics)
    return {
        "cluster_count": len(labels),
        "cluster_sizes": [assignments.count(label) for label in labels],
        "synthetic": syn,
        "empirical": emp,
        "silhouette_like": silhouette_like(combined_distance, assignments),
        "passes_within_type_threshold": bool(
            syn["within_median_cosine"] >= 0.95 and syn["within_median_nrmse"] <= 0.20
            and emp["within_median_cosine"] >= 0.95 and emp["within_median_nrmse"] <= 0.20
        ),
    }


def classify_gradient_compatibility(rows: list[dict], gate: dict) -> str:
    values = np.asarray([row["median_cosine"] for row in rows], dtype=np.float64)
    strong = np.asarray([row["fraction_strongly_negative"] for row in rows], dtype=np.float64)
    median = float(np.median(values))
    strong_fraction = float(np.mean(strong))
    if median >= float(gate["compatible_median_cosine_min"]) and strong_fraction <= float(gate["compatible_strong_negative_fraction_max"]):
        return "compatible"
    if median <= float(gate["conflicted_median_cosine_max"]) or strong_fraction >= float(gate["conflicted_strong_negative_fraction_min"]):
        return "conflicted"
    return "mixed"


def _fit_common(metric: dict, gate: dict) -> bool:
    return bool(metric["median_pairwise_cosine"] >= gate["cosine_min"] and metric["median_pairwise_normalized_rmse"] <= gate["nrmse_max"])


def _fit_materially_below(metric: dict, gate: dict) -> bool:
    return bool(metric["median_pairwise_cosine"] < gate["diverse_cosine_max"] or metric["median_pairwise_normalized_rmse"] > gate["diverse_nrmse_min"])


def classify_candidate_law(fits: list[dict], clustering: dict, gate: dict) -> dict:
    required = int(gate["required_fit_count"])
    direct_count = 0
    aligned_count = 0
    gain_count = 0
    raw_null_count = 0
    aligned_null_count = 0
    materially_below_count = 0
    for fit in fits:
        raw_syn, raw_emp = fit["trained_raw_synthetic"], fit["trained_raw_empirical"]
        aligned_syn, aligned_emp = fit["trained_aligned_synthetic"], fit["trained_aligned_empirical"]
        initial_raw_syn, initial_raw_emp = fit["initial_raw_synthetic"], fit["initial_raw_empirical"]
        initial_aligned_syn, initial_aligned_emp = fit["initial_aligned_synthetic"], fit["initial_aligned_empirical"]
        direct_count += int(_fit_common(raw_syn, gate) and _fit_common(raw_emp, gate))
        aligned_count += int(_fit_common(aligned_syn, gate) and _fit_common(aligned_emp, gate))
        syn_gain = (raw_syn["median_pairwise_normalized_rmse"] - aligned_syn["median_pairwise_normalized_rmse"]) / (raw_syn["median_pairwise_normalized_rmse"] + EPS)
        emp_gain = (raw_emp["median_pairwise_normalized_rmse"] - aligned_emp["median_pairwise_normalized_rmse"]) / (raw_emp["median_pairwise_normalized_rmse"] + EPS)
        gain_count += int(syn_gain >= gate["alignment_relative_nrmse_reduction_min"] and emp_gain >= gate["alignment_relative_nrmse_reduction_min"])
        def null_exceeded(trained: dict, initial: dict) -> bool:
            return bool(
                trained["median_pairwise_normalized_rmse"] <= gate["null_nrmse_ratio_max"] * initial["median_pairwise_normalized_rmse"]
                or trained["median_pairwise_cosine"] >= initial["median_pairwise_cosine"] + gate["null_cosine_absolute_gain_min"]
            )
        raw_null_count += int(null_exceeded(raw_syn, initial_raw_syn) and null_exceeded(raw_emp, initial_raw_emp))
        aligned_null_count += int(null_exceeded(aligned_syn, initial_aligned_syn) and null_exceeded(aligned_emp, initial_aligned_emp))
        materially_below_count += int(_fit_materially_below(aligned_syn, gate) or _fit_materially_below(aligned_emp, gate))
    direct = direct_count >= required and raw_null_count >= required
    common_aligned = (not direct) and aligned_count >= required and gain_count >= required and aligned_null_count >= required
    stable_small = False
    selected_k = None
    if not direct and not common_aligned:
        for k in range(2, 6):
            row = clustering.get(str(k), {})
            if row.get("passes"):
                stable_small = True
                selected_k = k
                break
    diverse = (
        not direct and not common_aligned and not stable_small
        and materially_below_count >= int(gate["diverse_required_fit_count"])
    )
    if direct:
        diagnosis = "DIRECT_COMMON_BASIS"
        recommendation = "V837ab_DIRECT_SHARED_CANDIDATE_CORE"
    elif common_aligned:
        stable_relative = bool(clustering.get("relative_basis_stable", False))
        diagnosis = "COMMON_LAW_UP_TO_SIGNED_PERMUTATION"
        recommendation = (
            "V837ab_CANONICAL_SHARED_CORE_WITH_FIXED_BASIS_ADAPTERS"
            if stable_relative else "V837ab_COORDINATE_ADAPTIVE_SHARED_CORE_REQUIRED"
        )
    elif stable_small:
        diagnosis = "STABLE_SMALL_TYPE_VOCABULARY"
        recommendation = "V837ab_GROUP_SHARED_CANDIDATE_TYPES"
    elif diverse:
        diagnosis = "GENUINELY_DIVERSE_CANDIDATE_LAWS"
        recommendation = "NEXT_AXIS_SHARED_INPUT_REPRESENTATION"
    else:
        diagnosis = "INCONCLUSIVE"
        recommendation = "NO_ARCHITECTURE_CHANGE_TARGETED_DIAGNOSTIC_REQUIRED"
    return {
        "direct_common_basis": direct,
        "common_law_after_signed_permutation": common_aligned,
        "stable_small_type_vocabulary": stable_small,
        "selected_type_count": selected_k,
        "candidate_law_diagnosis": diagnosis,
        "recommended_next_axis": recommendation,
        "counts": {
            "direct_common_fit_count": direct_count,
            "aligned_common_fit_count": aligned_count,
            "meaningful_alignment_gain_fit_count": gain_count,
            "raw_null_exceeded_fit_count": raw_null_count,
            "aligned_null_exceeded_fit_count": aligned_null_count,
            "materially_below_common_fit_count": materially_below_count,
        },
    }


def block_diagonal_transform(permutation_indices: list[int]) -> np.ndarray:
    if len(permutation_indices) != NUM_CELLS:
        raise ValueError("expected ten local basis assignments")
    block = np.zeros((NUM_CELLS * CELL_DIM, NUM_CELLS * CELL_DIM), dtype=np.float64)
    for cell, index in enumerate(permutation_indices):
        start = cell * CELL_DIM
        block[start:start + CELL_DIM, start:start + CELL_DIM] = SIGNED_PERMUTATIONS[int(index)].matrix
    return block


def transform_global_matrix(matrix: np.ndarray, permutation_indices: list[int]) -> np.ndarray:
    p = block_diagonal_transform(permutation_indices)
    a = _array(matrix)
    return p @ a @ p.T


def global_matrix_diagnostics(matrix: np.ndarray) -> dict:
    a = _array(matrix)
    singular = np.linalg.svd(a, compute_uv=False)
    tolerance = max(a.shape) * np.finfo(np.float64).eps * (float(singular.max()) if singular.size else 0.0)
    rank = int(np.sum(singular > tolerance))
    block_energy = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.float64)
    for dst in range(NUM_CELLS):
        for src in range(NUM_CELLS):
            block = a[dst * CELL_DIM:(dst + 1) * CELL_DIM, src * CELL_DIM:(src + 1) * CELL_DIM]
            block_energy[dst, src] = float(np.sum(block * block))
    destinations = [a[i * CELL_DIM:(i + 1) * CELL_DIM, :] for i in range(NUM_CELLS)]
    sources = [a[:, i * CELL_DIM:(i + 1) * CELL_DIM] for i in range(NUM_CELLS)]
    dst_cos = pairwise_cosine_matrix(destinations)
    src_cos = pairwise_cosine_matrix(sources)
    pair = np.triu_indices(NUM_CELLS, 1)
    return {
        "singular_values": [float(v) for v in singular.tolist()],
        "effective_rank": rank,
        "spectral_norm": float(singular.max()) if singular.size else 0.0,
        "frobenius_norm": float(np.linalg.norm(a)),
        "block_energies": block_energy.tolist(),
        "destination_block_cosine": dst_cos.tolist(),
        "source_block_cosine": src_cos.tolist(),
        "median_destination_block_cosine": float(np.median(dst_cos[pair])),
        "median_source_block_cosine": float(np.median(src_cos[pair])),
    }


validate_signed_permutations()
