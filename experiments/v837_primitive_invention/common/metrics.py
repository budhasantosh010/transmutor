from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def binary_summary(successes: list[bool] | np.ndarray) -> dict:
    values = np.asarray(successes, dtype=bool)
    n = int(values.size)
    count = int(values.sum())
    rate = count / n if n else float("nan")
    low, high = wilson_interval(count, n)
    return {"n": n, "success_count": count, "success_rate": rate, "ci95": [low, high]}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def continuous_summary(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def bootstrap_mean_ci(values: list[float] | np.ndarray, *, resamples: int = 2000, seed: int = 0, confidence: float = 0.95) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"mean": None, "median": None, "ci": [None, None], "confidence": confidence, "n": 0}
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    n = len(array)
    for index in range(resamples):
        sample = generator.integers(0, n, size=n)
        means[index] = array[sample].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci": [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))],
        "confidence": confidence,
        "n": int(n),
    }


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, *, resamples: int = 2000, seed: int = 0, confidence: float = 0.95) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    diff = a - b
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    n = len(diff)
    for index in range(resamples):
        sample = generator.integers(0, n, size=n)
        means[index] = diff[sample].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci": [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))],
        "confidence": confidence,
        "n": n,
    }


def permutation_rate_difference(success_hits: np.ndarray, random_hits: np.ndarray, *, permutations: int = 1999, seed: int = 0) -> dict:
    success_hits = np.asarray(success_hits, dtype=float)
    random_hits = np.asarray(random_hits, dtype=float)
    observed = float(success_hits.mean() - random_hits.mean())
    combined = np.concatenate([success_hits, random_hits])
    n_left = len(success_hits)
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        perm = generator.permutation(combined)
        delta = float(perm[:n_left].mean() - perm[n_left:].mean())
        if delta >= observed - 1e-12:
            extreme += 1
    return {"observed_difference": observed, "p_value": (extreme + 1) / (permutations + 1), "permutations": permutations}
