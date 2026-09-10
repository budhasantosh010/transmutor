from __future__ import annotations

import numpy as np


def normalized_rmse(pred, truth, semantic_range: float) -> float:
    p = np.asarray(pred, dtype=np.float64); z = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((p - z) ** 2)) / max(float(semantic_range), 1e-12))


def causal_recovery(base_pred, expected_target, patched_pred, eps: float = 1e-8) -> np.ndarray:
    b = np.asarray(base_pred, dtype=np.float64); t = np.asarray(expected_target, dtype=np.float64); p = np.asarray(patched_pred, dtype=np.float64)
    return 1.0 - np.abs(t - p) / (np.abs(t - b) + float(eps))


def direction_agreement(base_pred, expected_target, patched_pred) -> np.ndarray:
    b = np.asarray(base_pred, dtype=np.float64); t = np.asarray(expected_target, dtype=np.float64); p = np.asarray(patched_pred, dtype=np.float64)
    desired = np.sign(t - b); got = np.sign(p - b)
    neutral = np.abs(t - b) < 1e-10
    return np.where(neutral, np.abs(p - t) < 1e-6, desired == got).astype(np.float64)


def safe_median(values, default: float = 0.0) -> float:
    a = np.asarray(values, dtype=np.float64)
    return float(np.median(a)) if a.size else float(default)
