from __future__ import annotations

import numpy as np

RIDGE_LAMBDA = 1e-6


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for i, count in enumerate(counts):
            if count > 1:
                idx = np.flatnonzero(inv == i)
                ranks[idx] = ranks[idx].mean()
    return ranks


def fit_k1_reader(states: np.ndarray, semantic: np.ndarray, q: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    s = np.asarray(states, dtype=np.float64)
    z = np.asarray(semantic, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    h = s @ q
    X = np.stack([h, np.ones_like(h)], axis=1)
    reg = np.diag([float(lam), 0.0])
    coef = np.linalg.solve(X.T @ X + reg, X.T @ z)
    return {"a": float(coef[0]), "b": float(coef[1]), "q": q.tolist(), "ridge_lambda": float(lam), "gradient_steps": 0, "kind": "K1_AFFINE"}


def fit_full40_reader(states: np.ndarray, semantic: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    s = np.asarray(states, dtype=np.float64)
    z = np.asarray(semantic, dtype=np.float64).reshape(-1)
    X = np.concatenate([s, np.ones((len(s), 1))], axis=1)
    reg = float(lam) * np.eye(X.shape[1]); reg[-1, -1] = 0.0
    coef = np.linalg.solve(X.T @ X + reg, X.T @ z)
    return {"coef": coef[:-1].tolist(), "b": float(coef[-1]), "ridge_lambda": float(lam), "gradient_steps": 0, "diagnostic_only": True, "kind": "FULL40_AFFINE"}


def read_k1(states: np.ndarray, reader: dict) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64)
    q = np.asarray(reader["q"], dtype=np.float64)
    return float(reader["a"]) * (s @ q) + float(reader["b"])


def read_full40(states: np.ndarray, reader: dict) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64)
    return s @ np.asarray(reader["coef"], dtype=np.float64) + float(reader["b"])


def reader_metrics(pred: np.ndarray, truth: np.ndarray, semantic_range: float, binary: bool) -> dict:
    p = np.asarray(pred, dtype=np.float64).reshape(-1); z = np.asarray(truth, dtype=np.float64).reshape(-1)
    nrmse = float(np.sqrt(np.mean((p - z) ** 2)) / max(float(semantic_range), 1e-12))
    if len(z) > 1 and np.std(z) > 1e-12 and np.std(p) > 1e-12:
        pearson = float(np.corrcoef(z, p)[0, 1])
        rz, rp = _rankdata(z), _rankdata(p)
        spearman = float(np.corrcoef(rz, rp)[0, 1]) if np.std(rz) > 0 and np.std(rp) > 0 else 0.0
    else:
        pearson = 0.0; spearman = 0.0
    sign_accuracy = float(np.mean(np.sign(p) == np.sign(z))) if len(z) else 0.0
    passed = (nrmse <= 0.10 and sign_accuracy >= 0.95) if binary else (nrmse <= 0.10 and abs(pearson) >= 0.95 and spearman >= 0.95)
    return {"normalized_rmse": nrmse, "pearson": pearson, "spearman": spearman, "sign_accuracy": sign_accuracy, "pass": bool(passed), "n": int(len(z))}
