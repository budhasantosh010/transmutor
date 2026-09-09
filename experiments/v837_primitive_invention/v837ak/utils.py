from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition")
DIRECTED = "DIRECTED_STRUCTURAL_SEARCH"
RANDOM = "RANDOM_STRUCTURAL_SAMPLER"


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        arr = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(b"\0"); digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        arr = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(b"\0"); digest.update(str(arr.dtype).encode()); digest.update(b"\0")
        digest.update(str(tuple(arr.shape)).encode()); digest.update(b"\0"); digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def organism_id(row: dict) -> str:
    return sha256_json({
        "engine": row["engine"], "family": row["family"], "run_index": int(row["run_index"]),
        "topology_id": row["topology_id"], "finalization_seed": int(row["finalization_seed"]),
    })


def occurrence_id(org_id: str, nodes: Sequence[int]) -> str:
    return sha256_json({"organism_id": org_id, "nodes": [int(x) for x in sorted(nodes)]})


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed_range(pair: Sequence[int]) -> list[int]:
    return list(range(int(pair[0]), int(pair[1]) + 1))


def robust_center_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0); q75 = np.nanpercentile(matrix, 75, axis=0)
    iqr = q75 - q25
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-9), iqr, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)
    return median.astype(np.float64), iqr.astype(np.float64)


def normalize(matrix: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    out = (matrix - median) / iqr
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return rmse / (float(np.std(target)) + 1e-8)


def one_sided_paired_sign_permutation(deltas: Sequence[float]) -> float:
    vals = np.asarray(list(deltas), dtype=np.float64)
    if vals.size == 0:
        return 1.0
    observed = float(np.mean(vals))
    n = int(vals.size)
    if n <= 20:
        ge = 0
        total = 1 << n
        for mask in range(total):
            signs = np.fromiter((1.0 if (mask >> i) & 1 else -1.0 for i in range(n)), dtype=np.float64, count=n)
            if float(np.mean(vals * signs)) >= observed - 1e-15:
                ge += 1
        return ge / total
    rng = np.random.default_rng(837_001)
    trials = 100_000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(trials, n))
    sampled = np.mean(signs * vals[None, :], axis=1)
    return float((np.sum(sampled >= observed - 1e-15) + 1) / (trials + 1))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def size_category(k: int) -> str:
    if k <= 3: return "MICRO"
    if k <= 6: return "MESO"
    if k <= 9: return "MACRO"
    return "WHOLE"


def safe_spectrum(matrix: np.ndarray, length: int | None = None) -> np.ndarray:
    if matrix.size == 0:
        vals = np.zeros(0, dtype=np.float64)
    else:
        vals = np.linalg.svd(matrix, compute_uv=False).astype(np.float64)
    if length is None: return vals
    out = np.zeros(length, dtype=np.float64); out[:min(length, vals.size)] = vals[:length]
    return out


def covariance_spectrum(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, x.shape[0] - 1)
    vals = np.linalg.eigvalsh(cov)[::-1]
    vals = np.clip(vals, 0.0, None)
    total = float(vals.sum())
    return vals / total if total > 1e-12 else vals


def participation_ratio(spectrum: np.ndarray) -> float:
    s = np.asarray(spectrum, dtype=np.float64)
    den = float(np.sum(s * s))
    return float((np.sum(s) ** 2) / den) if den > 1e-12 else 0.0
