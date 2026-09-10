from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837an.random_controls import haar_subspace
from .utils import deterministic_seed

RANDOM_DIRECTION_COUNT = 32


def random_directions(organism_id: str, backend_variant: str, phase: str | None = None) -> np.ndarray:
    rows = []
    for i in range(RANDOM_DIRECTION_COUNT):
        q = haar_subspace(40, 1, deterministic_seed("v837ao-random-direction", organism_id, backend_variant, phase or "GLOBAL", i))
        rows.append(q[:, 0])
    return np.stack(rows)


def norm_matched_updates(direction: np.ndarray, canonical_updates: np.ndarray) -> np.ndarray:
    q = np.asarray(direction, dtype=np.float64).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    updates = np.asarray(canonical_updates, dtype=np.float64)
    norms = np.linalg.norm(updates, axis=1)
    return norms[:, None] * q[None, :]
