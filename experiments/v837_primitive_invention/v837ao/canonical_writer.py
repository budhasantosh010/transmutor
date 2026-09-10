from __future__ import annotations

import numpy as np


def raw_writer_from_compiler(q: np.ndarray, compiler: dict) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    coef = np.asarray(compiler["coef"], dtype=np.float64).reshape(1, -1)
    return (q @ coef.T).reshape(-1)


def set_state(states: np.ndarray, targets: np.ndarray | float, reader: dict, writer: np.ndarray) -> np.ndarray:
    from .canonical_reader import read_k1
    s = np.asarray(states, dtype=np.float64)
    one = s.ndim == 1
    if one:
        s = s[None, :]
    z = read_k1(s, reader)
    target = np.asarray(targets, dtype=np.float64)
    if target.ndim == 0:
        target = np.full(len(s), float(target))
    out = s + (target.reshape(-1, 1) - z.reshape(-1, 1)) * np.asarray(writer, dtype=np.float64).reshape(1, -1)
    return out[0] if one else out


def delta_state(states: np.ndarray, delta_z: np.ndarray | float, writer: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64)
    dz = np.asarray(delta_z, dtype=np.float64)
    if s.ndim == 1:
        return s + float(dz) * np.asarray(writer, dtype=np.float64)
    if dz.ndim == 0:
        dz = np.full(len(s), float(dz))
    return s + dz.reshape(-1, 1) * np.asarray(writer, dtype=np.float64).reshape(1, -1)
