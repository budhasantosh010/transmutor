from __future__ import annotations

import numpy as np


def reader_gradient(reader: dict) -> np.ndarray:
    return float(reader["a"]) * np.asarray(reader["q"], dtype=np.float64)


def gauge_fix_writer(reader: dict, raw_writer: np.ndarray, eps: float = 1e-6) -> dict:
    g = reader_gradient(reader)
    raw = np.asarray(raw_writer, dtype=np.float64).reshape(-1)
    gamma = float(g @ raw)
    if abs(gamma) < eps:
        return {"valid": False, "gamma": gamma, "writer": None, "failure_code": "READER_WRITER_GAIN_DEGENERATE"}
    writer = raw / gamma
    return {"valid": True, "gamma": gamma, "writer": writer.tolist(), "unit_gain": float(g @ writer)}


def biorthogonal_projector(reader: dict, writer: np.ndarray) -> np.ndarray:
    g = reader_gradient(reader).reshape(1, -1)
    w = np.asarray(writer, dtype=np.float64).reshape(-1, 1)
    return w @ g
