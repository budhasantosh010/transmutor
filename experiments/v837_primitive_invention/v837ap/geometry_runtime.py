from __future__ import annotations

import numpy as np

from .chart_reader import read_chart
from .nonlinear_setter import set_state_geometry

BINARY_FAMILIES = {"conditional_routing", "delayed_recall"}


def chart_for_phase(geometry: dict, phase: str) -> dict:
    if geometry.get("phase_atlas"):
        chart = geometry.get("charts_by_phase", {}).get(phase)
        if not isinstance(chart, dict):
            raise KeyError(f"missing atlas chart for {phase}")
        return chart
    return geometry["chart"]


def tangent_for_phase(geometry: dict, phase: str):
    fields = geometry.get("tangent_fields")
    if not isinstance(fields, dict):
        return geometry.get("tangent_field")
    return fields.get(phase, fields.get("GLOBAL"))


def read_state(geometry: dict, q: np.ndarray, state40: np.ndarray, phase: str) -> float:
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    h = np.asarray(state40, dtype=np.float64).reshape(40) @ Q
    chart = chart_for_phase(geometry, phase)
    return float(read_chart(chart, h.reshape(1, -1))[0])


def set_state(geometry: dict, q: np.ndarray, state40: np.ndarray, phase: str, target: float, d90: float) -> dict:
    writer = geometry.get("writer_family", "AUTO")
    tangent = tangent_for_phase(geometry, phase)
    if writer.startswith("TANGENT_") or writer == "TANGENT_FIELD":
        writer_family = "TANGENT_FIELD"
    else:
        writer_family = "AUTO"
    return set_state_geometry(
        np.asarray(state40, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
        chart_for_phase(geometry, phase),
        float(target),
        float(d90),
        geometry["family"] in BINARY_FAMILIES,
        writer_family=writer_family,
        tangent=tangent,
        max_iterations=4,
    )
