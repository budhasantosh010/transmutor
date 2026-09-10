from __future__ import annotations

import json
import math
import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES

from .chart_data import phase_dataset
from .chart_reader import read_chart, reader_metrics
from .data_roles import seeds
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .setpoint_grid import family_grid
from .utils import HERE, write_json


def closest_global_reader_candidate(family: str, summaries: list[dict]) -> dict:
    """Freeze the simplest candidate among those closest to the reader family gate.

    Closeness is defined only from AP_CHART_SELECT outcomes: smallest pass-count
    shortfall first, then the already-frozen candidate order. No setpoint/outcome
    metric participates.
    """
    ranked = []
    for index, row in enumerate(summaries):
        gate = row["reader_family_gate"]
        shortfall = max(0, int(gate["required"]) - int(gate["passing"]))
        distance = float(row.get("median_reader_gate_distance", row.get("median_reader_nrmse", 1e6)))
        ranked.append((shortfall, distance, index, row))
    if not ranked:
        raise RuntimeError(f"AP_ATLAS_NO_GLOBAL_READER_CANDIDATES:{family}")
    return min(ranked, key=lambda x: (x[0], x[1], x[2]))[3]


def fit_phase_atlas_organism(organism_id: str, family: str, engine: str, candidate: dict, q: np.ndarray) -> dict:
    binary = family in {"conditional_routing", "delayed_recall"}
    q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    charts: dict[str, dict] = {}
    phase_metrics: list[dict] = []
    valid = True
    for phase in PHASES[family]:
        fit = phase_dataset(organism_id, family, seeds("AP_CHART_FIT"), q, phase)
        sel = phase_dataset(organism_id, family, seeds("AP_CHART_SELECT"), q, phase)
        if len(fit["semantic"]) < 8 or len(sel["semantic"]) < 8:
            charts[phase] = {"valid": False, "failure_code": "INSUFFICIENT_CHART_SAMPLES"}
            phase_metrics.append({"phase": phase, "pass": False, "n": 0})
            valid = False
            continue
        try:
            if int(candidate["k"]) == 1:
                chart = fit_scalar_chart(fit["h"].reshape(-1), fit["semantic"], candidate["chart_family"], binary)
            else:
                chart = fit_projected_chart(fit["h"], fit["semantic"], candidate["chart_family"], binary)
        except Exception as exc:
            charts[phase] = {"valid": False, "failure_code": "CHART_FIT_INVALID", "error": f"{type(exc).__name__}:{exc}"}
            phase_metrics.append({"phase": phase, "pass": False, "n": 0})
            valid = False
            continue
        if chart.get("derivative_sign_consistent") is False:
            chart["valid"] = False
            chart["failure_code"] = "NONMONOTONIC_CANONICAL_CHART"
            charts[phase] = chart
            phase_metrics.append({"phase": phase, "pass": False, "n": len(sel["semantic"])})
            valid = False
            continue
        if binary and int(candidate["k"]) == 1:
            h = fit["h"].reshape(-1)
            z = fit["semantic"]
            chart["prototype_minus"] = float(np.median(h[z < 0])) if np.any(z < 0) else None
            chart["prototype_plus"] = float(np.median(h[z > 0])) if np.any(z > 0) else None
            chart["prototype_separation"] = None if chart["prototype_minus"] is None or chart["prototype_plus"] is None else abs(chart["prototype_plus"] - chart["prototype_minus"])
        pred = read_chart(chart, sel["h"] if int(candidate["k"]) > 1 else sel["h"].reshape(-1))
        metrics = reader_metrics(pred, sel["semantic"], family_grid(family)["semantic_range"], binary)
        chart["valid"] = True
        charts[phase] = chart
        phase_metrics.append({"phase": phase, **metrics})
        valid = bool(valid and metrics["pass"])
    parameter_count = sum(int(c.get("parameter_count", 0)) for c in charts.values())
    stored_bytes = len(json.dumps(charts, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "version": "V837ap",
        "branch": "AP_D_PHASE_ATLAS",
        "organism_id": organism_id,
        "family": family,
        "engine": engine,
        "k": int(candidate["k"]),
        "chart_family": candidate["chart_family"],
        "writer_family": candidate.get("writer_family", "GRADIENT_NEWTON"),
        "phase_atlas": True,
        "charts_by_phase": charts,
        "phase_reader_metrics": phase_metrics,
        "reader_pass": bool(valid),
        "valid": bool(valid),
        "parameter_count": parameter_count,
        "stored_bytes": stored_bytes,
        "semantic_dimension": 1,
        "same_model_family_across_phases": True,
        "phase_specific_degree_search": False,
        "cross_organism_alignment": False,
    }


def summarize_atlas(rows: list[dict]) -> dict:
    passing = [r for r in rows if r.get("reader_pass")]
    n = len(rows)
    required = max(1, math.ceil(0.60 * n))
    engines = sorted({r["engine"] for r in passing})
    return {
        "organisms": n,
        "passing": len(passing),
        "required": required,
        "pass_fraction": 0.0 if not n else len(passing) / n,
        "passing_engines": engines,
        "pass": bool(n and len(passing) >= required and len(engines) >= 2),
    }


def save_phase_atlas(rows: list[dict], selections: dict) -> dict:
    payload = {"version": "V837ap", "stage": "AP6_PHASE_ATLAS", "rows": rows, "selection_from_ap_chart_select_only": selections}
    write_json(HERE / "raw/phase_atlas_fit.json", payload)
    write_json(HERE / "diagnostics/phase_atlas.json", payload)
    return payload
