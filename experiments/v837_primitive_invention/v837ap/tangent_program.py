from __future__ import annotations

import json
import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES

from .chart_data import phase_dataset
from .data_roles import seeds
from .setpoint_eval import evaluate_geometry
from .tangent_field import TANGENT_FAMILIES, fit_tangent_field
from .utils import HERE, write_json


def fit_global_tangent(geometry: dict, q: np.ndarray, kind: str) -> dict:
    oid = geometry["organism_id"]
    family = geometry["family"]
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    hb=[]; hc=[]; zb=[]; zc=[]
    for phase in PHASES[family]:
        ch = phase_dataset(oid, family, seeds("AP_WRITER_FIT"), Q, phase)
        if len(ch["base_states"]):
            hb.append(ch["base_states"] @ Q)
            hc.append(ch["cf_states"] @ Q)
            zb.append(ch["base_semantic"])
            zc.append(ch["cf_semantic"])
    if not hb:
        return {"valid": False, "kind": kind, "failure_code": "NO_TANGENT_OBSERVATIONS"}
    return fit_tangent_field(np.concatenate(hb), np.concatenate(hc), np.concatenate(zb), np.concatenate(zc), kind)


def tangent_geometry(base_geometry: dict, q: np.ndarray, kind: str) -> dict:
    field = fit_global_tangent(base_geometry, q, kind)
    return {
        **base_geometry,
        "branch": "AP_C_STATE_DEPENDENT_TANGENT",
        "writer_family": f"TANGENT_{kind}",
        "tangent_field": field,
        "tangent_parameter_count": int(field.get("parameter_count", 0)),
        "parameter_count": int(base_geometry.get("parameter_count", 0)) + int(field.get("parameter_count", 0)),
        "valid": bool(base_geometry.get("valid", False) and field.get("valid", False)),
    }


def evaluate_tangent_ladder(base_geometry: dict, q: np.ndarray) -> list[dict]:
    rows=[]
    for kind in TANGENT_FAMILIES:
        geom=tangent_geometry(base_geometry,q,kind)
        if geom.get("valid"):
            result=evaluate_geometry(geom,q)
        else:
            result={"organism_id":geom["organism_id"],"family":geom["family"],"engine":geom["engine"],"k":geom["k"],"chart_family":geom["chart_family"],"writer_family":geom["writer_family"],"phase_atlas":False,"pass":False,"failure_code":geom.get("tangent_field",{}).get("failure_code","TANGENT_FIELD_INVALID")}
        rows.append({"geometry":geom,"setpoint":result,"full_preliminary_pass":bool(geom.get("reader_pass") and result.get("pass"))})
    return rows


def save_tangent_rows(rows: list[dict]) -> None:
    payload={"version":"V837ap","stage":"AP5_STATE_DEPENDENT_TANGENT","rows":rows}
    write_json(HERE/"raw/tangent_field_fit.json",payload)
    write_json(HERE/"diagnostics/tangent_field.json",payload)
