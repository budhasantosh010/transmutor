from __future__ import annotations

from .program_ir import canonical_bytes
from .utils import HERE, read_json, write_json

PARAMETER_CEILING = 256


def _count_params(ir):
    if not ir:
        return 0
    complexity = ir.get("complexity") or {}
    if "continuous_parameters" in complexity:
        return int(complexity["continuous_parameters"])
    return len(ir.get("parameters", {}))


def _compactness_class(n):
    if n <= 16:
        return "TINY_IR"
    if n <= 64:
        return "VERY_COMPACT_IR"
    if n <= 256:
        return "COMPACT_IR"
    if n <= 1024:
        return "HEAVY_IR"
    return "NOT_COMPACT"


def _cost(ir):
    cls = (ir or {}).get("operator_class")
    if cls == "LINEAR_STATE_UPDATE_1D":
        return {"scalar_additions": 2, "scalar_multiplies": 2, "nonlinear_functions": 0, "state_scalars": 1, "temporary_scalars": 1}
    if cls == "BILINEAR_STATIC_MAP":
        return {"scalar_additions": 5, "scalar_multiplies": 8, "nonlinear_functions": 0, "state_scalars": 0, "temporary_scalars": 3}
    if cls == "SECOND_ORDER_STATIC_MAP":
        return {"scalar_additions": 8, "scalar_multiplies": 14, "nonlinear_functions": 0, "state_scalars": 0, "temporary_scalars": 6}
    if cls == "LINEAR_PHASE_MACHINE_R1":
        return {"scalar_additions": 1, "scalar_multiplies": 2, "nonlinear_functions": 0, "state_scalars": 1, "temporary_scalars": 1}
    return {"scalar_additions": 0, "scalar_multiplies": 0, "nonlinear_functions": 0, "state_scalars": 0, "temporary_scalars": 0}


def _compute(ir_map: dict) -> dict:
    refs = read_json(HERE / "raw/reference_response_tables.json")
    out = {"version": "V837ar", "parameter_ceiling": PARAMETER_CEILING, "families": {}}
    for family, ref in refs["families"].items():
        ir = ir_map.get(family)
        n = _count_params(ir)
        serialized = len(canonical_bytes(ir)) if ir else 0
        raw_bytes = ref["raw_serialized_bytes"]
        archiveable = bool(ir and n <= PARAMETER_CEILING and ir.get("operator_class") != "EMPIRICAL_RESPONSE_TABLE")
        out["families"][family] = {
            "raw_rows": ref["raw_row_count"],
            "raw_numeric_scalars": ref["raw_numeric_scalar_count"],
            "raw_bytes": raw_bytes,
            "ir_parameters": n,
            "ir_bytes": serialized,
            "byte_compression_ratio": (raw_bytes / serialized if serialized else None),
            "scalar_compression_ratio": (ref["raw_numeric_scalar_count"] / n if n else None),
            "compactness_class": (_compactness_class(n) if ir else "NO_IR"),
            "archiveable": archiveable,
            "reference_response_table_archiveable": False,
            "execution_cost": _cost(ir),
        }
    return out


def compute_pre_meta_minimality() -> dict:
    winners = read_json(HERE / "raw/discovery_program_ir_winners.json")
    ir_map = {family: (winner or {}).get("ir") for family, winner in winners.get("families", {}).items()}
    out = _compute(ir_map)
    out.update({
        "stage": "AR7_COMPRESSION_MINIMALITY",
        "pre_meta": True,
        "one_candidate_per_family": True,
        "coefficients_refit": False,
    })
    write_json(HERE / "raw/pre_meta_compression.json", out)
    write_json(HERE / "diagnostics/minimality.json", out)
    return out


def compute_compression(ir_map=None):
    irs = ir_map or read_json(HERE / "raw/frozen_canonical_program_irs.json")["families"]
    out = _compute(irs)
    out.update({"stage": "AR15_FINAL_COMPRESSION", "pre_meta": False})
    write_json(HERE / "raw/compression_results.json", out)
    write_json(HERE / "diagnostics/compression.json", out)
    return out
