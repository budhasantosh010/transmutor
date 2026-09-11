from __future__ import annotations

from collections import defaultdict
from .utils import AQ, HERE, canonical_json, read_json, sha256_json, write_json
from .response_basis import _first_order_key

FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state")


def _interaction_summary(family: str) -> dict:
    src = read_json(AQ / "raw/interaction_results.json")
    rows = [r for r in src.get("rows", []) if r.get("family") == family]
    fam = src.get("families", {}).get(family, {})
    if not rows:
        return {"second_order_required": bool(fam.get("second_order_required")), "organism_rows": 0}
    keys = (
        "oracle_interaction_median_abs_normalized",
        "oracle_interaction_p90_abs_normalized",
        "predicted_interaction_median_abs_normalized",
        "interaction_nrmse",
        "interaction_direction",
    )
    return {
        "second_order_required": bool(fam.get("second_order_required")),
        "organism_rows": len(rows),
        "equal_organism_weighting": True,
        "means": {key: sum(float(r[key]) for r in rows) / len(rows) for key in keys},
        "gate": fam.get("gate", {}),
    }


def build_reference_tables() -> dict:
    out = {
        "version": "V837ar",
        "reference_kind": "IR0_EMPIRICAL_RESPONSE_TABLE",
        "archiveable": False,
        "generalizing": False,
        "families": {},
    }
    for family in FAMILIES:
        src = read_json(AQ / f"raw/response_tensor_{family}.json")
        by_org = defaultdict(list)
        for row in src["rows"]:
            by_org[row["organism_id"]].append(row)
        pooled = defaultdict(list)
        for _organism_id, rows in by_org.items():
            local = defaultdict(list)
            for row in rows:
                local[_first_order_key(row)].append(float(row["predicted_response"]))
            for key, values in local.items():
                pooled[key].append(sum(values) / len(values))
        table = []
        for key, values in pooled.items():
            table.append({"basis_key": list(key), "mean_response": sum(values) / len(values), "organism_count": len(values)})
        table.sort(key=lambda row: tuple("" if value is None else str(value) for value in row["basis_key"]))
        interaction = _interaction_summary(family)
        payload = {
            "raw_row_count": len(src["rows"]),
            "raw_numeric_scalar_count": sum(
                sum(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row.values())
                for row in src["rows"]
            ),
            "raw_serialized_bytes": len(canonical_json(src)),
            "discovery_organisms": len(by_org),
            "equal_organism_weighting": True,
            "pooled_table": table,
            "required_second_order_summary": interaction,
            "archiveable": False,
            "generalizing": False,
        }
        payload["pooled_table_sha256"] = sha256_json({"first_order": table, "second_order": interaction})
        out["families"][family] = payload
    write_json(HERE / "raw/reference_response_tables.json", out)
    return out
