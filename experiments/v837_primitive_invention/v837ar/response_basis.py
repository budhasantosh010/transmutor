from __future__ import annotations

from .utils import AQ, HERE, read_json, sha256_json, write_json

ORDER = (
    "family",
    "phase",
    "intervention_channel",
    "magnitude",
    "secondary_intervention_channel",
    "secondary_magnitude",
    "response_horizon",
    "response_semantic_variable",
    "interaction_order",
)
FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state")


def _first_order_key(row):
    return (
        row.get("family"),
        row.get("phase"),
        row.get("intervention"),
        None if row.get("actual_delta") is None else round(float(row["actual_delta"]), 8),
        None,
        None,
        int(row.get("horizon", 1)),
        "semantic_output",
        1,
    )


def _required_second_order_keys() -> list[tuple]:
    """Materialize only interactions V837aq actually declared required.

    V837aq's interaction experiment uses CONTROL_FLIP x PAYLOAD_A_PERTURB for
    conditional routing. The interaction summary is organism-level rather than
    a raw response tensor, so the semantic basis records the frozen intervention
    identity without inventing organism-specific neural coordinates.
    """
    interactions = read_json(AQ / "raw/interaction_results.json")
    out: list[tuple] = []
    for family in FAMILIES:
        fam = interactions.get("families", {}).get(family, {})
        if not fam.get("second_order_required"):
            continue
        if family == "conditional_routing":
            out.append(
                (
                    family,
                    "CONTROL+PAYLOAD_A",
                    "CONTROL_FLIP",
                    None,
                    "PAYLOAD_A_PERTURB",
                    0.4,
                    1,
                    "semantic_output_interaction_residual",
                    2,
                )
            )
        else:
            raise RuntimeError(f"V837AR_UNDECLARED_REQUIRED_SECOND_ORDER_SEMANTICS:{family}")
    return out


def build_basis():
    keys = set()
    counts = {}
    for family in FAMILIES:
        src = read_json(AQ / f"raw/response_tensor_{family}.json")
        rows = src["rows"]
        counts[family] = len(rows)
        for row in rows:
            keys.add(_first_order_key(row))

    second_order = _required_second_order_keys()
    keys.update(second_order)
    keys_sorted = sorted(keys, key=lambda x: tuple("" if v is None else str(v) for v in x))
    payload = {
        "version": "V837ar",
        "ordering": list(ORDER),
        "semantic_units_only": True,
        "organism_id_in_basis": False,
        "hidden_state_coordinates_in_basis": False,
        "keys": [dict(zip(ORDER, key)) for key in keys_sorted],
        "source_row_counts": counts,
        "required_second_order_dimensions": len(second_order),
        "required_second_order_families": [k[0] for k in second_order],
    }
    payload["response_basis_sha256"] = sha256_json(payload["keys"])
    write_json(HERE / "raw/canonical_response_basis.json", payload)
    write_json(
        HERE / "diagnostics/response_basis_integrity.json",
        {
            "version": "V837ar",
            "pass": True,
            "basis_hash": payload["response_basis_sha256"],
            "dimensions": len(keys_sorted),
            "required_second_order_dimensions": len(second_order),
            "semantic_units_only": True,
            "organism_id_in_basis": False,
        },
    )
    return payload
