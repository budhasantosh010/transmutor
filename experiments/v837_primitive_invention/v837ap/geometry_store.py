from __future__ import annotations

from .utils import HERE, read_json


def _match(row: dict, family: str, organism_id: str, winner: dict) -> bool:
    return (
        row.get("family") == family
        and row.get("organism_id") == organism_id
        and int(row.get("k", -1)) == int(winner["k"])
        and row.get("chart_family") == winner["chart_family"]
    )


def load_discovery_geometry(family: str, organism_id: str, winner: dict) -> dict:
    if winner.get("phase_atlas"):
        payload = read_json(HERE / "raw/phase_atlas_fit.json")
        for row in payload.get("rows", []):
            if _match(row, family, organism_id, winner):
                return row
        raise KeyError(f"missing phase-atlas geometry {family}:{organism_id}")
    writer = str(winner.get("writer_family", ""))
    if writer.startswith("TANGENT_"):
        payload = read_json(HERE / "raw/tangent_field_fit.json")
        for record in payload.get("rows", []):
            row = record.get("geometry", {})
            if _match(row, family, organism_id, winner) and row.get("writer_family") == writer:
                return row
        raise KeyError(f"missing tangent geometry {family}:{organism_id}:{writer}")
    payload = read_json(HERE / "diagnostics/chart_conditioning.json")
    for row in payload.get("rows", []):
        if _match(row, family, organism_id, winner) and not row.get("phase_atlas"):
            return row
    raise KeyError(f"missing global geometry {family}:{organism_id}")
