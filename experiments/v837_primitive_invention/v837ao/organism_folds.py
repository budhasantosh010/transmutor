from __future__ import annotations

from collections import defaultdict

from .source_contracts import POWERED_FAMILIES, validate_v837an_source
from .utils import HERE, read_json, sha256_json, write_json

RAW = HERE / "raw/frozen_organism_folds.json"
DIAG = HERE / "diagnostics/organism_fold_integrity.json"
ENGINES = ("DIRECTED_STRUCTURAL_SEARCH", "RANDOM_STRUCTURAL_SAMPLER")


def _build() -> dict:
    source = validate_v837an_source()
    competent = [r for r in source["population"]["rows"] if bool(r.get("competent")) and r["family"] in POWERED_FAMILIES]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in competent:
        grouped[(row["family"], row["engine"])].append(row)
    families = {}
    seen = set()
    for family in POWERED_FAMILIES:
        families[family] = {"engines": {}, "discovery": [], "holdout": []}
        for engine in ENGINES:
            rows = sorted(grouped[(family, engine)], key=lambda r: r["organism_id"])
            if len(rows) < 2:
                raise RuntimeError(f"AO_ORGANISM_SPLIT_INVALID:{family}:{engine}:{len(rows)}")
            holdout = rows[-2:]
            discovery = rows[:-2]
            if not discovery:
                raise RuntimeError(f"AO_ORGANISM_SPLIT_INVALID:no discovery:{family}:{engine}")
            d_ids = [r["organism_id"] for r in discovery]
            h_ids = [r["organism_id"] for r in holdout]
            families[family]["engines"][engine] = {
                "source_count": len(rows),
                "discovery": d_ids,
                "holdout": h_ids,
            }
            families[family]["discovery"].extend(d_ids)
            families[family]["holdout"].extend(h_ids)
            for oid in d_ids + h_ids:
                if oid in seen:
                    raise RuntimeError(f"AO_ORGANISM_SPLIT_INVALID:duplicate:{oid}")
                seen.add(oid)
        families[family]["discovery"] = sorted(families[family]["discovery"])
        families[family]["holdout"] = sorted(families[family]["holdout"])
    payload = {
        "version": "V837ao",
        "algorithm": "for each powered family and V837aj engine: sort competent organism IDs lexicographically; last 2 -> BACKEND_HOLDOUT; remainder -> BACKEND_DISCOVERY",
        "performance_sorting": False,
        "holdout_per_engine_when_available": 2,
        "families": families,
        "all_accounted": len(seen) == len(competent),
        "source_competent_powered": len(competent),
    }
    payload["fold_sha256"] = sha256_json(payload)
    return payload


def freeze_organism_folds() -> dict:
    payload = read_json(RAW) if RAW.is_file() else _build()
    if not RAW.is_file():
        write_json(RAW, payload)
    expected = _build()
    if payload != expected:
        raise RuntimeError("AO_ORGANISM_SPLIT_INVALID:frozen fold drift")
    write_json(DIAG, {**payload, "pass": True, "split_frozen_before_science": True, "holdout_metrics_inspected_before_freeze": False})
    return payload


def discovery_ids(family: str) -> list[str]:
    return list(freeze_organism_folds()["families"][family]["discovery"])


def holdout_ids(family: str) -> list[str]:
    return list(freeze_organism_folds()["families"][family]["holdout"])
