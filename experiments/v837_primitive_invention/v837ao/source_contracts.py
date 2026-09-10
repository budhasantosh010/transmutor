from __future__ import annotations

from .utils import ROOT, git_blob_sha256, read_json, sha256_file

POWERED_FAMILIES = (
    "conditional_routing",
    "delayed_recall",
    "iterative_state",
    "variable_composition",
)
PARTIAL_FAMILY = "partial_observation"
V837AN = ROOT / "experiments/v837_primitive_invention/v837an"
SOURCE_PATHS = {
    "decision": "experiments/v837_primitive_invention/v837an/diagnostics/decision_state.json",
    "results": "experiments/v837_primitive_invention/v837an/results.json",
    "frozen_abstractions": "experiments/v837_primitive_invention/v837an/raw/frozen_family_abstractions.json",
    "source_population": "experiments/v837_primitive_invention/v837an/raw/source_population.json",
    "semantic_compiler_impl": "experiments/v837_primitive_invention/v837an/semantic_compiler.py",
}


def validate_v837an_source() -> dict:
    decision = read_json(ROOT / SOURCE_PATHS["decision"])
    results = read_json(ROOT / SOURCE_PATHS["results"])
    frozen = read_json(ROOT / SOURCE_PATHS["frozen_abstractions"])
    population = read_json(ROOT / SOURCE_PATHS["source_population"])
    required = {
        "diagnosis": "GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN",
        "causal_subspace_families": 4,
        "semantic_compiler_families": 4,
        "validated_family_abstractions": 4,
        "phase_conditional_representation": True,
        "primitive_archive_allowed_next": False,
        "primitives_promoted": 0,
        "fresh_audit_consumed": False,
        "v838_started": False,
        "next_program": "V837ao_LATENT_PRIMITIVE_CANONICALIZATION",
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            raise RuntimeError(f"AO_SOURCE_INTEGRITY_FAILURE:{key}:{decision.get(key)!r}")
    if results.get("diagnosis") != required["diagnosis"]:
        raise RuntimeError("AO_SOURCE_INTEGRITY_FAILURE:results diagnosis")
    for family in POWERED_FAMILIES:
        row = frozen.get("families", {}).get(family)
        cand = None if row is None else row.get("candidate")
        if not row or row.get("family_pass") is not True or not cand:
            raise RuntimeError(f"AO_SOURCE_INTEGRITY_FAILURE:frozen family {family}")
        if cand.get("config_id") != "STATE40-K1" or cand.get("carrier") != "STATE40" or int(cand.get("k", -1)) != 1 or cand.get("semantic_compiler") is not True:
            raise RuntimeError(f"AO_SOURCE_INTEGRITY_FAILURE:wrong V837an winner {family}")
    if frozen.get("families", {}).get(PARTIAL_FAMILY, "MISSING") is not None:
        raise RuntimeError("AO_SOURCE_INTEGRITY_FAILURE:partial observation must remain unresolved/null")
    rows = population.get("rows", [])
    competent = [r for r in rows if bool(r.get("competent"))]
    if len(rows) != 50 or len(competent) != 40:
        raise RuntimeError("AO_SOURCE_INTEGRITY_FAILURE:source population")
    return {
        "decision": decision,
        "results": results,
        "frozen": frozen,
        "population": population,
        "source_hashes": {k: git_blob_sha256(v) for k, v in SOURCE_PATHS.items()},
        "frozen_file_sha256": sha256_file(ROOT / SOURCE_PATHS["frozen_abstractions"]),
    }
