from __future__ import annotations

from .authorization import assert_authorized
from .episode_partitions import PARTITIONS, REUSED_HISTORICAL_VALIDATION, assert_disjoint_development_partitions
from .failure_ledger import initialize_failure_memory
from .source_contracts import POWERED_FAMILIES, SOURCE_PATHS, validate_v837an_source
from .utils import HERE, sha256_file, write_json


def verify_source_integrity() -> dict:
    initialize_failure_memory()
    auth = assert_authorized()
    assert_disjoint_development_partitions()
    src = validate_v837an_source()
    population = src["population"]["rows"]
    competent = [r for r in population if bool(r.get("competent"))]
    by_family = {}
    for family in sorted({r["family"] for r in competent}):
        rows = [r for r in competent if r["family"] == family]
        by_family[family] = {
            "competent": len(rows),
            "engines": {e: sum(r["engine"] == e for r in rows) for e in sorted({r["engine"] for r in rows})},
            "powered_in_v837ao": family in POWERED_FAMILIES,
        }
    payload = {
        "version": "V837ao", "pass": True, "authorization": auth,
        "source_organisms": len(population), "competent_organisms": len(competent),
        "families": by_family, "powered_families": list(POWERED_FAMILIES),
        "partial_observation_unresolved": True,
        "v837an_source_hashes": src["source_hashes"],
        "v837an_frozen_file_sha256": src["frozen_file_sha256"],
        "historical_validation_reused": REUSED_HISTORICAL_VALIDATION,
        "historical_validation_label": "REUSED_HISTORICAL_VALIDATION",
        "fresh_audit_consumed": False, "v838_started": False,
    }
    write_json(HERE / "raw/source_state.json", payload)
    write_json(HERE / "raw/v837an_contract_hashes.json", {"version":"V837ao", **src["source_hashes"], "frozen_file_sha256":src["frozen_file_sha256"]})
    write_json(HERE / "diagnostics/source_integrity.json", payload)
    write_json(HERE / "diagnostics/v837an_reproduction.json", {
        "version":"V837ao",
        "source_version":"V837an",
        "source_diagnosis":src["decision"]["diagnosis"],
        "causal_subspace_families":src["decision"]["causal_subspace_families"],
        "semantic_compiler_families":src["decision"]["semantic_compiler_families"],
        "validated_family_abstractions":src["decision"]["validated_family_abstractions"],
        "phase_conditional_representation":src["decision"]["phase_conditional_representation"],
        "frozen_abstraction_sha256":src["frozen_file_sha256"],
        "source_hashes":src["source_hashes"],
        "reproduction_type":"EXACT_COMMITTED_ARTIFACT_REPRODUCTION",
        "pass":True,
        "new_model_fits":0,
        "fresh_audit_consumed":False,
    })
    partition = {"version":"V837ao", "partitions":PARTITIONS, "development_disjoint":True, "historical_validation_reused":True, "fresh_audit_unused":True}
    write_json(HERE / "raw/data_partition_lock.json", partition)
    write_json(HERE / "diagnostics/data_partition_integrity.json", partition)
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(verify_source_integrity(), indent=2))
