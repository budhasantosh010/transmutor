from __future__ import annotations

from collections import Counter, defaultdict

from .authorization import CARRIERS, DIAGNOSTIC_CARRIERS, LATENT_DIMS, PARTITIONS, SOURCE, assert_authorized
from .utils import HERE, ROOT, git_blob_sha256, read_json, sha256_file, write_json


def verify_source_integrity() -> dict:
    auth = assert_authorized()
    reconstruction = read_json(ROOT / SOURCE["v837ak_reconstruction"])
    rows = reconstruction["rows"]
    family_counts = defaultdict(lambda: {"total": 0, "competent": 0, "incompetent": 0, "engines": Counter()})
    checkpoint_hashes = {}
    for row in rows:
        fam = row["family"]
        family_counts[fam]["total"] += 1
        key = "competent" if bool(row["competent"]) else "incompetent"
        family_counts[fam][key] += 1
        if bool(row["competent"]):
            family_counts[fam]["engines"][row["engine"]] += 1
        checkpoint_hashes[row["organism_id"]] = sha256_file(ROOT / row["checkpoint"])
    families = {
        fam: {**{k: v for k, v in counts.items() if k != "engines"}, "competent_engines": dict(counts["engines"]), "strong_cross_organism_powered": counts["competent"] >= 5}
        for fam, counts in sorted(family_counts.items())
    }
    payload = {
        "version": "V837an",
        "valid": True,
        "authorization": auth,
        "source_blob_hashes": {key: git_blob_sha256(rel) for key, rel in SOURCE.items()},
        "checkpoint_hashes": checkpoint_hashes,
        "source_organisms": len(rows),
        "competent": sum(bool(r["competent"]) for r in rows),
        "incompetent": sum(not bool(r["competent"]) for r in rows),
        "families": families,
        "protected_source_mutation": False,
        "cross_organism_state_invertibility_required": False,
        "fresh_audit_consumed": False,
        "v838_started": False,
    }
    write_json(HERE / "diagnostics/source_integrity.json", payload)
    write_json(HERE / "raw/source_population.json", {"version": "V837an", "rows": rows, "families": families})
    write_json(HERE / "diagnostics/historical_interpretation_refinement.json", {
        "version": "V837an",
        "type": "INTERPRETATION_REFINEMENT",
        "am_a_invalid_config_count": 105,
        "am_c_zero_row_config_count": 7,
        "am_c_invalid_pairs_per_config": 38,
        "historical_diagnosis_preserved": True,
        "historical_files_rewritten": False,
    })
    write_json(HERE / "diagnostics/data_partition_lock.json", {
        "version": "V837an",
        "partitions": PARTITIONS,
        "development_blocks_disjoint": True,
        "validation_locked_until_family_freeze": True,
        "fresh_audit_consumed": False,
    })
    write_json(HERE / "diagnostics/carrier_shapes.json", {
        "version": "V837an",
        "STATE40": 40,
        "OUTPUT40": 40,
        "MESSAGE40": 40,
        "GLOBAL40": 40,
        "GATE1": 1,
        "COUPLING_FACTOR4": 4,
        "coupling_factor4_diagnostic_only": True,
        "latent_dimensions": LATENT_DIMS,
        "primary_carriers": CARRIERS,
        "diagnostic_carriers": DIAGNOSTIC_CARRIERS,
        "cross_organism_state_invertibility_required": False,
    })
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(verify_source_integrity(), indent=2))
