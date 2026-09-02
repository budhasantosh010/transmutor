from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECOVERY = ROOT / "experiments" / "v836_recovery"
EXPECTED_SOURCE_SHA256 = "dcb0bfc2e4927ee89511ae7f87701ba465f6a7193a838cdb1b2dc0e1c122f953"
EXPECTED_RESULT_SHA256 = "0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d"
HISTORICAL_RESULT = ROOT / "archive" / "preserved_artifacts" / "transmutor_experiments_v836plus" / "v836_results.json"
DEFAULT_SOURCE_CANDIDATES = [
    ROOT / "transmutor_v828_v836_experiments.zip",
    ROOT / "archive" / "preserved_artifacts" / "transmutor_v828_v836_experiments.zip",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence-safe V836 reproduction gate.")
    parser.add_argument("--source-archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RECOVERY / "v836_reproduction_results.json")
    args = parser.parse_args()

    if not HISTORICAL_RESULT.exists():
        raise SystemExit(f"Historical result missing: {HISTORICAL_RESULT}")
    result_hash = sha256(HISTORICAL_RESULT)
    if result_hash != EXPECTED_RESULT_SHA256:
        raise SystemExit(f"Historical V836 result integrity failure: {result_hash}")

    historical = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    candidates = [args.source_archive] if args.source_archive else DEFAULT_SOURCE_CANDIDATES
    source = next((path for path in candidates if path is not None and path.exists()), None)

    output = {
        "experiment": "V836",
        "historical_status": "PASS" if historical.get("V836_PASS") is True else "UNKNOWN",
        "historical_result_sha256": result_hash,
        "expected_source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "source_archive_found": bool(source),
        "reproduction_classification": None,
        "historical_metric": historical.get("results", {}).get("3", {}).get("test_mean_regret"),
        "reproduction_metric": None,
        "absolute_difference": None,
        "relative_difference": None,
        "notes": [],
    }

    if source is None:
        output["reproduction_classification"] = "CANNOT_REPRODUCE_MISSING_SOURCE"
        output["notes"].append(
            "The registry/inventory references transmutor_v828_v836_experiments.zip, but it is absent from the migrated repository and migration package."
        )
        output["notes"].append(
            "No historical executable, dataset generator, seed policy, exact pass-gate logic, or per-task library-cost matrix is preserved, so replacement numbers would be fabrication rather than reproduction."
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
        return 2

    source = source.resolve()
    source_hash = sha256(source)
    output["source_archive_path"] = str(source)
    output["source_archive_sha256"] = source_hash
    if source_hash != EXPECTED_SOURCE_SHA256:
        output["reproduction_classification"] = "CANNOT_REPRODUCE_MISSING_SOURCE"
        output["notes"].append("A candidate source archive was found, but its SHA-256 does not match the preserved inventory hash.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
        return 3

    output["reproduction_classification"] = "CANNOT_REPRODUCE_MISSING_DEPENDENCY"
    output["notes"].append(
        "The historical source archive hash matches, but an executable entrypoint contract must be reconstructed and attributed before any scientific rerun."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
