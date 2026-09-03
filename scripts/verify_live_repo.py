from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "verification" / "live_repo_manifest.json"
SHA_PATH = ROOT / "verification" / "active_research_sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_bytes(relative: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"unable to read committed Git blob: {relative}") from exc


def git_blob_sha256(relative: str) -> str:
    return hashlib.sha256(git_blob_bytes(relative)).hexdigest()


def load_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing required file: {relative}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {relative}: {exc}") from exc


def require_path(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise RuntimeError(f"missing required path: {relative}")
    return path


def verify_sha_manifest() -> int:
    if not SHA_PATH.is_file():
        raise RuntimeError("missing verification/active_research_sha256.txt")
    checked = 0
    for lineno, raw in enumerate(SHA_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "  " not in line:
            raise RuntimeError(f"malformed SHA line {lineno}")
        expected, relative = line.split("  ", 1)
        path = require_path(relative)
        if not path.is_file():
            raise RuntimeError(f"SHA entry is not a file: {relative}")
        actual = git_blob_sha256(relative)
        if actual != expected:
            raise RuntimeError(f"committed Git-blob SHA mismatch: {relative}: {actual} != {expected}")
        checked += 1
    if checked == 0:
        raise RuntimeError("active-research SHA manifest is empty")
    return checked


def main() -> int:
    manifest = load_json("verification/live_repo_manifest.json")
    if manifest.get("repository") != "budhasantosh010/transmutor":
        raise RuntimeError("unexpected repository identity")

    critical_paths = manifest.get("critical_paths", [])
    for relative in critical_paths:
        require_path(relative)

    historical_git = manifest["historical_git_blob_sha256"]
    historical_paths = {
        "v836_result_sha256": "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json",
        "v837_result_sha256": "experiments/v837_primitive_invention/v837/results.json",
        "v837b_result_sha256": "experiments/v837_primitive_invention/v837b/results.json",
        "v837c_result_sha256": "experiments/v837_primitive_invention/v837c/results.json",
        "frozen_gate_sha256": "experiments/v837_primitive_invention/frozen_gates.json",
    }
    for key, relative in historical_paths.items():
        require_path(relative)
        actual = git_blob_sha256(relative)
        expected = historical_git[key]
        if actual != expected:
            raise RuntimeError(f"historical committed-blob mismatch for {relative}: {actual} != {expected}")

    for variant_name in ("V837d", "V837g", "V837h"):
        record = manifest["current_variants"][variant_name]
        for relative in record.get("source", []):
            require_path(relative)
        require_path(record["config"])
        result = load_json(record["results"])
        if result.get("version") != variant_name:
            raise RuntimeError(f"{variant_name} result version mismatch")
        if result.get("pass") is not False:
            raise RuntimeError(f"{variant_name} is expected to be a preserved FAIL result")
        if result.get("fresh_audit_consumed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly consumed fresh-audit data")
        if result.get("primitive_mining_allowed") is not False:
            raise RuntimeError(f"{variant_name} unexpectedly reopened primitive mining")
        for relative in record.get("documentation", []):
            require_path(relative)
        for relative in record.get("plots", []):
            require_path(relative)

    v837d = load_json(manifest["current_variants"]["V837d"]["results"])
    if v837d["resource_accounting"].get("historical_parameter_count") != 856:
        raise RuntimeError("V837d historical parameter count is not 856")

    v837g = load_json(manifest["current_variants"]["V837g"]["results"])
    if v837g["resource_accounting"].get("parameter_count") != 866:
        raise RuntimeError("V837g parameter count is not 866")

    v837h = load_json(manifest["current_variants"]["V837h"]["results"])
    matching = v837h.get("parameter_matching", {})
    if matching.get("multiplicative_parameter_count") != 1096:
        raise RuntimeError("V837h multiplicative parameter count is not 1096")
    if matching.get("additive_control_parameter_count") != 1096:
        raise RuntimeError("V837h additive control parameter count is not 1096")
    if matching.get("additive_equals_multiplicative") is not True:
        raise RuntimeError("V837h parameter-matched control is not actually matched")

    audit = load_json("experiments/v837_primitive_invention/audit/audit_results.json")
    if audit.get("episodes_consumed") != 0:
        raise RuntimeError("fresh audit episodes have been consumed")

    lineage = load_json("experiments/v837_primitive_invention/lineage_status.json")
    promoted = lineage.get("primitives_promoted")
    if promoted not in (0, [], None):
        raise RuntimeError("primitive promotions must remain zero")

    recovery = load_json("experiments/v837_primitive_invention/representation_recovery_status.json")
    if recovery.get("primitive_mining_allowed") is not False:
        raise RuntimeError("representation recovery unexpectedly reopened primitive mining")
    if recovery.get("fresh_audit_episodes_consumed") != 0:
        raise RuntimeError("representation recovery consumed fresh-audit data")
    if recovery.get("primitives_promoted") != 0:
        raise RuntimeError("representation recovery promoted primitives")

    for relative in manifest.get("validators", []):
        require_path(relative)
    for relative in manifest.get("tests", []):
        require_path(relative)

    checked = verify_sha_manifest()
    print(f"live repository verification: PASS ({checked} SHA-256 entries checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
