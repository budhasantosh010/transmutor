from __future__ import annotations

from .utils import HERE, ROOT, read_json, write_json

SOURCE = ROOT / "experiments/v837_primitive_invention/v837ap/raw/frozen_source_folds.json"
EXPECTED_FOLD_SHA = "b96eed0f053ec48b5cb183929296a371857e549ed05bbe9650df2ef32dee1c95"


def freeze_source_folds() -> dict:
    src = read_json(SOURCE)
    if src.get("fold_sha256") != EXPECTED_FOLD_SHA:
        raise RuntimeError("V837AQ_V837AP_FOLD_DRIFT")
    payload = {
        "version": "V837aq",
        "source": "V837ap exact inherited V837ao engine-stratified folds",
        "fold_sha256": src["fold_sha256"],
        "families": src["families"],
        "new_split_created": False,
        "performance_sorting": False,
        "heldout_operator_evidence_opened_before_freeze": False,
    }
    write_json(HERE / "raw/frozen_source_folds.json", payload)
    write_json(HERE / "diagnostics/fold_integrity.json", {**payload, "pass": True})
    return payload


def discovery_ids(family: str) -> list[str]:
    return list(freeze_source_folds()["families"][family]["discovery"])


def holdout_ids(family: str) -> list[str]:
    return list(freeze_source_folds()["families"][family]["holdout"])
