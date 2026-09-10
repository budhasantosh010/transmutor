from __future__ import annotations

from .utils import HERE, ROOT, read_json, sha256_file, write_json

AO_FOLDS=ROOT/"experiments/v837_primitive_invention/v837ao/raw/frozen_organism_folds.json"
RAW=HERE/"raw/frozen_source_folds.json"
DIAG=HERE/"diagnostics/fold_integrity.json"
EXPECTED_AO_FOLD_SHA="b96eed0f053ec48b5cb183929296a371857e549ed05bbe9650df2ef32dee1c95"

def freeze_source_folds()->dict:
    source=read_json(AO_FOLDS)
    if source.get("fold_sha256")!=EXPECTED_AO_FOLD_SHA:
        raise RuntimeError("V837AP_V837AO_FOLD_DRIFT")
    payload={
      "version":"V837ap","source":"V837ao frozen engine-stratified folds","v837ao_fold_sha256":source["fold_sha256"],
      "fold_sha256":source["fold_sha256"],"same_as_v837ao":True,
      "families":source["families"],"performance_sorting":False,"new_split_created":False,
      "all_accounted":bool(source.get("all_accounted",True)),
    }
    if RAW.is_file():
        prior=read_json(RAW)
        if prior.get("families")!=source["families"] or prior.get("v837ao_fold_sha256")!=source["fold_sha256"]:
            raise RuntimeError("V837AP_FOLD_DRIFT")
    write_json(RAW,payload)
    diag={**payload,"pass":True,"heldouts_unopened_for_geometry_selection":True}
    write_json(DIAG,diag);write_json(HERE/"diagnostics/source_fold_integrity.json",diag)
    return payload

def discovery_ids(family:str)->list[str]: return list(freeze_source_folds()["families"][family]["discovery"])
def holdout_ids(family:str)->list[str]: return list(freeze_source_folds()["families"][family]["holdout"])
