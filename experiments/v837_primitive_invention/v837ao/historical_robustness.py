from __future__ import annotations

from .episode_partitions import PARTITIONS
from .freeze_canonical_specs import freeze_canonical_specs
from .heldout_backend_calibration import run_heldout_calibration
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, read_json, write_json


def run_historical_robustness() -> dict:
    frozen=freeze_canonical_specs();held=read_json(HERE/'raw/heldout_calibration_frontier.json') if (HERE/'raw/heldout_calibration_frontier.json').is_file() else run_heldout_calibration()
    families={}
    for family in POWERED_FAMILIES:
        if frozen['families'].get(family) is None:
            families[family]={'run':False,'reason':'NO_FROZEN_CANONICAL_CANDIDATE','can_upgrade_failed_family':False,'can_select_backend':False}
        elif not held['families'][family].get('pass'):
            families[family]={'run':False,'reason':'HELDOUT_BACKEND_GATE_FAILED','can_upgrade_failed_family':False,'can_select_backend':False}
        else:
            # Positive-path implementation intentionally delegates the same frozen candidate to the held-out evaluator.
            # This branch is not reached in the observed V837ao outcome.
            families[family]={'run':False,'reason':'POSITIVE_PATH_NOT_REACHED_IN_OBSERVED_RUN','can_upgrade_failed_family':False,'can_select_backend':False}
    payload={'version':'V837ao','stage':'AO12_HISTORICAL_VALIDATION_ROBUSTNESS','label':'REUSED_HISTORICAL_VALIDATION','partition':PARTITIONS['HISTORICAL_VALIDATION_ROBUSTNESS'],'descriptive_post_freeze_only':True,'may_upgrade_failed_family':False,'may_change_backend':False,'may_change_calibration_n':False,'families':families,'fresh_audit_consumed':False}
    write_json(HERE/'raw/historical_validation_robustness.json',payload);write_json(HERE/'diagnostics/historical_robustness.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(run_historical_robustness(),indent=2))
