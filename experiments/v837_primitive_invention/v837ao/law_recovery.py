from __future__ import annotations

from .freeze_canonical_specs import freeze_canonical_specs
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, write_json


def run_law_recovery() -> dict:
    frozen=freeze_canonical_specs();families={}
    for family in POWERED_FAMILIES:
        if frozen['families'].get(family) is None:
            families[family]={'run':False,'pass':False,'reason':'NO_CANONICAL_TRAJECTORY_CANDIDATE','diagnostic_only':True}
        else:
            families[family]={'run':False,'pass':False,'reason':'POSITIVE_PATH_NOT_REACHED_IN_OBSERVED_RUN','diagnostic_only':True}
    payload={'version':'V837ao','stage':'AO13_TRANSITION_LAW_EXTRACTION','diagnostic_only':True,'gating':False,'families':families,'law_recovery_families':sum(1 for v in families.values() if v.get('pass')),'claim':'No law-recovery claim without a frozen canonical trajectory candidate.'}
    write_json(HERE/'raw/law_recovery.json',payload);write_json(HERE/'diagnostics/law_recovery.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(run_law_recovery(),indent=2))
