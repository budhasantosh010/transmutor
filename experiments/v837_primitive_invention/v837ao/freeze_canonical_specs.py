from __future__ import annotations

from .canonical_ir import canonical_ir
from .meta_confirm import run_meta_confirm
from .setpoint_grid import family_grid
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, read_json, sha256_json, write_json


def freeze_canonical_specs() -> dict:
    path=HERE/'raw/frozen_canonical_family_specs.json'
    if path.is_file(): return read_json(path)
    meta=read_json(HERE/'raw/meta_confirmation.json') if (HERE/'raw/meta_confirmation.json').is_file() else run_meta_confirm()
    families={}
    for family in POWERED_FAMILIES:
        cand=meta['confirmed_families'].get(family)
        if cand is None:
            families[family]=None;continue
        ir=canonical_ir(family).to_dict();grid=family_grid(family)
        families[family]={'family':family,'canonical_ir':ir,'semantic_variable':ir['semantic_variable'],'semantic_dimension':1,'semantic_range':grid,'backend_type':cand['variant'],'reader_algorithm':'affine K1 reader E(s)=a(q^T s)+b; ridge/OLS lambda=1e-6','writer_algorithm':'V837an zero-intercept semantic compiler followed by g^T w=1 gauge fix','ridge_lambda':1e-6,'setpoint_grid':grid['targets'],'quotient_test':'P=w g^T; same-phase max-residual donor; semantic align; common future context','commutativity_gates':{'natural_nrmse':.10,'natural_median_abs':.075,'natural_direction':.90,'interventional_one_step':.10,'rollout_nrmse':.15,'output_recovery':.80,'task_success':.75,'random_margin':.20,'ood':2.0},'meta_evidence':cand}
    payload={'version':'V837ao','stage':'AO9_FREEZE','families':families,'frozen_before_holdout_backend_read':True,'heldout_backend_evidence_read_before_freeze':False,'primitives_promoted':0}
    payload['frozen_sha256']=sha256_json(payload);write_json(path,payload);write_json(HERE/'diagnostics/canonical_freeze.json',{'version':'V837ao','pass':True,'frozen_sha256':payload['frozen_sha256'],'frozen_before_holdout_backend_read':True,'candidate_count':sum(v is not None for v in families.values())});return payload


if __name__=='__main__':
    import json;print(json.dumps(freeze_canonical_specs(),indent=2))
