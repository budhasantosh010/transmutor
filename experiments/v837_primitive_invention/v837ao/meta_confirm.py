from __future__ import annotations

import math

from .commutativity import evaluate_dynamics
from .episode_partitions import seeds
from .k1_backend import evaluate_reader_on_partition, fit_backend
from .organism_folds import freeze_organism_folds
from .quotient_eval import evaluate_quotient
from .setpoint_eval import evaluate_setpoints
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, read_json, write_json


def _aggregate(rows: list[dict]) -> dict:
    n=len(rows); passing=[r for r in rows if r.get('pass')]; required=max(1,math.ceil(.60*n)); engines=sorted({r['engine'] for r in passing})
    return {'organisms':n,'passing':len(passing),'required':required,'pass_fraction':0.0 if not n else len(passing)/n,'pass_engines':engines,'pass':bool(n and len(passing)>=required and len(engines)>=2)}


def run_meta_confirm() -> dict:
    discovery=read_json(HERE/'raw/discovery_family_backend_winners.json'); folds=freeze_organism_folds(); rows=[]; confirmed={}
    for family in POWERED_FAMILIES:
        winner=discovery['family_winners'].get(family)
        if winner is None:
            confirmed[family]=None
            rows.append({'family':family,'candidate':None,'partition':'AO_META_CONFIRM','no_refit':True,'pass':False,'reason':'NO_DISCOVERY_FAMILY_CANDIDATE'})
            continue
        variant=winner['variant']; organism_rows=[]
        # Reconstruct coefficients from the original AO_BACKEND_FIT only; META does not fit on META episodes.
        for oid in folds['families'][family]['discovery']:
            backend=fit_backend(oid,family,variant,seeds('AO_BACKEND_FIT'))
            reader=evaluate_reader_on_partition(backend,seeds('AO_META_CONFIRM')); setp=evaluate_setpoints(backend,seeds('AO_META_CONFIRM')); quot=evaluate_quotient(backend,seeds('AO_META_CONFIRM'),partition_name='AO_META_CONFIRM'); dyn=evaluate_dynamics(backend,seeds('AO_META_CONFIRM'),partition_name='AO_META_CONFIRM')
            passed=bool(reader.get('reader_pass') and reader.get('algebra_pass') and setp.get('pass') and quot.get('pass') and dyn.get('pass'))
            organism_rows.append({'organism_id':oid,'family':family,'engine':backend['engine'],'variant':variant,'reader':reader,'setpoint':setp,'quotient':quot,'dynamics':dyn,'pass':passed})
        agg=_aggregate(organism_rows); candidate={'family':family,'variant':variant,'meta_gate':agg,'support_organisms':[r['organism_id'] for r in organism_rows if r['pass']]} if agg['pass'] else None; confirmed[family]=candidate
        rows.append({'family':family,'candidate':winner,'variant':variant,'partition':'AO_META_CONFIRM','no_refit':True,'organism_results':organism_rows,'aggregate':agg,'pass':bool(candidate)})
    payload={'version':'V837ao','stage':'AO8_META_CONFIRM','partition':'AO_META_CONFIRM','no_refit':True,'no_variant_fallback':True,'results':rows,'confirmed_families':confirmed,'meta_confirmed_families':sum(v is not None for v in confirmed.values())}
    write_json(HERE/'raw/meta_confirmation.json',payload);write_json(HERE/'diagnostics/meta_confirmation.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(run_meta_confirm(),indent=2))
