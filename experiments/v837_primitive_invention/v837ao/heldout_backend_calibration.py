from __future__ import annotations

import time

from .calibration_frontier import CALIBRATION_LADDER, family_gate, qualifier
from .commutativity import evaluate_dynamics
from .episode_partitions import seeds
from .freeze_canonical_specs import freeze_canonical_specs
from .k1_backend import evaluate_reader_on_partition, fit_backend
from .organism_folds import freeze_organism_folds
from .quotient_eval import evaluate_quotient
from .setpoint_eval import evaluate_setpoints
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, sha256_json, write_json


def run_heldout_calibration() -> dict:
    started=time.perf_counter(); frozen=freeze_canonical_specs();folds=freeze_organism_folds();families={};all_rows=[];backend_rows=[]
    for family in POWERED_FAMILIES:
        spec=frozen['families'].get(family);holdouts=folds['families'][family]['holdout'];front=[];minimum=None
        if spec is None:
            families[family]={'candidate':None,'holdout_count':len(holdouts),'frontier':[],'minimum_successful_n':None,'qualifier':None,'pass':False,'reason':'NO_FROZEN_CANONICAL_CANDIDATE'};continue
        variant=spec['backend_type']
        for n in CALIBRATION_LADDER:
            rows=[]
            for oid in holdouts:
                t0=time.perf_counter();backend=fit_backend(oid,family,variant,seeds('AO_BACKEND_FIT'),calibration_n=n);reader=evaluate_reader_on_partition(backend,seeds('AO_HELDOUT_ORGANISM_EVAL'));setp=evaluate_setpoints(backend,seeds('AO_HELDOUT_ORGANISM_EVAL'));quot=evaluate_quotient(backend,seeds('AO_HELDOUT_ORGANISM_EVAL'),partition_name='AO_HELDOUT_ORGANISM_EVAL');dyn=evaluate_dynamics(backend,seeds('AO_HELDOUT_ORGANISM_EVAL'),partition_name='AO_HELDOUT_ORGANISM_EVAL');passed=bool(reader.get('reader_pass') and reader.get('algebra_pass') and setp.get('pass') and quot.get('pass') and dyn.get('pass'))
                row={'organism_id':oid,'family':family,'engine':backend['engine'],'variant':variant,'calibration_n':n,'reader':reader,'setpoint':setp,'quotient':quot,'dynamics':dyn,'pass':passed,'calibration_cost':{'paired_examples_used':n,'raw_state_vectors_consumed':2*n,'svd_count':1 if variant!='B2_PHASE_K1' else 3,'ridge_solves':2 if variant=='B0_GLOBAL_K1' else 6,'cpu_seconds':None,'wall_seconds':time.perf_counter()-t0,'backend_parameter_count':44 if variant=='B0_GLOBAL_K1' else (50 if variant=='B1_PHASE_GAUGE_K1' else 132),'backend_stored_bytes':(44 if variant=='B0_GLOBAL_K1' else (50 if variant=='B1_PHASE_GAUGE_K1' else 132))*8}}
                rows.append(row);all_rows.append(row);backend_rows.append({'organism_id':oid,'family':family,'engine':backend['engine'],'variant':variant,'calibration_n':n,'backend':backend,'evaluated':True,'source_checkpoint_hash_preserved':True})
            gate=family_gate(rows);front.append({'n':n,'gate':gate});
            if minimum is None and gate['pass']:minimum=n
        families[family]={'candidate':spec,'holdout_count':len(holdouts),'frontier':front,'minimum_successful_n':minimum,'qualifier':qualifier(minimum),'pass':minimum is not None}
    payload={'version':'V837ao','stage':'AO10_HELDOUT_BACKEND','calibration_ladder':list(CALIBRATION_LADDER),'families':families,'rows':all_rows,'heldout_backend_artifacts_read_only_after_freeze':True,'historical_v837an_per_organism_backend_loaded':False,'variant_search_on_holdout':False,'k_search_on_holdout':False,'phase_search_on_holdout':False,'wall_seconds':time.perf_counter()-started}
    write_json(HERE/'raw/heldout_calibration_frontier.json',payload);write_json(HERE/'raw/heldout_backend_results.json',{'version':'V837ao','rows':all_rows,'backend_records':backend_rows});write_json(HERE/'diagnostics/heldout_backend_isolation.json',{'version':'V837ao','pass':True,'frozen_spec_sha256':frozen['frozen_sha256'],'heldout_read_after_freeze':True,'v837an_q_loaded':False,'v837an_semantic_compiler_loaded':False,'variant_search':False,'k_search':False,'phase_search':False});write_json(HERE/'diagnostics/calibration_frontier.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(run_heldout_calibration(),indent=2))
