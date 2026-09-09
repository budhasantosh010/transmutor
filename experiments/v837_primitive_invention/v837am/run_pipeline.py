from __future__ import annotations

import argparse,json

from .authorization import assert_authorized
from .source_integrity import run_source_integrity
from .pairing import freeze_pairs
from .failure_ledger import initialize_historical,record_preselection_engineering_failures,record_schema1_invalidation
from .reality_gates import run as run_reality
from .branch_a_context import run as run_a
from .branch_b_boundary import run as run_b
from .branch_c_temporal_interaction import run as run_c
from .meta_confirm import run_meta
from .final_validation import run_dev_confirm,run_final_validation
from .context_closed_loop import run_closed_loop
from .analyze_results import analyze
from .utils import HERE,read_json

STAGES=("source","context","boundary","temporal","meta","dev-confirm","final","closed-loop","analyze")

def need(rel,msg):
    p=HERE/rel
    if not p.is_file():raise RuntimeError(msg)
    return read_json(p) if p.suffix=='.json' else p

def run(stage):
    assert_authorized()
    if stage=="source":
        initialize_historical();record_preselection_engineering_failures();freeze_pairs();out=run_source_integrity();run_reality();return out
    need("raw/source_state.json","V837AM_SOURCE_STAGE_REQUIRED")
    if stage=="context":return run_a()
    if stage=="boundary":return run_b()
    if stage=="temporal":return run_c()
    if stage=="meta":
        need("raw/am_a_selection.json","V837AM_AM_A_REQUIRED");need("raw/am_b_selection.json","V837AM_AM_B_REQUIRED");need("raw/am_c_selection.json","V837AM_AM_C_REQUIRED");return run_meta()
    if stage=="dev-confirm":need("raw/selected_final_hypothesis.json","V837AM_META_REQUIRED");return run_dev_confirm()
    if stage=="final":need("raw/final_dev_confirmation.json","V837AM_DEV_CONFIRM_REQUIRED");return run_final_validation()
    if stage=="closed-loop":need("raw/final_validation.json","V837AM_FINAL_STAGE_REQUIRED");return run_closed_loop()
    if stage=="analyze":
        for x in ("am_a_selection.json","am_b_selection.json","am_c_selection.json","meta_confirmation.json","selected_final_hypothesis.json","final_dev_confirmation.json","final_validation.json","closed_loop_results.json"):need("raw/"+x,"V837AM_PIPELINE_INCOMPLETE:"+x)
        return analyze()
    raise ValueError(stage)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=STAGES,required=True);args=ap.parse_args();print(json.dumps(run(args.stage),indent=2)[:6000]);return 0
if __name__=='__main__':raise SystemExit(main())
