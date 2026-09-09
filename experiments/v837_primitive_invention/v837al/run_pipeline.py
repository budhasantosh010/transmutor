from __future__ import annotations

import argparse,json
from .authorization import assert_authorized,freeze_gate
from .source_integrity import run_source_integrity
from .pairing import freeze_pairs
from .legacy_baseline_reproduction import reproduce_baselines
from .adapter_fit import fit_all_adapters
from .scope_localization import run_scope_localization
from .select_interface import freeze_selected
from .heldout_pairwise import run_pairwise_test
from .port_necessity import run_port_necessity
from .canonical_interface import run_canonical_test
from .closed_loop import run_closed_loop
from .alignment_data_frontier import run_alignment_data_frontier
from .analyze_results import analyze
from .utils import HERE,read_json


def _need(path,msg):
    if not (HERE/path).is_file():raise RuntimeError(msg)


def run(stage:str):
    if stage=='authorization':return run_source_integrity()
    assert_authorized()
    if stage=='baseline':
        _need('diagnostics/source_integrity.json','AL1 blocked: AL0 source integrity missing');freeze_pairs();return reproduce_baselines()
    if stage=='fit':
        _need('diagnostics/legacy_alignment_reproduction.json','AL4 blocked: AL1 baseline missing');return fit_all_adapters()
    if stage=='select':
        _need('raw/adapter_fits.json','AL5 blocked: AL4 fits missing');run_scope_localization();return freeze_selected()
    if stage=='test':
        _need('raw/selected_interface_configs.json','AL6 blocked: AL5 selection not frozen');p=run_pairwise_test();run_port_necessity();return p
    if stage=='canonical':
        _need('raw/pairwise_test_results.json','AL7 blocked: AL6 missing');return run_canonical_test()
    if stage=='transplant':
        _need('raw/canonical_test_results.json','AL8 blocked: AL7 missing');return run_closed_loop()
    if stage=='data-frontier':
        _need('raw/closed_loop_results.json','AL9 blocked: AL8 missing');return run_alignment_data_frontier()
    if stage=='analyze':
        for p in ('raw/pairwise_test_results.json','raw/canonical_test_results.json','raw/closed_loop_results.json'):_need(p,'AL10 blocked: predecessor artifact missing')
        if not (HERE/'raw/alignment_data_frontier.json').is_file():run_alignment_data_frontier()
        return analyze()
    raise ValueError(stage)


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['authorization','baseline','fit','select','test','canonical','transplant','data-frontier','analyze']);args=ap.parse_args();print(json.dumps(run(args.stage),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
