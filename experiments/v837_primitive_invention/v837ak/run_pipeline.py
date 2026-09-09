from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.v837ak.analyze_results import analyze
from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.boundary_substitution import run_boundary_substitution
from experiments.v837_primitive_invention.v837ak.candidate_discovery import freeze_candidate_classes
from experiments.v837_primitive_invention.v837ak.causal_interventions import run_causal
from experiments.v837_primitive_invention.v837ak.closed_loop_substitution import run_closed_loop
from experiments.v837_primitive_invention.v837ak.confirm_candidates import confirm_candidates
from experiments.v837_primitive_invention.v837ak.dynamic_clustering import cluster_dynamic_classes
from experiments.v837_primitive_invention.v837ak.fingerprint_reliability import calibrate_reliability
from experiments.v837_primitive_invention.v837ak.ported_primitive import run_replay_gate
from experiments.v837_primitive_invention.v837ak.probe_partition import freeze_probe_partition
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import reconstruct_all
from experiments.v837_primitive_invention.v837ak.subset_census import run_census
from experiments.v837_primitive_invention.v837ak.utils import HERE,write_json

STAGES=(
    ("AK0_reconstruct",reconstruct_all),
    ("AK1_probe_freeze",freeze_probe_partition),
    ("AK2_replay_gate",run_replay_gate),
    ("AK3_census",run_census),
    ("AK4_fingerprint_reliability",calibrate_reliability),
    ("AK5_dynamic_clustering",cluster_dynamic_classes),
    ("AK5_candidate_freeze",freeze_candidate_classes),
    ("AK6_confirmation",confirm_candidates),
    ("AK7_causal",run_causal),
    ("AK8_boundary",run_boundary_substitution),
    ("AK9_closed_loop",run_closed_loop),
    ("AK10_analyze",analyze),
)


def run(start_at:str|None=None,stop_after:str|None=None)->dict:
    assert_v837ak_authorized();timing_path=HERE/"diagnostics/stage_timings.json"
    timings=json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {"version":"V837ak","stages":{}}
    active=start_at is None;last=None
    for name,fn in STAGES:
        if name==start_at:active=True
        if not active:continue
        print(f"RUN {name}",flush=True);w=time.perf_counter();c=time.process_time();fn();entry={"wall_seconds":float(time.perf_counter()-w),"cpu_seconds":float(time.process_time()-c)};timings["stages"][name]=entry;write_json(timing_path,timings);last=name
        if stop_after==name:break
    return {"version":"V837ak","last_stage":last,"timings":timings}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--start-at",choices=[x[0] for x in STAGES]);p.add_argument("--stop-after",choices=[x[0] for x in STAGES]);a=p.parse_args();r=run(a.start_at,a.stop_after);print(json.dumps({"last_stage":r["last_stage"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
