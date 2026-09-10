from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES

from .chart_data import phase_dataset
from .chart_reader import read_chart, reader_metrics
from .geometry_runtime import chart_for_phase
from .setpoint_grid import family_grid

BINARY = {"conditional_routing", "delayed_recall"}


def evaluate_reader_geometry(geometry: dict, q: np.ndarray, eval_seeds: list[int], partition_name: str) -> dict:
    family = geometry["family"]
    oid = geometry["organism_id"]
    Q = np.asarray(q, dtype=np.float64).reshape(40, -1)
    binary = family in BINARY
    rz = float(family_grid(family)["semantic_range"])
    phase_rows=[]; pooled_pred=[]; pooled_truth=[]; coverage_values=[]
    for phase in PHASES[family]:
        data=phase_dataset(oid,family,eval_seeds,Q,phase)
        if not len(data["semantic"]):
            phase_rows.append({"phase":phase,"n":0,"pass":False});continue
        chart=chart_for_phase(geometry,phase)
        pred=read_chart(chart,data["h"] if Q.shape[1]>1 else data["h"].reshape(-1))
        metrics=reader_metrics(pred,data["semantic"],rz,binary)
        coverage=1.0
        if Q.shape[1]==1 and not binary:
            lo,hi=sorted([float(chart.get("fit_z_min",-np.inf)),float(chart.get("fit_z_max",np.inf))])
            coverage=float(np.mean((data["semantic"]>=lo)&(data["semantic"]<=hi)))
        metrics["chart_coverage"]=coverage;metrics["coverage_pass"]=coverage>=.95
        metrics["pass"]=bool(metrics["pass"] and metrics["coverage_pass"])
        phase_rows.append({"phase":phase,**metrics});pooled_pred.extend(np.asarray(pred,dtype=np.float64).tolist());pooled_truth.extend(np.asarray(data["semantic"],dtype=np.float64).tolist());coverage_values.append(coverage)
    pooled=reader_metrics(np.asarray(pooled_pred),np.asarray(pooled_truth),rz,binary) if pooled_truth else {"pass":False,"n":0}
    pooled["chart_coverage"]=float(min(coverage_values)) if coverage_values else 0.0
    pooled["coverage_pass"]=bool(coverage_values and min(coverage_values)>=.95)
    passed=bool(pooled.get("pass") and pooled.get("coverage_pass") and phase_rows and all(r.get("pass",False) for r in phase_rows))
    return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry.get("engine"),"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"writer_family":geometry.get("writer_family"),"phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"pooled":pooled,"phases":phase_rows,"pass":passed,"failure_code":None if passed else "READER_GATE_FAIL"}
