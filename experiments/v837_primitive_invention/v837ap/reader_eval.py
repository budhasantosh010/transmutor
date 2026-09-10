from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES
from .chart_data import phase_dataset
from .chart_reader import read_chart, reader_metrics
from .geometry_runtime import chart_for_phase
from .setpoint_grid import family_grid

BINARY={"conditional_routing","delayed_recall"}


def evaluate_reader(geometry:dict,q:np.ndarray,eval_seeds:list[int],partition_name:str)->dict:
    family=geometry["family"];Q=np.asarray(q,dtype=np.float64).reshape(40,-1);grid=family_grid(family);binary=family in BINARY;rows=[];all_pred=[];all_truth=[]
    for phase in PHASES[family]:
        ch=phase_dataset(geometry["organism_id"],family,eval_seeds,Q,phase)
        if not len(ch["semantic"]):
            rows.append({"phase":phase,"n":0,"pass":False});continue
        chart=chart_for_phase(geometry,phase);h=ch["h"] if Q.shape[1]>1 else ch["h"].reshape(-1);pred=read_chart(chart,h);m=reader_metrics(pred,ch["semantic"],grid["semantic_range"],binary);rows.append({"phase":phase,**m});all_pred.append(np.asarray(pred));all_truth.append(np.asarray(ch["semantic"]))
    if all_pred:
        pooled=reader_metrics(np.concatenate(all_pred),np.concatenate(all_truth),grid["semantic_range"],binary)
    else:pooled={"pass":False,"n":0}
    # Discovery reader contract is pooled across frozen semantic-active phases; phase rows are diagnostics.
    passed=bool(pooled.get("pass"))
    return {"version":"V837ap","organism_id":geometry["organism_id"],"family":family,"engine":geometry["engine"],"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"writer_family":geometry.get("writer_family","AUTO"),"phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"pooled":pooled,"phases":rows,"pass":passed}
