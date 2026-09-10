from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.v837an.authorization import PRIMARY_PROBES
from experiments.v837_primitive_invention.v837an.oracle_macrostate import instrument_episode
from experiments.v837_primitive_invention.v837ao.phase_backends import representative_phase_timesteps

from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .source_folds import freeze_source_folds
from .utils import HERE, read_json, sha256_json, write_json

RAW=HERE/"raw/canonical_target_grids.json"

def freeze_target_grids()->dict:
    if RAW.is_file():
        out=read_json(RAW);write_json(HERE/"diagnostics/setpoint_grid.json",out);return out
    folds=freeze_source_folds();out={"version":"V837ap","fit_partition":"AP_CHART_FIT","families":{}}
    for family in POWERED_FAMILIES:
        if family in {"conditional_routing","delayed_recall"}:
            out["families"][family]={"semantic_range":2.0,"q05":-1.0,"q95":1.0,"targets":[-1.0,1.0],"target_definition":"EXACT_BINARY"};continue
        values=[]
        for _oid in folds["families"][family]["discovery"]:
            for seed in seeds("AP_CHART_FIT"):
                ep=instrument_episode(family,seed,"development");probe=ep.macrostate_traces[PRIMARY_PROBES[family]]
                for t in representative_phase_timesteps(ep).values(): values.append(float(probe[t]))
        a=np.asarray(values,dtype=np.float64);q05,q95=np.quantile(a,[.05,.95]);targets=np.quantile(a,[.10,.25,.40,.50,.60,.75,.90])
        out["families"][family]={"semantic_range":float(q95-q05),"q05":float(q05),"q95":float(q95),"targets":[float(x) for x in targets],"target_definition":"DISCOVERY_POOLED_Q10_Q25_Q40_Q50_Q60_Q75_Q90","pooled_values":int(len(a))}
    out["grid_sha256"]=sha256_json(out);write_json(RAW,out);write_json(HERE/"diagnostics/setpoint_grid.json",out);return out

def family_grid(family:str)->dict:return freeze_target_grids()["families"][family]
