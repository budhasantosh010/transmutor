from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837an.oracle_macrostate import instrument_episode
from experiments.v837_primitive_invention.v837an.authorization import PRIMARY_PROBES

from .episode_partitions import seeds
from .organism_folds import freeze_organism_folds
from .phase_backends import representative_phase_timesteps
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, read_json, sha256_json, write_json

RAW = HERE / "raw/canonical_target_grids.json"


def freeze_target_grids() -> dict:
    if RAW.is_file():
        return read_json(RAW)
    folds = freeze_organism_folds()
    out = {"version":"V837ao", "fit_partition":"AO_BACKEND_FIT", "families":{}}
    for family in POWERED_FAMILIES:
        if family in {"conditional_routing", "delayed_recall"}:
            out["families"][family] = {"semantic_range":2.0, "p05":-1.0, "p95":1.0, "targets":[-1.0,1.0], "target_definition":"EXACT_BINARY"}
            continue
        values = []
        # Pooling across discovery organisms is intentional; semantic episode values repeat across implementations.
        for _oid in folds["families"][family]["discovery"]:
            for seed in seeds("AO_BACKEND_FIT"):
                ep = instrument_episode(family, seed, "development")
                times = representative_phase_timesteps(ep).values()
                probe = ep.macrostate_traces[PRIMARY_PROBES[family]]
                values.extend(float(probe[t]) for t in times)
        a = np.asarray(values, dtype=np.float64)
        p05, p95 = np.quantile(a, [0.05,0.95])
        qs = np.quantile(a, [0.10,0.30,0.50,0.70,0.90])
        out["families"][family] = {"semantic_range":float(p95-p05), "p05":float(p05), "p95":float(p95), "targets":[float(x) for x in qs], "target_definition":"DISCOVERY_POOLED_Q10_Q30_Q50_Q70_Q90", "pooled_values":int(len(a))}
    out["grid_sha256"] = sha256_json(out)
    write_json(RAW, out)
    return out


def family_grid(family: str) -> dict:
    return freeze_target_grids()["families"][family]
