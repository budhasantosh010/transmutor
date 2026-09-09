from __future__ import annotations
import json
from .branch_a_context import configs as a_configs
from .branch_b_boundary import configs as b_configs,_dummy
from .branch_c_temporal_interaction import configs as c_configs
from .context_replay import evaluate_pair,aggregate
from .expanded_primitive import expanded_pair,ranking_index
from .utils import HERE,read_json

def _cfg(branch,config_id):
    pool={"AM-A":a_configs(),"AM-B":b_configs(),"AM-C":c_configs()}[branch];return next(c for c in pool if c["config_id"]==config_id)
def _load(branch,pi,cfg,pair):
    if branch=="AM-A":path=HERE/"raw/cache/am_a/fits"/f"p{pi:02d}__{cfg['family']}__{cfg['context_set']}.json"
    elif branch=="AM-B":
        if cfg["adapter"]=="IDENTITY":return _dummy(pair)
        path=HERE/"raw/cache/am_b/fits"/f"{cfg['boundary'].replace('+','p')}__p{pi:02d}.json"
    else:path=HERE/"raw/cache/am_c/fits"/f"{cfg['config_id'].replace('+','p')}__p{pi:02d}.json"
    if not path.is_file():raise RuntimeError(f"V837AM_NO_REFIT_CACHE_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))
def evaluate_candidate(branch,winner,seeds_used,p_threshold=None):
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];cfg=_cfg(branch,winner["config_id"]);idx=ranking_index() if branch in {"AM-B","AM-C"} else None;rows=[]
    for pi,p0 in enumerate(pairs):
        if branch=="AM-B":p=expanded_pair(p0,cfg["boundary"],idx);ports=set() if cfg["adapter"]=="IDENTITY" else {"state","external_messages","global_term","projected_input","gate","output"};ctx="NONE";hist=1
        elif branch=="AM-C":p=p0 if cfg["boundary"]=="ORIGINAL" else expanded_pair(p0,cfg["boundary"],idx);ports={"state","external_messages","global_term","projected_input","gate","output"};ctx="C5_FULL_LOCAL_CONTEXT";hist=cfg["history"]
        else:p=p0;ports=set(cfg["ports"]);ctx=cfg["context_set"];hist=cfg["history"]
        fit=_load(branch,pi,cfg,p);rows.append(evaluate_pair(p,fit,ports,ctx,hist,seeds_used))
    return {"branch":branch,"config_id":cfg["config_id"],"rows":rows,"aggregate":aggregate(rows,p_threshold),"refit":False}
