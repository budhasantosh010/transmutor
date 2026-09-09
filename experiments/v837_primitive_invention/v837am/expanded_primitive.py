from __future__ import annotations
from .utils import HERE,read_json

def ranking_index():return {r["occurrence_id"]:r for r in read_json(HERE/"raw/frozen_context_cell_rankings.json")["rows"]}
def expand_occurrence(occ,boundary,idx=None):
    idx=idx or ranking_index();base=set(map(int,occ["nodes"]));o={**occ}
    if boundary=="WHOLE_SYSTEM":nodes=list(range(10))
    else:
        typ,n=boundary.split("+");n=int(n);key={"MESSAGE":"message_ranking","GLOBAL":"global_ranking","COMBINED":"combined_ranking"}[typ];nodes=sorted(base|set(idx[occ["occurrence_id"]][key][:n]))
    o["nodes"]=nodes;o["size"]=len(nodes);o["added_cells"]=len(nodes)-len(base);o["boundary_definition"]=boundary;return o
def expanded_pair(pair,boundary,idx=None):
    idx=idx or ranking_index();return {**pair,"recipient":expand_occurrence(pair["recipient"],boundary,idx),"same_class_donor":expand_occurrence(pair["same_class_donor"],boundary,idx),"different_class_donor":expand_occurrence(pair["different_class_donor"],boundary,idx)}
