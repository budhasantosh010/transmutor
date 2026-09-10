from __future__ import annotations

from collections import defaultdict
import numpy as np

from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .canonical_reader import read_k1
from .heldout_backend_calibration import run_heldout_calibration
from .k1_backend import component_for_phase
from .phase_backends import representative_phase_timesteps
from .setpoint_grid import family_grid
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, read_json, write_json


def run_cross_organism_agreement() -> dict:
    held=read_json(HERE/'raw/heldout_calibration_frontier.json') if (HERE/'raw/heldout_calibration_frontier.json').is_file() else run_heldout_calibration();backend_payload=read_json(HERE/'raw/heldout_backend_results.json');results={};rows=[]
    # Agreement is only a gating claim for families with a successful heldout calibration budget.
    for family in POWERED_FAMILIES:
        info=held['families'][family];n=info.get('minimum_successful_n')
        if n is None:
            results[family]={'pass':False,'reason':'NO_SUCCESSFUL_HELDOUT_BACKEND','median_pairwise_disagreement':None,'p90_pairwise_disagreement':None};continue
        recs=[r for r in backend_payload.get('backend_records',[]) if r['family']==family and r['calibration_n']==n];by_seed=defaultdict(list);R=float(family_grid(family)['semantic_range'])
        for rec in recs:
            b=rec['backend'];data=pair_traces(rec['organism_id'],family,list(range(10448,10512)));states=data['base_trace'].states.detach().cpu().numpy().reshape(len(data['pairs']),data['base_trace'].states.shape[1],40)
            for i,p in enumerate(data['pairs']):
                for phase,t in representative_phase_timesteps(p.base_episode).items():
                    comp=component_for_phase(b,phase)
                    if comp.get('valid'):by_seed[(p.base_seed,phase)].append((rec['engine'],float(read_k1(states[i,t][None,:],comp['reader'])[0])))
        disagreements=[];engine_types=[]
        for key,vals in by_seed.items():
            for i in range(len(vals)):
                for j in range(i+1,len(vals)):
                    disagreements.append(abs(vals[i][1]-vals[j][1])/max(R,1e-12));engine_types.append(f"{vals[i][0]}__{vals[j][0]}")
        med=float(np.median(disagreements)) if disagreements else float('inf');p90=float(np.quantile(disagreements,.90)) if disagreements else float('inf');passed=med<=.10 and p90<=.25 and any('DIRECTED_STRUCTURAL_SEARCH' in x and 'RANDOM_STRUCTURAL_SAMPLER' in x for x in engine_types)
        results[family]={'pass':bool(passed),'median_pairwise_disagreement':med,'p90_pairwise_disagreement':p90,'pairs':len(disagreements)};rows.append({'family':family,'disagreements':disagreements,'engine_pair_types':engine_types})
    payload={'version':'V837ao','stage':'AO11_CROSS_ORGANISM_AGREEMENT','families':results,'rows':rows,'microstate_comparison_used':False,'q_alignment_used':False};write_json(HERE/'raw/cross_organism_agreement.json',payload);write_json(HERE/'diagnostics/cross_organism_agreement.json',payload);return payload


if __name__=='__main__':
    import json;print(json.dumps(run_cross_organism_agreement(),indent=2))
