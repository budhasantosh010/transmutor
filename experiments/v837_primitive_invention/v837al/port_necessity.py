from __future__ import annotations

from .pairwise_replay import SELECT, aggregate, evaluate_pair
from .heldout_pairwise import _fit_bundles
from .utils import HERE, read_json, write_json

PORTS=('state','external_messages','global_term','projected_input','gate','output')
QUALIFIERS={'state':'STATE_BASIS_REQUIRED','external_messages':'MESSAGE_PROTOCOL_REQUIRED','global_term':'GLOBAL_CONTEXT_PORT_REQUIRED','projected_input':'PRIVATE_INPUT_BASIS_REQUIRED','gate':'GATE_PROTOCOL_REQUIRED','output':'OUTPUT_PROTOCOL_REQUIRED'}


def _track(name, selected, pairs):
    if selected is None:return {'track':name,'run':False,'required_ports':[],'leave_one_out':[]}
    base_ports=list(selected['ports']);base_scope=selected['scope'];rows=[]
    target_pairs=[(i,p) for i,p in enumerate(pairs) if name=='GLOBAL_TRACK' or p.get('primary_causal')]
    base=[]
    for pi,p in target_pairs:base.append(evaluate_pair(p,_fit_bundles(pi,selected['family']),base_scope,SELECT,'development'))
    base_gate=aggregate(base,False);base_out=base_gate['median_output'];required=[]
    for port in base_ports:
        bits=list(base_scope);bits[PORTS.index(port)]='0';scope=''.join(bits);rr=[]
        for pi,p in target_pairs:rr.append(evaluate_pair(p,_fit_bundles(pi,selected['family']),scope,SELECT,'development'))
        gate=aggregate(rr,False);rel=((gate['median_output']-base_out)/max(base_out,1e-12)) if gate['median_output'] is not None and base_out is not None else None
        req=(rel is not None and rel>=0.15) or not gate['pass']
        if req:required.append(port)
        rows.append({'removed_port':port,'scope':scope,'metrics':gate,'relative_output_nrmse_worsening':rel,'required':req,'qualifier':QUALIFIERS[port] if req else None})
    return {'track':name,'run':True,'selected_config':selected,'base_metrics':base_gate,'required_ports':required,'required_qualifiers':[QUALIFIERS[p] for p in required],'leave_one_out':rows}


def run_port_necessity():
    sel=read_json(HERE/'raw/selected_interface_configs.json');pairs=read_json(HERE/'raw/frozen_pairs.json')['pairs']
    payload={'version':'V837al','stage':'AL6_PORT_NECESSITY','data':'ALIGN_SELECT_ONLY','global_track':_track('GLOBAL_TRACK',sel.get('global_track'),pairs),'causal_track':_track('CAUSAL_TRACK',sel.get('causal_track'),pairs)}
    write_json(HERE/'diagnostics/port_necessity.json',payload);return payload

if __name__=='__main__':run_port_necessity()
