from __future__ import annotations


def initialize(ir, semantic_inputs=None):
    cls=ir["operator_class"];p=ir.get("parameters",{})
    if cls=="LINEAR_STATE_UPDATE_1D": return float((semantic_inputs or {}).get("z",p.get("initial_state",0.0)))
    if cls=="LINEAR_PHASE_MACHINE_R1": return float((semantic_inputs or {}).get("p",0.0))
    return None

def step(ir,state,semantic_input,phase=None):
    cls=ir["operator_class"];p=ir["parameters"]
    if cls=="LINEAR_STATE_UPDATE_1D": return p["a"]*float(state)+p["b"]*float(semantic_input["x"])+p.get("c",0.0)
    if cls=="LINEAR_PHASE_MACHINE_R1":
        ph=(phase or semantic_input.get("phase","")).upper()
        if ph=="WRITE": return p["w"]*float(semantic_input["value"])+p.get("bw",0.0)
        if ph=="HOLD": return p["lambda"]*float(state)+p.get("bh",0.0)
        if ph=="READ": return float(state)+p.get("br",p.get("b",0.0))
    if cls in {"BILINEAR_STATIC_MAP","SECOND_ORDER_STATIC_MAP"}:
        c=float(semantic_input["control"]);a=float(semantic_input["A"]);b=float(semantic_input["B"])
        y=p.get("b0",0.0)+p.get("bc",0.0)*c+p.get("bA",0.0)*a+p.get("bB",0.0)*b+p.get("bcA",0.0)*c*a+p.get("bcB",0.0)*c*b
        if cls=="SECOND_ORDER_STATIC_MAP": y+=p.get("bAB",0.0)*a*b+p.get("bAA",0.0)*a*a+p.get("bBB",0.0)*b*b
        return y
    raise ValueError("V837AR_UNSUPPORTED_STEP")

def run(ir,operator_word):
    cls=ir["operator_class"]
    if cls in {"BILINEAR_STATIC_MAP","SECOND_ORDER_STATIC_MAP"}: return step(ir,None,operator_word)
    state=initialize(ir,operator_word.get("initial",{}))
    out=None
    for op in operator_word.get("ops",[]):
        out=step(ir,state,op,op.get("phase"));
        if op.get("phase","").upper()!="READ": state=out
    return state if out is None else out

def predict_response(ir,base_word,intervention_word): return run(ir,intervention_word)-run(ir,base_word)
def compose(ir,op_a,op_b): return run(ir,{"initial":op_a.get("initial",{}),"ops":op_a.get("ops",[])+op_b.get("ops",[])})
