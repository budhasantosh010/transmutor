from __future__ import annotations

PROGRAM_IR = {
    "conditional_routing": {
        "name": "SELECT", "inputs": ["control", "payload_a", "payload_b"],
        "causal_response_law": "control selects which payload has unit downstream influence; the unselected payload has near-zero influence",
        "temporal_contract": ["CONTROL", "PAYLOAD_A", "PAYLOAD_B", "SELECT"],
        "composition_rule": "SELECT(LOAD(A,B), CONTROL)",
    },
    "delayed_recall": {
        "name": "WRITE_HOLD_READ", "inputs": ["write_value", "distractors", "delay", "query"],
        "causal_response_law": "write value persists through delay; distractors and query nuisance do not alter the remembered semantic value",
        "temporal_contract": ["WRITE", "HOLD^n", "READ"],
        "composition_rule": "READ(HOLD^n(WRITE(v))) = v",
    },
    "iterative_state": {
        "name": "ACCUMULATE", "inputs": ["x_t"],
        "causal_response_law": "delta z_{t+h}=0.35*(0.65**(h-1))*delta x_t",
        "temporal_contract": ["ITERATE"],
        "composition_rule": "U(x2) o U(x1)",
    },
    "variable_composition": {
        "name": "APPLY_TRANSFORM", "inputs": ["initial_value", "gain", "drive"],
        "causal_response_law": "z_next=tanh(g*z+d)",
        "temporal_contract": ["APPLY(g,d)"] ,
        "composition_rule": "A(g2,d2) o A(g1,d1)",
    },
}


def program_ir(family: str) -> dict:
    return dict(PROGRAM_IR[family])
