from __future__ import annotations

from dataclasses import dataclass, asdict
from .utils import canonical_json, sha256_json

ALLOWED_CLASSES={"LINEAR_STATE_UPDATE_1D","BILINEAR_STATIC_MAP","SECOND_ORDER_STATIC_MAP","LINEAR_PHASE_MACHINE_R1","EMPIRICAL_RESPONSE_TABLE"}
FORBIDDEN_KEYS={"q","q_vector","state40","cell_ids","message_edge_ids","checkpoint_parameters","checkpoint","state_dict"}

@dataclass(frozen=True)
class ProgramIR:
    version:str="V837ar"; ir_schema_version:int=1; family:str=""; operator_name:str=""; operator_class:str=""; granularity:str=""
    semantic_inputs:tuple=(); semantic_outputs:tuple=(); predictive_state:dict|None=None; parameters:dict|None=None; phase_contract:tuple=(); composition_contract:dict|None=None; validity_domain:dict|None=None; response_basis_hash:str=""; fit_provenance:dict|None=None; complexity:dict|None=None; evidence:dict|None=None
    def to_dict(self):
        d=asdict(self); d["semantic_inputs"]=list(self.semantic_inputs);d["semantic_outputs"]=list(self.semantic_outputs);d["phase_contract"]=list(self.phase_contract)
        validate_ir(d);return d

def _walk(x):
    if isinstance(x,dict):
        for k,v in x.items(): yield str(k);yield from _walk(v)
    elif isinstance(x,(list,tuple)):
        for v in x: yield from _walk(v)

def validate_ir(ir:dict):
    if ir.get("operator_class") not in ALLOWED_CLASSES: raise ValueError("V837AR_BAD_OPERATOR_CLASS")
    if FORBIDDEN_KEYS & {k.lower() for k in _walk(ir)}: raise ValueError("V837AR_NEURAL_STATE_IN_PROGRAM_IR")
    if ir.get("operator_class")=="EMPIRICAL_RESPONSE_TABLE" and ir.get("evidence",{}).get("archiveable",False): raise ValueError("V837AR_REFERENCE_TABLE_ARCHIVEABLE")
    return True

def canonical_bytes(ir:dict)->bytes: validate_ir(ir);return canonical_json(ir)
def ir_sha256(ir:dict)->str:return sha256_json(ir)
