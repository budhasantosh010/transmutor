from __future__ import annotations

from .utils import HERE,read_json,write_json
from .iterative_ir import evaluate_role as iterative_eval
from .routing_ir import evaluate_role as routing_eval
from .memory_ir import evaluate_role as memory_eval
from .controls import evaluate_controls


def _pass(ev,controls):
    if ev.get("undefined_unseen"):return False
    a=ev["organism_relative"];o=ev["oracle_relative"]
    base=bool(a["response_nrmse"]<=.10 and o["response_nrmse"]<=.10 and o["pearson"]>=.85 and o["direction_agreement"]>=.85 and o["perturbed_task_success"]>=.80)
    margin_ok=True
    invariance_ok=True
    for name,c in controls.items():
        if c.get("valid") and c.get("use_for_margin") and "nrmse" in c:
            margin_ok=margin_ok and (c["nrmse"]-a["response_nrmse"]>=.05)
        if c.get("valid") and c.get("control_kind")=="positive_invariance":
            invariance_ok=invariance_ok and bool(c.get("pass"))
    return bool(base and margin_ok and invariance_ok)
def run_meta():
    winners=read_json(HERE/"raw/discovery_program_ir_winners.json");out={"version":"V837ar","no_refit":True,"no_fallback":True,"families":{}}
    for fam,w in winners["families"].items():
        if not w or not w.get("ir"):out["families"][fam]={"pass":False,"candidate":None,"reason":"NO_DISCOVERY_CANDIDATE"};continue
        g=w["grammar"]
        if fam=="iterative_state":
            fits=read_json(HERE/"raw/iterative_ir_fits.json")["fits"];ev=iterative_eval(g,fits[g]["coefficients"],"META_WORDS")
        elif fam=="conditional_routing":
            fits=read_json(HERE/"raw/routing_ir_fits.json")["fits"];ev=routing_eval(g,fits[g]["coefficients"],"META_WORDS")
        else:
            fits=read_json(HERE/"raw/memory_ir_fits.json")["fits"];ev=memory_eval(g,fits[g],"META_WORDS")
        controls=evaluate_controls(fam,w["ir"],"META_WORDS");passed=_pass(ev,controls);out["families"][fam]={"pass":passed,"candidate":w,"metrics":ev,"controls":controls}
    write_json(HERE/"raw/meta_confirmation.json",out);write_json(HERE/"diagnostics/meta_confirmation.json",out);return out
