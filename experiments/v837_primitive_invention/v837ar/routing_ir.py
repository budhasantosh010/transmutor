from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.v837aq.operator_metrics import response_metrics
from .shared_fit import ridge_fit, per_organism_fit
from .semantic_execution import discovery_ids, engine_for, predict_episodes
from .operator_words import ROLE_SEEDS, routing_word_cases
from .program_ir import ProgramIR
from .utils import HERE, read_json, write_json

ORDER = ("R0_FINE", "R1_TWO_STAGE", "R2_FIRST_ORDER", "R2_BILINEAR", "R2_FULL_SECOND_ORDER")
ATOMIC_MODEL_LABELS = {
    "R2_FIRST_ORDER": "R2-L1_FIRST_ORDER_AFFINE",
    "R2_BILINEAR": "R2-L2_BILINEAR_ROUTE",
    "R2_FULL_SECOND_ORDER": "R2-L3_FULL_SECOND_ORDER",
}


def _X(kind, c, a, b):
    c = np.asarray(c, float); a = np.asarray(a, float); b = np.asarray(b, float)
    if kind == "R1_TWO_STAGE": return np.c_[np.ones(len(c)), c, a + b, c * (a - b)]
    if kind == "R2_FIRST_ORDER": return np.c_[np.ones(len(c)), c, a, b]
    if kind == "R2_BILINEAR": return np.c_[np.ones(len(c)), c, a, b, c * a, c * b]
    if kind == "R2_FULL_SECOND_ORDER": return np.c_[np.ones(len(c)), c, a, b, c * a, c * b, a * b, a * a, b * b]
    raise KeyError(kind)


def _rows(role):
    cs=[]; aa=[]; bb=[]; ys=[]; ids=[]; meta=[]
    for oid in discovery_ids("conditional_routing"):
        words=[w for s in ROLE_SEEDS[role] for w in routing_word_cases(s, role)]
        pred=predict_episodes(oid,[w["episode"] for w in words])
        for w,y in zip(words,pred):
            cs.append(w["control"]); aa.append(w["A"]); bb.append(w["B"]); ys.append(float(y)); ids.append(oid); meta.append((oid,w))
    return np.asarray(cs),np.asarray(aa),np.asarray(bb),np.asarray(ys),np.asarray(ids),meta


def fit_candidates():
    c,a,b,y,ids,_=_rows("FIT_WORDS")
    fits={"R0_FINE":{"pass_anchor":False,"source":"V837aq discovery composition 2/5 FAIL","parameter_count":0}}
    for grammar in ORDER[1:]:
        X=_X(grammar,c,a,b); coef=ridge_fit(X,y,ids); pred=X@coef
        fits[grammar]={
            "coefficients":coef.tolist(),
            "fit_nrmse":float(np.sqrt(np.mean((pred-y)**2))/2.0),
            "parameter_count":len(coef),
            "per_organism":per_organism_fit(X,y,ids),
            "granularity":"TWO_STAGE" if grammar=="R1_TWO_STAGE" else "ATOMIC",
            "atomic_model":ATOMIC_MODEL_LABELS.get(grammar),
        }
    write_json(HERE/"raw/routing_ir_fits.json",{
        "version":"V837ar","fits":fits,"third_order_used":False,"fit_rows":len(y),
        "granularity_ladder":["R0_FINE","R1_TWO_STAGE","R2_ATOMIC"],
        "atomic_model_ladder":["R2-L1_FIRST_ORDER_AFFINE","R2-L2_BILINEAR_ROUTE","R2-L3_FULL_SECOND_ORDER"],
        "R2_L3_rule":"evaluate only if R2-L2 fails",
    })
    return fits


def _expanded_params(grammar,coef):
    q=list(map(float,coef))
    if grammar=="R1_TWO_STAGE":
        b0,bc,s,d=q; return {"b0":b0,"bc":bc,"bA":s,"bB":s,"bcA":d,"bcB":-d}
    names={
        "R2_FIRST_ORDER":["b0","bc","bA","bB"],
        "R2_BILINEAR":["b0","bc","bA","bB","bcA","bcB"],
        "R2_FULL_SECOND_ORDER":["b0","bc","bA","bB","bcA","bcB","bAB","bAA","bBB"],
    }[grammar]
    return dict(zip(names,q))


def _predict(grammar,coef,w):
    return float(_X(grammar,[w["control"]],[w["A"]],[w["B"]])[0]@np.asarray(coef))


def evaluate_role(grammar,coef,role):
    source=[]; oracle=[]; pred=[]; success=[]; rows=[]
    for oid in discovery_ids("conditional_routing"):
        words=[w for s in ROLE_SEEDS[role] for w in routing_word_cases(s, role)]
        actual=predict_episodes(oid,[w["episode"] for w in words])
        for w,a in zip(words,actual):
            p=_predict(grammar,coef,w); source.append(float(a)); oracle.append(w["oracle"]); pred.append(p); success.append(abs(p-w["oracle"])<=.25)
            rows.append({"organism_id":oid,"engine":engine_for("conditional_routing",oid),"word":w["name"],"source":float(a),"oracle":w["oracle"],"ir":p})
    return {"role":role,"organism_relative":response_metrics(source,pred,success),"oracle_relative":response_metrics(oracle,pred,success),"rows":rows}


def _passes(ev):
    m=ev["organism_relative"]; o=ev["oracle_relative"]
    return bool(m["response_nrmse"]<=.10 and o["response_nrmse"]<=.10 and o["pearson"]>=.85 and o["direction_agreement"]>=.85 and o["perturbed_task_success"]>=.80)


def select_routing():
    fits=fit_candidates(); basis=read_json(HERE/"raw/canonical_response_basis.json")["response_basis_sha256"]
    rows=[{"grammar":"R0_FINE","granularity":"FINE_CONTROL_LOAD_SELECT","pass":False,"historical_anchor":"V837aq discovery composition 2/5 FAIL"}]
    winner=None; evaluated={}

    # R1 is the first newly fitted granularity.
    ev=evaluate_role("R1_TWO_STAGE",fits["R1_TWO_STAGE"]["coefficients"],"SELECT_WORDS"); evaluated["R1_TWO_STAGE"]=ev
    r1_pass=_passes(ev); rows.append({"grammar":"R1_TWO_STAGE","granularity":"TWO_STAGE","pass":r1_pass,"metrics":ev})
    if r1_pass:
        winner="R1_TWO_STAGE"
    else:
        # R2 is atomic. Test the frozen low-order model ladder inside R2.
        for grammar in ("R2_FIRST_ORDER","R2_BILINEAR"):
            ev=evaluate_role(grammar,fits[grammar]["coefficients"],"SELECT_WORDS"); evaluated[grammar]=ev
            passed=_passes(ev); rows.append({"grammar":grammar,"granularity":"ATOMIC","atomic_model":ATOMIC_MODEL_LABELS[grammar],"pass":passed,"metrics":ev})
            if passed:
                winner=grammar; break
        if winner is None:
            grammar="R2_FULL_SECOND_ORDER"
            ev=evaluate_role(grammar,fits[grammar]["coefficients"],"SELECT_WORDS"); evaluated[grammar]=ev
            passed=_passes(ev); rows.append({"grammar":grammar,"granularity":"ATOMIC","atomic_model":ATOMIC_MODEL_LABELS[grammar],"pass":passed,"metrics":ev,"evaluated_because":"R2-L2_FAILED"})
            if passed: winner=grammar

    ir=None; selected_granularity=None; selected_model=None
    if winner:
        selected_granularity="TWO_STAGE" if winner=="R1_TWO_STAGE" else "ATOMIC"
        selected_model=ATOMIC_MODEL_LABELS.get(winner,"R1_TWO_STAGE")
        cls="SECOND_ORDER_STATIC_MAP" if winner=="R2_FULL_SECOND_ORDER" else "BILINEAR_STATIC_MAP"
        params=_expanded_params(winner,fits[winner]["coefficients"])
        ir=ProgramIR(
            family="conditional_routing",operator_name="ROUTE",operator_class=cls,granularity=selected_granularity,
            semantic_inputs=("control","A","B"),semantic_outputs=("y",),predictive_state={"dimension":0},parameters=params,
            phase_contract=("ROUTE",),composition_contract={"C3_fine_anchor":False,"minimum_closed_granularity":selected_granularity},
            validity_domain={"control":[-1,1],"payload":[-.85,.85]},response_basis_hash=basis,
            fit_provenance={"fold":"AQ_DISCOVERY","word_partition":"FIT_WORDS","equal_organism_weighting":True,"granularity_ladder":["R0_FINE","R1_TWO_STAGE","R2_ATOMIC"]},
            complexity={"continuous_parameters":fits[winner]["parameter_count"]},evidence={"archiveable":fits[winner]["parameter_count"]<=256},
        ).to_dict()

    l1=evaluated.get("R2_FIRST_ORDER",{}).get("organism_relative",{}).get("response_nrmse")
    l2=evaluated.get("R2_BILINEAR",{}).get("organism_relative",{}).get("response_nrmse")
    diagnosis=None
    if winner=="R2_BILINEAR" and not r1_pass:
        diagnosis="ROUTING_PRIMITIVE_GRANULARITY_TOO_FINE_IN_V837AQ"
    elif winner and winner.startswith("R2_"):
        diagnosis="ROUTING_ATOMIC_PRIMITIVE_REQUIRED"
    elif winner=="R1_TWO_STAGE":
        diagnosis="ROUTING_TWO_STAGE_PRIMITIVE_SUFFICIENT"
    elif winner is None:
        diagnosis="ROUTING_NO_COMPACT_PROGRAM_IR"

    payload={
        "version":"V837ar","candidates":rows,"selected_grammar":winner,"selected_granularity":selected_granularity,"selected_model":selected_model,
        "selected_ir":ir,"pass":winner is not None,"diagnosis":diagnosis,"R0_FINE_pass":False,"R1_TWO_STAGE_pass":r1_pass,
        "R2_L3_evaluated":"R2_FULL_SECOND_ORDER" in evaluated,"first_order_nrmse":l1,"bilinear_nrmse":l2,
        "bilinear_material_improvement":bool(l1 is not None and l2 is not None and l1-l2>=.03),"third_order_used":False,
    }
    write_json(HERE/"diagnostics/routing_granularity.json",payload); write_json(HERE/"diagnostics/routing_interaction_order.json",payload)
    return payload
