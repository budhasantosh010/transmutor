from __future__ import annotations

import math
from collections import defaultdict

from .commutativity import evaluate_dynamics, evaluate_natural_commutativity
from .episode_partitions import seeds
from .failure_ledger import add, make_entry
from .k1_backend import evaluate_reader_on_partition, fit_backend
from .organism_folds import freeze_organism_folds
from .phase_backends import BACKEND_ORDER, PHASES
from .quotient_eval import evaluate_quotient
from .quotient_pairs import select_quotient_pairs
from .residual_projector import projector_diagnostics
from .setpoint_eval import evaluate_setpoints
from .source_contracts import POWERED_FAMILIES
from .utils import HERE, sha256_json, write_json


def _family_gate(rows:list[dict], field:str)->dict:
    passing=[r for r in rows if bool(r.get(field))];n=len(rows);required=max(1,math.ceil(.60*n));engines=sorted({r["engine"] for r in passing});return {"organisms":n,"passing":len(passing),"required":required,"pass_fraction":0.0 if n==0 else len(passing)/n,"pass_engines":engines,"pass":bool(n>=1 and len(passing)>=required and len(engines)>=2)}


def _log(code:str,stage:str,row:dict,*,phase=None,setpoint=None,failed=None,metrics=None,status="DEFINITIVE_WITHIN_FROZEN_SCOPE"):
    family=row.get("family");oid=row.get("organism_id");variant=row.get("variant");tag=sha256_json({"code":code,"stage":stage,"family":family,"oid":oid,"variant":variant,"phase":phase,"setpoint":setpoint})[:12]
    add(make_entry(failure_id=f"V837ao-{stage}-{tag}",stage=stage,failure_code=code,failure_type="SCIENTIFIC_FAILURE",result_status=status,family=family,organism_id=oid,backend_variant=variant,phase=phase,setpoint=setpoint,calibration_budget=None,hypothesis="The frozen V837ao canonical backend realizes a substrate-independent scalar causal state.",why="Resolve global/phase-local coordinate, quotient, and dynamics gates without cross-organism microstate alignment.",configuration={"variant":variant,"phase":phase,"setpoint":setpoint},data={"backend_fit":"AO_BACKEND_FIT","backend_select":"AO_BACKEND_SELECT","quotient":"AO_QUOTIENT_FIT/SELECT","dynamics":"AO_DYNAMICS_FIT/SELECT"},metrics=metrics or row,gate="V837ao frozen gate",failed=failed or [code],distance="see measured metrics",confounds_ruled_out=["cross-organism state alignment","model retraining","backend gradients","performance-selected organism folds"],uncertainty="More complex frozen backend may still pass unless this is B2/full-family closure.",next_experiment="Escalate only according to B0<B1<B2; otherwise preserve null family candidate.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837ao --stage backends --execute",artifacts=["experiments/v837_primitive_invention/v837ao/raw/discovery_family_backend_winners.json"],artifact_hashes={"metrics_sha256":sha256_json(metrics or row)}))


def run_discovery_backends()->dict:
    folds=freeze_organism_folds();all_fits=[];reader_rows=[];set_rows=[];phase_rows=[];quotient_fit_rows=[];quotient_rows=[];dyn_rows=[];winners={};evaluated={}
    for family in POWERED_FAMILIES:
        ids=folds["families"][family]["discovery"];evaluated[family]=[];winner=None
        for variant in BACKEND_ORDER:
            evaluated[family].append(variant);candidate_rows=[]
            for index,oid in enumerate(ids,1):
                backend=fit_backend(oid,family,variant,seeds("AO_BACKEND_FIT"));all_fits.append(backend)
                if not backend.get("valid"):
                    rr={"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant,"reader_pass":False,"algebra_pass":False};sp={"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant,"pass":False,"sample_count":0,"failure_code":backend.get("failure_code","K1_READER_NOT_GLOBAL")}
                else:
                    rr=evaluate_reader_on_partition(backend,seeds("AO_BACKEND_SELECT"));sp=evaluate_setpoints(backend,seeds("AO_BACKEND_SELECT"))
                reader_rows.append(rr);set_rows.append(sp)
                prelim=bool(backend.get("valid") and rr.get("reader_pass") and rr.get("algebra_pass") and sp.get("pass"));candidate_rows.append({"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant,"preliminary_pass":prelim,"backend":backend,"reader":rr,"setpoint":sp})
                for ph in rr.get("phases",[]):
                    phase_rows.append({"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant,**ph})
                    if not ph.get("pass"):_log("K1_READER_NOT_GLOBAL" if variant=="B0_GLOBAL_K1" else ("PHASE_GAUGE_FAIL" if variant=="B1_PHASE_GAUGE_K1" else "PHASE_K1_FAIL"),"AO4_READER",{"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant},phase=ph.get("phase"),failed=["reader/algebra phase gate"],metrics=ph)
                if not sp.get("pass"):
                    _log("ABSOLUTE_SETPOINT_FAIL","AO4_SETPOINT",{"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant},failed=[sp.get("failure_code")],metrics=sp.get("metrics",sp))
                    for ti,tr in enumerate(sp.get("target_results",[])):
                        if not tr.get("pass_recovery_060"):_log("ABSOLUTE_SETPOINT_FAIL","AO4_SETPOINT_TARGET",{"organism_id":oid,"family":family,"engine":backend.get("engine"),"variant":variant},setpoint=tr.get("target"),failed=["target median recovery <0.60"],metrics=tr)
                print(f"V837ao discovery {family} {variant} {index}/{len(ids)} prelim={prelim}",flush=True)
            prelim_gate=_family_gate(candidate_rows,"preliminary_pass")
            if not prelim_gate["pass"]:
                code="GLOBAL_BACKEND_FAIL" if variant=="B0_GLOBAL_K1" else ("PHASE_GAUGE_FAIL" if variant=="B1_PHASE_GAUGE_K1" else "PHASE_K1_FAIL")
                _log(code,"AO5_FAMILY_PRELIM",{"organism_id":None,"family":family,"engine":None,"variant":variant},failed=["<60% preliminary pass or both engines absent"],metrics=prelim_gate)
                continue
            # Only variants clearing reader+SET+controls earn quotient/dynamics compute.
            full_rows=[]
            for row in candidate_rows:
                if not row["preliminary_pass"]:
                    full_rows.append({**row,"full_pass":False});continue
                backend=row["backend"];qfit=select_quotient_pairs(backend,seeds("AO_QUOTIENT_FIT"),"AO_QUOTIENT_FIT");quotient_fit_rows.append(qfit);qres=evaluate_quotient(backend,seeds("AO_QUOTIENT_SELECT"));quotient_rows.append(qres);dfit=evaluate_natural_commutativity(backend,seeds("AO_DYNAMICS_FIT"),"AO_DYNAMICS_FIT");dsel=evaluate_dynamics(backend,seeds("AO_DYNAMICS_SELECT"));dyn_rows.append({"organism_id":row["organism_id"],"family":family,"engine":row["engine"],"variant":variant,"fit_natural":dfit,"select":dsel})
                full=bool(qres.get("pass") and dsel.get("pass"));full_rows.append({**row,"quotient":qres,"dynamics":dsel,"full_pass":full})
                if not qres.get("pass"):_log(qres.get("failure_code") or "QUOTIENT_RESIDUAL_SENSITIVITY","AO6_QUOTIENT",row,failed=[qres.get("failure_code")],metrics=qres.get("metrics",qres))
                if not dsel.get("pass"):_log(dsel.get("failure_code") or "CANONICAL_DYNAMICS_ROLLOUT_FAIL","AO7_DYNAMICS",row,failed=[dsel.get("failure_code")],metrics={"natural":dsel.get("natural",{}).get("metrics"),"interventional":dsel.get("interventional",{}).get("metrics")})
                print(f"V837ao full-gate {family} {variant} {row['organism_id'][:8]} quotient={qres.get('pass')} dynamics={dsel.get('pass')} full={full}",flush=True)
            full_gate=_family_gate(full_rows,"full_pass")
            if full_gate["pass"]:
                winner={"family":family,"variant":variant,"discovery_gate":full_gate,"support_organisms":[r["organism_id"] for r in full_rows if r.get("full_pass")],"support_engines":full_gate["pass_engines"]};break
            _log("GLOBAL_BACKEND_FAIL" if variant=="B0_GLOBAL_K1" else ("PHASE_GAUGE_FAIL" if variant=="B1_PHASE_GAUGE_K1" else "PHASE_K1_FAIL"),"AO7_FAMILY_FULL",{"organism_id":None,"family":family,"engine":None,"variant":variant},failed=["full canonicalization gate failed"],metrics=full_gate)
        winners[family]=winner
        print(f"V837ao discovery family {family} winner={None if winner is None else winner['variant']}",flush=True)
    payload={"version":"V837ao","stage":"AO3-AO7_DISCOVERY","evaluated_variants":evaluated,"family_winners":winners,"discovery_family_candidates":sum(v is not None for v in winners.values())}
    write_json(HERE/"raw/discovery_backend_fits.json",{"version":"V837ao","rows":all_fits,"gradient_steps":0,"heldout_backend_artifacts_read":False})
    write_json(HERE/"raw/discovery_reader_results.json",{"version":"V837ao","rows":reader_rows})
    write_json(HERE/"raw/discovery_setpoint_results.json",{"version":"V837ao","rows":set_rows})
    write_json(HERE/"raw/phase_backend_results.json",{"version":"V837ao","rows":phase_rows,"evaluated_variants":evaluated})
    write_json(HERE/"raw/quotient_pairs.json",{"version":"V837ao","fit_pair_sets":quotient_fit_rows})
    write_json(HERE/"raw/quotient_results.json",{"version":"V837ao","rows":quotient_rows})
    write_json(HERE/"raw/commutativity_results.json",{"version":"V837ao","rows":dyn_rows})
    write_json(HERE/"raw/discovery_family_backend_winners.json",payload)
    write_json(HERE/"diagnostics/reader_fit.json",{"version":"V837ao","rows":reader_rows})
    write_json(HERE/"diagnostics/reader_writer_gain.json",{"version":"V837ao","rows":[{"organism_id":b["organism_id"],"family":b["family"],"variant":b["variant"],"components":{k:{"gamma":v.get("gauge",{}).get("gamma"),"unit_gain":v.get("gauge",{}).get("unit_gain"),"valid":v.get("valid")} for k,v in b.get("components",{}).items()}} for b in all_fits]})
    write_json(HERE/"diagnostics/setter_algebra.json",{"version":"V837ao","rows":[{"organism_id":b["organism_id"],"family":b["family"],"variant":b["variant"],"components":{k:v.get("algebra") for k,v in b.get("components",{}).items()}} for b in all_fits]})
    write_json(HERE/"diagnostics/setpoint_generalization.json",{"version":"V837ao","rows":set_rows})
    write_json(HERE/"diagnostics/random_controls.json",{"version":"V837ao","count":32,"rows":[{"organism_id":r["organism_id"],"family":r["family"],"variant":r["variant"],"random_median_recovery":r.get("metrics",{}).get("random_median_recovery"),"shuffled_median_recovery":r.get("metrics",{}).get("shuffled_median_recovery"),"control_margin":r.get("metrics",{}).get("control_margin")} for r in set_rows]})
    write_json(HERE/"diagnostics/phase_realization.json",{"version":"V837ao","rows":phase_rows,"winners":winners})
    write_json(HERE/"diagnostics/residual_projector.json",{"version":"V837ao","rows":[{"organism_id":b["organism_id"],"family":b["family"],"variant":b["variant"],"components":{k:(projector_diagnostics(v) if v.get("valid") else {"pass":False}) for k,v in b.get("components",{}).items()}} for b in all_fits]})
    write_json(HERE/"diagnostics/quotient_sufficiency.json",{"version":"V837ao","rows":quotient_rows})
    write_json(HERE/"diagnostics/commutativity_natural.json",{"version":"V837ao","rows":[r["fit_natural"] for r in dyn_rows]})
    write_json(HERE/"diagnostics/commutativity_interventional.json",{"version":"V837ao","rows":[r["select"]["interventional"] for r in dyn_rows]})
    return payload
