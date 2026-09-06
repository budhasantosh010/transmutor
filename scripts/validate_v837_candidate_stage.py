from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v837_primitive_invention"
HERE = BASE / "v837z"


def fail(msg): raise ValueError(msg)

def blob(path): return hashlib.sha256(subprocess.check_output(["git","show","HEAD:"+path], cwd=ROOT)).hexdigest()


def main():
    if not HERE.exists(): return 0
    c = json.loads((HERE / "config.json").read_text())
    ydec = json.loads((BASE / "v837y/diagnostics/decision_state.json").read_text())
    if c.get("parent") != "V837y": fail("parent mismatch")
    if not (ydec.get("v837y_complete") is True and ydec.get("representation_adequacy_pass") is False and ydec.get("v837z_allowed") is True): fail("invalid V837y authorization")
    if ydec.get("selected_v837z_parent") != c.get("selected_parent"): fail("selected parent mismatch")
    if c.get("selected_parent") != "Y3_global_control_rank4_candidate": fail("unexpected parent for this closed decision")
    if set(c.get("conditions",{})) != {"Z0_historical_candidate_stage","Z1_synchronous_candidate_stage"}: fail("condition set drift")
    fixed=c["fixed_parent_mechanisms"]
    if fixed != {"global_scalar_control":True,"global_controller_mode":"JOINT_INPUT_STATE_GLOBAL_SCALAR","candidate_coupling":"rank4_cross_block","candidate_coupling_rank":4,"candidate_coupling_scaling":1.0,"input_projection":"historical_per_cell","graph_edges_unchanged":True}: fail("frozen parent mechanisms drift")
    tr=c["training"]
    if (tr["steps"],tr["train_episodes"],tr["validation_episodes"],tr["development_seed_range"],tr["validation_seed_range"],tr["replicates"]) != (192,512,128,[10000,10511],[20000,20127],5): fail("training/data drift")
    if c.get("unique_seed_defined_episodes") != 3200: fail("unique data drift")
    if any(c.get(k) is not False for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started")): fail("science lock violated")
    parent_hashes={
      "experiments/v837_primitive_invention/v837y/results.json":c["v837y_results_sha256"],
      "experiments/v837_primitive_invention/v837y/diagnostics/decision_state.json":c["v837y_decision_sha256"],
      "experiments/v837_primitive_invention/v837y/candidate_interaction.py":c["v837y_model_sha256"],
      "experiments/v837_primitive_invention/v837y/config.json":c["v837y_config_sha256"],
    }
    for p,e in parent_hashes.items():
        if blob(p)!=e: fail("frozen V837y parent hash drift: "+p)
    src=(HERE/"candidate_stage.py").read_text()
    for token in ("snapshot_outputs = prev_outputs","snapshot_outputs[edge.src]","Simultaneous commit","stage_mode == \"historical_mixed\""):
        if token not in src: fail("required stage semantics missing: "+token)
    if (HERE/"results.json").exists():
        rows=[]
        for f in (HERE/"raw/z0_runs.json", HERE/"raw/z1_runs.json"):
            rows.extend(json.loads(f.read_text())["rows"])
        if len(rows)!=50: fail("raw run count != 50")
        if any(r.get("fresh_audit_consumed") is not False or r.get("structural_search_allowed") is not False or r.get("primitive_mining_allowed") is not False or r.get("v838_started") is not False for r in rows): fail("raw science lock violation")
        d=json.loads((HERE/"diagnostics/decision_state.json").read_text())
        allowed={"SYNCHRONOUS_CANDIDATE_STAGE_SUFFICIENT","SYNCHRONOUS_CANDIDATE_STAGE_PARTIAL_BENEFIT","CANDIDATE_STAGE_SYNCHRONY_INSUFFICIENT","HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL"}
        if d.get("diagnosis") not in allowed: fail("invalid diagnosis")
        if d.get("representation_adequacy_pass") is True and d.get("sample_efficiency_retest_allowed") is not True: fail("sample efficiency not opened after pass")
        depth=json.loads((HERE/"diagnostics/stage_depth.json").read_text())
        if depth["Z1_synchronous_candidate_stage"]["per_cell"] != [1]*10: fail("Z1 stage depth is not uniformly one")
    print("V837z candidate stage validation: PASS"); return 0

if __name__=="__main__": raise SystemExit(main())
