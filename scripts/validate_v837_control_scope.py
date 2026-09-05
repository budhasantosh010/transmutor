from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v837_primitive_invention"
HERE = BASE / "v837v"
GATE = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
CAPACITY = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
EXPECTED_DOMAINS = {
    "V0_10_domains": ([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],[0,1,2,3,4,5,6,7,8,9]),
    "V1_5_domains": ([[0,1],[2,3],[4,5],[6,7],[8,9]],[0,2,4,6,8]),
    "V2_2_domains": ([[0,1,2,3,4],[5,6,7,8,9]],[0,5]),
    "V3_1_domain": ([[0,1,2,3,4,5,6,7,8,9]],[0]),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not HERE.exists():
        return 0
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    frozen = json.loads((HERE / "frozen_control_scope_gate.json").read_text(encoding="utf-8"))
    if config.get("parent") != "V837u": raise ValueError("V837v parent must be V837u")
    if config.get("historical_gate_hash") != GATE or frozen.get("historical_v837_gate_sha256") != GATE: raise ValueError("V837v historical gate drift")
    if config.get("capacity_criterion_hash") != CAPACITY or frozen.get("capacity_criterion_sha256") != CAPACITY: raise ValueError("V837v capacity criterion drift")
    if config.get("data_regime") != "4x_unique" or int(config.get("unique_seed_defined_episodes",0)) != 3200: raise ValueError("V837v must use frozen 4x unique-data regime")
    tr=config["training"]
    if (tr.get("development_seed_range"),tr.get("validation_seed_range"),tr.get("steps"),tr.get("replicates")) != ([10000,10511],[20000,20127],192,5): raise ValueError("V837v training regime drift")
    for name,(domains,sources) in EXPECTED_DOMAINS.items():
        row=config.get("conditions",{}).get(name)
        if row is None or row.get("domains") != domains or row.get("sources") != sources: raise ValueError(f"V837v domain map changed: {name}")
    for key in ("global_state_visibility","global_controller","global_recurrent_coupling","vector_modulation","gate_pooling","structural_search","primitive_mining","fresh_audit","fresh_audit_consumed","v838_started"):
        if config.get(key) is not False: raise ValueError(f"V837v scientific lock violated: {key}")
    source=(HERE/"control_scope.py").read_text(encoding="utf-8")
    if "source_gates" not in source or "domain_spec" not in source: raise ValueError("V837v source-domain implementation missing")
    if "mean(g_" in source or "stack(source_gates).mean" in source: raise ValueError("V837v primary path appears to pool gates")
    audit=json.loads((BASE/"audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0: raise ValueError("fresh audit consumed")

    raw=HERE/"raw/runs.json"
    result=HERE/"results.json"
    if not raw.exists() and not result.exists():
        print("V837v framework validation: PASS (pre-run)")
        return 0
    if not raw.exists() or not result.exists(): raise ValueError("V837v partial result state")
    rows=json.loads(raw.read_text(encoding="utf-8")).get("rows",[])
    if len(rows)!=100: raise ValueError(f"V837v expected 100 raw fits, found {len(rows)}")
    seen=set()
    for row in rows:
        key=(row.get("condition"),row.get("family"),row.get("replicate"))
        if key in seen: raise ValueError(f"duplicate V837v row {key}")
        seen.add(key)
        if row.get("controller_information_scope") != "source_cell_local_only" or row.get("gate_pooling") is not False or row.get("global_state_visibility") is not False: raise ValueError("V837v widened controller information scope")
        if row.get("fresh_audit_consumed") is not False: raise ValueError("V837v raw row consumed fresh audit")
        if row.get("development_seed_range") != [10000,10511] or row.get("validation_seed_range") != [20000,20127]: raise ValueError("V837v raw seeds not paired")
        if int(row.get("resources",{}).get("optimizer_steps",-1)) != 192: raise ValueError("V837v optimizer-step mismatch")
        if int(row.get("resources",{}).get("examples_processed",-1)) != 192*512: raise ValueError("V837v processed-example mismatch")
    data=json.loads(result.read_text(encoding="utf-8"))
    if data.get("version")!="V837v" or data.get("parent")!="V837u": raise ValueError("V837v result version/parent mismatch")
    if data.get("controller_information_scope")!="source_cell_local_only" or data.get("gate_pooling") is not False: raise ValueError("V837v result widened information scope")
    if data.get("fresh_audit_consumed") is not False or data.get("primitives_promoted") != 0 or data.get("v838_started") is not False: raise ValueError("V837v downstream lock violation")
    if data.get("structural_search_allowed") is not False or data.get("primitive_mining_allowed") is not False: raise ValueError("V837v reopened downstream research")
    if data.get("baseline_compatibility",{}).get("compatible") is not True: raise ValueError("V837v interpreted without compatible V0 baseline")
    decision=json.loads((HERE/"diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if decision.get("diagnosis") not in {"GLOBAL_CONTROL_SCOPE_SUFFICIENT","INTERMEDIATE_CONTROL_DOMAIN_SCALE_SUFFICIENT","CONTROL_SCOPE_PARTIAL_BENEFIT","CONTROL_SCOPE_ALONE_INSUFFICIENT","CONTROL_SCOPE_COARSENING_HARMFUL"}: raise ValueError("invalid V837v diagnosis")
    if bool(decision.get("v837w_allowed")) == bool(decision.get("representation_adequacy_pass")): raise ValueError("V837v downstream authorization inconsistent")
    expected_counts={"V0_10_domains":10,"V1_5_domains":5,"V2_2_domains":2,"V3_1_domain":1}
    expected_active_params={"V0_10_domains":150,"V1_5_domains":75,"V2_2_domains":30,"V3_1_domain":15}
    expected_macs={"V0_10_domains":140,"V1_5_domains":70,"V2_2_domains":28,"V3_1_domain":14}
    for name,summary in data.get("conditions",{}).items():
        if summary.get("active_controller_count")!=expected_counts[name] or summary.get("active_controller_parameters")!=expected_active_params[name] or summary.get("controller_macs_per_timestep")!=expected_macs[name]: raise ValueError(f"V837v accounting mismatch {name}")
    print("V837v control-scope validation: PASS")
    return 0


if __name__=="__main__": raise SystemExit(main())
