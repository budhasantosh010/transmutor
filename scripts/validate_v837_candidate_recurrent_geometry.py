from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837ad.candidate_recurrent_geometry import CONDITION_SPECS, candidate_mask, mask_integrity

HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837ad"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GATE = json.loads((HERE / "frozen_candidate_geometry_gate.json").read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists(): raise RuntimeError(f"missing V837ad artifact: {path.relative_to(ROOT)}")


def load(path: Path) -> dict:
    require(path); return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    for name in ("README.md","RESEARCH_SPEC.md","config.json","frozen_candidate_geometry_gate.json","candidate_recurrent_geometry.py","run_candidate_geometry.py","analyze_results.py"):
        require(HERE/name)
    if CONFIG.get("data_regime") != "4x_unique" or CONFIG.get("unique_seed_defined_episodes") != 3200:
        raise RuntimeError("V837ad data regime changed")
    tr=CONFIG["training"]
    if tr.get("steps") != 192 or tr.get("replicates") != 5 or tr.get("train_episodes") != 512 or tr.get("validation_episodes") != 128:
        raise RuntimeError("V837ad training budget changed")
    if tr.get("development_seed_range") != [10000,10511] or tr.get("validation_seed_range") != [20000,20127]:
        raise RuntimeError("V837ad task seeds changed")
    if any(CONFIG.get(k) is not False for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started")):
        raise RuntimeError("V837ad science lock changed")
    audit=load(ROOT/"experiments/v837_primitive_invention/audit/audit_results.json")
    if audit.get("episodes_consumed") != 0: raise RuntimeError("fresh audit consumed")
    if (ROOT/"experiments/v837_primitive_invention/v838").exists(): raise RuntimeError("V838 exists")

    expected_active={"AD0_H13_dense":169,"AD1_H40_dense":1600,"AD2_H40_2x20":800,"AD3_H40_5x8":320,"AD4_H40_10x4":160}
    for name,count in expected_active.items():
        if int(candidate_mask(CONDITION_SPECS[name]).sum().item()) != count: raise RuntimeError(f"candidate mask count changed: {name}")
    for i in range(5):
        x=mask_integrity(CONDITION_SPECS[f"AD4S_S{i}"])
        if x["active_weights"] != 160 or set(x["fan_in"]) != {4} or set(x["fan_out"]) != {4} or x["same_logical_4d_block_edges"] != 0 or not x["strongly_connected"]:
            raise RuntimeError(f"invalid sparse topology S{i}")

    mask_path=HERE/"diagnostics/mask_integrity.json"
    if mask_path.exists() and load(mask_path).get("valid") is not True: raise RuntimeError("mask preflight failed")
    pair_path=HERE/"diagnostics/h40_pairing.json"
    if pair_path.exists() and load(pair_path).get("valid") is not True: raise RuntimeError("H40 pairing failed")

    ad0_path=HERE/"raw/ad0_runs.json"
    anchor_path=HERE/"diagnostics/anchor_compatibility.json"
    if ad0_path.exists():
        rows=load(ad0_path).get("rows",[])
        if len(rows)!=25: raise RuntimeError("AD0 must contain 25 fits")
        anchor=load(anchor_path)
        if anchor.get("ad0_anchor_valid") is not True: raise RuntimeError("AD0 anchor invalid")

    ad1_path=HERE/"raw/ad1_runs.json"; width_path=HERE/"diagnostics/width_gate.json"
    if ad1_path.exists():
        if not ad0_path.exists(): raise RuntimeError("AD1 exists without AD0")
        if len(load(ad1_path).get("rows",[]))!=25: raise RuntimeError("AD1 must contain 25 fits")
        width=load(width_path); ad1=int(width.get("ad1_dense_h40_families",-1)); allowed=width.get("geometry_stage_allowed") is True
        if allowed != (ad1>=4): raise RuntimeError("width-stage decision inconsistent")
        geo_path=HERE/"raw/geometry_runs.json"
        if not allowed and geo_path.exists(): raise RuntimeError("geometry results exist despite failed width gate")

    geo_path=HERE/"raw/geometry_runs.json"
    if geo_path.exists():
        width=load(width_path)
        if width.get("geometry_stage_allowed") is not True: raise RuntimeError("geometry stage unauthorized")
        rows=load(geo_path).get("rows",[])
        if len(rows)!=100: raise RuntimeError("primary geometry stage must contain 100 fits")
        for condition in CONFIG["stage_b_conditions"]:
            cr=[r for r in rows if r.get("condition")==condition]
            if len(cr)!=25: raise RuntimeError(f"incomplete geometry condition {condition}")
        ad4=[r for r in rows if r["condition"]=="AD4_H40_10x4"][0]; s0=[r for r in rows if r["condition"]=="AD4S_S0"][0]
        if ad4["active_candidate_recurrent_weights"]!=s0["active_candidate_recurrent_weights"] or ad4["candidate_recurrent_macs"]!=s0["candidate_recurrent_macs"] or ad4["total_active_macs_per_timestep"]!=s0["total_active_macs_per_timestep"]:
            raise RuntimeError("AD4/AD4S compute mismatch")
        trigger=load(HERE/"diagnostics/sparse_robustness_gate.json")
        robust_path=HERE/"raw/robustness_runs.json"
        if trigger.get("robustness_required") is not True and robust_path.exists(): raise RuntimeError("robustness results exist without trigger")
        if robust_path.exists() and len(load(robust_path).get("rows",[]))!=100: raise RuntimeError("robustness stage must contain 100 fits")

    results_path=HERE/"results.json"
    if results_path.exists():
        results=load(results_path); decision=load(HERE/"diagnostics/decision_state.json")
        if results.get("unique_seed_defined_episodes")!=3200 or results.get("fresh_audit_consumed") is not False or results.get("v838_started") is not False:
            raise RuntimeError("V837ad final lock state changed")
        if decision.get("ad0_anchor_valid") is not True: raise RuntimeError("final V837ad lacks valid AD0")
        ad1=int(decision.get("ad1_dense_h40_families",-1))
        if ad1<4 and decision.get("geometry_stage_run") is not False: raise RuntimeError("failed width gate may not run geometry")
        if decision.get("authorized_v837ae_mode") is not None:
            if decision.get("diagnosis") != "GLOBAL_SPARSE_CANDIDATE_GEOMETRY_SUFFICIENT" or decision.get("ad4s_topology_pass_count",0)<4:
                raise RuntimeError("V837ae authorization is not robustly supported")
        resources=load(ROOT/"experiments/v837_primitive_invention/v837ad_resource_accounting.json")
        expected_fits=50 if ad1<4 else (250 if decision.get("ad4s_robustness_run") else 150)
        if int(resources.get("model_fits",-1)) != expected_fits: raise RuntimeError("V837ad resource fit count inconsistent")
        if int(resources.get("unique_seed_defined_episodes",-1)) != 3200: raise RuntimeError("unique episode accounting changed")

    print("V837ad candidate-recurrent-geometry validation: PASS")
    return 0


if __name__ == "__main__": raise SystemExit(main())
