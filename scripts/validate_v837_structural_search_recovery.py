from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/v837_primitive_invention/v837aj"


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required V837aj artifact: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(preflight: bool = False) -> None:
    cfg = load(HERE / "config.json")
    gate = load(HERE / "frozen_structural_search_gate.json")
    ai = load(ROOT / "experiments/v837_primitive_invention/v837ai/diagnostics/decision_state.json")
    if ai.get("structural_search_recovery_allowed") is not True or int(ai.get("recommended_structural_search_multiplier", -1)) != 4:
        raise RuntimeError("V837ai did not authorize V837aj structural search at 4x")
    if cfg.get("required_start_sha") != "f87f1506d9f1ee9e85ac1668bff76e03d3f20993" or gate.get("starting_sha") != cfg.get("required_start_sha"):
        raise RuntimeError("V837aj start SHA lock changed")
    if cfg.get("architecture") != "AF1D_deshared_candidate_input_factorization" or int(cfg.get("cell_count", -1)) != 10 or int(cfg.get("max_message_edges", -1)) != 64:
        raise RuntimeError("V837aj frozen architecture/search-space lock changed")
    if cfg.get("fresh_audit_allowed") is not False or cfg.get("primitive_mining_allowed_during_v837aj") is not False or cfg.get("v838_started") is not False:
        raise RuntimeError("V837aj science locks changed")
    if gate.get("candidate_training_from_scratch") is not True or gate.get("lamarckian_inheritance") is not False or gate.get("equal_budget_random_required") is not True:
        raise RuntimeError("V837aj candidate/random control lock changed")
    data = load(HERE / "diagnostics/data_partition_lock.json") if (HERE / "diagnostics/data_partition_lock.json").exists() else None
    guard = load(HERE / "diagnostics/final_validation_guard.json") if (HERE / "diagnostics/final_validation_guard.json").exists() else None
    init = load(HERE / "diagnostics/initialization_pairing.json") if (HERE / "diagnostics/initialization_pairing.json").exists() else None
    arch = load(HERE / "diagnostics/architecture_lock.json") if (HERE / "diagnostics/architecture_lock.json").exists() else None
    if any(item is not None for item in (data, guard, init, arch)):
        if not all(item is not None for item in (data, guard, init, arch)):
            raise RuntimeError("partial V837aj preflight artifacts")
        if data.get("exact") is not True or int(data.get("union_unique_family_seed_episodes", -1)) != 3200 or int(data.get("fresh_audit_overlap", -1)) != 0:
            raise RuntimeError("V837aj data partition lock failed")
        if guard.get("guard_valid") is not True or guard.get("calibration_blocked") is not True or guard.get("search_blocked") is not True or guard.get("random_search_blocked") is not True:
            raise RuntimeError("V837aj final-validation guard failed")
        if init.get("pairing_valid") is not True:
            raise RuntimeError("V837aj initialization pairing failed")
        if arch.get("compatible") is not True or arch.get("source_hash_lock", {}).get("compatible") is not True:
            raise RuntimeError("V837aj AF1D architecture/source lock failed")
    if preflight:
        audit = load(ROOT / "experiments/v837_primitive_invention/audit/audit_results.json")
        if int(audit.get("episodes_consumed", -1)) != 0:
            raise RuntimeError("fresh audit consumed during V837aj")
        for forbidden in ("v837ae", "v837ag", "v837ah", "v838"):
            if (ROOT / "experiments/v837_primitive_invention" / forbidden).exists():
                raise RuntimeError(f"unauthorized {forbidden} exists")
        print("V837aj structural-search recovery preflight validation: PASS")
        return

    anchor = load(HERE / "diagnostics/anchor_reproduction.json")
    fidelity = load(HERE / "diagnostics/fidelity_decision.json")
    decision = load(HERE / "diagnostics/decision_state.json")
    results = load(HERE / "results.json")
    if anchor.get("anchor_reproduced") is not True or int(anchor.get("families_passing", -1)) != 4:
        raise RuntimeError("V837aj AF1D anchor reproduction failed")
    if fidelity.get("calibration_complete") is not True or fidelity.get("target_fidelity") != "F4":
        raise RuntimeError("V837aj fidelity calibration incomplete")
    if decision.get("fidelity_calibration_complete") is not True or decision.get("af1d_anchor_valid") is not True:
        raise RuntimeError("V837aj final decision lost anchor/fidelity state")
    if fidelity.get("proxy_valid") is True:
        selected = fidelity.get("selected_search_fidelity")
        if selected not in {"F0", "F1", "F2", "F3"} or decision.get("search_stage_allowed") is not True:
            raise RuntimeError("V837aj search ran without valid selected proxy")
        search = load(HERE / "raw/search_proxy_runs.json")
        random = load(HERE / "raw/random_proxy_runs.json")
        sruns = search.get("runs", []); rruns = random.get("runs", [])
        if len(sruns) not in {25, 50} or len(rruns) != len(sruns):
            raise RuntimeError("V837aj search/random run counts invalid")
        if any(int(run.get("candidate_budget", -1)) != 64 or len(run.get("records", [])) != 64 for run in sruns + rruns):
            raise RuntimeError("V837aj search/random candidate budget mismatch")
        budget = load(HERE / "diagnostics/search_random_budget_match.json")
        if budget.get("all_exact_64") is not True or budget.get("all_slot_matched") is not True:
            raise RuntimeError("V837aj equal-budget/complexity pairing failed")
        sfinal = load(HERE / "raw/search_finalized.json").get("rows", [])
        rfinal = load(HERE / "raw/random_finalized.json").get("rows", [])
        if len(sfinal) != len(sruns) or len(rfinal) != len(rruns):
            raise RuntimeError("V837aj champion finalization incomplete")
        for row in sfinal + rfinal:
            if row.get("champion_frozen_before_final_validation") is not True or row.get("proxy_weights_discarded") is not True:
                raise RuntimeError("V837aj champion freeze/reinitialization violated")
            if int(row.get("final_validation_data_accesses", -1)) != 1 or int(row.get("final_validation_episode_count", -1)) != 128:
                raise RuntimeError("V837aj final validation access count changed")
            if int(row.get("optimizer_steps", -1)) != 192 or int(row.get("processed_examples", -1)) != 98304:
                raise RuntimeError("V837aj full finalization protocol changed")
    else:
        if decision.get("search_stage_allowed") is not False or decision.get("constructive_search_run") is not False or decision.get("diagnosis") != "SEARCH_FIDELITY_PROXY_INVALID":
            raise RuntimeError("V837aj failed proxy but did not hard-stop Stage B")
    if decision.get("fresh_audit_consumed") is not False or int(decision.get("primitives_promoted", -1)) != 0 or decision.get("v838_started") is not False:
        raise RuntimeError("V837aj audit/primitive/V838 lock violated")
    if results.get("primitive_mining_allowed_next") is True and results.get("diagnosis") not in {"STRUCTURAL_SEARCH_RECOVERED", "RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT"}:
        raise RuntimeError("V837aj unlocked primitive mining without automated discovery")
    audit = load(ROOT / "experiments/v837_primitive_invention/audit/audit_results.json")
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise RuntimeError("fresh audit consumed during V837aj")
    for forbidden in ("v837ae", "v837ag", "v837ah", "v838"):
        if (ROOT / "experiments/v837_primitive_invention" / forbidden).exists():
            raise RuntimeError(f"unauthorized {forbidden} exists")
    print("V837aj structural-search recovery validation: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    validate(preflight=bool(args.preflight)); return 0

if __name__ == "__main__": raise SystemExit(main())
