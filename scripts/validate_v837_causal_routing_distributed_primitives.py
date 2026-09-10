from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837an.authorization import (
    CARRIERS,
    LATENT_DIMS,
    MIN_ELIGIBLE,
    PARTITIONS,
    START_SHA,
    assert_authorized,
)
from experiments.v837_primitive_invention.v837an.coalition_scan import all_cell_subsets
from experiments.v837_primitive_invention.v837an.failure_ledger import REQUIRED
from experiments.v837_primitive_invention.v837an.utils import sha256_json

HERE = ROOT / "experiments/v837_primitive_invention/v837an"


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def req(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    auth = assert_authorized()
    gate = load("experiments/v837_primitive_invention/v837an/frozen_causal_primitive_gate.json")
    req(gate["start_sha"] == START_SHA == "f556c92a895b141f73f214e5eda3ac29b1927ea9", "V837an START SHA drift")
    req(auth["authorized"] is True and auth["source_organisms"] == 50 and auth["competent_organisms"] == 40, "V837an source authorization invalid")
    req(gate["source_organisms_expected"] == 50 and gate["competent_organisms_expected"] == 40 and gate["incompetent_organisms_expected"] == 10, "V837an population lock drift")
    req(gate["cross_organism_state_invertibility_required"] is False, "V837an illegally requires cross-organism state invertibility")
    req(gate["new_model_fits"] == 0 and gate["optimizer_steps"] == 0 and gate["adapter_gradient_steps"] == 0, "V837an zero-training lock drift")
    req(gate["fresh_audit_consumed"] is False and gate["primitive_archive_allowed"] is False and gate["primitives_promoted"] == 0 and gate["v838_started"] is False, "V837an downstream lock drift")
    req(gate["data_partitions"] == PARTITIONS and gate["minimum_pair_counts"] == MIN_ELIGIBLE, "V837an data partition/power drift")
    req(gate["carrier_list"] == CARRIERS and gate["latent_dimensions"] == LATENT_DIMS, "V837an carrier grid drift")
    req(gate["random_subspace_controls"] == 32 and gate["routing_random_controls"] == 64 and gate["cell_coalitions"]["nonempty_subsets"] == 1023, "V837an control/coalition grid drift")
    req(len(all_cell_subsets()) == 1023 and tuple(range(10)) in set(all_cell_subsets()), "V837an exact coalition universe invalid")

    source = load("experiments/v837_primitive_invention/v837an/diagnostics/source_integrity.json")
    req(source["valid"] is True and source["source_organisms"] == 50 and source["competent"] == 40 and source["incompetent"] == 10, "V837an source integrity invalid")
    req(source["cross_organism_state_invertibility_required"] is False and source["protected_source_mutation"] is False, "V837an source/invertibility integrity invalid")
    req(sum(v["competent"] for v in source["families"].values()) == 40, "V837an per-family competent count drift")

    ledger_raw = load("experiments/v837_primitive_invention/v837an/raw/failure_ledger.json")
    ledger_diag = load("experiments/v837_primitive_invention/v837an/diagnostics/failure_ledger.json")
    req(ledger_raw == ledger_diag and ledger_raw.get("append_only") is True, "V837an failure ledger copies drift")
    ids = {e["failure_id"] for e in ledger_raw["entries"]}
    req({"REF-AN-001", "REF-AN-002", "REF-AN-003"}.issubset(ids), "V837an historical refinements missing")
    for entry in ledger_raw["entries"]:
        missing = [k for k in REQUIRED if k not in entry]
        req(not missing, f"V837an incomplete failure entry {entry.get('failure_id')}: {missing}")
        req(entry["failure_type"] in {"SCIENTIFIC_FAILURE", "ENGINEERING_FAILURE", "UNDERPOWERED", "INTERPRETATION_REFINEMENT"}, f"V837an failure type invalid: {entry['failure_id']}")
    central = (ROOT / "docs/V837_FAILURE_LEDGER.md").read_text(encoding="utf-8")
    req(all(x in central for x in ("REF-AN-001", "REF-AN-002", "REF-AN-003")), "V837an central refinement ledger missing")

    oracle = load("experiments/v837_primitive_invention/v837an/diagnostics/oracle_equivalence.json")
    req(oracle["pass"] is True and oracle["episodes_verified"] == 2560 and oracle["max_observation_abs_error"] <= 1e-7 and oracle["max_target_abs_error"] <= 1e-12, "V837an oracle instrumentation invalid")
    cf = load("experiments/v837_primitive_invention/v837an/diagnostics/counterfactual_validity.json")
    req(cf["pass"] is True and cf["new_random_task_seeds"] == 0 and cf["fresh_audit_consumed"] is False, "V837an counterfactual constructor invalid")
    runtime = load("experiments/v837_primitive_invention/v837an/diagnostics/instrumented_runtime_equivalence.json")
    req(runtime["pass"] is True and all(float(v) <= 1e-6 for v in runtime["max_abs"].values()), "V837an instrumented runtime drift")
    req(set(runtime["intervention_hooks"]) == {"patch_state", "patch_output", "patch_message", "patch_global_term", "patch_gate"}, "V837an intervention hooks incomplete")

    a = load("experiments/v837_primitive_invention/v837an/raw/an_a_selection.json")
    req(a["stage"] == "AN-A-SELECT", "V837an AN-A missing")
    for row in a["organism_results"]:
        req(row["controls"]["random_subspaces"] == 32, f"V837an random control count drift {row['organism_id']}/{row['config_id']}")
        req(row["carrier"] in CARRIERS and int(row["k"]) in LATENT_DIMS[row["carrier"]], "V837an hidden carrier/dimension config")
    fit = load("experiments/v837_primitive_invention/v837an/raw/an_a_fit.json")
    req(fit["gradient_steps"] == 0 and fit["cross_organism_state_invertibility_required"] is False, "V837an AN-A training/invertibility lock violated")

    routing = load("experiments/v837_primitive_invention/v837an/raw/routing_selection.json")
    req(routing["fit_seeds"] == PARTITIONS["AN_ROUTING_FIT"] and routing["select_seeds"] == PARTITIONS["AN_ROUTING_SELECT"], "V837an routing partition drift")
    msg = load("experiments/v837_primitive_invention/v837an/diagnostics/message_decomposition.json")
    glob = load("experiments/v837_primitive_invention/v837an/diagnostics/global_decomposition.json")
    req(msg["pass"] is True and float(msg["max_abs_error"]) <= 1e-6, "V837an message decomposition invalid")
    req(glob["pass"] is True and float(glob["max_abs_error"]) <= 1e-6, "V837an global decomposition invalid")
    for row in routing["organism_results"]:
        if not row.get("powered"):
            continue
        for cfg in row.get("configs", {}).values():
            req("metrics" in cfg and "intervention_dof" in cfg, "V837an routing config incomplete")

    synergy = load("experiments/v837_primitive_invention/v837an/raw/synergy_results.json")
    coalition = load("experiments/v837_primitive_invention/v837an/raw/coalition_scan.json")
    for scan in coalition.get("scans", []):
        rows = scan.get("rows", [])
        req(len(rows) == 1023, f"V837an coalition scan incomplete: {scan.get('organism_id')}")
        req(len({tuple(r["nodes"]) for r in rows}) == 1023, f"V837an duplicate/missing coalition: {scan.get('organism_id')}")
    req("communication_channel_synergy" in synergy, "V837an communication-channel synergy missing")

    meta = load("experiments/v837_primitive_invention/v837an/raw/meta_confirmation.json")
    req(meta["seeds"] == PARTITIONS["AN_META_CONFIRM"] and meta["no_refit"] is True, "V837an META_CONFIRM leakage/refit")
    dev = load("experiments/v837_primitive_invention/v837an/raw/final_dev_confirmation.json")
    req(dev["seeds"] == PARTITIONS["AN_FINAL_DEV"] and dev["no_refit"] is True, "V837an FINAL_DEV leakage/refit")
    freeze = load("experiments/v837_primitive_invention/v837an/raw/frozen_family_abstractions.json")
    semantic = dict(freeze); expected = semantic.pop("frozen_sha256")
    req(freeze["frozen_before_validation"] is True and sha256_json(semantic) == expected, "V837an final family freeze hash invalid")
    final_diag = load("experiments/v837_primitive_invention/v837an/diagnostics/final_freeze.json")
    req(final_diag["frozen_sha256"] == expected and final_diag["validation_unlocked"] is True, "V837an final freeze diagnostic mismatch")
    val = load("experiments/v837_primitive_invention/v837an/raw/final_validation.json")
    if val.get("run"):
        req(val["seeds"] == PARTITIONS["AN_FINAL_VALIDATION"] and val["freeze_sha256"] == expected and val["no_refit"] is True and val["no_second_best_retry"] is True, "V837an final-validation firewall violation")
    else:
        req(val.get("validation_seeds_accessed") is False, "V837an validation accessed without selected family abstractions")

    # No hidden training/optimization is permitted anywhere in V837an.
    forbidden = ("AdamW(", ".backward(", "optimizer.step(", "torch.optim.")
    for path in HERE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        req(not any(token in text for token in forbidden), f"V837an optimizer/backprop forbidden: {path.name}")

    decision = load("experiments/v837_primitive_invention/v837an/diagnostics/decision_state.json")
    req(decision["new_model_fits"] == 0 and decision["optimizer_steps"] == 0 and decision["adapter_gradient_steps"] == 0, "V837an final training lock violated")
    req(decision["primitive_archive_allowed_next"] is False and decision["primitives_promoted"] == 0 and decision["fresh_audit_consumed"] is False and decision["v838_started"] is False, "V837an final downstream locks violated")
    req(decision["failure_entries"] == len(ledger_raw["entries"]), "V837an failure accounting drift")

    required_plots = {
        "causal_abstraction_evidence_ladder.png", "decodability_vs_causality.png", "carrier_recovery_by_family.png",
        "recovery_vs_latent_dimension.png", "random_subspace_control_distribution.png", "semantic_compiler_vs_source_swap.png",
        "phase_specific_causal_carriers.png", "message_edge_mediation.png", "global_source_mediation.png",
        "message_vs_global_vs_gate.png", "rank4_bus_evidence.png", "routing_subset_size_vs_recovery.png",
        "coalition_size_vs_recovery.png", "cell_pair_synergy_heatmap.png", "minimum_causal_granularity.png",
        "implementation_diversity_vs_shared_semantics.png", "failure_map_v837ak_to_v837an.png",
    }
    req(required_plots == {p.name for p in (HERE / "plots").glob("*.png")}, "V837an required plot set drift")
    req((ROOT / "docs/V837_CAUSAL_ROUTING_DISTRIBUTED_PRIMITIVE_REDEFINITION_REPORT.md").is_file(), "V837an main report missing")
    req((HERE / "FAILURE_ANALYSIS.md").is_file(), "V837an failure analysis missing")

    audit = load("experiments/v837_primitive_invention/audit/audit_results.json")
    req(audit.get("episodes_consumed") == 0, "V837an fresh audit consumed")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", START_SHA, "--", "experiments/v837_primitive_invention/v837am", "experiments/v837_primitive_invention/v837al", "experiments/v837_primitive_invention/v837ak"],
        cwd=ROOT, text=True,
    ).strip()
    req(changed == "", f"V837an protected historical science changed: {changed}")

    print("V837an causal/routing/distributed primitive redefinition validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
