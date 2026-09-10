from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from experiments.v837_primitive_invention.v837an.authorization import (
    CARRIERS,
    COUNTERFACTUALS,
    LATENT_DIMS,
    MIN_ELIGIBLE,
    PARTITIONS,
    START_SHA,
)
from experiments.v837_primitive_invention.v837an.coalition_scan import all_cell_subsets
from experiments.v837_primitive_invention.v837an.counterfactual_tasks import make_counterfactual
from experiments.v837_primitive_invention.v837an.failure_ledger import REQUIRED
from experiments.v837_primitive_invention.v837an.random_controls import random_subspaces
from experiments.v837_primitive_invention.v837an.routing_interventions import channel_universe
from experiments.v837_primitive_invention.v837an.semantic_compiler import compile_delta, fit_semantic_compiler
from experiments.v837_primitive_invention.v837an.utils import HERE, ROOT, read_json, sha256_json


class TestV837anFinalSourceContract(unittest.TestCase):
    def test_start_sha(self):
        self.assertEqual(START_SHA, "f556c92a895b141f73f214e5eda3ac29b1927ea9")

    def test_source_population(self):
        d = read_json(HERE / "diagnostics/source_integrity.json")
        self.assertEqual((d["source_organisms"], d["competent"], d["incompetent"]), (50, 40, 10))

    def test_partial_observation_power_not_lowered(self):
        d = read_json(HERE / "diagnostics/source_integrity.json")
        self.assertEqual(d["families"]["partial_observation"]["competent"], 3)
        self.assertFalse(d["families"]["partial_observation"]["strong_cross_organism_powered"])

    def test_protected_source_tree_unchanged(self):
        out = subprocess.check_output(
            ["git", "diff", "--name-only", START_SHA, "HEAD", "--", "experiments/v837_primitive_invention/v837am", "experiments/v837_primitive_invention/v837al", "experiments/v837_primitive_invention/v837ak"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertEqual(out, "")

    def test_interpretation_refinement_not_rewrite(self):
        d = read_json(HERE / "diagnostics/historical_interpretation_refinement.json")
        self.assertEqual(d["am_a_invalid_config_count"], 105)
        self.assertEqual(d["am_c_zero_row_config_count"], 7)
        self.assertEqual(d["am_c_invalid_pairs_per_config"], 38)
        self.assertTrue(d["historical_diagnosis_preserved"])
        self.assertFalse(d["historical_files_rewritten"])


class TestV837anDataFirewall(unittest.TestCase):
    def test_all_partitions_exact(self):
        self.assertEqual(PARTITIONS["AN_FIT"], [10000, 10127])
        self.assertEqual(PARTITIONS["AN_SELECT"], [10128, 10255])
        self.assertEqual(PARTITIONS["AN_ROUTING_FIT"], [10256, 10319])
        self.assertEqual(PARTITIONS["AN_ROUTING_SELECT"], [10320, 10383])
        self.assertEqual(PARTITIONS["AN_META_CONFIRM"], [10384, 10447])
        self.assertEqual(PARTITIONS["AN_FINAL_DEV"], [10448, 10511])
        self.assertEqual(PARTITIONS["AN_FINAL_VALIDATION"], [20000, 20127])
        self.assertEqual(PARTITIONS["FRESH_AUDIT"], [90000, 90499])

    def test_development_partitions_are_pairwise_disjoint(self):
        names = ["AN_FIT", "AN_SELECT", "AN_ROUTING_FIT", "AN_ROUTING_SELECT", "AN_META_CONFIRM", "AN_FINAL_DEV"]
        sets = {n: set(range(PARTITIONS[n][0], PARTITIONS[n][1] + 1)) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertTrue(sets[a].isdisjoint(sets[b]))

    def test_final_validation_disjoint(self):
        val = set(range(PARTITIONS["AN_FINAL_VALIDATION"][0], PARTITIONS["AN_FINAL_VALIDATION"][1] + 1))
        dev = set().union(*(set(range(PARTITIONS[n][0], PARTITIONS[n][1] + 1)) for n in ["AN_FIT", "AN_SELECT", "AN_ROUTING_FIT", "AN_ROUTING_SELECT", "AN_META_CONFIRM", "AN_FINAL_DEV"]))
        self.assertTrue(val.isdisjoint(dev))

    def test_freeze_hash_before_validation(self):
        freeze = read_json(HERE / "raw/frozen_family_abstractions.json")
        expected = freeze["frozen_sha256"]
        semantic = dict(freeze); semantic.pop("frozen_sha256")
        self.assertEqual(sha256_json(semantic), expected)
        self.assertTrue(freeze["frozen_before_validation"])

    def test_validation_uses_frozen_hash(self):
        freeze = read_json(HERE / "raw/frozen_family_abstractions.json")
        val = read_json(HERE / "raw/final_validation.json")
        if val.get("run"):
            self.assertEqual(val["freeze_sha256"], freeze["frozen_sha256"])
            self.assertTrue(val["no_refit"])
            self.assertTrue(val["no_second_best_retry"])
        else:
            self.assertFalse(val["validation_seeds_accessed"])


class TestV837anCounterfactualContract(unittest.TestCase):
    def test_counterfactual_kinds_exact(self):
        self.assertEqual(COUNTERFACTUALS, {
            "conditional_routing": "CONTROL_FLIP",
            "delayed_recall": "MEMORY_VALUE_FLIP",
            "iterative_state": "MID_INPUT_FLIP",
            "partial_observation": "INITIAL_LATENT_SIGN_FLIP",
            "variable_composition": "INITIAL_VALUE_SIGN_FLIP",
        })

    def test_routing_payloads_unchanged(self):
        p = make_counterfactual("conditional_routing", 10041)
        self.assertEqual(p.base_episode.causal_inputs["payload_a"], p.counterfactual_episode.causal_inputs["payload_a"])
        self.assertEqual(p.base_episode.causal_inputs["payload_b"], p.counterfactual_episode.causal_inputs["payload_b"])
        self.assertEqual(p.counterfactual_episode.causal_inputs["control"], -p.base_episode.causal_inputs["control"])

    def test_partial_innovations_and_noise_unchanged(self):
        p = make_counterfactual("partial_observation", 10041)
        np.testing.assert_array_equal(p.base_episode.causal_inputs["innovation"], p.counterfactual_episode.causal_inputs["innovation"])
        np.testing.assert_array_equal(p.base_episode.causal_inputs["observation_noise"], p.counterfactual_episode.causal_inputs["observation_noise"])
        self.assertEqual(p.base_episode.causal_inputs["rho"], p.counterfactual_episode.causal_inputs["rho"])

    def test_composition_gain_drive_unchanged(self):
        p = make_counterfactual("variable_composition", 10041)
        np.testing.assert_array_equal(p.base_episode.causal_inputs["gain"], p.counterfactual_episode.causal_inputs["gain"])
        np.testing.assert_array_equal(p.base_episode.causal_inputs["drive"], p.counterfactual_episode.causal_inputs["drive"])

    def test_no_fresh_audit_counterfactuals(self):
        d = read_json(HERE / "diagnostics/counterfactual_validity.json")
        self.assertEqual(d["new_random_task_seeds"], 0)
        self.assertFalse(d["fresh_audit_consumed"])


class TestV837anCarrierAndCompilerContract(unittest.TestCase):
    def test_primary_carrier_dimensions(self):
        shapes = read_json(HERE / "diagnostics/carrier_shapes.json")
        self.assertEqual([shapes[x] for x in CARRIERS], [40, 40, 40, 40, 1])

    def test_coupling_factor_diagnostic_only(self):
        shapes = read_json(HERE / "diagnostics/carrier_shapes.json")
        self.assertEqual(shapes["COUPLING_FACTOR4"], 4)
        self.assertTrue(shapes["coupling_factor4_diagnostic_only"])

    def test_latent_dimensions_exact(self):
        self.assertEqual(LATENT_DIMS["STATE40"], [1, 2, 4, 8])
        self.assertEqual(LATENT_DIMS["OUTPUT40"], [1, 2, 4, 8])
        self.assertEqual(LATENT_DIMS["MESSAGE40"], [1, 2, 4, 8])
        self.assertEqual(LATENT_DIMS["GLOBAL40"], [1, 2, 4, 8])
        self.assertEqual(LATENT_DIMS["GATE1"], [1])
        self.assertEqual(LATENT_DIMS["COUPLING_FACTOR4"], [1, 2, 4])

    def test_semantic_compiler_exact_zero_intercept(self):
        rng = np.random.default_rng(123)
        q, _ = np.linalg.qr(rng.normal(size=(4, 2)))
        dz = rng.normal(size=64)
        dx = dz[:, None] * np.array([[1., 2., 3., 4.]])
        model = fit_semantic_compiler(dz, dx, q)
        out = compile_delta(np.zeros(5), model)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)
        self.assertEqual(model["ridge_lambda"], 1e-6)
        self.assertEqual(model["intercept"], 0.0)

    def test_an_a_never_requires_invertibility(self):
        fit = read_json(HERE / "raw/an_a_fit.json")
        self.assertFalse(fit["cross_organism_state_invertibility_required"])
        self.assertTrue(all(r["invertibility_required"] is False for r in fit["records"]))

    def test_32_random_subspaces_per_an_a_row(self):
        d = read_json(HERE / "raw/an_a_selection.json")
        self.assertTrue(d["organism_results"])
        for r in d["organism_results"]:
            self.assertEqual(r["controls"]["random_subspaces"], 32)


class TestV837anRoutingContract(unittest.TestCase):
    def test_message_decomposition_exact(self):
        d = read_json(HERE / "diagnostics/message_decomposition.json")
        self.assertTrue(d["pass"])
        self.assertLessEqual(d["max_abs_error"], 1e-6)

    def test_global_decomposition_exact(self):
        d = read_json(HERE / "diagnostics/global_decomposition.json")
        self.assertTrue(d["pass"])
        self.assertLessEqual(d["max_abs_error"], 1e-6)

    def test_rank4_bus_is_intervention_based(self):
        d = read_json(HERE / "diagnostics/rank4_bus.json")
        self.assertTrue(d["intervention_based"])

    def test_routing_control_count_and_prefix_semantics(self):
        d = read_json(HERE / "raw/routing_selection.json")
        self.assertEqual(d["fit_seeds"], PARTITIONS["AN_ROUTING_FIT"])
        self.assertEqual(d["select_seeds"], PARTITIONS["AN_ROUTING_SELECT"])
        for row in d["organism_results"]:
            if not row.get("powered"):
                continue
            self.assertIn("COMBINED_PREFIX_1", row["configs"])
            self.assertIn("BUS_ALL", row["configs"])


class TestV837anCoalitionContract(unittest.TestCase):
    def test_exact_1023_subset_universe(self):
        subsets = all_cell_subsets()
        self.assertEqual(len(subsets), 1023)
        self.assertEqual(len(set(subsets)), 1023)

    def test_all_cardinalities_present(self):
        subsets = all_cell_subsets()
        self.assertEqual({len(s) for s in subsets}, set(range(1, 11)))

    def test_full_set_unique(self):
        subsets = all_cell_subsets()
        self.assertEqual(sum(s == tuple(range(10)) for s in subsets), 1)

    def test_every_executed_coalition_scan_complete(self):
        d = read_json(HERE / "raw/coalition_scan.json")
        for scan in d.get("scans", []):
            self.assertEqual(len(scan["rows"]), 1023)
            self.assertEqual(len({tuple(r["nodes"]) for r in scan["rows"]}), 1023)


class TestV837anFailureMemoryContract(unittest.TestCase):
    def test_raw_and_diagnostic_ledgers_equal(self):
        self.assertEqual(read_json(HERE / "raw/failure_ledger.json"), read_json(HERE / "diagnostics/failure_ledger.json"))

    def test_every_failure_entry_complete(self):
        d = read_json(HERE / "raw/failure_ledger.json")
        for e in d["entries"]:
            self.assertFalse([k for k in REQUIRED if k not in e], e.get("failure_id"))

    def test_refinements_present(self):
        ids = {e["failure_id"] for e in read_json(HERE / "raw/failure_ledger.json")["entries"]}
        self.assertTrue({"REF-AN-001", "REF-AN-002", "REF-AN-003"}.issubset(ids))

    def test_failure_analysis_survives_success_or_failure(self):
        text = (HERE / "FAILURE_ANALYSIS.md").read_text(encoding="utf-8")
        self.assertIn("Historical interpretation refinements", text)
        self.assertIn("What remains unknown", text)


class TestV837anFinalLocks(unittest.TestCase):
    def test_zero_training_and_archive_locks(self):
        d = read_json(HERE / "diagnostics/decision_state.json")
        self.assertEqual(d["new_model_fits"], 0)
        self.assertEqual(d["optimizer_steps"], 0)
        self.assertEqual(d["adapter_gradient_steps"], 0)
        self.assertEqual(d["primitives_promoted"], 0)
        self.assertFalse(d["primitive_archive_allowed_next"])
        self.assertFalse(d["fresh_audit_consumed"])
        self.assertFalse(d["v838_started"])

    def test_resource_accounting_zero_training(self):
        d = read_json(HERE / "diagnostics/resource_accounting.json")
        self.assertEqual(d["new_organism_fits"], 0)
        self.assertEqual(d["optimizer_steps"], 0)
        self.assertEqual(d["adapter_gradient_steps"], 0)
        self.assertEqual(d["processed_training_examples"], 0)
        self.assertEqual(d["gpu_seconds"], 0.0)

    def test_validation_count_matches_decision(self):
        val = read_json(HERE / "raw/final_validation.json")
        dec = read_json(HERE / "diagnostics/decision_state.json")
        self.assertEqual(dec["validated_family_abstractions"], val.get("validated_family_count", 0))


if __name__ == "__main__":
    unittest.main()
