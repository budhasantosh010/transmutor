from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"


class V837nMechanismOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((BASE / "v837n" / "results.json").read_text(encoding="utf-8"))
        cls.status = json.loads((BASE / "gru_mechanism_localization_status.json").read_text(encoding="utf-8"))

    def test_full_gru_positive_control_is_5_of_5_and_875_parameters(self):
        positive = self.result["full_gru_positive_control"]
        self.assertTrue(positive["compatible"])
        self.assertEqual(positive["families_passing"], 5)
        self.assertEqual(positive["parameter_count"], 875)

    def test_no_individual_named_gru_gate_is_necessary(self):
        counts = self.result["families_passing"]
        self.assertEqual(counts["no_update"], 5)
        self.assertEqual(counts["no_reset"], 5)
        self.assertEqual(counts["static_reset_vector"], 5)
        self.assertEqual(counts["static_update_vector"], 4)
        self.assertEqual(counts["static_update_scalar"], 4)

    def test_double_ablation_falls_below_representation_gate(self):
        self.assertEqual(self.result["families_passing"]["no_update_no_reset"], 3)
        self.assertEqual(self.result["mechanism_diagnosis"], "MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY")
        self.assertTrue(self.result["diagnostic_pass"])

    def test_gate_dynamics_and_counterfactual_replay_are_recorded(self):
        full = self.result["gate_statistics"]["full_gru"]["aggregate"]
        self.assertGreater(full["update_temporal_variance"]["median"], 0.0)
        self.assertGreater(full["reset_temporal_variance"]["median"], 0.0)
        replay = self.result["counterfactual_gate_replay"]
        self.assertIn("update_time_shuffle", replay)
        self.assertIn("reset_time_shuffle", replay)
        self.assertLess(replay["update_time_shuffle"]["conditional_routing"]["median"], 0.85)

    def test_v837n_did_not_directly_transfer_a_named_gru_gate(self):
        self.assertEqual(self.status["outcome"], "C_NO_INDIVIDUAL_GRU_MECHANISM_EXPLAINS_SUCCESS")
        self.assertFalse(self.status["full_structural_search_allowed"])
        self.assertFalse(self.status["primitive_mining_allowed"])
        v837o = json.loads((BASE / "v837o" / "results.json").read_text(encoding="utf-8"))
        v837p = json.loads((BASE / "v837p" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(v837o["parent"], "V837n")
        self.assertEqual(v837o["mechanism_diagnosis"], "DYNAMIC_STATE_MODULATION_REQUIRED")
        self.assertEqual(v837p["parent"], "V837o")
        self.assertEqual(v837p["selection_basis"], "V837o DYNAMIC_STATE_MODULATION_REQUIRED")
        v837q = json.loads((BASE / "v837q" / "results.json").read_text(encoding="utf-8"))
        q_config = json.loads((BASE / "v837q" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(v837q["parent"], "V837p")
        self.assertEqual(v837q["diagnosis"], "STATE_FRAGMENTATION_HYPOTHESIS_NOT_SUPPORTED")
        self.assertFalse(q_config["dynamic_modulation_allowed"])
        v837r = json.loads((BASE / "v837r" / "results.json").read_text(encoding="utf-8"))
        r_config = json.loads((BASE / "v837r" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(v837r["parent"], "V837q")
        self.assertEqual(v837r["diagnosis"], "GLOBAL_COUPLING_PARTIAL_BENEFIT")
        self.assertFalse(r_config["dynamic_modulation_allowed"])
        v837s = json.loads((BASE / "v837s" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(v837s["parent"], "V837r")
        self.assertEqual(v837s["diagnosis"], "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT")
        self.assertFalse(v837s["multiplicative_specificity_established"])
        for suffix in ("v837t", "v837u", "v837v", "v838"):
            self.assertFalse((BASE / suffix).exists(), suffix)

    def test_fresh_audit_and_primitives_remain_locked(self):
        self.assertFalse(self.result["fresh_audit_consumed"])
        self.assertFalse(self.result["primitive_mining_allowed"])
        self.assertEqual(self.status["fresh_audit_episodes_consumed"], 0)
        self.assertEqual(self.status["primitives_promoted"], 0)

    def test_final_report_exists(self):
        report = ROOT / "docs" / "V837_GRU_MECHANISM_LOCALIZATION_REPORT.md"
        self.assertTrue(report.exists())
        text = report.read_text(encoding="utf-8")
        self.assertIn("MECHANISM_REDUNDANCY_OR_COMPLEMENTARITY", text)
        self.assertIn("No V837o, V837p, V837q", text)


if __name__ == "__main__":
    unittest.main()
