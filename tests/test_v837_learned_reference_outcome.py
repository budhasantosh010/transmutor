from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"


class LearnedReferenceOutcomeTests(unittest.TestCase):
    def load(self, relative: str) -> dict:
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))

    def test_v837j_matched_budget_remains_unresolved(self):
        data = self.load("experiments/v837_primitive_invention/v837j/results.json")
        self.assertFalse(data["pass"])
        self.assertEqual(data["diagnosis"], "BENCHMARK_LEARNABILITY_UNRESOLVED")
        self.assertEqual(data["models"]["gru_reference"]["families_passing"], 2)
        self.assertEqual(data["models"]["neutral_high_capacity"]["families_passing"], 1)

    def test_v837k_more_steps_do_not_rescue_reference(self):
        data = self.load("experiments/v837_primitive_invention/v837k/results.json")
        self.assertFalse(data["pass"])
        self.assertEqual(data["diagnosis"], "BENCHMARK_LEARNABILITY_UNRESOLVED")
        self.assertEqual(data["executed_multipliers"], [2, 4])
        for multiplier in ("1x", "2x", "4x"):
            self.assertEqual(data["conditions"][multiplier]["models"]["gru_reference"]["families_passing"], 2)

    def test_v837l_unique_data_establishes_learnability(self):
        data = self.load("experiments/v837_primitive_invention/v837l/results.json")
        self.assertTrue(data["pass"])
        self.assertEqual(data["diagnosis"], "SAMPLE_EFFICIENCY_FAILURE")
        self.assertEqual(data["resolved_at_data_multiplier"], 4)
        self.assertEqual(data["fixed_optimizer_steps"], 192)
        self.assertEqual(data["conditions"]["4x"]["gru_reference"]["families_passing"], 5)
        self.assertEqual(data["conditions"]["4x"]["neutral_high_capacity"]["families_passing"], 2)

    def test_reference_capacity_escalation_was_not_needed(self):
        status = self.load("experiments/v837_primitive_invention/learned_reference_calibration_status.json")
        self.assertFalse(status["capacity_escalation_run"])
        self.assertIn("875-parameter GRU", status["capacity_escalation_reason"])
        self.assertEqual(status["benchmark_learnability"], "ESTABLISHED_UNDER_4X_UNIQUE_DEVELOPMENT_DATA")

    def test_v837m_transport_is_exactly_parameter_matched_and_fails(self):
        data = self.load("experiments/v837_primitive_invention/v837m/results.json")
        self.assertFalse(data["pass"])
        self.assertEqual(data["diagnosis"], "LINEAR_STATE_TRANSPORT_INSUFFICIENT")
        self.assertTrue(data["parameter_matching"]["exact_match"])
        self.assertEqual(data["parameter_matching"]["linear_transport"], 1016)
        self.assertEqual(data["parameter_matching"]["parameter_matched_additive"], 1016)
        self.assertEqual(data["conditions"]["linear_transport"]["families_passing"], 2)
        self.assertFalse(data["full_structural_search_allowed"])

    def test_downstream_science_remains_locked(self):
        status = self.load("experiments/v837_primitive_invention/learned_reference_calibration_status.json")
        self.assertEqual(status["fresh_audit_episodes_consumed"], 0)
        self.assertEqual(status["primitives_promoted"], 0)
        self.assertFalse(status["primitive_mining_allowed"])
        self.assertFalse(status["v838_started"])

    def test_final_report_and_live_manifest_include_calibration(self):
        report = ROOT / "docs" / "V837_LEARNED_REFERENCE_CALIBRATION_REPORT.md"
        self.assertTrue(report.is_file())
        self.assertIn("SAMPLE_EFFICIENCY_FAILURE", report.read_text(encoding="utf-8"))
        manifest = self.load("verification/live_repo_manifest.json")
        for version in ("V837j", "V837k", "V837l", "V837m"):
            self.assertIn(version, manifest["current_variants"])
        self.assertIn("docs/V837_LEARNED_REFERENCE_CALIBRATION_REPORT.md", manifest["reports"])

    def test_reproduction_dispatcher_contains_all_calibration_variants(self):
        source = (ROOT / "scripts" / "reproduce_v837_recovery.py").read_text(encoding="utf-8")
        for version in ("v837j", "v837k", "v837l", "v837m"):
            self.assertIn(f'"{version}"', source)


if __name__ == "__main__":
    unittest.main()
