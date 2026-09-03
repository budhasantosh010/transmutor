from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


class RepresentationRecoveryOutcomeTests(unittest.TestCase):
    def test_three_distinct_recovery_variants_failed_and_stop_rule_triggered(self):
        status = load("experiments/v837_primitive_invention/representation_recovery_status.json")
        self.assertEqual(status["scientifically_distinct_failed_variants"], ["V837d", "V837g", "V837h"])
        self.assertTrue(status["stop_rule_triggered"])
        self.assertEqual(status["neutral_substrate_competence"], "FAIL")
        for version in ("v837d", "v837g", "v837h"):
            self.assertFalse(load(f"experiments/v837_primitive_invention/{version}/results.json")["pass"])

    def test_fresh_audit_and_primitive_mining_remain_locked(self):
        status = load("experiments/v837_primitive_invention/representation_recovery_status.json")
        audit = load("experiments/v837_primitive_invention/audit/audit_results.json")
        self.assertEqual(status["fresh_audit_episodes_consumed"], 0)
        self.assertFalse(status["primitive_mining_allowed"])
        self.assertEqual(status["primitives_promoted"], 0)
        self.assertFalse(status["v838_started"])
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_original_v837_gate_and_historical_results_unchanged(self):
        expected = {
            "experiments/v837_primitive_invention/frozen_gates.json": "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867",
            "experiments/v837_primitive_invention/v837/results.json": "5fed69cc990be5c6f64a5229f59ff7f27af0c1fc26398bdfbe80ee46255eef14",
            "experiments/v837_primitive_invention/v837b/results.json": "f131110969e7700ec0cd9a82825e8554a51a9c05bb308d54625452db54e35cb0",
            "experiments/v837_primitive_invention/v837c/results.json": "994195fdd0e32e12ec44521ea782c1fc3561b8f596fd4a70e9d59f335fe7d009",
            "experiments/v837_primitive_invention/BLOCKER_ANALYSIS.md": "4eea85cbfb2fb9e379675765038527daf2ad6a49aa0721e861c9cc61b0155a20",
            "experiments/v837_primitive_invention/final_resource_accounting.json": "c712ea3c0771ebc398e4ccb80a4d0ffe0d8ead946d42fd460633577fbb3d9b37",
        }
        for path, digest in expected.items():
            self.assertEqual(sha256(path), digest, path)

    def test_v836_result_unchanged(self):
        self.assertEqual(
            sha256("archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json"),
            "0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d",
        )

    def test_v837h_parameter_control_is_exactly_matched(self):
        result = load("experiments/v837_primitive_invention/v837h/results.json")
        additive = result["resource_accounting"]["parameter_matched_additive"]["parameter_count"]
        multiplicative = result["resource_accounting"]["multiplicative"]["parameter_count"]
        self.assertEqual(additive, multiplicative)
        self.assertEqual(additive, 1096)

    def test_recovery_cost_remained_diagnostic_scale(self):
        resource = load("experiments/v837_primitive_invention/representation_recovery_resource_accounting.json")
        self.assertLess(resource["optimizer_step_fraction_of_original_v837"], 0.10)
        self.assertLess(resource["environment_interaction_fraction_of_original_v837"], 0.10)
        self.assertEqual(resource["candidate_evaluations"], 360)


if __name__ == "__main__":
    unittest.main()
