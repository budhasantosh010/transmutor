from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from experiments.v837_primitive_invention.v837l import run_data_diagnostic as v837l

ROOT = Path(__file__).resolve().parents[1]


class V837lDataBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "experiments/v837_primitive_invention/v837l/config.json").read_text(encoding="utf-8"))

    def test_single_change_is_unique_data_only(self) -> None:
        self.assertEqual(self.config["data_multipliers"], [1, 2, 4])
        self.assertEqual(self.config["base_training"]["steps"], 192)
        self.assertIn("unique development episodes only", self.config["single_change"])

    def test_same_validation_episodes(self) -> None:
        self.assertEqual(self.config["base_training"]["validation_seed_range"], [20000, 20127])
        self.assertEqual(self.config["base_training"]["validation_episodes"], 128)

    def test_data_ranges_stay_inside_v837_development_region(self) -> None:
        start = self.config["base_training"]["development_seed_start"]
        largest = start + self.config["base_training"]["train_episodes"] * 4 - 1
        self.assertGreaterEqual(start, 10000)
        self.assertLessEqual(largest, 10999)

    def test_same_initialization_seed_namespace_as_v837j(self) -> None:
        source = inspect.getsource(v837l._worker)
        self.assertIn('deterministic_int("v837j-primary-init", family, replicate)', source)

    def test_reference_architectures_frozen(self) -> None:
        j = json.loads((ROOT / "experiments/v837_primitive_invention/v837j/config.json").read_text(encoding="utf-8"))
        for model in ("gru_reference", "residual_rnn_reference"):
            self.assertEqual(self.config["hidden_size_selection"][model]["hidden_size"], j["reference_hidden_size_selection"][model]["hidden_size"])
            self.assertEqual(self.config["hidden_size_selection"][model]["parameter_count"], j["reference_hidden_size_selection"][model]["parameter_count"])

    def test_fresh_audit_and_primitive_mining_locked(self) -> None:
        self.assertFalse(self.config["fresh_audit_allowed"])
        self.assertFalse(self.config["fresh_audit_consumed"])
        self.assertFalse(self.config["primitive_mining_allowed"])

    def test_four_x_runs_only_if_two_x_not_resolved(self) -> None:
        source = inspect.getsource(v837l.main)
        self.assertIn("if _learned_resolved(summary):", source)
        self.assertIn("break", source)


if __name__ == "__main__":
    unittest.main()
