from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from experiments.v837_primitive_invention.common.reference_models import build_reference_model
from experiments.v837_primitive_invention.v837k import run_training_budget_diagnostic as v837k

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
CONFIG = json.loads((BASE / "v837k" / "config.json").read_text(encoding="utf-8"))
J_CONFIG = json.loads((BASE / "v837j" / "config.json").read_text(encoding="utf-8"))


class V837kTrainingBudgetTests(unittest.TestCase):
    def test_single_change_is_optimizer_steps_only(self):
        self.assertEqual(CONFIG["step_multipliers"], [1, 2, 4])
        self.assertIn("optimizer step budget only", CONFIG["single_change"])
        self.assertEqual(CONFIG["base_training"], J_CONFIG["primary_training"])

    def test_reference_architectures_and_hidden_sizes_frozen_from_v837j(self):
        for name in ("gru_reference", "residual_rnn_reference"):
            self.assertEqual(CONFIG["hidden_size_selection"][name], J_CONFIG["reference_hidden_size_selection"][name])
            model = build_reference_model(name, int(CONFIG["hidden_size_selection"][name]["hidden_size"]), 6)
            self.assertEqual(model.parameter_count(), int(CONFIG["hidden_size_selection"][name]["parameter_count"]))

    def test_same_initialization_seed_namespace_as_v837j(self):
        source = inspect.getsource(v837k._worker)
        self.assertIn('deterministic_int("v837j-primary-init", family, replicate)', source)

    def test_same_training_and_validation_episode_ranges(self):
        self.assertEqual(CONFIG["base_training"]["development_seed_range"], [10000, 10127])
        self.assertEqual(CONFIG["base_training"]["validation_seed_range"], [20000, 20127])
        source = inspect.getsource(v837k._worker)
        self.assertIn('training["development_seed_range"]', source)
        self.assertIn('training["validation_seed_range"]', source)

    def test_fresh_audit_and_primitive_mining_locked(self):
        self.assertIs(CONFIG["fresh_audit_allowed"], False)
        self.assertIs(CONFIG["fresh_audit_consumed"], False)
        self.assertIs(CONFIG["primitive_mining_allowed"], False)

    def test_four_x_runs_only_if_two_x_does_not_resolve_reference(self):
        source = inspect.getsource(v837k.main)
        self.assertIn("if _learned_pass(summary):", source)
        self.assertIn("for multiplier in (2, 4)", source)


if __name__ == "__main__":
    unittest.main()
