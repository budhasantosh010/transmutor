from __future__ import annotations

import inspect
import json
import subprocess
import unittest
from pathlib import Path

from experiments.v837_primitive_invention.common.gates import v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_models import (
    GRUReferenceModel,
    ResidualRecurrentMLPReferenceModel,
    VanillaRNNReferenceModel,
    select_hidden_size_for_parameter_target,
)
from experiments.v837_primitive_invention.common.reference_training import matched_budget_signature, train_sequence_model
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import task_by_name

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention" / "v837j"
CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
START_SHA = "3d5864f231e6cbd8be7703ef2ab90ef9fa6977e1"


class ReferenceModelTests(unittest.TestCase):
    def test_reference_models_receive_no_task_label(self):
        for cls in (GRUReferenceModel, ResidualRecurrentMLPReferenceModel, VanillaRNNReferenceModel):
            signature = inspect.signature(cls.forward)
            self.assertEqual(set(signature.parameters), {"self", "observations", "lengths", "return_trace"})

    def test_parameter_count_reported_and_within_match_tolerance(self):
        for architecture, record in CONFIG["reference_hidden_size_selection"].items():
            again = select_hidden_size_for_parameter_target(
                input_dim=6,
                target_parameter_count=CONFIG["parameter_target"],
                architecture_type=architecture,
            )
            self.assertEqual(again, record)
            self.assertLessEqual(abs(record["percent_difference"]), 10.0)

    def test_gru_hidden_size_selection_deterministic(self):
        left = select_hidden_size_for_parameter_target(input_dim=6, target_parameter_count=856, architecture_type="gru_reference")
        right = select_hidden_size_for_parameter_target(input_dim=6, target_parameter_count=856, architecture_type="gru_reference")
        self.assertEqual(left, right)
        self.assertEqual(left["hidden_size"], 13)
        self.assertEqual(left["parameter_count"], 875)


class MatchingAndGuardTests(unittest.TestCase):
    def test_reference_and_neutral_use_same_task_seeds(self):
        training = CONFIG["primary_training"]
        self.assertEqual(training["development_seed_range"], [10000, 10127])
        self.assertEqual(training["validation_seed_range"], [20000, 20127])
        self.assertEqual(training["train_episodes"], 128)
        self.assertEqual(training["validation_episodes"], 128)

    def test_matched_optimizer_steps_and_examples_processed(self):
        training = CONFIG["primary_training"]
        signature = matched_budget_signature(
            optimizer=training["optimizer"], optimizer_steps=training["steps"],
            train_episodes=training["train_episodes"], validation_episodes=training["validation_episodes"],
            learning_rate=training["learning_rate"], weight_decay=training["weight_decay"],
            gradient_clip=training["gradient_clip"],
        )
        self.assertEqual(signature["optimizer_steps"], 192)
        self.assertEqual(signature["examples_processed"], 192 * 128)

    def test_training_equivalence_small_probe(self):
        task = task_by_name("iterative_state")
        train_seeds = [10000, 10001]
        validation_seeds = [20000, 20001]
        factories = [
            lambda: NeutralGraphModel(high_capacity_generic_graph(0), obs_dim=6, state_dim=4, message_dim=4),
            lambda: GRUReferenceModel(13),
            lambda: ResidualRecurrentMLPReferenceModel(26),
        ]
        records = []
        for factory in factories:
            result = train_sequence_model(
                model_factory=factory, task=task, train_seeds=train_seeds, validation_seeds=validation_seeds,
                initialization_seed=1234, steps=2, learning_rate=0.005, weight_decay=0.0001,
                gradient_clip=5.0, curve_steps=(0, 2),
            )
            records.append((result.resources.optimizer_steps, result.resources.examples_processed, result.resources.environment_steps))
        self.assertEqual(len(set(records)), 1)
        self.assertEqual(records[0][0], 2)
        self.assertEqual(records[0][1], 4)

    def test_fresh_audit_not_consumed_and_primitive_mining_blocked(self):
        self.assertFalse(CONFIG["fresh_audit_allowed"])
        self.assertFalse(CONFIG["reference_calibration_fresh_audit_consumed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])

    def test_historical_v837_gate_unchanged(self):
        self.assertEqual(CONFIG["historical_gate_hash"], "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867")
        self.assertEqual(CONFIG["capacity_criterion_hash"], v837_capacity_criterion_sha256())

    def test_historical_results_unchanged_from_calibration_start(self):
        paths = [
            "archive", "registry",
            "experiments/v837_primitive_invention/v837",
            "experiments/v837_primitive_invention/v837b",
            "experiments/v837_primitive_invention/v837c",
            "experiments/v837_primitive_invention/v837d",
            "experiments/v837_primitive_invention/v837g",
            "experiments/v837_primitive_invention/v837h",
            "experiments/v837_primitive_invention/frozen_gates.json",
        ]
        completed = subprocess.run(["git", "diff", "--quiet", START_SHA, "--", *paths], cwd=ROOT)
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
